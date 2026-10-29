"""Download three short, freely licensed videos for the end-to-end ingestion test (ARCHITECTURE.md §8).

The clips are Blender Foundation open movies hosted on Wikimedia Commons, all under Creative
Commons Attribution licences. For each file the Commons API is asked for the 480p VP9 WebM
transcode, the licence and the author, so ``SOURCES.md`` is written from what Commons reports at
download time rather than from a hand-typed list. Media land in ``samples/`` (gitignored);
nothing is committed. Files already present with the expected size are not fetched again.

Usage::

    python -m scenechain.ingest.download_sample [--out samples]
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "scenechain-ingest-sample/1.0 (Video Browser Showdown research prototype; python-urllib)"
TRANSCODE_KEY = "480p.vp9.webm"
MAX_TOTAL_BYTES = 150 * 1024 * 1024
DEFAULT_OUT = Path(__file__).resolve().parents[2] / "samples"

# (local file stem, Commons file title)
SAMPLES: tuple[tuple[str, str], ...] = (
    ("sintel_trailer", "File:Sintel trailer-1080p.ogv"),
    ("caminandes_llama_drama", "File:Caminandes- Llama Drama - Short Movie.ogv"),
    ("charge_open_movie", "File:Charge - Blender Open Movie-full movie.webm"),
)


class DownloadError(RuntimeError):
    """Commons did not return what the sample list expects."""


@dataclass(frozen=True)
class Sample:
    stem: str
    title: str
    page_url: str
    download_url: str
    size: int
    duration: float
    licence: str
    licence_url: str
    author: str

    @property
    def filename(self) -> str:
        return f"{self.stem}.webm"


def _get_json(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def _plain(value: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", value)).strip()


def resolve_samples() -> list[Sample]:
    """Ask the Commons API for transcode URL, size, licence and author of every sample."""
    query = {
        "action": "query",
        "format": "json",
        "prop": "videoinfo",
        "viprop": "url|size|derivatives|extmetadata",
        "titles": "|".join(title for _, title in SAMPLES),
    }
    pages = _get_json(f"{API}?{urllib.parse.urlencode(query)}")["query"]["pages"].values()
    by_title = {page["title"]: page for page in pages}
    out: list[Sample] = []
    for stem, title in SAMPLES:
        page = by_title.get(title)
        if page is None or "videoinfo" not in page:
            raise DownloadError(f"Commons has no video info for {title!r}")
        info = page["videoinfo"][0]
        meta = info.get("extmetadata", {})
        webm = [d for d in info.get("derivatives", []) if d.get("transcodekey") == TRANSCODE_KEY]
        if not webm:
            raise DownloadError(f"no {TRANSCODE_KEY} transcode listed for {title!r}")
        licence = _plain(meta.get("LicenseShortName", {}).get("value", ""))
        if not licence.upper().startswith(("CC BY", "CC0", "PUBLIC DOMAIN")):
            raise DownloadError(f"{title!r} is not under a free licence: {licence!r}")
        request = urllib.request.Request(webm[0]["src"], method="HEAD", headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=60) as response:
            size = int(response.headers["Content-Length"])
        out.append(Sample(
            stem=stem,
            title=title,
            page_url=info["descriptionurl"],
            download_url=webm[0]["src"],
            size=size,
            duration=float(info.get("duration") or 0.0),
            licence=licence,
            licence_url=_plain(meta.get("LicenseUrl", {}).get("value", "")),
            author=_plain(meta.get("Artist", {}).get("value", "")),
        ))
    total = sum(s.size for s in out)
    if total > MAX_TOTAL_BYTES:
        raise DownloadError(f"samples total {total / 2**20:.1f} MB, above the {MAX_TOTAL_BYTES / 2**20:.0f} MB budget")
    return out


def download(sample: Sample, out_dir: Path) -> bool:
    """Fetch one file unless it is already complete; return True if bytes were downloaded."""
    target = out_dir / sample.filename
    if target.is_file() and target.stat().st_size == sample.size:
        return False
    tmp = target.with_name(target.name + ".part")
    request = urllib.request.Request(sample.download_url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as response, tmp.open("wb") as fh:
        while chunk := response.read(1 << 20):
            fh.write(chunk)
    if tmp.stat().st_size != sample.size:
        raise DownloadError(f"{sample.filename}: got {tmp.stat().st_size} bytes, expected {sample.size}")
    tmp.replace(target)
    return True


def sources_markdown(samples: Sequence[Sample]) -> str:
    lines = [
        "# Sample videos",
        "",
        "Downloaded by `scenechain/ingest/download_sample.py` for the end-to-end ingestion test. The media files",
        "are not part of the repository. Licence and author are copied from the Wikimedia Commons API at",
        "download time; the files are unmodified 480p WebM transcodes served by Commons.",
        "",
        "| File | Source | Duration | Size | Licence | Author |",
        "|---|---|---|---|---|---|",
    ]
    for s in samples:
        licence = f"[{s.licence}]({s.licence_url})" if s.licence_url else s.licence
        lines.append(
            f"| `{s.filename}` | [{s.title}]({s.page_url}) ([file]({s.download_url})) | {s.duration:.0f} s "
            f"| {s.size / 2**20:.1f} MB | {licence} | {s.author.replace('|', '/')} |"
        )
    lines += ["", f"Total: {sum(s.size for s in samples) / 2**20:.1f} MB.", ""]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help=f"target directory (default {DEFAULT_OUT})")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    try:
        samples = resolve_samples()
        for s in samples:
            fetched = download(s, args.out)
            print(f"{s.filename}: {s.size / 2**20:.1f} MB, {s.duration:.0f} s, {s.licence} "
                  f"({'downloaded' if fetched else 'already present'})", flush=True)
    except (DownloadError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    (args.out / "SOURCES.md").write_bytes(sources_markdown(samples).encode("utf-8"))
    print(f"wrote {args.out / 'SOURCES.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
