"""Build the ``aic2026`` collection directory (``docs/ARCHITECTURE.md`` §3) from the AIC data folder.

Nothing large is copied: ``collection.json`` points at the existing embedding matrix and keyframe
folder in ``data/`` by absolute path. Written files:

* ``metadata.json``    the contract fields of ``data/metadata.json``, row for row;
* ``videos.json``      fps, ``n_frames`` = last keyframe index + 1 and ``duration`` = ``n_frames / fps``
                       from the metadata, i.e. the timeline the keyframes were cut from (the YouTube
                       length in the media-info archive ends between 0.3 s before and 8.4 s after
                       the last keyframe, so it is not used); title and YouTube id from
                       ``media-info-aic25-b1.zip``;
* ``invalid_rows.json`` the first ``--skip-first-n`` keyframes of every video (channel bumpers and
                       black frames, never an answer in AIC) plus ``blank_frame_indices.json``;
* ``collection.json``.

Usage (from the repository root)::

    python -m scenechain.ingest.build_from_aic --data data --out collections/aic2026
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

MODEL = "google/siglip2-so400m-patch14-384"
EMBEDDINGS = "embeddings_siglip2_384.npy"
MEDIA_INFO = "media-info-aic25-b1.zip"
BLANK_FRAMES = "blank_frame_indices.json"
META_KEYS = ("video_id", "n", "frame_idx", "pts_time", "fps", "frame_filename")
_YOUTUBE_ID = re.compile(r"(?:v=|youtu\.be/|/embed/)([A-Za-z0-9_-]{11})")


def load_media_info(zip_path: Path, video_ids: set[str]) -> dict[str, dict[str, Any]]:
    """``{video_id: {"title", "youtube_id"}}`` from the organisers' media-info archive."""
    info: dict[str, dict[str, Any]] = {}
    with zipfile.ZipFile(zip_path) as zf:
        for name in sorted(zf.namelist()):
            vid = Path(name).stem
            if not name.endswith(".json") or vid not in video_ids:
                continue
            entry = json.loads(zf.read(name))
            match = _YOUTUBE_ID.search(str(entry.get("watch_url") or ""))
            info[vid] = {
                "title": str(entry.get("title") or vid),
                "youtube_id": match.group(1) if match else None,
            }
    return info


def build(data_dir: Path, out_dir: Path, name: str = "aic2026", skip_first_n: int = 2) -> dict[str, Any]:
    """Write the collection files into ``out_dir`` and return a summary."""
    data_dir, out_dir = data_dir.resolve(), out_dir.resolve()
    emb_path = data_dir / EMBEDDINGS
    frames_dir = data_dir / "frames"
    for required in (emb_path, data_dir / "metadata.json", frames_dir):
        if not required.exists():
            raise FileNotFoundError(f"{required} is missing")
    n_rows, dim = np.load(emb_path, mmap_mode="r").shape

    source = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))
    if len(source) != n_rows:
        raise ValueError(f"{emb_path} has {n_rows} rows but metadata.json has {len(source)}")
    metadata = []
    for i, m in enumerate(source):
        missing = [k for k in META_KEYS if k not in m]
        if missing:
            raise ValueError(f"metadata row {i} lacks {missing}")
        metadata.append({k: m[k] for k in META_KEYS})

    videos: dict[str, dict[str, Any]] = {}
    for m in metadata:
        v = videos.setdefault(m["video_id"], {"fps": float(m["fps"]), "last_frame": -1})
        v["last_frame"] = max(v["last_frame"], int(m["frame_idx"]))

    media_zip = data_dir / MEDIA_INFO
    media = load_media_info(media_zip, set(videos)) if media_zip.is_file() else {}
    table = {}
    for vid, v in videos.items():
        n_frames = v["last_frame"] + 1
        table[vid] = {
            "fps": v["fps"],
            "duration": round(n_frames / v["fps"], 3),
            "n_frames": n_frames,
            "title": media.get(vid, {}).get("title", vid),
            "youtube_id": media.get(vid, {}).get("youtube_id"),
            "file": None,
        }

    invalid = {i for i, m in enumerate(metadata) if int(m["n"]) <= skip_first_n}
    blank_path = data_dir / BLANK_FRAMES
    n_blank = 0
    if blank_path.is_file():
        raw = json.loads(blank_path.read_text(encoding="utf-8"))
        blank = raw if isinstance(raw, list) else raw.get("blank_frame_indices", [])
        outside = [int(i) for i in blank if not 0 <= int(i) < n_rows]
        if outside:
            raise ValueError(f"{blank_path} has indices outside [0, {n_rows}): {outside[:5]}")
        n_blank = len(blank)
        invalid.update(int(i) for i in blank)

    out_dir.mkdir(parents=True, exist_ok=True)

    def write(filename: str, obj: Any, indent: int | None = None) -> None:
        text = json.dumps(obj, ensure_ascii=False, indent=indent,
                          separators=None if indent else (",", ":"))
        (out_dir / filename).write_text(text + "\n", encoding="utf-8", newline="\n")

    write("metadata.json", metadata)
    write("videos.json", table)
    write("invalid_rows.json", sorted(invalid))
    write("collection.json", {
        "name": name,
        "model": MODEL,
        "dim": int(dim),
        "embeddings": emb_path.as_posix(),
        "metadata": "metadata.json",
        "frames_dir": frames_dir.as_posix(),
        "videos": "videos.json",
        "invalid_rows": "invalid_rows.json",
    }, indent=2)
    return {
        "out": out_dir.as_posix(),
        "rows": n_rows,
        "dim": int(dim),
        "videos": len(table),
        "invalid_rows": len(invalid),
        "blank_frames": n_blank,
        "youtube_ids": sum(1 for v in table.values() if v["youtube_id"]),
        "media_info": media_zip.is_file(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", type=Path, default=Path("data"), help="AIC data folder")
    parser.add_argument("--out", type=Path, default=Path("collections/aic2026"),
                        help="collection directory to write")
    parser.add_argument("--name", default="aic2026", help="collection name (sent to DRES)")
    parser.add_argument("--skip-first-n", type=int, default=2,
                        help="mark the first N keyframes of every video invalid")
    args = parser.parse_args(argv)
    summary = build(args.data, args.out, args.name, args.skip_first_n)
    if not summary["media_info"]:
        print(f"warning: {MEDIA_INFO} not found; titles fall back to video ids", file=sys.stderr)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
