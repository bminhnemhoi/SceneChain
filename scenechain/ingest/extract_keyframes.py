"""Videos -> shots -> keyframes: the first stage of dataset-agnostic ingestion (ARCHITECTURE.md §3, §8).

For every video file under ``--videos`` the script

1. finds shot boundaries with one of four methods:

   * ``scenedetect`` - PySceneDetect ``ContentDetector`` (the default of ``auto`` when importable);
   * ``histogram``   - built-in detector: a cut is declared where BOTH the Bhattacharyya distance
     between HSV colour histograms of consecutive frames AND the mean absolute HSV pixel difference
     exceed their thresholds (the histogram test ignores motion, the pixel test ignores global
     colour-preserving changes; requiring both keeps false cuts low). Hard cuts only - dissolves
     and fades are not modelled;
   * ``shots``       - external boundaries, one ``<video_id>.csv`` per video in ``--shots-dir``;
   * ``uniform``     - no shots, one keyframe every ``1 / --fps`` seconds of video time;

2. picks one to three keyframes per shot: the middle frame, plus frames ``--edge`` seconds inside
   the start and the end of shots longer than ``--long-shot`` seconds (and, with ``--max-gap``,
   extra frames so that consecutive keyframes of a shot are never further apart than that);

3. decodes the video once more and writes the keyframes as JPEG (long side ``--long-side``) to
   ``<out>/frames/<video_id>/<n>.jpg`` plus a per-video record ``<out>/ingest/<video_id>.json``;
   with ``--encode-long-side N`` a second copy (long side N, JPEG quality ``--encode-quality``) goes
   to ``<out>/frames_encode/`` and is registered as ``encode_frames_dir`` for ``encode_images.py``.
   SigLIP-2 embeddings of 512 px JPEGs differ markedly from those of full-resolution frames
   (measured mean cosine 0.92 on 64 AIC keyframes, 0.997 for 1024 px at quality 95), so use the
   larger copy whenever disk allows;

4. assembles ``metadata.json``, ``videos.json`` and the non-embedding fields of
   ``collection.json`` from the per-video records, in sorted ``video_id`` order.

Frame numbers are 0-based indices in OpenCV's decode order, for detection and extraction alike
(PySceneDetect is fed the same decoded frames, so its cuts use the same numbering). ``pts_time``
is the decoder timestamp of the frame, which stays correct for variable-frame-rate files where
``frame_idx / fps`` drifts; shot lengths for keyframe selection use the nominal frame rate.

Resumable per video: a video whose record exists and was produced with the same parameters from
an input of the same size (and the same shot file) is skipped. A video interrupted half-way has no
record, so its partial frames are deleted and it is extracted again.

External shot files hold one shot per line as ``start_frame,end_frame`` (0-based, both ends
inclusive, sorted, non-overlapping); blank lines and lines starting with ``#`` are ignored.

Converting V3C master shot boundaries: the V3C releases ship one tab-separated file per video in
their ``msb`` archive, with a header row and the start/end frame numbers and times of every segment.
Look at the header of one file, then keep the two frame columns::

    import csv, pathlib
    for tsv in sorted(pathlib.Path("V3C1/msb").glob("*.tsv")):
        with tsv.open(newline="") as f:
            rows = list(csv.DictReader(f, delimiter="\\t"))
        start, end = "startframe", "endframe"          # use the column names found in the header
        lines = [f"{int(float(r[start]))},{int(float(r[end]))}" for r in rows]
        pathlib.Path("shots", tsv.stem + ".csv").write_text("\\n".join(lines) + "\\n")

An exclusive end column (next start == previous end) is rejected here as overlapping; subtract 1
from the end in that case. V3C frame numbers assume constant frame rate, which holds for the
decode-order numbering used here whenever the file has no dropped or duplicated frames.

Usage::

    python -m scenechain.ingest.extract_keyframes --videos samples --out collections/samples
    python -m scenechain.ingest.extract_keyframes --videos V3C1/videos --out collections/v3c1 \\
        --method shots --shots-dir shots
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterator, Protocol, Sequence

import cv2
import numpy as np

VIDEO_EXTENSIONS = frozenset({".mp4", ".m4v", ".mkv", ".webm", ".mov", ".avi", ".mpg", ".mpeg", ".ogv", ".ts"})
METHODS = ("auto", "scenedetect", "histogram", "shots", "uniform")
RECORD_VERSION = 2
FILENAME_DIGITS = 5
FRAMES_DIR = "frames"
# video ids become URL path segments: the server accepts only these (ARCHITECTURE.md §3)
VIDEO_ID_PATTERN = re.compile(r"[A-Za-z0-9_-][A-Za-z0-9._-]*")
ENCODE_FRAMES_DIR = "frames_encode"


class IngestError(RuntimeError):
    """An input problem the user has to fix (unreadable video, bad parameters, bad shot list)."""


@dataclass(frozen=True)
class Shot:
    """A shot as a closed interval of 0-based decode-order frame indices."""

    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start + 1


@dataclass(frozen=True)
class Params:
    """Everything that influences the output for one video; any change forces re-extraction."""

    method: str
    fps: float = 1.0
    threshold: float = 27.0
    hist_threshold: float = 0.25
    pixel_threshold: float = 20.0
    min_shot: float = 0.5
    long_shot: float = 4.0
    edge: float = 0.5
    max_gap: float = 0.0
    long_side: int = 512
    quality: int = 90
    encode_long_side: int = 0
    encode_quality: int = 95


# --------------------------------------------------------------------------- video access


def open_capture(path: Path) -> cv2.VideoCapture:
    """Open ``path`` with OpenCV or raise ``IngestError``."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IngestError(f"OpenCV cannot open video: {path}")
    return cap


def nominal_fps(path: Path) -> float:
    """The container frame rate; raises if the file does not declare a usable one."""
    cap = open_capture(path)
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    finally:
        cap.release()
    if not math.isfinite(fps) or fps <= 0.0:
        raise IngestError(f"video reports no frame rate: {path}")
    return fps


def iter_frames(
    path: Path, fps: float, select: Callable[[int, float], bool] | None = None
) -> Iterator[tuple[int, float, np.ndarray | None]]:
    """Yield ``(frame_index, time_s, BGR image or None)`` for every decodable frame in decode order.

    With ``select``, only frames for which ``select(index, time_s)`` is true are converted to an
    image (the others are decoded but yield None), which saves the colour conversion.
    ``time_s`` is the decoder timestamp; if a backend reports a non-increasing timestamp, the
    previous timestamp plus one nominal frame duration is used instead, so times stay monotone.
    """
    cap = open_capture(path)
    try:
        index = 0
        previous = -math.inf
        while cap.grab():
            t = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0) / 1000.0
            if not math.isfinite(t) or t <= previous:
                t = previous + 1.0 / fps if index else 0.0
            previous = t
            frame = None
            if select is None or select(index, t):
                ok, frame = cap.retrieve()
                if not ok:
                    raise IngestError(f"frame {index} of {path} was decoded but could not be retrieved")
            yield index, t, frame
            index += 1
    finally:
        cap.release()


def discover_videos(videos_dir: Path) -> dict[str, Path]:
    """Map ``video_id`` (file stem) -> path for every video file below ``videos_dir``, sorted by id."""
    if not videos_dir.is_dir():
        raise IngestError(f"--videos is not a directory: {videos_dir}")
    found: dict[str, Path] = {}
    for path in sorted(videos_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        if not VIDEO_ID_PATTERN.fullmatch(path.stem):
            raise IngestError(f"video id {path.stem!r} (file {path}) may only contain letters, digits, '.', '_' "
                              "and '-' and must not start with '.'; rename the file")
        if path.stem in found:
            raise IngestError(f"two videos share the id {path.stem!r}: {found[path.stem]} and {path}")
        found[path.stem] = path
    if not found:
        raise IngestError(f"no video files ({', '.join(sorted(VIDEO_EXTENSIONS))}) under {videos_dir}")
    return dict(sorted(found.items()))


# --------------------------------------------------------------------------- shot boundaries


class CutDetector(Protocol):
    """Streaming hard-cut detector over decode-order frames."""

    def process(self, index: int, frame: np.ndarray) -> list[int]:
        """Feed frame ``index``; return indices of shots that start at a newly confirmed cut."""
        ...

    def finish(self, last_index: int) -> list[int]:
        """Flush cuts still pending after the last frame."""
        ...


class HistogramCutDetector:
    """HSV histogram + frame difference detector (see module docstring).

    ``hist_threshold`` is a Bhattacharyya distance in [0, 1]; ``pixel_threshold`` is the mean
    absolute HSV difference on the 0-255 scale; ``min_frames`` suppresses a cut closer than that to
    the previous one.
    """

    def __init__(self, hist_threshold: float, pixel_threshold: float, min_frames: int, width: int = 128) -> None:
        self.hist_threshold = hist_threshold
        self.pixel_threshold = pixel_threshold
        self.min_frames = max(1, min_frames)
        self.width = width
        self._previous: tuple[np.ndarray, np.ndarray] | None = None
        self._last_cut = 0

    def _signature(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (self.width, max(1, round(h * self.width / w))), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [16, 4, 4], [0, 180, 0, 256, 0, 256])
        return (hist / max(float(hist.sum()), 1.0)).astype(np.float32), hsv.astype(np.int16)

    def process(self, index: int, frame: np.ndarray) -> list[int]:
        hist, hsv = self._signature(frame)
        previous, self._previous = self._previous, (hist, hsv)
        if previous is None or index - self._last_cut < self.min_frames:
            return []
        prev_hist, prev_hsv = previous
        d_hist = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
        d_pix = float(np.abs(hsv - prev_hsv).mean()) if hsv.shape == prev_hsv.shape else 255.0
        if d_hist >= self.hist_threshold and d_pix >= self.pixel_threshold:
            self._last_cut = index
            return [index]
        return []

    def finish(self, last_index: int) -> list[int]:
        return []


class ContentCutDetector:
    """PySceneDetect ``ContentDetector`` fed with our decoded frames.

    Frames are downscaled exactly as ``SceneManager`` does by default (effective width >= 256 px,
    linear interpolation), so thresholds mean the same as on the PySceneDetect command line, while
    the cut indices use this module's decode-order numbering.
    """

    def __init__(self, fps: float, threshold: float, min_frames: int) -> None:
        from scenedetect import ContentDetector, FrameTimecode
        from scenedetect.scene_manager import compute_downscale_factor

        self._timecode = FrameTimecode
        self._downscale = compute_downscale_factor
        self._detector = ContentDetector(threshold=threshold, min_scene_len=max(1, min_frames))
        self._fps = fps
        self._factor: float | None = None

    def process(self, index: int, frame: np.ndarray) -> list[int]:
        if self._factor is None:
            self._factor = float(self._downscale(frame.shape[1]))
        if self._factor > 1.0:
            size = (max(1, round(frame.shape[1] / self._factor)), max(1, round(frame.shape[0] / self._factor)))
            frame = cv2.resize(frame, size, interpolation=cv2.INTER_LINEAR)
        cuts = self._detector.process_frame(self._timecode(index, self._fps), frame)
        return [int(c.frame_num) for c in cuts]

    def finish(self, last_index: int) -> list[int]:
        return [int(c.frame_num) for c in self._detector.post_process(self._timecode(last_index, self._fps))]


def scenedetect_available() -> bool:
    try:
        import scenedetect  # noqa: F401
    except ImportError:
        return False
    return True


def detect_cuts(path: Path, fps: float, detector: CutDetector) -> tuple[list[int], int]:
    """Run ``detector`` over the whole video; return ``(sorted unique cut indices, frames decoded)``."""
    cuts: list[int] = []
    n_frames = 0
    for index, _, frame in iter_frames(path, fps):
        assert frame is not None
        cuts.extend(detector.process(index, frame))
        n_frames = index + 1
    if n_frames == 0:
        raise IngestError(f"no decodable frames in {path}")
    cuts.extend(detector.finish(n_frames - 1))
    return sorted({c for c in cuts if 0 < c < n_frames}), n_frames


def cuts_to_shots(cuts: Sequence[int], n_frames: int) -> list[Shot]:
    """Turn sorted first-frame-of-shot indices into consecutive shots covering ``[0, n_frames)``."""
    starts = [0] + [c for c in cuts if 0 < c < n_frames]
    ends = [s - 1 for s in starts[1:]] + [n_frames - 1]
    return [Shot(s, e) for s, e in zip(starts, ends)]


def read_shot_file(path: Path) -> list[Shot]:
    """Parse and validate a ``start_frame,end_frame`` file (see module docstring)."""
    if not path.is_file():
        raise IngestError(f"shot boundary file not found: {path}")
    shots: list[Shot] = []
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 2 or not all(p.isdigit() for p in parts):
            raise IngestError(f"{path}:{lineno}: expected 'start_frame,end_frame' (non-negative integers), got {raw!r}")
        start, end = int(parts[0]), int(parts[1])
        if end < start:
            raise IngestError(f"{path}:{lineno}: shot ends before it starts: {start},{end}")
        if shots and start <= shots[-1].end:
            raise IngestError(
                f"{path}:{lineno}: shot {start},{end} overlaps the previous shot ending at {shots[-1].end} "
                "(end frames are inclusive; subtract 1 if the source uses exclusive ends)"
            )
        shots.append(Shot(start, end))
    if not shots:
        raise IngestError(f"no shots in {path}")
    return shots


# --------------------------------------------------------------------------- keyframe selection


def shot_keyframes(shots: Sequence[Shot], fps: float, *, long_shot: float, edge: float, max_gap: float) -> list[int]:
    """Frame indices of the keyframes of ``shots`` (sorted, unique).

    Middle frame of every shot; for shots longer than ``long_shot`` seconds also the frames
    ``edge`` seconds inside both ends; if ``max_gap`` > 0, evenly spaced extra frames so that
    consecutive keyframes of a shot are at most ``max_gap`` seconds apart.
    """
    selected: set[int] = set()
    limit = max(1, round(max_gap * fps)) if max_gap > 0 else 0
    for shot in shots:
        middle = shot.start + (shot.length - 1) // 2
        picks = [middle]
        if shot.length / fps > long_shot:
            inset = min(round(edge * fps), (shot.length - 1) // 4)
            picks = [shot.start + inset, middle, shot.end - inset]
        if limit:
            filled = [picks[0]]
            for nxt in picks[1:]:
                base, gap = filled[-1], nxt - filled[-1]
                extra = math.ceil(gap / limit) - 1
                filled.extend(base + round(j * gap / (extra + 1)) for j in range(1, extra + 1))
                filled.append(nxt)
            picks = filled
        selected.update(picks)
    return sorted(selected)


class FrameSelector(Protocol):
    def __call__(self, index: int, time_s: float) -> bool: ...


class IndexSelector:
    """Select a fixed set of frame indices."""

    def __init__(self, indices: Sequence[int]) -> None:
        self.indices = frozenset(indices)

    def __call__(self, index: int, time_s: float) -> bool:
        return index in self.indices


class UniformSelector:
    """Select the first frame at or after every multiple of ``1 / sample_fps`` seconds."""

    def __init__(self, sample_fps: float) -> None:
        self.period = 1.0 / sample_fps
        self.next_time = 0.0

    def __call__(self, index: int, time_s: float) -> bool:
        if time_s + 1e-6 < self.next_time:
            return False
        self.next_time = (math.floor(time_s / self.period + 1e-6) + 1) * self.period
        return True


# --------------------------------------------------------------------------- output


def write_text_atomic(path: Path, text: str) -> None:
    """Write UTF-8 text (no BOM) via a temporary file and an atomic rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(text.encode("utf-8"))
    os.replace(tmp, path)


def write_json_atomic(path: Path, obj: object) -> None:
    write_text_atomic(path, json.dumps(obj, ensure_ascii=False))


def resize_long_side(frame: np.ndarray, long_side: int) -> np.ndarray:
    """Downscale (never upscale) so that the longer side is ``long_side`` pixels."""
    h, w = frame.shape[:2]
    scale = long_side / max(h, w)
    if scale >= 1.0:
        return frame
    return cv2.resize(frame, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)


def write_jpeg(target: Path, frame: np.ndarray, long_side: int, quality: int) -> None:
    ok, buf = cv2.imencode(".jpg", resize_long_side(frame, long_side), [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise IngestError(f"JPEG encoding failed for {target}")
    target.write_bytes(buf.tobytes())


def save_keyframes(
    path: Path, fps: float, select: FrameSelector, out: Path, video_id: str, params: Params
) -> tuple[list[tuple[int, float]], int, float]:
    """Decode ``path`` once and write selected frames as ``00001.jpg``, ``00002.jpg``, ...

    Keyframes go to ``out/frames/<video_id>/``; with ``params.encode_long_side`` > 0 a second,
    larger copy for the image encoder goes to ``out/frames_encode/<video_id>/`` under the same name.
    Returns ``([(frame_index, time_s) per keyframe], frames decoded, time of the last frame)``.
    """
    frames_dir = out / FRAMES_DIR / video_id
    encode_dir = out / ENCODE_FRAMES_DIR / video_id if params.encode_long_side > 0 else None
    for d in (frames_dir, encode_dir):
        if d is not None:
            d.mkdir(parents=True, exist_ok=True)
    written: list[tuple[int, float]] = []
    n_frames, last_time = 0, 0.0
    for index, time_s, frame in iter_frames(path, fps, select):
        n_frames, last_time = index + 1, time_s
        if frame is None:
            continue
        written.append((index, time_s))
        name = f"{len(written):0{FILENAME_DIGITS}d}.jpg"
        write_jpeg(frames_dir / name, frame, params.long_side, params.quality)
        if encode_dir is not None:
            write_jpeg(encode_dir / name, frame, params.encode_long_side, params.encode_quality)
    if n_frames == 0:
        raise IngestError(f"no decodable frames in {path}")
    return written, n_frames, last_time


def video_file_reference(path: Path, out: Path) -> str:
    """``videos.json`` ``file`` entry: the video path relative to the collection directory (§3),
    or absolute when no relative path exists (different drive)."""
    try:
        return Path(os.path.relpath(path.resolve(), out.resolve())).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def file_digest(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()


def shot_file_for(video_id: str, params: Params, shots_dir: Path | None) -> Path | None:
    """The external shot file of a video, or None when the method does not use one."""
    if params.method != "shots":
        return None
    if shots_dir is None:
        raise IngestError("--method shots needs --shots-dir")
    return shots_dir / f"{video_id}.csv"


def find_shots(path: Path, fps: float, params: Params, shot_file: Path | None) -> list[Shot] | None:
    """Shots of one video by the configured method; None for uniform sampling."""
    min_frames = round(params.min_shot * fps)
    if params.method == "uniform":
        return None
    if params.method == "shots":
        assert shot_file is not None
        return read_shot_file(shot_file)
    if params.method == "scenedetect":
        detector: CutDetector = ContentCutDetector(fps, params.threshold, min_frames)
    elif params.method == "histogram":
        detector = HistogramCutDetector(params.hist_threshold, params.pixel_threshold, min_frames)
    else:
        raise IngestError(f"unknown shot detection method: {params.method!r}")
    cuts, n_frames = detect_cuts(path, fps, detector)
    return cuts_to_shots(cuts, n_frames)


def process_video(
    video_id: str,
    path: Path,
    out: Path,
    params: Params,
    shots_dir: Path | None,
    log: Callable[[str], None] = print,
) -> dict:
    """Extract one video and return its record (also written to ``out/ingest/<video_id>.json``)."""
    fps = nominal_fps(path)
    shot_file = shot_file_for(video_id, params, shots_dir)
    shots = find_shots(path, fps, params, shot_file)
    if shots is None:
        select: FrameSelector = UniformSelector(params.fps)
    else:
        picks = shot_keyframes(shots, fps, long_shot=params.long_shot, edge=params.edge, max_gap=params.max_gap)
        select = IndexSelector(picks)

    written, n_frames, last_time = save_keyframes(path, fps, select, out, video_id, params)
    if shots is not None and shots[-1].end >= n_frames:
        log(f"  warning: shots of {video_id} reach frame {shots[-1].end} but only {n_frames} frames decode; clamped")
        shots = [Shot(s.start, min(s.end, n_frames - 1)) for s in shots if s.start < n_frames]
    record = {
        "version": RECORD_VERSION,
        "video_id": video_id,
        "params": asdict(params),
        "source": {
            "size": path.stat().st_size,
            "shot_file_sha1": file_digest(shot_file) if shot_file is not None else None,
        },
        "video": {
            "fps": fps,
            "duration": last_time + 1.0 / fps,
            "n_frames": n_frames,
            "title": video_id,
            "youtube_id": None,
            "file": video_file_reference(path, out),
        },
        "shots": [[s.start, s.end] for s in shots] if shots is not None else None,
        "keyframes": [
            {"n": n, "frame_idx": index, "pts_time": round(time_s, 6), "frame_filename": f"{n:0{FILENAME_DIGITS}d}.jpg"}
            for n, (index, time_s) in enumerate(written, start=1)
        ],
    }
    write_json_atomic(out / "ingest" / f"{video_id}.json", record)
    return record


def load_current_record(record_path: Path, path: Path, params: Params, shot_file: Path | None) -> dict | None:
    """The stored record if it was produced from the same input with the same parameters, else None."""
    if not record_path.is_file():
        return None
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    source = record.get("source", {})
    shot_sha = file_digest(shot_file) if shot_file is not None and shot_file.is_file() else None
    current = (
        record.get("version") == RECORD_VERSION
        and record.get("params") == asdict(params)
        and source.get("size") == path.stat().st_size
        and source.get("shot_file_sha1") == shot_sha
    )
    return record if current else None


def assemble_collection(
    out: Path, records: Sequence[dict], name: str, encode_frames: bool, log: Callable[[str], None] = print
) -> list[dict]:
    """Write ``metadata.json``, ``videos.json`` and update ``collection.json``; return the metadata rows.

    ``encode_frames`` registers ``frames_encode`` as ``encode_frames_dir`` (or unregisters it).

    When the metadata changes, the embedding entries of ``collection.json`` are removed so that a
    stale ``embeddings.npy`` is never served against new rows; run ``encode_images.py`` again.
    """
    metadata: list[dict] = []
    videos: dict[str, dict] = {}
    for record in records:
        vid = record["video_id"]
        fps = record["video"]["fps"]
        videos[vid] = record["video"]
        metadata.extend(
            {"video_id": vid, "n": kf["n"], "frame_idx": kf["frame_idx"], "pts_time": kf["pts_time"], "fps": fps,
             "frame_filename": kf["frame_filename"]}
            for kf in record["keyframes"]
        )

    metadata_path = out / "metadata.json"
    new_text = json.dumps(metadata, ensure_ascii=False)
    changed = not metadata_path.is_file() or metadata_path.read_text(encoding="utf-8") != new_text
    write_text_atomic(metadata_path, new_text)
    write_json_atomic(out / "videos.json", videos)

    collection_path = out / "collection.json"
    collection = json.loads(collection_path.read_text(encoding="utf-8")) if collection_path.is_file() else {}
    if changed and "embeddings" in collection:
        log("metadata changed: embedding entries removed from collection.json; run encode_images.py again")
        for key in ("embeddings", "model", "dim"):
            collection.pop(key, None)
    collection.update({"name": name, "metadata": "metadata.json", "videos": "videos.json", "frames_dir": FRAMES_DIR})
    if encode_frames:
        collection["encode_frames_dir"] = ENCODE_FRAMES_DIR
    else:
        collection.pop("encode_frames_dir", None)
    write_json_atomic(collection_path, collection)
    return metadata


def extract_collection(
    videos_dir: Path,
    out: Path,
    params: Params,
    *,
    shots_dir: Path | None = None,
    name: str | None = None,
    log: Callable[[str], None] = print,
) -> dict:
    """Run the whole stage; return ``{"videos", "extracted", "skipped", "keyframes", "seconds"}``."""
    if params.method not in METHODS or params.method == "auto":
        raise IngestError(f"method must be one of {METHODS[1:]}, got {params.method!r}")
    if params.method == "shots" and shots_dir is None:
        raise IngestError("--method shots needs --shots-dir")
    if params.method == "scenedetect" and not scenedetect_available():
        raise IngestError("--method scenedetect requested but PySceneDetect is not installed")
    videos = discover_videos(videos_dir)
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    records: list[dict] = []
    extracted = skipped = 0
    for i, (video_id, path) in enumerate(videos.items(), start=1):
        record_path = out / "ingest" / f"{video_id}.json"
        prefix = f"[{i}/{len(videos)}] {video_id}"
        stored = load_current_record(record_path, path, params, shot_file_for(video_id, params, shots_dir))
        if stored is not None:
            records.append(stored)
            skipped += 1
            log(f"{prefix}: up to date, skipped")
            continue
        record_path.unlink(missing_ok=True)
        for top in (FRAMES_DIR, ENCODE_FRAMES_DIR):
            shutil.rmtree(out / top / video_id, ignore_errors=True)
        t0 = time.perf_counter()
        record = process_video(video_id, path, out, params, shots_dir, log)
        dt = max(time.perf_counter() - t0, 1e-9)
        records.append(record)
        extracted += 1
        video, shots = record["video"], record["shots"]
        layout = "uniform sampling" if shots is None else f"{len(shots)} shots"
        log(f"{prefix}: {video['n_frames']} frames ({video['duration']:.1f} s), {layout}, "
            f"{len(record['keyframes'])} keyframes in {dt:.1f} s ({video['n_frames'] / dt:.0f} frames/s over all passes)")
    metadata = assemble_collection(out, records, name or out.resolve().name, params.encode_long_side > 0, log)
    seconds = time.perf_counter() - started
    log(f"done: {len(videos)} videos ({extracted} extracted, {skipped} skipped), {len(metadata)} keyframes, {seconds:.1f} s")
    return {"videos": len(videos), "extracted": extracted, "skipped": skipped, "keyframes": len(metadata), "seconds": seconds}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--videos", type=Path, required=True, help="directory searched recursively for video files")
    ap.add_argument("--out", type=Path, required=True, help="collection directory to create or update")
    ap.add_argument("--name", help="collection name (default: name of --out)")
    ap.add_argument("--method", choices=METHODS, default="auto",
                    help="shot boundaries; auto = scenedetect if installed, else histogram (default auto)")
    ap.add_argument("--fps", type=float, default=1.0, help="keyframes per second for --method uniform (default 1)")
    ap.add_argument("--shots-dir", type=Path, help="directory with <video_id>.csv shot files for --method shots")
    ap.add_argument("--threshold", type=float, default=27.0, help="scenedetect ContentDetector threshold (default 27)")
    ap.add_argument("--hist-threshold", type=float, default=0.25,
                    help="histogram detector: Bhattacharyya distance threshold (default 0.25)")
    ap.add_argument("--pixel-threshold", type=float, default=20.0,
                    help="histogram detector: mean absolute HSV difference threshold on 0-255 (default 20)")
    ap.add_argument("--min-shot", type=float, default=0.5, help="minimum shot length in seconds (default 0.5)")
    ap.add_argument("--long-shot", type=float, default=4.0,
                    help="shots longer than this many seconds also get start and end keyframes (default 4)")
    ap.add_argument("--edge", type=float, default=0.5, help="seconds between a cut and a start/end keyframe (default 0.5)")
    ap.add_argument("--max-gap", type=float, default=0.0,
                    help="maximum seconds between keyframes inside one shot; 0 disables (default 0)")
    ap.add_argument("--long-side", type=int, default=512, help="JPEG long side in pixels, never upscaled (default 512)")
    ap.add_argument("--quality", type=int, default=90, help="JPEG quality 1-100 (default 90)")
    ap.add_argument("--encode-long-side", type=int, default=0,
                    help="also write a copy with this long side (never upscaled) to frames_encode/ for "
                         "encode_images.py; 0 disables (default 0)")
    ap.add_argument("--encode-quality", type=int, default=95, help="JPEG quality of the encoder copy (default 95)")
    args = ap.parse_args(argv)
    if (args.fps <= 0 or args.long_side <= 0 or args.encode_long_side < 0 or args.min_shot < 0 or args.max_gap < 0
            or not 1 <= args.quality <= 100 or not 1 <= args.encode_quality <= 100):
        ap.error("--fps and --long-side must be positive, qualities within 1..100, "
                 "--encode-long-side, --min-shot and --max-gap >= 0")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    method = args.method
    if method == "auto":
        method = "scenedetect" if scenedetect_available() else "histogram"
    params = Params(
        method=method,
        fps=args.fps,
        threshold=args.threshold,
        hist_threshold=args.hist_threshold,
        pixel_threshold=args.pixel_threshold,
        min_shot=args.min_shot,
        long_shot=args.long_shot,
        edge=args.edge,
        max_gap=args.max_gap,
        long_side=args.long_side,
        quality=args.quality,
        encode_long_side=args.encode_long_side,
        encode_quality=args.encode_quality,
    )
    print(f"shot detection method: {method}", flush=True)
    try:
        extract_collection(args.videos, args.out, params, shots_dir=args.shots_dir, name=args.name,
                           log=lambda line: print(line, flush=True))
    except IngestError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
