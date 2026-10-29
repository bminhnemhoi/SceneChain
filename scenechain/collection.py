"""Load a keyframe collection (format of ``docs/ARCHITECTURE.md`` §3) and answer row-level questions.

The embedding matrix is memory-mapped, never loaded whole: the AIC 2026 index is 817 MB and a V3C
index is an order of magnitude larger. Similarity scans stream over the map in fixed-size chunks
and always return float32, whether the file stores float32 or float16.

Metadata is kept as columnar numpy arrays instead of 177k Python dicts. Rows are additionally
indexed per video in time order (``time_order`` and ``video_starts``), which is what the temporal
search in :mod:`scenechain.search` needs to scan every video at once without a Python loop.

Nothing here assumes AIC naming: video ids and frame file names are opaque strings.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np

REQUIRED_META_KEYS = ("video_id", "n", "frame_idx", "pts_time", "fps", "frame_filename")
REQUIRED_VIDEO_KEYS = ("fps", "duration", "n_frames", "title", "youtube_id", "file")

#: rows per similarity chunk; 32768 x 1152 float16 rows convert to a 151 MB float32 block
SIMILARITY_CHUNK_ROWS = 32_768


class CollectionError(ValueError):
    """A collection directory is malformed or inconsistent."""


def _read_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing")
    return json.loads(path.read_text(encoding="utf-8"))


class Collection:
    """An immutable, memory-mapped keyframe collection."""

    def __init__(
        self,
        root: Path,
        name: str,
        model: str,
        embeddings: np.ndarray,
        video_ids: list[str],
        videos: dict[str, dict],
        row_video: np.ndarray,
        n_in_video: np.ndarray,
        frame_idx: np.ndarray,
        times: np.ndarray,
        fps: np.ndarray,
        frame_filenames: list[str],
        valid: np.ndarray,
        frames_dir: Path,
        frame_url_template: Optional[str],
    ) -> None:
        self.root = root
        self.name = name
        self.model = model
        self.embeddings = embeddings
        #: video ids in order of first appearance in the metadata
        self.video_ids = video_ids
        self._videos = videos
        #: per row: index into ``video_ids``
        self.row_video = row_video
        self.n_in_video = n_in_video
        self.frame_idx = frame_idx
        #: per row: presentation time in seconds (float64)
        self.times = times
        self.fps = fps
        self.frame_filenames = frame_filenames
        #: per row: False for rows that must never be returned (bumpers, blank frames)
        self.valid = valid
        self.frames_dir = frames_dir
        self.frame_url_template = frame_url_template

        #: all rows grouped by video (in ``video_ids`` order), each group sorted by time
        self.time_order = np.lexsort((times, row_video)).astype(np.int64)  # stable: ties keep row order
        counts = np.bincount(row_video, minlength=len(video_ids)).astype(np.int64)
        #: ``time_order[video_starts[v]:video_starts[v + 1]]`` are the rows of video ``v``
        self.video_starts = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)
        self._video_index = {vid: i for i, vid in enumerate(video_ids)}
        self._row_index = {
            (video_ids[v], f): r
            for r, (v, f) in enumerate(zip(row_video.tolist(), frame_idx.tolist()))
        }
        self.valid.setflags(write=False)

    # ------------------------------------------------------------------ loading
    @classmethod
    def load(cls, directory: str | Path) -> "Collection":
        """Open ``directory``; raises ``FileNotFoundError`` or :class:`CollectionError` when invalid."""
        root = Path(directory).resolve()
        spec = _read_json(root / "collection.json")
        for key in ("name", "embeddings", "metadata", "frames_dir", "videos"):
            if key not in spec:
                raise CollectionError(f"{root / 'collection.json'} lacks the key {key!r}")

        def resolve(value: str) -> Path:
            p = Path(value)
            return p if p.is_absolute() else root / p

        emb_path = resolve(spec["embeddings"])
        if not emb_path.is_file():
            raise FileNotFoundError(f"{emb_path} is missing")
        embeddings = np.load(emb_path, mmap_mode="r")
        if embeddings.ndim != 2:
            raise CollectionError(f"{emb_path} must be a 2-D matrix, got shape {embeddings.shape}")
        if embeddings.dtype not in (np.float32, np.float16):
            raise CollectionError(f"{emb_path} must be float32 or float16, got {embeddings.dtype}")
        n, dim = embeddings.shape
        if "dim" in spec and int(spec["dim"]) != dim:
            raise CollectionError(f"collection.json says dim={spec['dim']} but {emb_path} has {dim}")

        metadata = _read_json(resolve(spec["metadata"]))
        if not isinstance(metadata, list) or len(metadata) != n:
            count = len(metadata) if isinstance(metadata, list) else type(metadata).__name__
            raise CollectionError(f"metadata has {count} rows but the embedding matrix has {n}")
        video_ids: list[str] = []
        video_index: dict[str, int] = {}
        row_video = np.empty(n, dtype=np.int32)
        try:
            for r, m in enumerate(metadata):
                vid = str(m["video_id"])
                v = video_index.get(vid)
                if v is None:
                    v = video_index[vid] = len(video_ids)
                    video_ids.append(vid)
                row_video[r] = v
            n_in_video = np.fromiter((m["n"] for m in metadata), dtype=np.int32, count=n)
            frame_idx = np.fromiter((m["frame_idx"] for m in metadata), dtype=np.int64, count=n)
            times = np.fromiter((m["pts_time"] for m in metadata), dtype=np.float64, count=n)
            fps = np.fromiter((m["fps"] for m in metadata), dtype=np.float32, count=n)
            frame_filenames = [str(m["frame_filename"]) for m in metadata]
        except KeyError as exc:
            raise CollectionError(
                f"metadata rows must carry {list(REQUIRED_META_KEYS)}; missing {exc}"
            ) from exc
        except (TypeError, ValueError) as exc:
            raise CollectionError(f"malformed metadata row: {exc}") from exc
        if not np.all(np.isfinite(times)):
            raise CollectionError("metadata contains non-finite pts_time values")

        videos_raw = _read_json(resolve(spec["videos"]))
        if not isinstance(videos_raw, dict):
            raise CollectionError("videos.json must map video ids to video records")
        absent = [vid for vid in video_ids if vid not in videos_raw]
        if absent:
            raise CollectionError(f"videos.json lacks {len(absent)} videos, e.g. {absent[:3]}")
        videos: dict[str, dict] = {}
        for vid in video_ids:
            entry = videos_raw[vid]
            lacking = [k for k in REQUIRED_VIDEO_KEYS if k not in entry]
            if lacking:
                raise CollectionError(f"videos.json entry {vid!r} lacks {lacking}")
            videos[vid] = {"video_id": vid, **{k: entry[k] for k in REQUIRED_VIDEO_KEYS}}

        valid = np.ones(n, dtype=bool)
        if spec.get("invalid_rows"):
            invalid = np.asarray(_read_json(resolve(spec["invalid_rows"])), dtype=np.int64)
            if invalid.size and (invalid.min() < 0 or invalid.max() >= n):
                raise CollectionError(f"invalid_rows contains indices outside [0, {n})")
            valid[invalid] = False

        return cls(
            root=root,
            name=str(spec["name"]),
            model=str(spec.get("model", "")),
            embeddings=embeddings,
            video_ids=video_ids,
            videos=videos,
            row_video=row_video,
            n_in_video=n_in_video,
            frame_idx=frame_idx,
            times=times,
            fps=fps,
            frame_filenames=frame_filenames,
            valid=valid,
            frames_dir=resolve(spec["frames_dir"]),
            frame_url_template=spec.get("frame_url") or None,
        )

    # --------------------------------------------------------------- properties
    @property
    def n(self) -> int:
        """Number of keyframes (rows)."""
        return int(self.embeddings.shape[0])

    @property
    def dim(self) -> int:
        """Embedding dimension."""
        return int(self.embeddings.shape[1])

    @property
    def n_videos(self) -> int:
        return len(self.video_ids)

    # ------------------------------------------------------------------ lookups
    def row(self, video_id: str, frame_idx: int) -> Optional[int]:
        """Row of the keyframe ``frame_idx`` of ``video_id``, or ``None`` if it is not a keyframe."""
        return self._row_index.get((video_id, int(frame_idx)))

    def video_position(self, video_id: str) -> int:
        """Index of ``video_id`` in :attr:`video_ids`; raises ``KeyError`` for an unknown video."""
        try:
            return self._video_index[video_id]
        except KeyError:
            raise KeyError(f"unknown video {video_id!r}") from None

    def rows_of(self, video_id: str) -> np.ndarray:
        """Rows of ``video_id`` sorted by time (a copy)."""
        v = self.video_position(video_id)
        return self.time_order[self.video_starts[v]:self.video_starts[v + 1]].copy()

    def video(self, video_id: str) -> dict:
        """``{video_id, fps, duration, title, youtube_id, file}``; raises ``KeyError`` if unknown."""
        self.video_position(video_id)
        return dict(self._videos[video_id])

    def video_id_of(self, row: int) -> str:
        return self.video_ids[int(self.row_video[row])]

    def meta(self, row: int) -> dict:
        """The metadata record of ``row`` in the on-disk format."""
        self._check_row(row)
        return {
            "video_id": self.video_id_of(row),
            "n": int(self.n_in_video[row]),
            "frame_idx": int(self.frame_idx[row]),
            "pts_time": float(self.times[row]),
            "fps": float(self.fps[row]),
            "frame_filename": self.frame_filenames[row],
        }

    def frame_path(self, row: int) -> Path:
        """Local path of the keyframe image of ``row``."""
        self._check_row(row)
        return self.frames_dir / self.video_id_of(row) / self.frame_filenames[row]

    def frame_url(self, row: int) -> str:
        """URL of the keyframe image: the collection's ``frame_url`` template or the server route."""
        self._check_row(row)
        vid = self.video_id_of(row)
        filename = self.frame_filenames[row]
        if self.frame_url_template:
            return self.frame_url_template.format(
                video_id=vid,
                frame_filename=filename,
                frame_idx=int(self.frame_idx[row]),
                n=int(self.n_in_video[row]),
            )
        return f"/frames/{vid}/{filename}"

    def embedding(self, row: int) -> np.ndarray:
        """The stored embedding of ``row`` as float32."""
        self._check_row(row)
        return np.asarray(self.embeddings[row], dtype=np.float32)

    # --------------------------------------------------------------- similarity
    def similarities(self, vec: np.ndarray, chunk_rows: int = SIMILARITY_CHUNK_ROWS) -> np.ndarray:
        """Dot product of every row with ``vec`` as a float32 ``(N,)`` array, computed chunk by chunk."""
        v = np.asarray(vec, dtype=np.float32).reshape(-1)
        if v.shape[0] != self.dim:
            raise ValueError(f"query vector has dimension {v.shape[0]}, collection has {self.dim}")
        out = np.empty(self.n, dtype=np.float32)
        for a in range(0, self.n, chunk_rows):
            b = min(a + chunk_rows, self.n)
            np.matmul(np.asarray(self.embeddings[a:b], dtype=np.float32), v, out=out[a:b])
        return out

    def _check_row(self, row: int) -> None:
        if not 0 <= int(row) < self.n:
            raise IndexError(f"row {row} outside [0, {self.n})")
