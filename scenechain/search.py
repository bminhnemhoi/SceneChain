"""Ranking logic of the VBS tool: pure numpy over a :class:`Collection`, no I/O.

Every function takes precomputed similarity vectors (one float per keyframe), so the whole module is
testable on a synthetic collection and the encoder never runs twice for the same query.

Temporal search ("scene A, then scene B within w seconds") is the part that has to be fast. It is
dynamic programming backwards over the scenes, evaluated for all candidate videos at once:

1. the rows of all candidate videos are laid out in one array, grouped by video and sorted by time;
2. each keyframe's successor window ``(t + w_min, t + w_max]`` becomes an index range ``[lo, hi)``
   by a single ``searchsorted`` on the key ``video_position * stride + time``, which keeps every
   window inside its own video;
3. the best successor score in each window is a range-maximum query answered from a sparse table
   (``O(M log L)`` to build for ``M`` rows and windows of at most ``L`` rows, ``O(1)`` per query),
   which also yields the argmax so the chain can be read back.

No Python loop runs over videos or keyframes except the final selection of the returned chains.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Sequence, TypedDict

import numpy as np

from .collection import Collection

if TYPE_CHECKING:  # the encoder module pulls in the AIC engine; search itself needs only numpy
    from .encoder import TextEncoder

#: chains returned per video
MAX_CHAINS_PER_VIDEO = 3


class FrameHit(TypedDict):
    row: int
    video_id: str
    frame_idx: int
    time: float
    score: float


class VideoGroup(TypedDict):
    video_id: str
    score: float
    hits: list[FrameHit]


class ChainFrame(TypedDict):
    row: int
    frame_idx: int
    time: float
    score: float


class ChainHit(TypedDict):
    video_id: str
    score: float
    frames: list[ChainFrame]


# ------------------------------------------------------------------------------------ frames
def text_similarities(collection: Collection, encoder: TextEncoder, text: str) -> np.ndarray:
    """Float32 similarity of every keyframe to ``text``; invalid rows are ``-inf``.

    Long text is scored chunk by chunk and the chunk similarities are averaged, which rewards a
    frame that satisfies the whole description rather than any single clause.
    """
    if not (text or "").strip():
        raise ValueError("empty query text")
    chunks = encoder.encode_chunks(text)
    if not chunks:
        raise ValueError(f"encoder {encoder.name!r} returned no vectors")
    sims = collection.similarities(chunks[0])
    for vec in chunks[1:]:
        sims += collection.similarities(vec)
    if len(chunks) > 1:
        sims /= np.float32(len(chunks))
    sims[~collection.valid] = -np.inf
    return sims


def _check_sims(sims: np.ndarray, collection: Collection) -> np.ndarray:
    arr = np.asarray(sims)
    if arr.shape != (collection.n,):
        raise ValueError(f"similarity vector has shape {arr.shape}, expected ({collection.n},)")
    return arr


def _frame_hit(collection: Collection, row: int, score: float) -> FrameHit:
    return {
        "row": row,
        "video_id": collection.video_id_of(row),
        "frame_idx": int(collection.frame_idx[row]),
        "time": float(collection.times[row]),
        "score": score,
    }


def _top_rows(scores: np.ndarray, rows: np.ndarray, top_k: int) -> np.ndarray:
    """The ``top_k`` entries of ``rows`` by score, best first, ties broken by lower row."""
    if rows.size > top_k:
        # everything strictly above the k-th best value, plus enough of its ties
        kth = np.partition(scores[rows], rows.size - top_k)[rows.size - top_k]
        rows = rows[scores[rows] >= kth]
    order = np.lexsort((rows, -scores[rows].astype(np.float64)))
    return rows[order[:top_k]]


def rank_frames(sims: np.ndarray, collection: Collection, top_k: int = 500) -> list[FrameHit]:
    """The ``top_k`` best valid keyframes, best first (ties: lower row first)."""
    if top_k < 1:
        raise ValueError("top_k must be positive")
    arr = _check_sims(sims, collection)
    rows = np.flatnonzero(collection.valid & np.isfinite(arr))
    return [_frame_hit(collection, int(r), float(arr[r])) for r in _top_rows(arr, rows, top_k)]


def group_by_video(
    hits: Sequence[FrameHit], max_videos: int = 60, frames_per_video: int = 8
) -> list[VideoGroup]:
    """Group best-first ``hits`` by video.

    Each group keeps its ``frames_per_video`` best hits, listed in time order; its score is the best
    hit's score. Groups are ordered by score, ties by first appearance in ``hits``.
    """
    if max_videos < 1 or frames_per_video < 1:
        raise ValueError("max_videos and frames_per_video must be positive")
    groups: dict[str, list[FrameHit]] = {}
    for hit in hits:
        bucket = groups.setdefault(hit["video_id"], [])
        if len(bucket) < frames_per_video:
            bucket.append(hit)
    ranked = sorted(
        enumerate(groups.items()),
        key=lambda item: (-max(h["score"] for h in item[1][1]), item[0]),
    )
    return [
        {
            "video_id": vid,
            "score": max(h["score"] for h in bucket),
            "hits": sorted(bucket, key=lambda h: (h["time"], h["row"])),
        }
        for _, (vid, bucket) in ranked[:max_videos]
    ]


# ---------------------------------------------------------------------------------- temporal
def _range_argmax(values: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Index of the maximum of ``values[lo[i]:hi[i]]`` for every ``i``; ``-1`` for empty ranges.

    Sparse table over ``values``; ties resolve to the earliest index.
    """
    m = values.shape[0]
    length = hi - lo
    out = np.full(lo.shape[0], -1, dtype=np.int64)
    if m == 0 or not np.any(length > 0):
        return out
    max_len = int(length.max())
    levels = max_len.bit_length()  # level k covers ranges of 2**k rows
    table = np.empty((levels, m), dtype=np.int64)
    table[0] = np.arange(m)
    for k in range(1, levels):
        span, half = 1 << k, 1 << (k - 1)
        count = m - span + 1
        table[k] = table[k - 1]  # tail positions are never queried at this level
        if count > 0:
            left = table[k - 1, :count]
            right = table[k - 1, half:half + count]
            table[k, :count] = np.where(values[right] > values[left], right, left)
    nonempty = np.flatnonzero(length > 0)
    q_lo, q_hi = lo[nonempty], hi[nonempty]
    level = np.frexp(length[nonempty].astype(np.float64))[1].astype(np.int64) - 1
    a = table[level, q_lo]
    b = table[level, q_hi - (np.int64(1) << level)]
    out[nonempty] = np.where(values[b] > values[a], b, a)
    return out


def _zscore(scores: np.ndarray) -> np.ndarray:
    """``(s - mean) / sd`` over the finite entries; ``-inf`` stays ``-inf``; sd 0 only centres."""
    out = scores.copy()
    live = np.isfinite(out)
    if live.any():
        mean = out[live].mean()
        sd = out[live].std()
        out[live] = (out[live] - mean) / sd if sd > 0 else out[live] - mean
    return out


def temporal_search(
    collection: Collection,
    scene_sims: Sequence[np.ndarray],
    window_s: tuple[float, float] = (0.0, 30.0),
    top_k: int = 200,
    candidate_videos: int = 300,
    scene_norm: Optional[str] = None,
) -> list[ChainHit]:
    """Chains of keyframes, one per scene, in scene order and within the time window.

    A successor of keyframe ``i`` for the next scene is any keyframe ``j`` of the same video with
    ``t_i + w_min < t_j <= t_i + w_max``. The chain score is the mean of the scene similarities
    along the best chain starting at the first keyframe (dynamic programming backwards over the
    scenes), so scores are comparable across the number of scenes.

    Candidate videos are the union of the ``candidate_videos`` best videos of every scene (by best
    keyframe). At most ``MAX_CHAINS_PER_VIDEO`` chains are returned per video and chains of the same
    video never overlap in time (a chain whose ``[first, last]`` time span intersects an already
    selected chain of that video is skipped), so the list shows distinct moments. Ties are broken by
    lower row of the first keyframe.

    With a single scene this is exactly :func:`rank_frames`, each hit wrapped as a one-frame chain
    and without the per-video limit.

    ``scene_norm="z"`` (opt-in, default ``None`` = raw similarities) z-normalises every scene's
    similarities over the keyframes of the candidate videos before the dynamic program, so no scene
    dominates the sum through a wider score spread; chain and frame scores are then z-scores. With a
    single scene it is ignored (it cannot change the order).
    """
    n_scenes = len(scene_sims)
    if n_scenes < 1:
        raise ValueError("temporal_search needs at least one scene")
    if top_k < 1 or candidate_videos < 1:
        raise ValueError("top_k and candidate_videos must be positive")
    if scene_norm not in (None, "z"):
        raise ValueError(f"scene_norm must be None or 'z', got {scene_norm!r}")
    w_min, w_max = float(window_s[0]), float(window_s[1])
    if not (np.isfinite(w_min) and w_min >= 0.0 and w_max > w_min):
        raise ValueError(f"window must satisfy 0 <= min < max, got {window_s}")
    sims = [_check_sims(s, collection) for s in scene_sims]

    if n_scenes == 1:
        return [
            {
                "video_id": hit["video_id"],
                "score": hit["score"],
                "frames": [{
                    "row": hit["row"],
                    "frame_idx": hit["frame_idx"],
                    "time": hit["time"],
                    "score": hit["score"],
                }],
            }
            for hit in rank_frames(sims[0], collection, top_k)
        ]
    if collection.n == 0:
        return []

    # scene scores in time order per video; invalid rows can never be part of a chain
    order, starts = collection.time_order, collection.video_starts
    invalid_sorted = ~collection.valid[order]
    ordered = []
    for s in sims:
        col = s[order].astype(np.float64)
        col[invalid_sorted | ~np.isfinite(col)] = -np.inf
        ordered.append(col)

    # candidate videos: union over scenes of the best videos by best keyframe
    selected = np.zeros(collection.n_videos, dtype=bool)
    for col in ordered:
        video_best = np.maximum.reduceat(col, starts[:-1])  # every video has at least one row
        live = np.flatnonzero(np.isfinite(video_best))
        ranked = live[np.lexsort((live, -video_best[live]))]
        selected[ranked[:candidate_videos]] = True
    cand = np.flatnonzero(selected)
    if cand.size == 0:
        return []

    # lay the candidate videos out contiguously: positions into ``order``
    lengths = starts[cand + 1] - starts[cand]
    video_pos = np.repeat(np.arange(cand.size), lengths)
    first = np.concatenate(([0], np.cumsum(lengths)[:-1]))
    pos = np.arange(int(lengths.sum())) - np.repeat(first, lengths) + np.repeat(starts[cand], lengths)
    rows = order[pos]
    scene_scores = [col[pos] for col in ordered]
    if scene_norm == "z":
        scene_scores = [_zscore(s) for s in scene_scores]

    # successor window [lo, hi) per keyframe, confined to its own video
    t = collection.times[rows]
    t_rel = t - t.min()
    horizon = float(t_rel.max()) + 1.0  # every t_rel < horizon
    stride = horizon + 1.0
    base = video_pos * stride
    keys = base + t_rel  # non-decreasing: videos ascending, time ascending within a video
    lo = np.searchsorted(keys, base + np.minimum(t_rel + w_min, horizon), side="right")
    hi = np.searchsorted(keys, base + np.minimum(t_rel + w_max, horizon), side="right")

    # dynamic programming backwards over scenes
    best = scene_scores[-1]
    successors: list[np.ndarray] = []
    for k in range(n_scenes - 2, -1, -1):
        nxt = _range_argmax(best, lo, hi)
        succ_score = np.where(nxt >= 0, best[np.maximum(nxt, 0)], -np.inf)
        best = scene_scores[k] + succ_score
        successors.append(nxt)
    successors.reverse()  # successors[k]: scene k -> scene k + 1
    final = best / n_scenes

    chain_starts = np.flatnonzero(np.isfinite(final))
    ranked_starts = chain_starts[np.lexsort((rows[chain_starts], -final[chain_starts]))]
    paths = [ranked_starts]
    for nxt in successors:
        paths.append(nxt[paths[-1]])
    path_arr = np.stack(paths, axis=1)  # (candidates, n_scenes) positions

    # greedy selection in score order; only scalars already converted to Python are touched
    # per candidate, the chain itself is materialised only when it is accepted
    start_video = video_pos[ranked_starts].tolist()
    first_time = t[path_arr[:, 0]].tolist()
    last_time = t[path_arr[:, -1]].tolist()
    live_videos = len(set(start_video))
    chains: list[ChainHit] = []
    taken: dict[int, list[tuple[float, float]]] = {}
    full_videos = 0
    for i, v in enumerate(start_video):
        spans = taken.setdefault(v, [])
        if len(spans) >= MAX_CHAINS_PER_VIDEO:
            continue
        t_first, t_last = first_time[i], last_time[i]
        if any(t_first <= b and a <= t_last for a, b in spans):
            continue
        spans.append((t_first, t_last))
        chains.append({
            "video_id": collection.video_ids[int(cand[v])],
            "score": float(final[ranked_starts[i]]),
            "frames": [
                {
                    "row": int(rows[p]),
                    "frame_idx": int(collection.frame_idx[rows[p]]),
                    "time": float(t[p]),
                    "score": float(scene_scores[k][p]),
                }
                for k, p in enumerate(path_arr[i].tolist())
            ],
        })
        if len(chains) >= top_k:
            break
        if len(spans) == MAX_CHAINS_PER_VIDEO:
            full_videos += 1
            if full_videos == live_videos:
                break
    return chains


# ------------------------------------------------------------------------------ by example
def similar_frames(
    collection: Collection, row: int, top_k: int = 200, exclude_same_video: bool = False
) -> list[FrameHit]:
    """Keyframes most similar to the stored embedding of ``row``, best first.

    The query keyframe itself is never returned; with ``exclude_same_video`` no keyframe of its
    video is.
    """
    sims = collection.similarities(collection.embedding(row))
    if exclude_same_video:
        sims[collection.rows_of(collection.video_id_of(row))] = -np.inf
    sims[row] = -np.inf
    return rank_frames(sims, collection, top_k)


# ------------------------------------------------------------------------------ video view
def video_timeline(
    collection: Collection, video_id: str, scene_sims: Optional[Sequence[np.ndarray]] = None
) -> dict:
    """All keyframes of ``video_id`` in time order, with per-scene scores for the score curve.

    Returns ``{"video": {...}, "keyframes": [{row, frame_idx, time, valid, scores}]}``; ``scores``
    is ``None`` without ``scene_sims``, otherwise one value per scene, ``None`` where the score is
    not finite (invalid rows), so the result is JSON-serialisable.
    """
    video = collection.video(video_id)
    rows = collection.rows_of(video_id)
    per_scene = [_check_sims(s, collection)[rows] for s in scene_sims] if scene_sims else None
    keyframes = []
    for i, r in enumerate(rows.tolist()):
        scores = None
        if per_scene is not None:
            scores = [float(col[i]) if np.isfinite(col[i]) else None for col in per_scene]
        keyframes.append({
            "row": r,
            "frame_idx": int(collection.frame_idx[r]),
            "time": float(collection.times[r]),
            "valid": bool(collection.valid[r]),
            "scores": scores,
        })
    return {"video": video, "keyframes": keyframes}


def submission_segment(
    collection: Collection, video_id: str, time_s: float, half_width_s: float = 0.0
) -> tuple[int, int]:
    """``(start_ms, end_ms)`` of ``time_s +- half_width_s`` clamped to ``[0, duration]``.

    The default is a point (``start_ms == end_ms``): DRES judges a temporal answer correct only
    when the whole ``[start, end]`` lies inside the target segment (confirmed on DRES 2.0.4), so any
    width only adds the risk of crossing a segment edge.
    """
    if not np.isfinite(time_s):
        raise ValueError(f"time must be finite, got {time_s}")
    if not (np.isfinite(half_width_s) and half_width_s >= 0):
        raise ValueError(f"half_width_s must be a non-negative number, got {half_width_s}")
    duration = float(collection.video(video_id)["duration"])
    start = min(max(time_s - half_width_s, 0.0), duration)
    end = min(max(time_s + half_width_s, 0.0), duration)
    return int(round(start * 1000.0)), int(round(end * 1000.0))
