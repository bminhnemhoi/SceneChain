"""Ranking, grouping, temporal chains, query by example, timeline and submission segments."""

from __future__ import annotations

import itertools
import json
import os
import shutil
import time

import numpy as np
import pytest

from tests.conftest import CONCEPT_PHRASES, PLANTED, FakeEncoder
from scenechain import search
from scenechain.collection import Collection


# --------------------------------------------------------------------------- helpers/fixtures
def kf(collection: Collection, row: int) -> int:
    """0-based keyframe number within the video (the numbering of the conftest docstring)."""
    return int(collection.n_in_video[row]) - 1


def chain_keyframes(collection: Collection, chain: dict) -> tuple[str, list[int]]:
    return chain["video_id"], [kf(collection, f["row"]) for f in chain["frames"]]


@pytest.fixture(scope="module")
def collection(synthetic_collection_dir) -> Collection:
    return Collection.load(synthetic_collection_dir)


def scene_sims(collection: Collection, *concepts: str) -> list[np.ndarray]:
    enc = FakeEncoder()
    return [search.text_similarities(collection, enc, CONCEPT_PHRASES[c]) for c in concepts]


# ------------------------------------------------------------------------------------ frames
def test_text_similarities_marks_invalid_rows(collection):
    (sims,) = scene_sims(collection, "boat")
    assert sims.dtype == np.float32 and sims.shape == (collection.n,)
    assert np.all(np.isneginf(sims[~collection.valid]))
    assert np.all(np.isfinite(sims[collection.valid]))


def test_text_similarities_averages_chunks(collection):
    enc = FakeEncoder()
    door, desk = enc.encode(CONCEPT_PHRASES["door"]), enc.encode(CONCEPT_PHRASES["desk"])

    class TwoChunks(FakeEncoder):
        def encode_chunks(self, text):
            return [door, desk]

    got = search.text_similarities(collection, TwoChunks(), "long description")
    want = (collection.similarities(door) + collection.similarities(desk)) / 2
    valid = collection.valid
    np.testing.assert_allclose(got[valid], want[valid], rtol=1e-6)


def test_text_similarities_rejects_empty_text(collection, fake_encoder):
    with pytest.raises(ValueError):
        search.text_similarities(collection, fake_encoder, "   ")


def test_rank_frames_best_first_with_planted_hit_on_top(collection):
    (sims,) = scene_sims(collection, "boat")
    hits = search.rank_frames(sims, collection, top_k=10)
    assert len(hits) == 10
    scores = [h["score"] for h in hits]
    assert scores == sorted(scores, reverse=True)
    assert {(h["video_id"], kf(collection, h["row"])) for h in hits[:3]} == {
        ("V001", 5), ("V001", 6), ("V001", 7)
    }
    first = hits[0]
    assert set(first) == {"row", "video_id", "frame_idx", "time", "score"}
    assert first["frame_idx"] == int(collection.frame_idx[first["row"]])
    assert first["time"] == float(collection.times[first["row"]])


def test_rank_frames_breaks_ties_by_row_and_skips_invalid(collection):
    hits = search.rank_frames(np.zeros(collection.n, dtype=np.float32), collection, top_k=5)
    assert [h["row"] for h in hits] == [1, 2, 3, 4, 5]  # row 0 is the invalid first keyframe


def test_rank_frames_validates_input(collection):
    with pytest.raises(ValueError):
        search.rank_frames(np.zeros(3, dtype=np.float32), collection)
    with pytest.raises(ValueError):
        search.rank_frames(np.zeros(collection.n, dtype=np.float32), collection, top_k=0)


def test_group_by_video(collection):
    (sims,) = scene_sims(collection, "door")
    hits = search.rank_frames(sims, collection, top_k=200)
    groups = search.group_by_video(hits, max_videos=4, frames_per_video=3)
    assert len(groups) == 4
    assert [g["score"] for g in groups] == sorted((g["score"] for g in groups), reverse=True)
    assert {g["video_id"] for g in groups[:3]} == {"V003", "V004", "V005"}
    for g in groups:
        assert 1 <= len(g["hits"]) <= 3
        assert all(h["video_id"] == g["video_id"] for h in g["hits"])
        assert [h["time"] for h in g["hits"]] == sorted(h["time"] for h in g["hits"])
        best = [h for h in hits if h["video_id"] == g["video_id"]][:3]
        assert {h["row"] for h in g["hits"]} == {h["row"] for h in best}
        assert g["score"] == best[0]["score"]


# ---------------------------------------------------------------------------------- temporal
def test_temporal_search_finds_planted_chain_first(collection):
    chains = search.temporal_search(collection, scene_sims(collection, "door", "desk"))
    video, frames = chain_keyframes(collection, chains[0])
    assert video == "V003"
    assert frames[0] in (10, 11) and frames[1] in (14, 15)
    others = [c for c in chains if c["video_id"] != "V003"]
    # a chain with one planted and one noise keyframe loses about half the planted-vs-noise gap
    assert chains[0]["score"] > others[0]["score"] + 0.15


def test_temporal_search_order_matters(collection):
    door, desk = scene_sims(collection, "door", "desk")
    chains = search.temporal_search(collection, [door, desk])
    rank = {c["video_id"]: i for i, c in reversed(list(enumerate(chains)))}
    assert rank["V003"] == 0
    assert rank["V005"] > rank["V003"]
    v005 = next(c for c in chains if c["video_id"] == "V005")
    assert v005["score"] < chains[0]["score"] - 0.2

    reversed_chains = search.temporal_search(collection, [desk, door])
    video, frames = chain_keyframes(collection, reversed_chains[0])
    assert video == "V005" and frames[0] in (10, 11) and frames[1] in (14, 15)


def test_temporal_search_respects_window(collection):
    door, desk = scene_sims(collection, "door", "desk")
    planted_v004 = ("V004", [2, 38])  # 72 s apart

    narrow = search.temporal_search(collection, [door, desk], window_s=(0.0, 30.0))
    assert planted_v004 not in [chain_keyframes(collection, c) for c in narrow]
    best_v004 = next(c for c in narrow if c["video_id"] == "V004")
    assert best_v004["score"] < narrow[0]["score"] - 0.15

    wide = search.temporal_search(collection, [door, desk], window_s=(0.0, 90.0))
    found = [chain_keyframes(collection, c) for c in wide]
    assert planted_v004 in found[:2]
    assert {video for video, _ in found[:2]} == {"V003", "V004"}

    # the lower bound is exclusive: door at 22 s -> desk at 30 s (8 s) needs min < 8
    late = search.temporal_search(collection, [door, desk], window_s=(8.0, 30.0))
    assert ("V003", [11, 15]) not in [chain_keyframes(collection, c) for c in late]
    assert all(9.99 < c["frames"][1]["time"] - c["frames"][0]["time"] <= 30.0 for c in late)
    assert chain_keyframes(collection, late[0]) == ("V003", [10, 15])  # 20 s -> 30 s


def _brute_force_best(collection, sims, window, first_row):
    """Best chain score (mean) starting at ``first_row`` by exhaustive enumeration."""
    w_min, w_max = window
    rows = collection.rows_of(collection.video_id_of(first_row))
    best = -np.inf

    def extend(k, row, total):
        nonlocal best
        if not np.isfinite(sims[k][row]):
            return
        total += float(sims[k][row])
        if k == len(sims) - 1:
            best = max(best, total / len(sims))
            return
        t = collection.times[row]
        for nxt in rows:
            if t + w_min < collection.times[nxt] <= t + w_max:
                extend(k + 1, nxt, total)

    extend(0, first_row, 0.0)
    return best


@pytest.mark.parametrize("concepts", [("door", "desk"), ("dog", "boat", "door")])
@pytest.mark.parametrize("window", [(0.0, 30.0), (2.0, 6.0), (0.0, 4.0), (10.0, 20.0), (0.0, 1000.0)])
def test_temporal_search_matches_brute_force(collection, concepts, window):
    sims = scene_sims(collection, *concepts)
    chains = search.temporal_search(collection, sims, window_s=window, top_k=1000, candidate_videos=10)
    assert chains
    per_video: dict[str, list[tuple[float, float]]] = {}
    for c in chains:
        frames = c["frames"]
        assert len(frames) == len(sims)
        for a, b in itertools.pairwise(frames):
            assert window[0] < b["time"] - a["time"] <= window[1]
        for k, f in enumerate(frames):
            assert f["score"] == pytest.approx(float(sims[k][f["row"]]))
            assert collection.valid[f["row"]]
        assert c["score"] == pytest.approx(_brute_force_best(collection, sims, window, frames[0]["row"]))
        span = (frames[0]["time"], frames[-1]["time"])
        spans = per_video.setdefault(c["video_id"], [])
        assert all(span[1] < a or b < span[0] for a, b in spans)
        spans.append(span)
    assert all(len(s) <= search.MAX_CHAINS_PER_VIDEO for s in per_video.values())
    assert [c["score"] for c in chains] == sorted((c["score"] for c in chains), reverse=True)
    # the top chain is the global optimum over every valid starting keyframe
    global_best = max(
        _brute_force_best(collection, sims, window, r) for r in np.flatnonzero(collection.valid)
    )
    assert chains[0]["score"] == pytest.approx(global_best)


def test_temporal_search_single_scene_equals_rank_frames(collection):
    (sims,) = scene_sims(collection, "dog")
    chains = search.temporal_search(collection, [sims], top_k=50)
    hits = search.rank_frames(sims, collection, top_k=50)
    assert [(c["video_id"], c["score"], c["frames"]) for c in chains] == [
        (h["video_id"], h["score"],
         [{"row": h["row"], "frame_idx": h["frame_idx"], "time": h["time"], "score": h["score"]}])
        for h in hits
    ]


def test_temporal_search_scene_norm_z(collection):
    door, desk = scene_sims(collection, "door", "desk")
    chains = search.temporal_search(collection, [door, desk], scene_norm="z")
    video, frames = chain_keyframes(collection, chains[0])
    assert video == "V003" and frames[0] in (10, 11) and frames[1] in (14, 15)
    # z-normalised chains do not depend on the scale or offset of one scene's similarities
    rescaled = search.temporal_search(collection, [door, desk * 4.0 + 0.5], scene_norm="z", top_k=30)
    assert [[f["row"] for f in c["frames"]] for c in rescaled] == [
        [f["row"] for f in c["frames"]] for c in chains[:30]
    ]
    # default stays the raw sum
    raw = search.temporal_search(collection, [door, desk])
    assert raw[0]["score"] == pytest.approx((raw[0]["frames"][0]["score"] + raw[0]["frames"][1]["score"]) / 2)
    with pytest.raises(ValueError):
        search.temporal_search(collection, [door, desk], scene_norm="minmax")


def test_temporal_search_candidate_videos(collection):
    door, desk = scene_sims(collection, "door", "desk")
    chains = search.temporal_search(collection, [door, desk], candidate_videos=1)
    top_by_scene = {search.rank_frames(s, collection, 1)[0]["video_id"] for s in (door, desk)}
    assert {c["video_id"] for c in chains} <= top_by_scene


def test_temporal_search_validates_arguments(collection):
    sims = scene_sims(collection, "door", "desk")
    for window in [(5.0, 5.0), (-1.0, 30.0), (10.0, 2.0)]:
        with pytest.raises(ValueError):
            search.temporal_search(collection, sims, window_s=window)
    with pytest.raises(ValueError):
        search.temporal_search(collection, [])
    with pytest.raises(ValueError):
        search.temporal_search(collection, [sims[0], sims[1][:10]])


def test_temporal_search_does_not_assume_row_order(synthetic_collection_dir, tmp_path, collection):
    """Rows of the same video need not be contiguous nor sorted by time."""
    dst = tmp_path / "shuffled"
    shutil.copytree(synthetic_collection_dir, dst, ignore=shutil.ignore_patterns("frames"))
    meta = json.loads((dst / "metadata.json").read_text(encoding="utf-8"))
    emb = np.load(dst / "embeddings.npy")
    invalid = set(json.loads((dst / "invalid_rows.json").read_text(encoding="utf-8")))
    perm = np.random.default_rng(3).permutation(len(meta))
    np.save(dst / "embeddings.npy", emb[perm])
    (dst / "metadata.json").write_text(json.dumps([meta[i] for i in perm]), encoding="utf-8")
    (dst / "invalid_rows.json").write_text(
        json.dumps([j for j, i in enumerate(perm.tolist()) if i in invalid]), encoding="utf-8"
    )
    shuffled = Collection.load(dst)

    def signature(coll):
        chains = search.temporal_search(coll, scene_sims(coll, "door", "desk"), top_k=40)
        return [(c["video_id"], round(c["score"], 5), [f["frame_idx"] for f in c["frames"]])
                for c in chains]

    assert signature(shuffled) == signature(collection)
    assert np.all(np.diff(shuffled.times[shuffled.rows_of("V002")]) > 0)


# ------------------------------------------------------------------------------- invalid rows
def test_invalid_rows_are_never_returned(collection):
    invalid_row = int(np.flatnonzero(~collection.valid)[2])
    # a query aimed exactly at an invalid keyframe
    sims = collection.similarities(collection.embedding(invalid_row))
    invalid = set(np.flatnonzero(~collection.valid).tolist())

    returned = {h["row"] for h in search.rank_frames(sims, collection, top_k=collection.n)}
    returned |= {h["row"] for h in search.similar_frames(collection, invalid_row, top_k=collection.n)}
    for n in (1, 2, 3):
        chains = search.temporal_search(collection, [sims] * n, window_s=(0, 1000), top_k=1000)
        returned |= {f["row"] for c in chains for f in c["frames"]}
    assert returned and not returned & invalid


# ----------------------------------------------------------------------------- by example
def test_similar_frames(collection):
    query = next(r for r in collection.rows_of("V003") if kf(collection, r) == 10)
    hits = search.similar_frames(collection, int(query), top_k=4)
    door_rows = {(f"V{v:03d}", k) for v, kfs, c in PLANTED if c == "door" for k in kfs}
    assert query not in {h["row"] for h in hits}
    assert {(h["video_id"], kf(collection, h["row"])) for h in hits} == door_rows - {("V003", 10)}

    other = search.similar_frames(collection, int(query), top_k=50, exclude_same_video=True)
    assert other and all(h["video_id"] != "V003" for h in other)


# ------------------------------------------------------------------------------ float16
def test_float16_embeddings_give_the_same_ranking(synthetic_collection_dir, tmp_path, collection):
    dst = tmp_path / "f16"
    shutil.copytree(synthetic_collection_dir, dst, ignore=shutil.ignore_patterns("frames"))
    np.save(dst / "embeddings.npy", np.load(dst / "embeddings.npy").astype(np.float16))
    half = Collection.load(dst)
    assert half.embeddings.dtype == np.float16

    for concept in CONCEPT_PHRASES:
        (a,) = scene_sims(collection, concept)
        (b,) = scene_sims(half, concept)
        assert b.dtype == np.float32
        assert [h["row"] for h in search.rank_frames(a, collection, 50)] == [
            h["row"] for h in search.rank_frames(b, half, 50)
        ]
    def chain_rows(coll: Collection) -> list[list[int]]:
        chains = search.temporal_search(coll, scene_sims(coll, "door", "desk"), top_k=30)
        return [[f["row"] for f in c["frames"]] for c in chains]

    assert chain_rows(collection) == chain_rows(half)


# ---------------------------------------------------------------------------- video view
def test_video_timeline(collection):
    door, desk = scene_sims(collection, "door", "desk")
    timeline = search.video_timeline(collection, "V003", [door, desk])
    assert timeline["video"]["video_id"] == "V003"
    frames = timeline["keyframes"]
    assert len(frames) == 40
    assert [f["time"] for f in frames] == sorted(f["time"] for f in frames)
    assert frames[0]["valid"] is False and frames[0]["scores"] == [None, None]
    assert frames[11]["scores"][0] == pytest.approx(float(door[frames[11]["row"]]))
    json.dumps(timeline)  # JSON-serialisable (no inf)

    plain = search.video_timeline(collection, "V003")
    assert all(f["scores"] is None for f in plain["keyframes"])
    with pytest.raises(KeyError):
        search.video_timeline(collection, "V999")


def test_submission_segment_clamps(collection):
    duration_ms = int(collection.video("V002")["duration"] * 1000)  # 80 s
    # default: a point at the frame time (DRES requires [start, end] inside the target segment)
    assert search.submission_segment(collection, "V002", 40.0) == (40000, 40000)
    assert search.submission_segment(collection, "V002", 12.3456) == (12346, 12346)
    assert search.submission_segment(collection, "V002", 500.0) == (duration_ms, duration_ms)
    assert search.submission_segment(collection, "V002", -3.0) == (0, 0)
    # an explicit width is still clamped to [0, duration]
    assert search.submission_segment(collection, "V002", 40.0, half_width_s=2.0) == (38000, 42000)
    assert search.submission_segment(collection, "V002", 1.0, half_width_s=2.0) == (0, 3000)
    assert search.submission_segment(collection, "V002", 79.0, half_width_s=2.0) == (77000, duration_ms)
    assert search.submission_segment(collection, "V002", -3.0, half_width_s=1.0) == (0, 0)
    with pytest.raises(ValueError):
        search.submission_segment(collection, "V002", 10.0, half_width_s=-1.0)
    with pytest.raises(ValueError):
        search.submission_segment(collection, "V002", float("nan"))
    with pytest.raises(KeyError):
        search.submission_segment(collection, "V999", 10.0)


# -------------------------------------------------------------------------------- timing
@pytest.mark.skipif(not os.environ.get("VBS_RUN_SLOW"), reason="timing test; set VBS_RUN_SLOW=1")
def test_temporal_search_speed_at_aic_scale(tmp_path):
    """Three scenes over 177,321 keyframes of 873 videos must take < 1.5 s on CPU."""
    n_rows, n_videos = 177_321, 873
    rng = np.random.default_rng(0)
    video_of_row = np.sort(rng.integers(0, n_videos, n_rows))
    video_of_row[:n_videos] = np.arange(n_videos)  # every video has keyframes
    video_of_row.sort()
    meta, previous, t = [], -1, 0.0
    for v in video_of_row.tolist():
        t = 0.0 if v != previous else t + float(rng.uniform(1.0, 9.0))
        n = 1 if v != previous else meta[-1]["n"] + 1
        previous = v
        meta.append({"video_id": f"L{v:04d}", "n": n, "frame_idx": int(t * 25), "pts_time": t,
                     "fps": 25.0, "frame_filename": f"{n:03d}.jpg"})
    videos = {m["video_id"]: {"fps": 25.0, "duration": 4000.0, "n_frames": 100000, "title": "",
                              "youtube_id": None, "file": None} for m in meta}
    np.save(tmp_path / "embeddings.npy", np.zeros((n_rows, 4), dtype=np.float16))
    (tmp_path / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    (tmp_path / "videos.json").write_text(json.dumps(videos), encoding="utf-8")
    (tmp_path / "invalid_rows.json").write_text(
        json.dumps([i for i, m in enumerate(meta) if m["n"] <= 2]), encoding="utf-8")
    (tmp_path / "collection.json").write_text(json.dumps({
        "name": "scale", "dim": 4, "embeddings": "embeddings.npy", "metadata": "metadata.json",
        "frames_dir": "frames", "videos": "videos.json", "invalid_rows": "invalid_rows.json"}),
        encoding="utf-8")
    coll = Collection.load(tmp_path)
    sims = [rng.standard_normal(n_rows).astype(np.float32) * 0.05 + 0.1 for _ in range(3)]

    search.temporal_search(coll, sims)  # warm-up
    timings = {}
    for n in (2, 3):
        start = time.perf_counter()
        chains = search.temporal_search(coll, sims[:n])
        timings[n] = time.perf_counter() - start
        assert len(chains) == 200
    print(f"temporal_search on {n_rows} rows: 2 scenes {timings[2]:.3f} s, 3 scenes {timings[3]:.3f} s")
    assert timings[3] < 1.5
