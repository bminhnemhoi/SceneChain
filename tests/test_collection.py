"""Collection loading, lookups, chunked similarities and validation errors."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from tests.conftest import N_VIDEOS, PER_VIDEO, STEP_S
from scenechain.collection import Collection, CollectionError


@pytest.fixture(scope="module")
def collection(synthetic_collection_dir) -> Collection:
    return Collection.load(synthetic_collection_dir)


def copy_collection(src: Path, dst: Path) -> Path:
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("frames"))
    return dst


def edit_json(path: Path, change) -> None:
    """Replace the JSON document at ``path`` by ``change(document)``."""
    data = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps(change(data)), encoding="utf-8")


def without(mapping: dict, key: str) -> dict:
    return {k: v for k, v in mapping.items() if k != key}


def test_load_basic_properties(collection, synthetic_collection_dir):
    assert collection.name == "synthetic"
    assert (collection.n, collection.dim) == (N_VIDEOS * PER_VIDEO, 16)
    assert collection.n_videos == N_VIDEOS
    assert collection.video_ids == [f"V{v:03d}" for v in range(1, N_VIDEOS + 1)]
    assert collection.frames_dir == synthetic_collection_dir.resolve() / "frames"
    assert isinstance(collection.embeddings, np.memmap)


def test_valid_mask_from_invalid_rows(collection):
    assert collection.valid.dtype == bool
    assert int((~collection.valid).sum()) == N_VIDEOS
    assert np.all(collection.n_in_video[~collection.valid] == 1)
    with pytest.raises(ValueError):
        collection.valid[5] = False  # read-only


def test_row_lookup(collection):
    r = collection.row("V003", int(11 * STEP_S * 25))
    assert r == 2 * PER_VIDEO + 11
    assert collection.meta(r) == {
        "video_id": "V003", "n": 12, "frame_idx": 550, "pts_time": 22.0, "fps": 25.0,
        "frame_filename": "012.jpg",
    }
    assert collection.row("V003", 551) is None
    assert collection.row("V999", 0) is None


def test_rows_of_is_time_sorted(collection):
    rows = collection.rows_of("V004")
    assert len(rows) == PER_VIDEO
    assert {collection.video_id_of(r) for r in rows} == {"V004"}
    assert np.all(np.diff(collection.times[rows]) > 0)
    rows[0] = -1  # a copy: the collection is unaffected
    assert collection.rows_of("V004")[0] != -1
    with pytest.raises(KeyError):
        collection.rows_of("V999")


def test_video_record(collection):
    assert collection.video("V002") == {
        "video_id": "V002", "fps": 25.0, "duration": 80.0, "n_frames": 2000,
        "title": "Synthetic video 2", "youtube_id": None, "file": None,
    }
    with pytest.raises(KeyError):
        collection.video("nope")


def test_frame_path_and_url(collection, synthetic_collection_dir, tmp_path):
    r = collection.row("V001", 0)
    assert collection.frame_path(r).is_file()
    assert collection.frame_url(r) == "/frames/V001/001.jpg"
    with pytest.raises(IndexError):
        collection.frame_url(collection.n)

    dst = copy_collection(synthetic_collection_dir, tmp_path / "c")
    edit_json(dst / "collection.json",
              lambda spec: {**spec, "frame_url": "https://cdn.example.org/{video_id}/{frame_filename}"})
    assert Collection.load(dst).frame_url(r) == "https://cdn.example.org/V001/001.jpg"


def test_similarities_chunked_float32(collection, synthetic_collection_dir):
    emb = np.load(synthetic_collection_dir / "embeddings.npy")
    vec = emb[42]
    for chunk in (7, 64, 10_000):
        sims = collection.similarities(vec, chunk_rows=chunk)
        assert sims.dtype == np.float32 and sims.shape == (collection.n,)
        np.testing.assert_allclose(sims, emb @ vec, rtol=1e-5, atol=1e-6)
    with pytest.raises(ValueError):
        collection.similarities(np.ones(3, dtype=np.float32))


def test_float16_similarities(synthetic_collection_dir, tmp_path):
    dst = copy_collection(synthetic_collection_dir, tmp_path / "f16")
    emb = np.load(dst / "embeddings.npy")
    np.save(dst / "embeddings.npy", emb.astype(np.float16))
    half = Collection.load(dst)
    sims = half.similarities(emb[7], chunk_rows=50)
    assert sims.dtype == np.float32
    np.testing.assert_allclose(sims, emb @ emb[7], atol=1e-3)
    assert half.embedding(7).dtype == np.float32


def test_absolute_paths_in_collection_json(synthetic_collection_dir, tmp_path):
    dst = tmp_path / "pointer"
    dst.mkdir()
    for name in ("metadata.json", "videos.json", "invalid_rows.json"):
        shutil.copy(synthetic_collection_dir / name, dst / name)
    spec = json.loads((synthetic_collection_dir / "collection.json").read_text(encoding="utf-8"))
    spec["embeddings"] = (synthetic_collection_dir / "embeddings.npy").resolve().as_posix()
    spec["frames_dir"] = (synthetic_collection_dir / "frames").resolve().as_posix()
    (dst / "collection.json").write_text(json.dumps(spec), encoding="utf-8")
    pointer = Collection.load(dst)
    assert pointer.n == N_VIDEOS * PER_VIDEO
    assert pointer.frame_path(0).is_file()


@pytest.mark.parametrize(
    "file, change",
    [
        ("collection.json", lambda s: {**s, "dim": 32}),
        ("collection.json", lambda s: without(s, "videos")),
        ("metadata.json", lambda m: m[:-1]),
        ("metadata.json", lambda m: m[:3] + [without(m[3], "pts_time")] + m[4:]),
        ("metadata.json", lambda m: [{**m[0], "pts_time": "soon"}] + m[1:]),
        ("videos.json", lambda v: without(v, "V004")),
        ("videos.json", lambda v: {**v, "V004": without(v["V004"], "duration")}),
        ("videos.json", lambda v: list(v)),
        ("invalid_rows.json", lambda rows: rows + [10_000]),
    ],
)
def test_load_rejects_inconsistent_collections(synthetic_collection_dir, tmp_path, file, change):
    dst = copy_collection(synthetic_collection_dir, tmp_path / "bad")
    edit_json(dst / file, change)
    with pytest.raises(CollectionError):
        Collection.load(dst)


def test_load_rejects_missing_files_and_bad_matrices(synthetic_collection_dir, tmp_path):
    with pytest.raises(FileNotFoundError):
        Collection.load(tmp_path / "does-not-exist")
    dst = copy_collection(synthetic_collection_dir, tmp_path / "ints")
    np.save(dst / "embeddings.npy", np.zeros((N_VIDEOS * PER_VIDEO, 16), dtype=np.int32))
    with pytest.raises(CollectionError):
        Collection.load(dst)
    (dst / "embeddings.npy").unlink()
    with pytest.raises(FileNotFoundError):
        Collection.load(dst)
