"""Ingestion tests: shot detection, keyframe selection, collection files, resume logic.

No network and no model: videos are synthesised with OpenCV (colour-block scenes with a moving
square and pixel noise, hard cuts at known frames) and the image encoder is a small deterministic
stand-in that implements the ``ImageEncoder`` protocol of ``encode_images.py``.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

cv2 = pytest.importorskip("cv2", reason="ingestion tests need opencv-python(-headless)")
torch = pytest.importorskip("torch", reason="ingestion tests need torch (CPU build is enough)")

from scenechain.ingest import encode_images as enc  # noqa: E402
from scenechain.ingest import extract_keyframes as ek  # noqa: E402
from tests.conftest import N_VIDEOS, PER_VIDEO, build_synthetic_collection

FPS = 25.0
# (length in frames, BGR colour) per scene; cuts at 40, 160, 195, 255; 300 frames = 12 s
SCENES = [(40, (40, 40, 200)), (120, (200, 60, 30)), (35, (40, 180, 60)), (60, (170, 170, 170)), (45, (20, 120, 220))]
TRUE_CUTS = [40, 160, 195, 255]
N_FRAMES = 300
METADATA_KEYS = {"video_id", "n", "frame_idx", "pts_time", "fps", "frame_filename"}
VIDEO_KEYS = {"fps", "duration", "n_frames", "title", "youtube_id", "file"}


def write_synthetic_video(path: Path, seed: int = 0) -> Path:
    """MJPG AVI, 192x108, with the scenes of ``SCENES``."""
    width, height = 192, 108
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (width, height))
    if not writer.isOpened():
        pytest.skip("OpenCV build cannot write MJPG AVI files")
    rng = np.random.default_rng(seed)
    try:
        for length, colour in SCENES:
            for i in range(length):
                img = np.full((height, width, 3), colour, np.uint8)
                x = (i * 3) % (width - 30)
                cv2.rectangle(img, (x, 30), (x + 30, 70), tuple(255 - c for c in colour), -1)
                noise = rng.integers(-6, 7, img.shape)
                writer.write(np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8))
    finally:
        writer.release()
    return path


@pytest.fixture(scope="module")
def video_dir(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("videos")
    write_synthetic_video(d / "clip_a.avi", seed=1)
    (d / "nested").mkdir()
    write_synthetic_video(d / "nested" / "clip_b.avi", seed=2)
    return d


def assert_cuts_close(found: list[int], expected: list[int], tolerance: int = 2) -> None:
    assert len(found) == len(expected), f"found cuts {found}, expected {expected}"
    for f, e in zip(found, expected):
        assert abs(f - e) <= tolerance, f"found cuts {found}, expected {expected}"


# --------------------------------------------------------------------------- shot detection


def test_histogram_detector_finds_planted_cuts(video_dir):
    p = ek.Params(method="histogram")
    detector = ek.HistogramCutDetector(p.hist_threshold, p.pixel_threshold, round(p.min_shot * FPS))
    cuts, n_frames = ek.detect_cuts(video_dir / "clip_a.avi", FPS, detector)
    assert n_frames == N_FRAMES
    assert_cuts_close(cuts, TRUE_CUTS)


def test_scenedetect_detector_finds_planted_cuts(video_dir):
    pytest.importorskip("scenedetect")
    p = ek.Params(method="scenedetect")
    detector = ek.ContentCutDetector(FPS, p.threshold, round(p.min_shot * FPS))
    cuts, n_frames = ek.detect_cuts(video_dir / "clip_a.avi", FPS, detector)
    assert n_frames == N_FRAMES
    assert_cuts_close(cuts, TRUE_CUTS)


def test_min_shot_suppresses_close_cuts(video_dir):
    detector = ek.HistogramCutDetector(0.25, 20.0, min_frames=50)
    cuts, _ = ek.detect_cuts(video_dir / "clip_a.avi", FPS, detector)
    assert all(b - a >= 50 for a, b in zip([0] + cuts, cuts))


def test_cuts_to_shots_cover_every_frame_once():
    shots = ek.cuts_to_shots([40, 160], 200)
    assert shots == [ek.Shot(0, 39), ek.Shot(40, 159), ek.Shot(160, 199)]
    assert ek.cuts_to_shots([], 10) == [ek.Shot(0, 9)]


def test_read_shot_file(tmp_path):
    good = tmp_path / "good.csv"
    good.write_text("# start,end\n0,39\n\n40,159\n160,299\n", encoding="utf-8")
    assert ek.read_shot_file(good) == [ek.Shot(0, 39), ek.Shot(40, 159), ek.Shot(160, 299)]

    exclusive = tmp_path / "exclusive.csv"
    exclusive.write_text("0,40\n40,160\n", encoding="utf-8")
    with pytest.raises(ek.IngestError, match="exclusive"):
        ek.read_shot_file(exclusive)

    for name, text in {"words.csv": "start,end\n", "three.csv": "0,1,2\n", "backwards.csv": "9,3\n", "empty.csv": "# nothing\n"}.items():
        bad = tmp_path / name
        bad.write_text(text, encoding="utf-8")
        with pytest.raises(ek.IngestError):
            ek.read_shot_file(bad)


# --------------------------------------------------------------------------- keyframe selection


def test_short_shot_gets_its_middle_frame():
    assert ek.shot_keyframes([ek.Shot(0, 39)], FPS, long_shot=4.0, edge=0.5, max_gap=0.0) == [19]


def test_long_shot_gets_start_middle_end():
    # 120 frames = 4.8 s > 4 s: edge 0.5 s = 12 frames inside each end
    assert ek.shot_keyframes([ek.Shot(40, 159)], FPS, long_shot=4.0, edge=0.5, max_gap=0.0) == [52, 99, 147]


def test_max_gap_fills_long_shots():
    shot = ek.Shot(0, 499)  # 20 s
    picks = ek.shot_keyframes([shot], FPS, long_shot=4.0, edge=0.5, max_gap=2.0)
    assert picks[0] == 12 and picks[-1] == 487 and 249 in picks
    assert max(b - a for a, b in zip(picks, picks[1:])) <= 50
    assert picks == sorted(set(picks))


def test_uniform_selector_follows_timestamps_not_frame_counts():
    select = ek.UniformSelector(sample_fps=1.0)
    # a variable-frame-rate stream: frames every 0.4 s, then a 2.1 s gap
    times = [0.0, 0.4, 0.8, 1.2, 1.6, 3.7, 4.1, 4.5, 4.9, 5.3]
    chosen = [t for i, t in enumerate(times) if select(i, t)]
    assert chosen == [0.0, 1.2, 3.7, 4.1, 5.3]


# --------------------------------------------------------------------------- collection output


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_extract_collection_writes_documented_schema(video_dir, tmp_path):
    out = tmp_path / "col"
    summary = ek.extract_collection(video_dir, out, ek.Params(method="histogram"), log=lambda _: None)
    assert summary["videos"] == 2 and summary["extracted"] == 2 and summary["skipped"] == 0

    metadata = read_json(out / "metadata.json")
    videos = read_json(out / "videos.json")
    collection = read_json(out / "collection.json")
    assert collection["name"] == "col" and collection["metadata"] == "metadata.json"
    assert collection["videos"] == "videos.json" and collection["frames_dir"] == "frames"
    assert set(videos) == {"clip_a", "clip_b"}
    assert [m["video_id"] for m in metadata] == sorted(m["video_id"] for m in metadata)
    assert summary["keyframes"] == len(metadata)

    for vid, video in videos.items():
        assert set(video) == VIDEO_KEYS
        assert video["n_frames"] == N_FRAMES and video["fps"] == pytest.approx(FPS)
        assert video["duration"] == pytest.approx(N_FRAMES / FPS, abs=0.05)
        assert video["youtube_id"] is None and (out / video["file"]).resolve().is_file()
        rows = [m for m in metadata if m["video_id"] == vid]
        record = read_json(out / "ingest" / f"{vid}.json")
        assert_cuts_close([s[0] for s in record["shots"][1:]], TRUE_CUTS)
        shots = [ek.Shot(*s) for s in record["shots"]]
        expected = ek.shot_keyframes(shots, FPS, long_shot=4.0, edge=0.5, max_gap=0.0)
        assert [m["frame_idx"] for m in rows] == expected
        assert [m["n"] for m in rows] == list(range(1, len(rows) + 1))
        for m in rows:
            assert set(m) == METADATA_KEYS
            assert isinstance(m["n"], int) and isinstance(m["frame_idx"], int)
            assert isinstance(m["pts_time"], float) and isinstance(m["fps"], float)
            assert m["pts_time"] == pytest.approx(m["frame_idx"] / FPS, abs=1e-3)
            image = out / "frames" / vid / m["frame_filename"]
            with Image.open(image) as im:
                assert im.format == "JPEG" and im.size == (192, 108)  # never upscaled


def test_uniform_method_samples_by_time(video_dir, tmp_path):
    out = tmp_path / "uniform"
    ek.extract_collection(video_dir, out, ek.Params(method="uniform", fps=2.0), log=lambda _: None)
    rows = [m for m in read_json(out / "metadata.json") if m["video_id"] == "clip_a"]
    assert len(rows) == 24
    for k, m in enumerate(rows):
        assert 0.0 <= m["pts_time"] - k * 0.5 < 1.0 / FPS + 1e-6
    assert read_json(out / "ingest" / "clip_a.json")["shots"] is None


def test_external_shot_files(video_dir, tmp_path):
    shots_dir = tmp_path / "shots"
    shots_dir.mkdir()
    for vid in ("clip_a", "clip_b"):
        (shots_dir / f"{vid}.csv").write_text("0,149\n150,299\n", encoding="utf-8")
    out = tmp_path / "external"
    ek.extract_collection(video_dir, out, ek.Params(method="shots"), shots_dir=shots_dir, log=lambda _: None)
    rows = [m["frame_idx"] for m in read_json(out / "metadata.json") if m["video_id"] == "clip_a"]
    assert rows == [12, 74, 137, 162, 224, 287]

    (shots_dir / "clip_b.csv").unlink()
    with pytest.raises(ek.IngestError, match="clip_b"):
        ek.extract_collection(video_dir, tmp_path / "missing", ek.Params(method="shots"), shots_dir=shots_dir,
                              log=lambda _: None)


def test_extraction_resumes_per_video(video_dir, tmp_path):
    out = tmp_path / "resume"
    params = ek.Params(method="histogram")
    ek.extract_collection(video_dir, out, params, log=lambda _: None)
    frame = out / "frames" / "clip_a" / "00001.jpg"
    stamp = frame.stat().st_mtime_ns
    metadata = (out / "metadata.json").read_bytes()

    again = ek.extract_collection(video_dir, out, params, log=lambda _: None)
    assert again["extracted"] == 0 and again["skipped"] == 2
    assert frame.stat().st_mtime_ns == stamp
    assert (out / "metadata.json").read_bytes() == metadata

    # an interrupted video has frames but no record: its leftovers are removed and it is redone
    (out / "ingest" / "clip_b.json").unlink()
    (out / "frames" / "clip_b" / "99999.jpg").write_bytes(b"partial")
    partial = ek.extract_collection(video_dir, out, params, log=lambda _: None)
    assert partial["extracted"] == 1 and partial["skipped"] == 1
    assert not (out / "frames" / "clip_b" / "99999.jpg").exists()
    assert (out / "metadata.json").read_bytes() == metadata

    # changed parameters invalidate every record
    changed = ek.extract_collection(video_dir, out, ek.Params(method="histogram", quality=70), log=lambda _: None)
    assert changed["extracted"] == 2


def test_encoder_copies_are_written_and_registered(video_dir, tmp_path):
    out = tmp_path / "encode_copies"
    ek.extract_collection(video_dir, out, ek.Params(method="histogram", long_side=96, encode_long_side=150),
                          log=lambda _: None)
    metadata = read_json(out / "metadata.json")
    assert read_json(out / "collection.json")["encode_frames_dir"] == "frames_encode"
    for m in metadata:
        with Image.open(out / "frames" / m["video_id"] / m["frame_filename"]) as small, \
                Image.open(out / "frames_encode" / m["video_id"] / m["frame_filename"]) as large:
            assert small.size == (96, 54) and large.size == (150, 84)

    ek.extract_collection(video_dir, out, ek.Params(method="histogram", long_side=96), log=lambda _: None)
    assert "encode_frames_dir" not in read_json(out / "collection.json")
    assert not any((out / "frames_encode").rglob("*.jpg"))


def test_changed_metadata_unregisters_stale_embeddings(video_dir, tmp_path):
    out = tmp_path / "stale"
    ek.extract_collection(video_dir, out, ek.Params(method="histogram"), log=lambda _: None)
    collection = read_json(out / "collection.json") | {"embeddings": "embeddings.npy", "model": "m", "dim": 8}
    (out / "collection.json").write_text(json.dumps(collection), encoding="utf-8")

    ek.extract_collection(video_dir, out, ek.Params(method="histogram"), log=lambda _: None)
    assert "embeddings" in read_json(out / "collection.json")  # same metadata: registration kept

    ek.extract_collection(video_dir, out, ek.Params(method="histogram", long_shot=10.0), log=lambda _: None)
    assert not {"embeddings", "model", "dim"} & set(read_json(out / "collection.json"))


def test_discover_videos_rejects_duplicate_ids(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    (tmp_path / "a" / "same.mp4").write_bytes(b"")
    (tmp_path / "b" / "same.webm").write_bytes(b"")
    with pytest.raises(ek.IngestError, match="same"):
        ek.discover_videos(tmp_path)
    with pytest.raises(ek.IngestError):
        ek.discover_videos(tmp_path / "missing")


def test_discover_videos_rejects_ids_the_server_cannot_serve(tmp_path):
    (tmp_path / "my clip (1).mp4").write_bytes(b"")
    with pytest.raises(ek.IngestError, match="rename"):
        ek.discover_videos(tmp_path)


# --------------------------------------------------------------------------- image encoding


class MeanColourPreprocess:
    """Top-level (picklable) stand-in for the image processor: 4x4 average-pooled RGB in [0, 1]."""

    def __call__(self, image: Image.Image) -> torch.Tensor:
        small = image.convert("RGB").resize((4, 4), Image.BILINEAR)
        return torch.from_numpy(np.asarray(small, dtype=np.float32) / 255.0).permute(2, 0, 1).contiguous()


class FakeImageEncoder:
    """Deterministic random projection of the pooled pixels; counts encoded images."""

    dim = 8
    device = "cpu"

    def __init__(self, fail_after: int | None = None) -> None:
        self.preprocess = MeanColourPreprocess()
        self.projection = np.random.default_rng(3).standard_normal((48, self.dim)).astype(np.float32)
        self.fail_after = fail_after
        self.images = 0

    def encode(self, pixel_values: torch.Tensor) -> np.ndarray:
        if self.fail_after is not None and self.images >= self.fail_after:
            raise KeyboardInterrupt("simulated interruption")
        x = pixel_values.reshape(len(pixel_values), -1).numpy() - 0.5
        vecs = x @ self.projection
        self.images += len(vecs)
        return (vecs / np.linalg.norm(vecs, axis=1, keepdims=True)).astype(np.float32)

    def identity(self) -> dict:
        return {"model": "fake-image-encoder", "precision": "fp32"}


@pytest.fixture()
def image_collection(tmp_path) -> Path:
    root = build_synthetic_collection(tmp_path / "images")
    (root / "embeddings.npy").unlink()
    collection = read_json(root / "collection.json")
    for key in ("embeddings", "model", "dim"):
        collection.pop(key)
    (root / "collection.json").write_text(json.dumps(collection), encoding="utf-8")
    return root


def quiet(_: str) -> None:
    return None


def test_encode_collection_writes_float16_embeddings(image_collection):
    encoder = FakeImageEncoder()
    report = enc.encode_collection(image_collection, encoder, batch_size=16, shard_size=50, log=quiet)
    n = N_VIDEOS * PER_VIDEO
    assert report["encoded"] == n and encoder.images == n
    assert report["images_per_s"] > 0

    emb = np.load(image_collection / "embeddings.npy")
    assert emb.shape == (n, FakeImageEncoder.dim) and emb.dtype == np.float16
    assert np.allclose(np.linalg.norm(emb.astype(np.float32), axis=1), 1.0, atol=2e-3)
    collection = read_json(image_collection / "collection.json")
    assert collection["embeddings"] == "embeddings.npy" and collection["dim"] == 8
    assert collection["model"] == "fake-image-encoder" and collection["name"] == "synthetic"
    assert not (image_collection / enc.PARTS_DIR).exists()

    # row order follows metadata.json: re-encode one image directly and compare
    metadata = read_json(image_collection / "metadata.json")
    m = metadata[123]
    with Image.open(image_collection / "frames" / m["video_id"] / m["frame_filename"]) as im:
        direct = FakeImageEncoder().encode(MeanColourPreprocess()(im)[None])[0]
    assert np.allclose(emb[123].astype(np.float32), direct, atol=2e-3)

    again = FakeImageEncoder()
    assert enc.encode_collection(image_collection, again, batch_size=16, shard_size=50, log=quiet)["encoded"] == 0
    assert again.images == 0


def test_encode_resumes_from_completed_shards(image_collection, tmp_path):
    reference_dir = tmp_path / "reference"
    shutil.copytree(image_collection, reference_dir)
    enc.encode_collection(reference_dir, FakeImageEncoder(), batch_size=16, shard_size=50, float16=False, log=quiet)
    reference = np.load(reference_dir / "embeddings.npy")

    with pytest.raises(KeyboardInterrupt):
        enc.encode_collection(image_collection, FakeImageEncoder(fail_after=100), batch_size=16, shard_size=50,
                              float16=False, log=quiet)
    parts = sorted(p.name for p in (image_collection / enc.PARTS_DIR).glob("part-*.npy"))
    assert parts == ["part-00000.npy", "part-00001.npy"]
    assert not (image_collection / "embeddings.npy").exists()

    resumed = FakeImageEncoder()
    report = enc.encode_collection(image_collection, resumed, batch_size=16, shard_size=50, float16=False, log=quiet)
    assert resumed.images == N_VIDEOS * PER_VIDEO - 100 == report["encoded"]
    # equal up to BLAS rounding: the resumed run batches the remaining rows differently
    assert np.allclose(np.load(image_collection / "embeddings.npy"), reference, atol=1e-6)


def test_encode_refuses_mismatched_shards_unless_restarted(image_collection):
    with pytest.raises(KeyboardInterrupt):
        enc.encode_collection(image_collection, FakeImageEncoder(fail_after=60), batch_size=16, shard_size=50, log=quiet)
    with pytest.raises(enc.EncodeError, match="shard_size"):
        enc.encode_collection(image_collection, FakeImageEncoder(), batch_size=16, shard_size=60, log=quiet)
    fresh = FakeImageEncoder()
    enc.encode_collection(image_collection, fresh, batch_size=16, shard_size=60, restart=True, log=quiet)
    assert fresh.images == N_VIDEOS * PER_VIDEO


def test_encode_prefers_encode_frames_dir(image_collection):
    enc.encode_collection(image_collection, FakeImageEncoder(), batch_size=16, shard_size=50, log=quiet)
    from_frames = np.load(image_collection / "embeddings.npy")

    # an encoder copy with inverted colours: same names, different pixels
    for src in (image_collection / "frames").rglob("*.jpg"):
        target = image_collection / "frames_encode" / src.relative_to(image_collection / "frames")
        target.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(src) as im:
            Image.eval(im.convert("RGB"), lambda v: 255 - v).save(target, "JPEG")
    collection = read_json(image_collection / "collection.json") | {"encode_frames_dir": "frames_encode"}
    (image_collection / "collection.json").write_text(json.dumps(collection), encoding="utf-8")

    encoder = FakeImageEncoder()
    enc.encode_collection(image_collection, encoder, batch_size=16, shard_size=50, log=quiet)
    assert encoder.images == N_VIDEOS * PER_VIDEO  # a different frame source is not "up to date"
    from_copies = np.load(image_collection / "embeddings.npy")
    assert not np.allclose(from_frames, from_copies, atol=1e-2)
    assert read_json(image_collection / "encode_manifest.json")["frames"] == "frames_encode"


def test_encode_reports_missing_frames(image_collection):
    metadata = read_json(image_collection / "metadata.json")
    os.remove(image_collection / "frames" / metadata[7]["video_id"] / metadata[7]["frame_filename"])
    with pytest.raises(enc.EncodeError, match="cannot read keyframe"):
        enc.encode_collection(image_collection, FakeImageEncoder(), batch_size=16, shard_size=50, log=quiet)


def test_verify_against_reports_parity(image_collection):
    enc.encode_collection(image_collection, FakeImageEncoder(), batch_size=16, shard_size=50, log=quiet)
    report = enc.verify_against(image_collection / "embeddings.npy", image_collection / "metadata.json",
                                image_collection / "frames", FakeImageEncoder(), k=20, seed=1, batch_size=8, log=quiet)
    assert report["k"] == 20 and report["cosine_min"] > 0.999 and report["share_above_0_999"] == 1.0
    assert len(report["worst"]) == 5 and {"row", "video_id", "frame_filename", "cosine"} == set(report["worst"][0])


def test_end_to_end_video_to_loadable_collection(video_dir, tmp_path):
    out = tmp_path / "e2e"
    ek.extract_collection(video_dir, out, ek.Params(method="histogram"), log=quiet)
    enc.encode_collection(out, FakeImageEncoder(), batch_size=4, shard_size=5, log=quiet)
    metadata = read_json(out / "metadata.json")
    emb = np.load(out / "embeddings.npy", mmap_mode="r")
    assert emb.shape == (len(metadata), FakeImageEncoder.dim)
    collection_module = pytest.importorskip("scenechain.collection")
    col = collection_module.Collection.load(out)
    assert col.n == len(metadata) and col.dim == FakeImageEncoder.dim
    assert len(col.rows_of("clip_a")) == sum(m["video_id"] == "clip_a" for m in metadata)
