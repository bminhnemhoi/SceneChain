"""Shared fixtures for the VBS tool tests.

Everything here is synthetic and tiny: no 817 MB index, no SigLIP-2 model, no network.

The synthetic collection has 6 videos x 40 keyframes (one keyframe every 2 s, 25 fps) with
D = 16 embeddings. Concept vectors are planted so that retrieval results are known in advance:

* ``door``  – "a man opens a wooden door"
* ``desk``  – "the man sits down at a desk"
* ``boat``  – "a blue boat on a river"
* ``dog``   – "a dog runs across a field"

Planted structure (keyframe numbers are 0-based within the video):

* V003: ``door`` at keyframes 10–11 (t = 20–22 s), then ``desk`` at 14–15 (t = 28–30 s)
  -> the correct door -> desk chain, gap 8 s.
* V005: ``desk`` at 10–11, then ``door`` at 14–15 -> the same two scenes in the WRONG order.
* V001: ``boat`` at 5–7.   V002: ``dog`` at 30–31.
* V004: ``door`` at 2 only, ``desk`` at 38 only (t = 76 s) -> right order but 72 s apart,
  outside a 30 s window.
* V006: noise only.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DIM = 16
N_VIDEOS = 6
PER_VIDEO = 40
FPS = 25.0
STEP_S = 2.0
#: per-dimension Gaussian noise added to every keyframe; at 0.10 the planted scenes beat every noise
#: keyframe (at 0.35 some noise keyframes outscored planted ones and made chain assertions ambiguous)
NOISE_SCALE = 0.10

CONCEPT_PHRASES = {
    "door": "a man opens a wooden door",
    "desk": "the man sits down at a desk",
    "boat": "a blue boat on a river",
    "dog": "a dog runs across a field",
}

# (video number, keyframe numbers, concept)
PLANTED = [
    (3, [10, 11], "door"),
    (3, [14, 15], "desk"),
    (5, [10, 11], "desk"),
    (5, [14, 15], "door"),
    (1, [5, 6, 7], "boat"),
    (2, [30, 31], "dog"),
    (4, [2], "door"),
    (4, [38], "desk"),
]


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def concept_vectors() -> dict:
    rng = np.random.default_rng(20270105)
    return {name: _unit(rng.standard_normal(DIM)).astype(np.float32) for name in CONCEPT_PHRASES}


class FakeEncoder:
    """Maps a phrase that contains a concept keyword to its planted vector.

    Unknown text gets a deterministic pseudo-random unit vector derived from sha1(text), so the
    encoder is total and repeatable.
    """

    name = "fake-encoder"

    def __init__(self) -> None:
        self.vectors = concept_vectors()
        self.calls = 0

    def encode(self, text: str) -> np.ndarray:
        self.calls += 1
        low = (text or "").lower()
        for name, vec in self.vectors.items():
            if name in low:
                return vec.copy()
        seed = int(hashlib.sha1(low.encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        return _unit(rng.standard_normal(DIM)).astype(np.float32)

    def encode_chunks(self, text: str) -> list:
        return [self.encode(text)]


def build_synthetic_collection(root: Path, with_images: bool = True) -> Path:
    """Write a complete collection directory (format of docs/ARCHITECTURE.md §3) under ``root``."""
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    concepts = concept_vectors()

    emb = np.zeros((N_VIDEOS * PER_VIDEO, DIM), dtype=np.float32)
    metadata, videos = [], {}
    planted = {}
    for vnum, kfs, concept in PLANTED:
        for k in kfs:
            planted[(vnum, k)] = concept

    row = 0
    for v in range(1, N_VIDEOS + 1):
        vid = f"V{v:03d}"
        for k in range(PER_VIDEO):
            noise = rng.standard_normal(DIM).astype(np.float32) * NOISE_SCALE
            concept = planted.get((v, k))
            vec = noise + (concepts[concept] * 1.0 if concept else 0.0)
            emb[row] = _unit(vec)
            t = k * STEP_S
            metadata.append({
                "video_id": vid,
                "n": k + 1,
                "frame_idx": int(round(t * FPS)),
                "pts_time": float(t),
                "fps": FPS,
                "frame_filename": f"{k + 1:03d}.jpg",
            })
            row += 1
        videos[vid] = {
            "fps": FPS,
            "duration": PER_VIDEO * STEP_S,
            "n_frames": int(PER_VIDEO * STEP_S * FPS),
            "title": f"Synthetic video {v}",
            "youtube_id": None,
            "file": None,
        }

    np.save(root / "embeddings.npy", emb)
    (root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (root / "videos.json").write_text(json.dumps(videos), encoding="utf-8")
    # the first keyframe of every video is marked invalid, mirroring the AIC skip rule
    invalid = [i for i, m in enumerate(metadata) if m["n"] == 1]
    (root / "invalid_rows.json").write_text(json.dumps(invalid), encoding="utf-8")
    (root / "collection.json").write_text(json.dumps({
        "name": "synthetic",
        "model": "fake",
        "dim": DIM,
        "embeddings": "embeddings.npy",
        "metadata": "metadata.json",
        "frames_dir": "frames",
        "videos": "videos.json",
        "invalid_rows": "invalid_rows.json",
    }), encoding="utf-8")

    if with_images:
        try:
            from PIL import Image
        except Exception:  # noqa: BLE001
            with_images = False
    if with_images:
        for m in metadata:
            d = root / "frames" / m["video_id"]
            d.mkdir(parents=True, exist_ok=True)
            shade = (m["n"] * 6) % 255
            Image.new("RGB", (16, 9), (shade, 80, 160)).save(d / m["frame_filename"], "JPEG")
    return root


@pytest.fixture(scope="session")
def synthetic_collection_dir(tmp_path_factory) -> Path:
    return build_synthetic_collection(tmp_path_factory.mktemp("collection"))


@pytest.fixture()
def fake_encoder() -> FakeEncoder:
    return FakeEncoder()


@pytest.fixture(scope="session")
def concept_phrases() -> dict:
    return dict(CONCEPT_PHRASES)
