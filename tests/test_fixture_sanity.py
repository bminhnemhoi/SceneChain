import json

import numpy as np

from tests.conftest import CONCEPT_PHRASES, PER_VIDEO, N_VIDEOS


def test_synthetic_collection_is_well_formed(synthetic_collection_dir):
    d = synthetic_collection_dir
    emb = np.load(d / "embeddings.npy")
    meta = json.loads((d / "metadata.json").read_text(encoding="utf-8"))
    assert emb.shape == (N_VIDEOS * PER_VIDEO, 16)
    assert len(meta) == emb.shape[0]
    assert np.allclose(np.linalg.norm(emb, axis=1), 1.0, atol=1e-5)
    assert (d / "frames" / "V003" / "011.jpg").exists()


def test_fake_encoder_hits_planted_rows(synthetic_collection_dir, fake_encoder):
    d = synthetic_collection_dir
    emb = np.load(d / "embeddings.npy")
    meta = json.loads((d / "metadata.json").read_text(encoding="utf-8"))
    sims = emb @ fake_encoder.encode(CONCEPT_PHRASES["boat"])
    best = meta[int(np.argmax(sims))]
    assert best["video_id"] == "V001" and best["n"] - 1 in (5, 6, 7)
