"""SiglipTextEncoder reproduces the AIC 2026 query vectors without loading SigLIP-2.

A stub stands in for the model: ``encode_texts`` maps every prompt to a deterministic unit vector
and ``chunk_text`` counts words instead of tokens. Equality with ``StubEngine.query_vector``, an
independent statement of the AIC pipeline's 4-prompt ensemble, then proves that the prompt layout,
slot order, weights and chunk handling are identical, whatever the real model produces.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from scenechain.encoder import SiglipTextEncoder, TextEncoder
from scenechain.siglip_text import SiglipTextTower


class WordTokenizer:
    def __call__(self, text):
        return {"input_ids": text.split()}


class StubProcessor:
    tokenizer = WordTokenizer()


class StubEngine(SiglipTextTower):
    def __init__(self):
        super().__init__()
        self.processor = StubProcessor()
        self.batches: list[int] = []

    def encode_texts(self, texts):
        self.batches.append(len(texts))
        out = []
        for t in texts:
            seed = int(hashlib.sha1(t.encode("utf-8")).hexdigest()[:8], 16)
            v = np.random.default_rng(seed).standard_normal(8).astype(np.float32)
            out.append(v / np.linalg.norm(v))
        return np.stack(out)

    def chunk_text(self, text, max_tokens=12):
        return super().chunk_text(text, max_tokens=max_tokens)

    def query_vector(self, query, query_en):
        """The AIC pipeline's ensemble: (English, original-language slot, keyframe prefix, photo prefix)."""
        prompts = [query_en, query, f"a high quality video keyframe of {query_en}", f"a photo of {query_en}"]
        w = np.array((0.45, 0.35, 0.10, 0.10), dtype=np.float32).reshape(-1, 1)
        v = (self.encode_texts(prompts) * w).sum(axis=0)
        return v / max(float(np.linalg.norm(v)), 1e-6)


def test_short_text_matches_query_vector():
    engine = StubEngine()
    enc = SiglipTextEncoder(engine=engine)
    text = "a man opens a wooden door"
    reference = engine.query_vector(text, text)
    engine.batches.clear()
    (vec,) = enc.encode_chunks(text)
    np.testing.assert_allclose(vec, reference, atol=1e-6)
    np.testing.assert_allclose(enc.encode(text), reference, atol=1e-6)
    assert engine.batches == [4, 4]
    assert isinstance(enc, TextEncoder)


def test_long_text_matches_query_similarities_chunking_in_one_forward_pass():
    engine = StubEngine()
    enc = SiglipTextEncoder(engine=engine)
    text = ("A woman in a red coat walks along a sandy beach at sunset. "
            "A small white dog follows her closely. "
            "Two fishing boats are anchored near the shore, one of them blue.")
    chunks = engine.chunk_text(text)
    assert len(chunks) == 3
    # as in the AIC pipeline: the first chunk keeps the full query in the second slot
    reference = [engine.query_vector(text, chunks[0])] + [engine.query_vector(c, c) for c in chunks[1:]]
    engine.batches.clear()
    vectors = enc.encode_chunks(text)
    assert engine.batches == [12]
    assert len(vectors) == 3
    for got, want in zip(vectors, reference):
        assert got.dtype == np.float32
        np.testing.assert_allclose(got, want, atol=1e-6)
        assert np.linalg.norm(got) == pytest.approx(1.0, abs=1e-5)


def test_empty_text_is_rejected():
    enc = SiglipTextEncoder(engine=StubEngine())
    with pytest.raises(ValueError):
        enc.encode_chunks("  ")
    with pytest.raises(ValueError):
        enc.encode("")
