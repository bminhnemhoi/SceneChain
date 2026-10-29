"""Text encoders: English text -> L2-normalised vectors in the collection's embedding space.

:class:`SiglipTextEncoder` reproduces the query encoding the authors measured in the AIC 2026
pipeline (text tower in :mod:`scenechain.siglip_text`):

* the 4-prompt ensemble (text, text, "a high quality video keyframe of ...", "a photo of ...")
  weighted by ``PROMPT_WEIGHTS``, with the English text in both language slots;
* long descriptions split by ``SiglipTextTower.chunk_text`` into pieces that fit the 64-token text
  tower. The first chunk keeps the full text in its second slot and later chunks use the chunk in
  both slots; averaging the per-chunk similarities is done by ``search.text_similarities``.

Only the text tower is loaded; the keyframe index is never opened. All prompts of all chunks go
through the model in one forward pass.
"""

from __future__ import annotations

from typing import Optional, Protocol, Sequence, runtime_checkable

import numpy as np

from .siglip_text import DEFAULT_MODEL, PROMPT_WEIGHTS, SiglipTextTower


@runtime_checkable
class TextEncoder(Protocol):
    """What the search layer needs from an encoder."""

    name: str

    def encode(self, text: str) -> np.ndarray:
        """One L2-normalised ``(D,)`` float32 vector for ``text``."""
        ...

    def encode_chunks(self, text: str) -> list[np.ndarray]:
        """One L2-normalised vector per chunk of ``text`` (a single chunk for short text)."""
        ...


def ensemble_prompts(text: str, second_slot: str) -> list[str]:
    """The four ensemble prompts, in the order of ``PROMPT_WEIGHTS``."""
    return [
        text,
        second_slot,
        f"a high quality video keyframe of {text}",
        f"a photo of {text}",
    ]


class SiglipTextEncoder:
    """SigLIP-2 text tower with the AIC 4-prompt ensemble and long-text chunking."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: Optional[str] = None,
        engine: Optional[SiglipTextTower] = None,
    ) -> None:
        """Load the text model on ``device`` (CUDA when available if ``None``).

        ``engine`` injects an already prepared :class:`SiglipTextTower` (its ``processor`` and
        ``encode_texts`` must be usable); the model is then not loaded again.
        """
        if engine is None:
            engine = SiglipTextTower(model_name=model_name, device=device)
            engine._load_model()
        self.engine = engine
        self.name = f"siglip2:{engine.model_name}"
        self._weights = np.asarray(PROMPT_WEIGHTS, dtype=np.float32)

    def encode(self, text: str) -> np.ndarray:
        """Ensemble vector of the whole text (the tower truncates it at 64 tokens)."""
        clean = _require_text(text)
        return self._ensemble([(clean, clean)])[0]

    def encode_chunks(self, text: str) -> list[np.ndarray]:
        """Ensemble vector per ``SiglipTextTower.chunk_text`` chunk, in text order."""
        clean = _require_text(text)
        chunks = self.engine.chunk_text(clean)
        if len(chunks) <= 1:
            pairs = [(clean, clean)]
        else:
            pairs = [(chunks[0], clean)] + [(c, c) for c in chunks[1:]]
        return list(self._ensemble(pairs))

    def _ensemble(self, pairs: Sequence[tuple[str, str]]) -> np.ndarray:
        prompts = [p for text, second in pairs for p in ensemble_prompts(text, second)]
        n_prompts = len(self._weights)
        emb = self.engine.encode_texts(prompts).reshape(len(pairs), n_prompts, -1)
        vec = np.einsum("cpd,p->cd", emb.astype(np.float32), self._weights)
        norms = np.maximum(np.linalg.norm(vec, axis=1, keepdims=True), 1e-6)
        return (vec / norms).astype(np.float32)


def _require_text(text: str) -> str:
    clean = (text or "").strip()
    if not clean:
        raise ValueError("cannot encode an empty query")
    return clean
