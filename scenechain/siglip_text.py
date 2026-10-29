"""SigLIP-2 text tower: tokenisation-aware chunking and L2-normalised text embeddings.

Vendored from the retrieval engine the authors used in the AI Challenge HCMC 2026 (only the text
side; the keyframe index, translation and ranking code are not needed here), so the query vectors
are the ones that were measured there:

* ``encode_texts`` pads to the tower's 64-token window, takes the pooled text features and
  L2-normalises them;
* ``chunk_text`` packs whole sentences greedily into pieces of at most ``max_tokens`` tokens (and
  splits an over-long sentence on commas), because the tower silently drops everything past 64
  tokens and long descriptions tend to put the distinguishing detail last.

``torch`` and ``transformers`` are imported only when the model is loaded, so the rest of the
package (and the test suite) runs without them.
"""

from __future__ import annotations

import re
from typing import List, Optional, Sequence

import numpy as np

DEFAULT_MODEL = "google/siglip2-so400m-patch14-384"

#: weights of the 4-prompt ensemble: (text, second slot, "a high quality video keyframe of ...",
#: "a photo of ..."). The second slot carried a Vietnamese rendering in the AIC pipeline; for
#: English-only VBS queries the description is repeated there (see :mod:`scenechain.encoder`).
PROMPT_WEIGHTS = (0.45, 0.35, 0.10, 0.10)

#: SigLIP-2 text window
MAX_LENGTH = 64


class SiglipTextTower:
    """The SigLIP-2 text model and its processor. Load once with :meth:`_load_model`."""

    def __init__(self, model_name: str = DEFAULT_MODEL, device: Optional[str] = None) -> None:
        self.model_name = model_name
        self.device = device
        self.model = None
        self.processor = None
        self._torch = None

    def _load_model(self) -> None:
        import torch
        from transformers import AutoModel, AutoProcessor

        self._torch = torch
        dev = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = dev
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name, dtype=torch.float32).to(dev).eval()

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        """L2-normalised SigLIP-2 text embeddings, one row per input."""
        torch = self._torch
        inp = self.processor(
            text=list(texts),
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
        ).to(self.device)
        with torch.inference_mode():
            f = self.model.get_text_features(**inp)
        if not isinstance(f, torch.Tensor):  # transformers 5.x returns a ModelOutput
            for attr in ("pooler_output", "text_embeds", "last_hidden_state"):
                v = getattr(f, attr, None)
                if isinstance(v, torch.Tensor):
                    f = v
                    break
            else:
                f = f[0]
        if f.ndim == 3:
            f = f.mean(dim=1)
        f = f / f.norm(dim=-1, keepdim=True)
        return f.cpu().numpy().astype(np.float32)

    def chunk_text(self, text: str, max_tokens: int = 58) -> List[str]:
        """Split a long description into pieces the text tower can see in full.

        Sentences are packed greedily so each chunk stays under ``max_tokens`` and no clause is cut
        in half; a single sentence that is still too long is split on commas.
        """
        tok = self.processor.tokenizer
        text = (text or "").strip()
        sentences = re.split(r"(?<=[.!?;])\s+|\n+", text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            return [text]

        chunks: List[str] = []
        cur = ""
        for s in sentences:
            cand = f"{cur} {s}".strip()
            if cur and len(tok(cand)["input_ids"]) > max_tokens:
                chunks.append(cur)
                cur = s
            else:
                cur = cand
        if cur:
            chunks.append(cur)

        out: List[str] = []
        for c in chunks:
            if len(tok(c)["input_ids"]) <= max_tokens:
                out.append(c)
                continue
            piece = ""
            for part in c.split(","):
                cand = f"{piece}, {part}".strip(", ")
                if piece and len(tok(cand)["input_ids"]) > max_tokens:
                    out.append(piece)
                    piece = part.strip()
                else:
                    piece = cand
            if piece:
                out.append(piece)
        return out or [text]
