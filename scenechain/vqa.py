"""Answer a question about one or more keyframes with a vision-language model.

Used for VBS VQA tasks: the operator finds the moment, selects up to five keyframes and gets an
answer suggestion that they can edit before submitting.

The prompt asks for a concise answer and, as learned in AIC 2026, for a best guess in every case:
an empty or "unknown" answer can never score, a guess from partial evidence can. The model also
states the visual evidence it used, so the operator can check the answer at a glance instead of
trusting it.

Providers: Gemini first (model order of ``llm.GEMINI_MODELS``), then OpenAI when its SDK is installed and
a key is set. A reply without a non-empty answer counts as a failure of that model.
"""

from __future__ import annotations

import io
from functools import lru_cache
from pathlib import Path
from typing import Optional, Sequence, TypedDict

from .config import load_settings
from .llm import GeminiClient, LLMClient, LLMError, extract_json_object, optional_openai

MAX_FRAMES = 5
#: long side of the images sent; enough for small text, about 1k input tokens per image on Gemini
MAX_SIDE = 1024
JPEG_QUALITY = 90

PROMPT = """You help an operator in a live video retrieval competition answer a question about a video.
The {n} attached image(s) are keyframes of the same video in time order (image 1 is the earliest).

Question: {question}

Rules:
- Give the shortest answer that fully answers the question: a number, a colour, a name or a few words.
- Always commit to your best guess, even when the images are unclear. Never answer "unknown" and never refuse.
- In "evidence", say in one sentence what you see, and in which image, that supports the answer.

Return only JSON of the form {{"answer": "...", "evidence": "..."}}."""


class VQAResult(TypedDict):
    answer: str
    evidence: str
    model: str


class VQAUnavailable(RuntimeError):
    """No provider is configured, or none produced an answer; the message says why."""


def load_image(path: str | Path) -> bytes:
    """JPEG bytes of the image at ``path``, downscaled to :data:`MAX_SIDE` on its long side."""
    from PIL import Image

    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"image {p} not found")
    with Image.open(p) as im:
        rgb = im.convert("RGB")
    rgb.thumbnail((MAX_SIDE, MAX_SIDE))
    buf = io.BytesIO()
    rgb.save(buf, "JPEG", quality=JPEG_QUALITY)
    return buf.getvalue()


def _parse_answer(reply: str) -> tuple[str, str]:
    obj = extract_json_object(reply)
    answer = str(obj.get("answer") or "").strip()
    if not answer:
        raise ValueError("empty answer")
    return answer, str(obj.get("evidence") or "").strip()


@lru_cache(maxsize=1)
def default_providers() -> tuple[LLMClient, ...]:
    """Gemini then OpenAI, each only when its key (and for OpenAI, its SDK) is available.

    Built once per process so a model whose daily quota is spent stays skipped.
    """
    settings = load_settings()
    providers: list[LLMClient] = []
    if settings.gemini_key:
        providers.append(GeminiClient(settings.gemini_key))
    openai_client = optional_openai(settings.openai_key)
    if openai_client is not None:
        providers.append(openai_client)
    return tuple(providers)


def answer(
    question: str,
    image_paths: Sequence[str | Path],
    *,
    providers: Optional[Sequence[LLMClient]] = None,
) -> VQAResult:
    """``{"answer", "evidence", "model"}`` for ``question`` about the images, in the given order.

    ``providers`` defaults to :func:`default_providers`. Raises ``ValueError`` for an empty question
    or a wrong number of images, ``FileNotFoundError`` for a missing image and
    :class:`VQAUnavailable` when no provider is configured or none answers.
    """
    q = (question or "").strip()
    if not q:
        raise ValueError("the question is empty")
    if not 1 <= len(image_paths) <= MAX_FRAMES:
        raise ValueError(f"between 1 and {MAX_FRAMES} images are required, got {len(image_paths)}")
    images = [load_image(p) for p in image_paths]
    chain = list(providers) if providers is not None else default_providers()
    if not chain:
        raise VQAUnavailable("no VQA model configured: set GEMINI_API_KEY or OPENAI_API_KEY")
    prompt = PROMPT.format(n=len(images), question=q)
    failures: list[str] = []
    for provider in chain:
        try:
            (text, evidence), model = provider.generate(prompt, _parse_answer, images)
        except LLMError as exc:
            failures.append(str(exc))
            continue
        return {"answer": text, "evidence": evidence, "model": model}
    raise VQAUnavailable("no VQA model answered: " + " | ".join(failures))
