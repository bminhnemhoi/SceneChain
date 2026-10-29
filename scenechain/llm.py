"""Thin, interactive-friendly wrappers around the Gemini and OpenAI chat APIs.

Shared by :mod:`scenechain.decompose` and :mod:`scenechain.vqa`. Both clients expose
``generate(prompt, parse, images=()) -> (parsed, model)``: the reply text goes through ``parse``, and
a reply that ``parse`` rejects (by raising ``ValueError``) counts as a failure of that model, so the
next model is tried.

Gemini uses the model order of the authors' AIC 2026 pipeline (:data:`GEMINI_MODELS`: the default
flash-lite model first, then its fallbacks) and the same test to tell a daily quota (the model is
skipped for the rest of the process) from a per-minute one. Unlike the batch pipeline, nothing here
sleeps and retries: an operator is waiting on stage, so any failure moves straight to the next
model, each of which has its own quota. Calls use temperature 0.
"""

from __future__ import annotations

import base64
import importlib.util
import json
from typing import Any, Callable, Optional, Protocol, Sequence, TypeVar

T = TypeVar("T")

#: Gemini model order of the AIC 2026 pipeline: the default flash-lite model first, then fallbacks
#: (each model name has its own free-tier quota). Model availability changes over time; edit here.
DEFAULT_MODEL = "gemini-3.5-flash-lite"
FALLBACK_MODELS: tuple[str, ...] = (
    "gemini-2.5-flash-lite",
    "gemini-flash-lite-latest",
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash",
    "gemini-flash-latest",
)


def _is_daily_quota(msg: str) -> bool:
    """True only for a quota error that will persist for the rest of the day.

    Google uses the same 429 / RESOURCE_EXHAUSTED status for the per-minute and the per-day limit;
    only the quota id in the message tells them apart.
    """
    low = msg.lower().replace(" ", "").replace("_", "")
    return "perday" in low or "dailylimit" in low


class LLMClient(Protocol):
    """What :mod:`decompose` and :mod:`vqa` need from a provider client."""

    def generate(
        self, prompt: str, parse: Callable[[str], T], images: Sequence[bytes] = ()
    ) -> tuple[T, str]:
        """``(parse(reply_text), model_name)``; raises :class:`LLMError` when no reply is usable."""
        ...

#: Gemini models in order of preference
GEMINI_MODELS: tuple[str, ...] = (DEFAULT_MODEL,) + tuple(
    m for m in FALLBACK_MODELS if m != DEFAULT_MODEL
)

#: OpenAI fallback model; gpt-5.x spends part of the completion budget on reasoning, so a budget
#: under about 2000 tokens can come back as an empty string
OPENAI_MODEL = "gpt-5.2"
OPENAI_MAX_COMPLETION_TOKENS = 2000

REQUEST_TIMEOUT_S = 45.0


class LLMError(RuntimeError):
    """No model produced an acceptable reply; the message lists what each model returned."""


class GeminiClient:
    """Gemini through ``google-genai`` with model fallback."""

    def __init__(
        self,
        api_key: str,
        models: Sequence[str] = GEMINI_MODELS,
        client: Any = None,
        timeout_s: float = REQUEST_TIMEOUT_S,
        max_output_tokens: int = 2048,
    ) -> None:
        """``client`` injects a ``genai.Client``-compatible object (used by tests)."""
        if not api_key and client is None:
            raise ValueError("a Gemini API key is required")
        if not models:
            raise ValueError("at least one Gemini model is required")
        self._api_key = api_key
        self._client = client
        self._timeout_ms = int(timeout_s * 1000)
        self._max_output_tokens = max_output_tokens
        self.models = tuple(models)
        #: models whose daily quota is spent; skipped until the process restarts
        self.exhausted: set[str] = set()

    def _get_client(self) -> Any:
        if self._client is None:
            from google import genai
            from google.genai import types

            self._client = genai.Client(
                api_key=self._api_key, http_options=types.HttpOptions(timeout=self._timeout_ms)
            )
        return self._client

    def generate(
        self, prompt: str, parse: Callable[[str], T], images: Sequence[bytes] = ()
    ) -> tuple[T, str]:
        """Ask each model in order until one reply passes ``parse``; raises :class:`LLMError`."""
        from google.genai import types

        client = self._get_client()
        contents: list[Any] = [types.Part.from_bytes(data=b, mime_type="image/jpeg") for b in images]
        contents.append(prompt)
        config = types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=self._max_output_tokens,
            response_mime_type="application/json",
            # no tools are passed; disabling AFC also silences the SDK's per-call warning
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        failures: list[str] = []
        for model in self.models:
            if model in self.exhausted:
                continue
            try:
                reply = client.models.generate_content(model=model, contents=contents, config=config)
            except Exception as exc:  # noqa: BLE001 - every API failure means: try the next model
                message = str(exc)
                if _is_daily_quota(message):
                    self.exhausted.add(model)
                failures.append(f"{model}: {type(exc).__name__}: {message[:160]}")
                continue
            try:
                return parse(getattr(reply, "text", None) or ""), model
            except ValueError as exc:
                failures.append(f"{model}: unusable reply: {exc}")
        raise LLMError("no Gemini model answered: " + ("; ".join(failures) or "all models exhausted"))


class OpenAIClient:
    """OpenAI chat completions through the ``openai`` SDK (optional dependency)."""

    def __init__(
        self,
        api_key: str,
        model: str = OPENAI_MODEL,
        client: Any = None,
        timeout_s: float = REQUEST_TIMEOUT_S,
    ) -> None:
        """``client`` injects an ``openai.OpenAI``-compatible object (used by tests)."""
        if client is None:
            if not api_key:
                raise ValueError("an OpenAI API key is required")
            if not self.sdk_available():
                raise LLMError("the openai package is not installed")
            from openai import OpenAI

            client = OpenAI(api_key=api_key, timeout=timeout_s)
        self._client = client
        self.model = model

    @staticmethod
    def sdk_available() -> bool:
        return importlib.util.find_spec("openai") is not None

    def generate(
        self, prompt: str, parse: Callable[[str], T], images: Sequence[bytes] = ()
    ) -> tuple[T, str]:
        """One request; raises :class:`LLMError` on API failure or a reply ``parse`` rejects."""
        content: list[dict] = [
            {"type": "image_url",
             "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(b).decode("ascii")}}
            for b in images
        ]
        content.append({"type": "text", "text": prompt})
        try:
            reply = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": content}],
                max_completion_tokens=OPENAI_MAX_COMPLETION_TOKENS,
                response_format={"type": "json_object"},
            )
            text = reply.choices[0].message.content or ""
        except Exception as exc:  # noqa: BLE001 - surfaced as LLMError with the provider message
            raise LLMError(f"{self.model}: {type(exc).__name__}: {str(exc)[:160]}") from exc
        try:
            return parse(text), self.model
        except ValueError as exc:
            raise LLMError(f"{self.model}: unusable reply: {exc}") from exc


def extract_json_object(text: str) -> dict:
    """The JSON object from the first ``{`` to the last ``}`` of ``text``; ``ValueError`` if none."""
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError(f"no JSON object in reply {text[:80]!r}")
    try:
        obj = json.loads(text[start:end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in reply: {exc}") from exc
    if not isinstance(obj, dict):
        raise ValueError("reply JSON is not an object")
    return obj


def optional_openai(api_key: Optional[str]) -> Optional[OpenAIClient]:
    """An :class:`OpenAIClient` when a key is set and the SDK is installed, else ``None``."""
    if not api_key or not OpenAIClient.sdk_available():
        return None
    return OpenAIClient(api_key)
