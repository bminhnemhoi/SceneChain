"""VQA answers with mocked Gemini and OpenAI clients; no network."""

from __future__ import annotations

import io
import json
from types import SimpleNamespace

import pytest
from PIL import Image

from scenechain import vqa
from scenechain.config import Settings
from scenechain.llm import GeminiClient, LLMError, OpenAIClient

DAILY_QUOTA = "429 RESOURCE_EXHAUSTED: Quota exceeded for metric requests, limit: PerDay"


class FakeModels:
    """Stands in for ``genai.Client().models``; replies per model name, exceptions are raised."""

    def __init__(self, replies: dict):
        self.replies = replies
        self.calls: list[tuple[str, list, object]] = []

    def generate_content(self, model, contents, config):
        self.calls.append((model, contents, config))
        result = self.replies.get(model, RuntimeError("404 NOT_FOUND"))
        if isinstance(result, Exception):
            raise result
        return SimpleNamespace(text=result)


class FakeCompletions:
    def __init__(self, content):
        self.content = content
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.content, Exception):
            raise self.content
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))])


def gemini(replies: dict) -> tuple[GeminiClient, FakeModels]:
    fake = FakeModels(replies)
    return GeminiClient("test-key", models=("model-a", "model-b"), client=SimpleNamespace(models=fake)), fake


def openai(content) -> tuple[OpenAIClient, FakeCompletions]:
    fake = FakeCompletions(content)
    client = SimpleNamespace(chat=SimpleNamespace(completions=fake))
    return OpenAIClient("test-key", model="gpt-test", client=client), fake


@pytest.fixture()
def frames(synthetic_collection_dir) -> list[str]:
    return [str(synthetic_collection_dir / "frames" / "V003" / f"{n:03d}.jpg") for n in (11, 12)]


def reply(answer: str, evidence: str = "seen in image 1") -> str:
    return json.dumps({"answer": answer, "evidence": evidence})


def test_gemini_answer(frames):
    client, fake = gemini({"model-a": reply("two", "two people stand at the door in image 2")})
    result = vqa.answer("How many people are at the door?", frames, providers=[client])
    assert result == {"answer": "two", "evidence": "two people stand at the door in image 2",
                      "model": "model-a"}
    _, contents, config = fake.calls[0]
    assert len(contents) == 3 and "How many people are at the door?" in contents[-1]
    assert "best guess" in contents[-1]
    assert config.temperature == 0.0


def test_empty_answer_moves_on_to_the_next_model_and_provider(frames):
    client, fake = gemini({"model-a": RuntimeError(DAILY_QUOTA), "model-b": reply("  ")})
    backup, completions = openai(reply("blue"))
    result = vqa.answer("What colour is the boat?", frames[:1], providers=[client, backup])
    assert result == {"answer": "blue", "evidence": "seen in image 1", "model": "gpt-test"}
    assert [m for m, _, _ in fake.calls] == ["model-a", "model-b"]
    call = completions.calls[0]
    content = call["messages"][0]["content"]
    assert content[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert "What colour is the boat?" in content[-1]["text"]
    assert call["max_completion_tokens"] >= 2000


def test_non_json_reply_is_rejected(frames):
    backup, _ = openai("The boat is blue.")
    with pytest.raises(vqa.VQAUnavailable, match="gpt-test"):
        vqa.answer("What colour is the boat?", frames, providers=[backup])


def test_no_provider_or_all_failing(frames):
    with pytest.raises(vqa.VQAUnavailable, match="GEMINI_API_KEY"):
        vqa.answer("What is shown?", frames, providers=[])
    client, _ = gemini({})
    backup, _ = openai(RuntimeError("401 invalid key"))
    with pytest.raises(vqa.VQAUnavailable) as err:
        vqa.answer("What is shown?", frames, providers=[client, backup])
    assert "404 NOT_FOUND" in str(err.value) and "401 invalid key" in str(err.value)


def test_input_validation(frames, tmp_path):
    client, _ = gemini({"model-a": reply("x")})
    with pytest.raises(ValueError):
        vqa.answer("  ", frames, providers=[client])
    with pytest.raises(ValueError):
        vqa.answer("What?", [], providers=[client])
    with pytest.raises(ValueError):
        vqa.answer("What?", frames * 3, providers=[client])
    with pytest.raises(FileNotFoundError):
        vqa.answer("What?", [str(tmp_path / "missing.jpg")], providers=[client])


def test_large_images_are_downscaled(tmp_path):
    path = tmp_path / "big.png"
    Image.new("RGB", (3000, 1500), (10, 20, 30)).save(path)
    blob = vqa.load_image(path)
    with Image.open(io.BytesIO(blob)) as im:
        assert im.format == "JPEG" and max(im.size) == vqa.MAX_SIDE


def test_default_providers_follow_settings(monkeypatch):
    vqa.default_providers.cache_clear()
    monkeypatch.setattr(vqa, "load_settings", lambda: Settings(gemini_key="k", openai_key="o"))
    monkeypatch.setattr(OpenAIClient, "sdk_available", staticmethod(lambda: False))
    try:
        providers = vqa.default_providers()
        assert len(providers) == 1 and isinstance(providers[0], GeminiClient)
        assert vqa.default_providers() is providers  # built once per process
    finally:
        vqa.default_providers.cache_clear()


def test_openai_client_requires_sdk_or_injected_client(monkeypatch):
    monkeypatch.setattr(OpenAIClient, "sdk_available", staticmethod(lambda: False))
    with pytest.raises(LLMError):
        OpenAIClient("key")
    with pytest.raises(ValueError):
        OpenAIClient("")
