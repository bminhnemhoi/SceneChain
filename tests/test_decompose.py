"""Scene decomposition: rule fallback, LLM path with a mocked Gemini client, disk cache."""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

from scenechain import decompose as dec
from scenechain.llm import GeminiClient

DAILY_QUOTA = "429 RESOURCE_EXHAUSTED: Quota exceeded for metric requests, limit: PerDay"


class FakeModels:
    """Stands in for ``genai.Client().models``; replies per model name, exceptions are raised."""

    def __init__(self, replies: dict):
        self.replies = replies
        self.calls: list[tuple[str, list, object]] = []

    def generate_content(self, model, contents, config):
        self.calls.append((model, contents, config))
        reply = self.replies.get(model, RuntimeError("404 NOT_FOUND"))
        if isinstance(reply, Exception):
            raise reply
        return SimpleNamespace(text=reply)


def gemini(replies: dict, models=("model-a", "model-b")) -> tuple[GeminiClient, FakeModels]:
    fake = FakeModels(replies)
    return GeminiClient("test-key", models=models, client=SimpleNamespace(models=fake)), fake


def scenes_json(*scenes: str) -> str:
    return json.dumps({"scenes": list(scenes)})


# ------------------------------------------------------------------------------------ rules
@pytest.mark.parametrize(
    "text, expected",
    [
        ("A man opens a wooden door. Then he sits down at a desk.",
         ["A man opens a wooden door", "He sits down at a desk"]),
        ("A woman in a red coat walks along a beach, and afterwards a dog runs across a field. "
         "Finally a blue boat drifts on a river.",
         ["A woman in a red coat walks along a beach", "A dog runs across a field",
          "A blue boat drifts on a river"]),
        ("A chef slices onions followed by a close-up of a frying pan",
         ["A chef slices onions", "A close-up of a frying pan"]),
        ("A cyclist climbs a hill. Later, he rests under a tree. After that he drinks water.",
         ["A cyclist climbs a hill", "He rests under a tree", "He drinks water"]),
        ("A girl feeds ducks next to a pond. Next, she waves at the camera.",
         ["A girl feeds ducks next to a pond", "She waves at the camera"]),
    ],
)
def test_rule_split_on_cue_words(text, expected):
    assert dec.split_scenes_by_rule(text) == expected


def test_rule_keeps_details_of_one_moment_together():
    text = "A cat sleeps on a sofa. It has white fur. Two years later, nothing has changed."
    assert dec.split_scenes_by_rule(text) == [
        "A cat sleeps on a sofa. It has white fur. Two years later, nothing has changed"
    ]


def test_rule_before_that_puts_the_scene_first():
    text = "A crowd cheers in a stadium. Before that, a player scores a goal."
    assert dec.split_scenes_by_rule(text) == ["A player scores a goal", "A crowd cheers in a stadium"]
    assert dec.split_scenes_by_rule("A man washes his hands before eating lunch") == [
        "A man washes his hands", "Eating lunch"
    ]


def test_rule_caps_at_four_scenes():
    scenes = dec.split_scenes_by_rule("One. Then two. Then three. Then four. Then five. Then six.")
    assert scenes == ["One", "Two", "Three", "Four. Five. Six"]


# -------------------------------------------------------------------------------------- LLM
def test_llm_split_is_used_and_cached(tmp_path):
    client, fake = gemini({"model-a": scenes_json("A man opens a door", "A man sits at a desk")})
    text = "A man opens a door and then he sits down at a desk."

    first = dec.decompose(text, llm=client, cache_dir=tmp_path)
    assert first == {"scenes": ["A man opens a door", "A man sits at a desk"], "source": "llm",
                     "model": "model-a"}
    model, contents, config = fake.calls[0]
    assert model == "model-a" and text in contents[-1]
    assert config.temperature == 0.0 and config.response_mime_type == "application/json"
    assert (tmp_path / f"{hashlib.sha1(text.encode('utf-8')).hexdigest()}.json").is_file()

    second = dec.decompose(text, llm=client, cache_dir=tmp_path)
    assert second == first
    assert len(fake.calls) == 1  # cache hit, no second call

    fresh_client, fresh = gemini({"model-a": scenes_json("different")})
    assert dec.decompose(f"  {text}\n", llm=fresh_client, cache_dir=tmp_path) == first
    assert fresh.calls == []


def test_cache_of_an_older_prompt_version_is_ignored(tmp_path):
    text = "A bus stops. Then children get off."
    path = tmp_path / f"{hashlib.sha1(text.encode('utf-8')).hexdigest()}.json"
    path.write_text(json.dumps({"prompt_version": dec.PROMPT_VERSION - 1, "scenes": ["old"]}),
                    encoding="utf-8")
    client, fake = gemini({"model-a": scenes_json("A bus stops", "Children get off a bus")})
    assert dec.decompose(text, llm=client, cache_dir=tmp_path)["scenes"] == [
        "A bus stops", "Children get off a bus"
    ]
    assert len(fake.calls) == 1


def test_falls_through_models_and_remembers_daily_quota(tmp_path):
    client, fake = gemini({"model-a": RuntimeError(DAILY_QUOTA), "model-b": scenes_json("A red car")})
    result = dec.decompose("A red car.", llm=client, cache_dir=tmp_path)
    assert result == {"scenes": ["A red car"], "source": "llm", "model": "model-b"}
    assert client.exhausted == {"model-a"}

    dec.decompose("A green car.", llm=client, cache_dir=tmp_path)
    assert [m for m, _, _ in fake.calls] == ["model-a", "model-b", "model-b"]


def test_unusable_replies_fall_back_to_rules_and_are_not_cached(tmp_path):
    client, fake = gemini({"model-a": "not json", "model-b": json.dumps({"scenes": []})})
    text = "A kite flies over a beach. Then it falls into the sea."
    result = dec.decompose(text, llm=client, cache_dir=tmp_path)
    assert result == {"scenes": ["A kite flies over a beach", "It falls into the sea"],
                      "source": "rule", "model": None}
    assert list(tmp_path.iterdir()) == []
    dec.decompose(text, llm=client, cache_dir=tmp_path)
    assert len(fake.calls) == 4  # retried, because rule results are not cached


def test_llm_output_is_capped_to_four_scenes(tmp_path):
    client, _ = gemini({"model-a": scenes_json("s1", "s2", "s3", "s4", "s5")})
    assert dec.decompose("five scenes", llm=client, cache_dir=None)["scenes"] == ["s1", "s2", "s3", "s4. s5"]


def test_never_raises(tmp_path, monkeypatch):
    class Broken:
        def generate(self, prompt, parse, images=()):
            raise RuntimeError("socket closed")

    result = dec.decompose("A horse jumps a fence", llm=Broken(), cache_dir=tmp_path)
    assert result == {"scenes": ["A horse jumps a fence"], "source": "rule", "model": None}

    monkeypatch.setattr(dec, "default_llm", lambda: None)  # no API key configured
    assert dec.decompose("A tram passes, then it rains.", cache_dir=tmp_path)["source"] == "rule"
    assert dec.decompose("   ") == {"scenes": [], "source": "rule", "model": None}
