"""verify.py (pre-submission check) and conversation.py (KIS-C dialogue) without network or models."""

from __future__ import annotations

import json
import shutil
import threading
import time

import pytest

from tests.conftest import CONCEPT_PHRASES
from scenechain import conversation, decompose, verify
from scenechain.collection import Collection
from scenechain.llm import LLMError

DOOR, DESK = CONCEPT_PHRASES["door"], CONCEPT_PHRASES["desk"]


class ReplyLLM:
    """Replies with a fixed text (or raises) and records prompts and image counts."""

    def __init__(self, reply=None, error: Exception | None = None, model: str = "fake"):
        self.reply, self.error, self.model = reply, error, model
        self.calls: list[tuple[str, int]] = []

    def generate(self, prompt, parse, images=()):
        self.calls.append((prompt, len(images)))
        if self.error is not None:
            raise self.error
        text = self.reply if isinstance(self.reply, str) else json.dumps(self.reply)
        try:
            return parse(text), self.model
        except ValueError as exc:
            raise LLMError(f"{self.model}: unusable reply: {exc}") from exc


@pytest.fixture(autouse=True)
def no_default_models(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("a test reached a real LLM factory")

    monkeypatch.setattr(decompose, "default_llm", forbidden)
    monkeypatch.setattr(verify, "default_providers", forbidden)


@pytest.fixture(scope="module")
def collection(synthetic_collection_dir) -> Collection:
    return Collection.load(synthetic_collection_dir)


# ------------------------------------------------------------------------------------ verify
def test_moment_rows_are_the_nearest_valid_keyframe_and_its_neighbours(collection):
    rows = verify.moment_rows(collection, "V003", 21.2)
    assert [int(collection.n_in_video[r]) - 1 for r in rows] == [9, 10, 11, 12, 13]
    start = verify.moment_rows(collection, "V003", 0.0)  # keyframe 0 is invalid
    assert [int(collection.n_in_video[r]) - 1 for r in start] == [1, 2, 3]
    with pytest.raises(KeyError):
        verify.moment_rows(collection, "V999", 1.0)


def test_verify_ok_warn_and_cache(collection, tmp_path):
    ok = ReplyLLM({"contradictions": []})
    first = verify.verify_moment(collection, [DOOR, DESK], "V003", 21.0, set(), providers=[ok], cache_dir=tmp_path,
                                 votes=1)
    assert first["verdict"] == "ok" and first["reasons"] == [] and first["model"] == "fake"
    assert first["rejected"] is False and first["notes"] == []
    prompt, n_images = ok.calls[0]
    assert n_images == 5 and "1. " + DOOR in prompt and "2. " + DESK in prompt and "0:22.0" in prompt
    again = verify.verify_moment(collection, [DOOR, DESK], "V003", 21.0, set(), providers=[ok], cache_dir=tmp_path,
                                 votes=1)
    assert again["verdict"] == "ok" and len(ok.calls) == 1  # answered from the disk cache

    warn = ReplyLLM({"contradictions": ["The door is blue, not wooden.", "", 7]})
    result = verify.verify_moment(collection, [DOOR], "V003", 21.0, [], providers=[warn], cache_dir=None, votes=1)
    assert result["verdict"] == "warn" and result["reasons"] == ["The door is blue, not wooden.", "7"]


def test_verify_moves_to_the_next_provider_and_never_raises(collection, monkeypatch):
    # one call per check, so that the call counts below are exact
    monkeypatch.setattr(verify.verify_moment, "__kwdefaults__", {**verify.verify_moment.__kwdefaults__, "votes": 1})
    broken = ReplyLLM("not json")
    backup = ReplyLLM({"contradictions": []}, model="backup")
    result = verify.verify_moment(collection, [DOOR], "V003", 21.0, providers=[broken, backup], cache_dir=None)
    assert result["verdict"] == "ok" and result["model"] == "backup"

    down = ReplyLLM(error=LLMError("quota"))
    result = verify.verify_moment(collection, [DOOR], "V003", 21.0, providers=[down], cache_dir=None)
    assert result["verdict"] == "warn" and "quota" in result["reasons"][0]

    crash = ReplyLLM(error=RuntimeError("boom"))
    result = verify.verify_moment(collection, [DOOR], "V003", 21.0, providers=[crash], cache_dir=None)
    assert result["verdict"] == "warn" and "boom" in result["reasons"][0]

    empty = verify.verify_moment(collection, ["  "], "V003", 1.0, providers=[down], cache_dir=None)
    assert empty["verdict"] == "ok" and empty["reasons"] == [] and len(down.calls) == 1  # no model call
    for scenes, video, t, providers in [([DOOR], "V999", 1.0, [down]),
                                        ([DOOR], "V003", float("nan"), [down]), ([DOOR], "V003", 1.0, [])]:
        result = verify.verify_moment(collection, scenes, video, t, providers=providers, cache_dir=None)
        assert result["verdict"] == "warn" and result["reasons"] and isinstance(result["took_ms"], int)


def test_verify_fails_only_near_a_moment_judged_wrong(collection):
    llm = ReplyLLM({"contradictions": []})
    wrong = {"V003": [21.0]}
    same = verify.verify_moment(collection, [DOOR], "V003", 21.0, wrong, providers=[llm], cache_dir=None, votes=1)
    near = verify.verify_moment(collection, [DOOR], "V003", 21.0 + verify.WRONG_RADIUS_S, wrong, providers=[llm],
                                cache_dir=None, votes=1)
    assert same["verdict"] == near["verdict"] == "fail" and same["rejected"] is near["rejected"] is True
    assert "already judged wrong" in same["reasons"][0] and "0:21.0" in same["reasons"][0]
    no_text = verify.verify_moment(collection, [], "V003", 23.0, wrong, providers=[llm], cache_dir=None)
    assert no_text["verdict"] == "fail" and no_text["rejected"] is True
    assert llm.calls == []  # a blocked moment is answered without the model

    # the right video at another moment: checked as usual, with a note naming the earlier wrong time
    far = verify.verify_moment(collection, [DOOR], "V003", 40.0, wrong, providers=[llm], cache_dir=None, votes=1)
    assert far["verdict"] == "ok" and far["rejected"] is False and len(llm.calls) == 1
    assert len(far["notes"]) == 1 and "0:21.0" in far["notes"][0] and far["reasons"] == []
    warn = ReplyLLM({"contradictions": ["No door is visible."]})
    flagged = verify.verify_moment(collection, [DOOR], "V003", 60.0, wrong, providers=[warn], cache_dir=None, votes=1)
    assert flagged["verdict"] == "warn" and flagged["reasons"] == ["No door is visible."] and flagged["notes"]
    # AVS basket items (no scene text): other shots of the same video are not blocked
    shot = verify.verify_moment(collection, [], "V003", 50.0, wrong, providers=[llm], cache_dir=None)
    assert shot["verdict"] == "ok" and shot["rejected"] is False and shot["notes"]
    other = verify.verify_moment(collection, [], "V002", 21.0, wrong, providers=[llm], cache_dir=None)
    assert other["verdict"] == "ok" and other["notes"] == []
    # without times nothing can be blocked; the radius is a parameter
    unknown = verify.verify_moment(collection, [DOOR], "V003", 21.0, {"V003"}, providers=[llm], cache_dir=None,
                                   votes=1)
    assert unknown["verdict"] == "ok" and unknown["rejected"] is False and "time unknown" in unknown["notes"][0]
    wide = verify.verify_moment(collection, [], "V003", 40.0, wrong, radius_s=20.0)
    assert wide["verdict"] == "fail"
    reason, note = verify.wrong_history(wrong, "V003", 26.01)
    assert reason is None and "0:21.0" in note
    assert verify.wrong_history(wrong, "V001", 21.0) == (None, None)


class ScriptedLLM:
    """Thread-safe provider whose n-th call answers ``replies[n]`` (a contradictions list, or an
    exception to raise) after ``delays[n]`` seconds; both lists repeat."""

    def __init__(self, replies, delays=None, model="scripted"):
        self.replies, self.delays, self.model = list(replies), list(delays or [0.0]), model
        self.calls = 0
        self._lock = threading.Lock()

    def generate(self, prompt, parse, images=()):
        with self._lock:
            n, self.calls = self.calls, self.calls + 1
        time.sleep(self.delays[n % len(self.delays)])
        reply = self.replies[n % len(self.replies)]
        if isinstance(reply, Exception):
            raise reply
        return parse(json.dumps({"contradictions": reply})), self.model


def test_verify_majority_of_votes(collection, tmp_path):
    assert verify.VOTES == 3
    one_warns = ScriptedLLM([["The door is blue."], [], []])
    result = verify.verify_moment(collection, [DOOR], "V003", 21.0, providers=[one_warns], cache_dir=None)
    assert result["verdict"] == "ok" and result["reasons"] == []
    assert wait_until(lambda: one_warns.calls == 3)

    two_warn = ScriptedLLM([["The door is blue."], ["The door is metal."], []])
    result = verify.verify_moment(collection, [DOOR], "V003", 21.0, providers=[two_warn], cache_dir=tmp_path)
    assert result["verdict"] == "warn" and result["reasons"][0] == "2 of 3 checks listed contradictions."
    assert result["reasons"][1] in ("The door is blue.", "The door is metal.")
    assert wait_until(lambda: two_warn.calls == 3)
    # the majority is cached: no new call, and a single call never reuses the answer of three
    again = verify.verify_moment(collection, [DOOR], "V003", 21.0, providers=[two_warn], cache_dir=tmp_path)
    assert again["verdict"] == "warn" and again["reasons"] == result["reasons"] and two_warn.calls == 3
    single = ScriptedLLM([[]])
    alone = verify.verify_moment(collection, [DOOR], "V003", 21.0, providers=[single], cache_dir=tmp_path, votes=1)
    assert alone["verdict"] == "ok" and single.calls == 1

    split = ScriptedLLM([["The door is blue."], []])
    result = verify.verify_moment(collection, [DOOR], "V003", 21.0, providers=[split], cache_dir=None, votes=2)
    assert result["verdict"] == "warn" and "split" in result["reasons"][1]


def test_verify_votes_answer_once_the_majority_is_known(collection, tmp_path):
    slow_third = ScriptedLLM([[], [], ["The door is blue."]], delays=[0.0, 0.05, 1.5])
    started = time.perf_counter()
    result = verify.verify_moment(collection, [DOOR], "V003", 21.0, providers=[slow_third], cache_dir=tmp_path,
                                  budget_s=1.0)
    assert result["verdict"] == "ok" and time.perf_counter() - started < 0.8  # not the slowest call
    assert len(list(tmp_path.glob("*.json"))) == 1

    # failed calls never make a majority, and such an answer is not cached
    flaky = ScriptedLLM([[], LLMError("quota"), LLMError("quota")])
    result = verify.verify_moment(collection, [DOOR], "V003", 30.0, providers=[flaky], cache_dir=tmp_path)
    assert result["verdict"] == "warn" and "Only 1 of 3 checks answered" in result["reasons"][0]
    assert len(list(tmp_path.glob("*.json"))) == 1
    down = ScriptedLLM([LLMError("quota")])
    result = verify.verify_moment(collection, [DOOR], "V003", 30.0, providers=[down], cache_dir=tmp_path)
    assert result["verdict"] == "warn" and "quota" in result["reasons"][0] and down.calls == 3

    # one deadline for all votes; their majority still reaches the cache afterwards
    late = ScriptedLLM([[]], delays=[0.3])
    first = verify.verify_moment(collection, [DOOR], "V003", 40.0, providers=[late], cache_dir=tmp_path,
                                 budget_s=0.05)
    assert first["verdict"] == "warn" and "did not answer" in first["reasons"][0]
    assert wait_until(lambda: len(list(tmp_path.glob("*.json"))) == 2)
    again = verify.verify_moment(collection, [DOOR], "V003", 40.0, providers=[late], cache_dir=tmp_path,
                                 budget_s=0.05)
    assert again["verdict"] == "ok" and late.calls == 3


class SlowGenai:
    """``genai.Client`` stand-in whose every call hangs for ``delay`` seconds and then times out."""

    def __init__(self, delay: float):
        self.delay, self.calls = delay, []
        self.models = self

    def generate_content(self, model, contents, config):
        self.calls.append(model)
        time.sleep(self.delay)
        raise TimeoutError("ReadTimeout")


def wait_until(condition, timeout=5.0):
    deadline = time.perf_counter() + timeout
    while not condition() and time.perf_counter() < deadline:
        time.sleep(0.02)
    return condition()


def test_verify_has_one_deadline_for_all_models_and_providers(collection):
    genai = SlowGenai(delay=0.4)
    gemini = verify.gemini_client("unused-key", client=genai)
    assert len(gemini.models) == verify.MAX_MODELS == 2  # not the six models of GEMINI_MODELS
    assert verify.TIMEOUT_S >= 10.0  # the Gemini API answers 400 to a request deadline under 10 s
    backup = ReplyLLM({"contradictions": []}, model="openai")
    started = time.perf_counter()
    result = verify.verify_moment(collection, [DOOR], "V003", 21.0, providers=[gemini, backup],
                                  cache_dir=None, budget_s=0.5, votes=1)
    elapsed = time.perf_counter() - started
    assert result["verdict"] == "warn" and "did not answer within 0.5 s" in result["reasons"][0]
    assert elapsed < 1.0 and result["model"] is None  # not 2 x 0.4 s, and no wait for the backup
    assert wait_until(lambda: len(genai.calls) == 2)
    time.sleep(0.5)
    assert backup.calls == []  # the budget was spent, so the worker did not start another provider


def test_verify_answer_arriving_after_the_deadline_is_cached(collection, tmp_path):
    class Late(ReplyLLM):
        def generate(self, prompt, parse, images=()):
            time.sleep(0.3)
            return super().generate(prompt, parse, images)

    late = Late({"contradictions": []}, model="late")
    first = verify.verify_moment(collection, [DOOR], "V003", 21.0, providers=[late], cache_dir=tmp_path, budget_s=0.05,
                                 votes=1)
    assert first["verdict"] == "warn" and "did not answer" in first["reasons"][0]
    assert wait_until(lambda: len(list(tmp_path.glob("*.json"))) == 1)
    again = verify.verify_moment(collection, [DOOR], "V003", 21.0, providers=[late], cache_dir=tmp_path, budget_s=0.05,
                                 votes=1)
    assert again["verdict"] == "ok" and again["model"] == "late" and len(late.calls) == 1


def test_verify_cache_is_not_shared_between_collections(synthetic_collection_dir, tmp_path, monkeypatch):
    monkeypatch.setattr(verify.verify_moment, "__kwdefaults__", {**verify.verify_moment.__kwdefaults__, "votes": 1})
    other_dir = tmp_path / "rebuilt"
    shutil.copytree(synthetic_collection_dir, other_dir)
    meta_path = other_dir / "metadata.json"
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    for item in metadata:
        item["pts_time"] += 500.0  # same video ids and rows, different keyframes
    meta_path.write_text(json.dumps(metadata), encoding="utf-8")
    a, b = Collection.load(synthetic_collection_dir), Collection.load(other_dir)
    assert verify.moment_rows(a, "V003", 21.0) == verify.moment_rows(b, "V003", 521.0)

    cache = tmp_path / "shared-cache"
    llm_a = ReplyLLM({"contradictions": ["from collection-A"]}, model="collection-A")
    llm_b = ReplyLLM({"contradictions": []}, model="collection-B")
    assert verify.verify_moment(a, [DOOR], "V003", 21.0, providers=[llm_a], cache_dir=cache)["model"] == "collection-A"
    result = verify.verify_moment(b, [DOOR], "V003", 521.0, providers=[llm_b], cache_dir=cache)
    assert result["model"] == "collection-B" and result["reasons"] == [] and len(llm_b.calls) == 1

    assert verify.verify_moment(b, [DOOR], "V003", 521.0, providers=[llm_b], cache_dir=cache)["model"] == "collection-B"
    assert len(llm_b.calls) == 1  # its own answer is cached
    # a keyframe image rewritten in place (new content, same name) is not answered from the cache
    frame = b.frame_path(verify.moment_rows(b, "V003", 521.0)[0])
    frame.write_bytes(frame.read_bytes() + b"\0")
    llm_c = ReplyLLM({"contradictions": []}, model="rewritten")
    assert verify.verify_moment(b, [DOOR], "V003", 521.0, providers=[llm_c], cache_dir=cache)["model"] == "rewritten"


# ------------------------------------------------------------------------------ conversation
TURNS =[{"role": "team", "text": "What colour is the door?"}, {"role": "oracle", "text": "red"},
         {"role": "team", "text": "Is it inside a house?"}, {"role": "oracle", "text": "No"}]


def test_update_from_dialogue_with_a_model(collection, tmp_path):
    llm = ReplyLLM({"scenes": ["a man opens a red wooden door", DESK, "c", "d", "e"],
                    "next_question": "Is it daytime?"})
    top = [{"video_id": "V003", "caption_hint": "office", "image_path": str(collection.frame_path(11))},
           {"video_id": "V005", "caption_hint": "", "image_path": str(tmp_path / "missing.jpg")}]
    result = conversation.update_from_dialogue(DOOR, TURNS, [DOOR, DESK], top, llm=llm, cache_dir=tmp_path)
    assert result == {"scenes": ["a man opens a red wooden door", DESK, "c", "d. e"],
                      "next_question": "Is it daytime?", "model": "fake"}
    prompt, n_images = llm.calls[0]
    assert n_images == 1 and "V003: office (image 1" in prompt and "V005: no caption" in prompt
    assert "ORACLE: red" in prompt
    conversation.update_from_dialogue(DOOR, TURNS, [DOOR, DESK], top, llm=llm, cache_dir=tmp_path)
    assert len(llm.calls) == 1  # cached


def test_update_from_dialogue_rule_fallback(tmp_path):
    for llm in (ReplyLLM({"scenes": [DOOR]}), ReplyLLM(error=RuntimeError("down"))):  # no question / crash
        result = conversation.update_from_dialogue(DOOR, TURNS, [DOOR, DESK], llm=llm, cache_dir=tmp_path)
        assert result["model"] is None
        assert result["scenes"] == [DOOR, f"{DESK}. Red"]  # "No" carries no detail
        assert result["next_question"] == "How many people are visible?"
    assert not list(tmp_path.glob("*.json"))  # rule results are not cached

    fresh = conversation.rule_update(f"{DOOR}, then {DESK}", [], [])
    assert fresh["scenes"] == ["A man opens a wooden door", "The man sits down at a desk"]
    empty = conversation.update_from_dialogue("", [], [], llm=ReplyLLM(error=RuntimeError("x")))
    assert empty["scenes"] == [] and empty["next_question"]
