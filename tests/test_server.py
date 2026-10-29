"""HTTP API of docs/ARCHITECTURE.md §6 over the synthetic collection, with fake encoder, LLM and DRES.

No network and no model: the LLM helpers' default factories are replaced by functions that fail the
test if called, every client is injected, and DRES is a real ``DresClient`` whose transport is
scripted (so request bodies still go through its validation).
"""

from __future__ import annotations

import json
import time

import pytest
from fastapi.testclient import TestClient

from tests.conftest import CONCEPT_PHRASES, FakeEncoder
from scenechain import decompose, server, verify, vqa
from scenechain.collection import Collection
from scenechain.config import Settings
from scenechain.dres import DresClient, DresError
from scenechain.llm import LLMError

DOOR, DESK, BOAT = CONCEPT_PHRASES["door"], CONCEPT_PHRASES["desk"], CONCEPT_PHRASES["boat"]


# ------------------------------------------------------------------------------------ fakes
class FakeLLM:
    """``LLMClient`` that answers by prompt kind and records every call."""

    def __init__(self, contradictions=(), fail: bool = False):
        self.contradictions = list(contradictions)
        self.fail = fail
        self.calls: list[tuple[str, int]] = []

    def generate(self, prompt, parse, images=()):
        self.calls.append((prompt, len(images)))
        if self.fail:
            raise LLMError("fake-llm: unavailable")
        if '"contradictions"' in prompt:
            reply = {"contradictions": self.contradictions}
        elif '"next_question"' in prompt:
            reply = {"scenes": [DOOR, "the man sits down at a wooden desk"],
                     "next_question": "Is the desk made of wood?"}
        elif '"answer"' in prompt:
            reply = {"answer": "blue", "evidence": "image 1 shows a blue boat"}
        else:
            reply = {"scenes": [DOOR, DESK]}
        return parse(json.dumps(reply)), "fake-llm"


class ScriptedDres(DresClient):
    """A real DresClient whose HTTP layer returns scripted DRES 2.0.x responses."""

    def __init__(self):
        super().__init__("http://dres.invalid")
        self.requests: list[tuple[str, str, object]] = []
        self.verdict = "WRONG"
        self.task_id = "task-1"
        self.submit_error: DresError | None = None

    def _request(self, method, path, *, body=None, authenticated=True):
        self.requests.append((method, path, body))
        if path == "/api/v2/login":
            if body["password"] != "pw":
                raise DresError(401, "Unauthorized request!", method, path)
            return {"id": "u1", "username": body["username"], "role": "PARTICIPANT", "sessionId": "tok"}
        if path == "/api/v2/client/evaluation/list":
            return [
                {"id": "ev1", "name": "Practice", "type": "SYNCHRONOUS", "status": "ACTIVE"},
                {"id": "ev0", "name": "Old run", "type": "SYNCHRONOUS", "status": "TERMINATED"},
            ]
        if path == "/api/v2/client/evaluation/currentTask/ev1":
            return {"name": "kis-t-01", "taskGroup": "kis", "taskType": "KIS-T", "duration": 300}
        if path == "/api/v2/evaluation/ev1/state":
            return {"evaluationStatus": "ACTIVE", "taskStatus": "RUNNING", "taskId": self.task_id,
                    "timeLeft": 250, "timeElapsed": 50}
        if path == "/api/v2/submit/ev1":
            if self.submit_error is not None:
                raise self.submit_error
            text = {"CORRECT": "Submission correct!", "WRONG": "Submission incorrect! Try again"}
            return {"status": True, "submission": self.verdict,
                    "description": text.get(self.verdict, "Submission received")}
        if path.startswith("/api/v2/log/"):
            return {"status": True, "description": "ok"}
        raise DresError(404, f"unexpected path {path}", method, path)

    def submissions(self):
        return [body for method, path, body in self.requests if path.startswith("/api/v2/submit/")]


@pytest.fixture(autouse=True)
def no_default_models(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("a test reached a real LLM factory")

    monkeypatch.setattr(decompose, "default_llm", forbidden)
    monkeypatch.setattr(vqa, "default_providers", forbidden)
    monkeypatch.setattr(verify, "default_providers", forbidden)


@pytest.fixture()
def settings(synthetic_collection_dir, tmp_path) -> Settings:
    return Settings(collection_dir=synthetic_collection_dir, log_dir=tmp_path / "logs")


@pytest.fixture()
def llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture()
def dres() -> ScriptedDres:
    return ScriptedDres()


@pytest.fixture()
def encoder() -> FakeEncoder:
    return FakeEncoder()


def make_client(settings, tmp_path, *, llm=None, providers=None, dres=None, encoder=None, **kwargs):
    app = server.create_app(
        settings,
        collection=Collection.load(settings.collection_dir),
        encoder=encoder or FakeEncoder(),
        llm=llm,
        vqa_providers=providers,
        dres_client=dres,
        cache_dir=tmp_path / "cache",
        **kwargs,
    )
    return TestClient(app)


@pytest.fixture()
def client(settings, tmp_path, llm, dres, encoder):
    with make_client(settings, tmp_path, llm=llm, providers=[llm], dres=dres, encoder=encoder) as c:
        yield c


def assert_error(response, status):
    assert response.status_code == status, response.text
    body = response.json()
    assert set(body) == {"error"} and isinstance(body["error"], str) and body["error"]


def login(client):
    assert client.post("/api/dres/login", json={"username": "team", "password": "pw"}).status_code == 200
    assert client.post("/api/dres/select", json={"evaluation_id": "ev1"}).json() == {"evaluation_id": "ev1"}


def kf(frame) -> int:
    """0-based keyframe number of a synthetic frame (one keyframe every 2 s)."""
    return int(round(frame["time"] / 2.0))


# ----------------------------------------------------------------------------- status, files
def test_status_shape(client):
    body = client.get("/api/status").json()
    assert body == {
        "collection": {"name": "synthetic", "n_keyframes": 240, "n_videos": 6, "dim": 16},
        "encoder": "fake-encoder",
        "dres": {"connected": False, "user": None, "evaluation_id": None},
        "llm": {"gemini": False, "openai": False},
    }


def test_web_ui_is_served_with_module_mime_type(client):
    index = client.get("/")
    assert index.status_code == 200 and "text/html" in index.headers["content-type"]
    script = client.get("/app.js")
    assert script.status_code == 200
    assert script.headers["content-type"].startswith("text/javascript")


def test_frames_are_served(client):
    response = client.get("/frames/V003/011.jpg")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert response.content[:2] == b"\xff\xd8"


@pytest.mark.parametrize("path", [
    "/frames/%2E%2E/collection.json",
    "/frames/%2E%2E/001.jpg",
    "/frames/V003/..%2F..%2Fcollection.json",
    "/frames/V003/..%5C..%5Cembeddings.npy",
    "/frames/V003/%2E%2E%5C001.jpg",
    "/frames/V003/embeddings.npy",
    "/frames/V003/.hidden.jpg",
    "/frames/V003/C:%5Cwindows%5Cwin.ini",
    "/frames/V999/001.jpg",
    "/frames/V003/999.jpg",
])
def test_frame_paths_outside_the_collection_are_rejected(client, path):
    response = client.get(path)
    assert response.status_code in (400, 404), (path, response.status_code)
    assert set(response.json()) == {"error"}
    assert b"embeddings" not in response.content and b"frames_dir" not in response.content


def test_error_shapes(client):
    assert_error(client.get("/api/nope"), 404)
    assert_error(client.get("/api/search"), 405)
    assert_error(client.post("/api/search", json={"scene": ["x"]}), 422)
    assert_error(client.post("/api/search", content=b"{not json", headers={"content-type": "application/json"}), 422)
    assert_error(client.get("/api/video/V999"), 404)


# ------------------------------------------------------------------------------------ search
def test_search_frames(client):
    body = client.post("/api/search", json={"scenes": [BOAT], "top_k": 10}).json()
    assert body["mode"] == "frames" and isinstance(body["query_id"], str) and body["took_ms"] >= 0
    frames = body["frames"]
    assert len(frames) == 10
    assert set(frames[0]) == {"row", "video_id", "frame_idx", "time", "score", "thumb"}
    assert {(f["video_id"], kf(f)) for f in frames[:3]} == {("V001", 5), ("V001", 6), ("V001", 7)}
    assert frames[0]["thumb"].startswith("/frames/V001/")
    assert client.get(frames[0]["thumb"]).status_code == 200


def test_search_chains_by_default_for_two_scenes(client):
    body = client.post("/api/search", json={"scenes": [DOOR, DESK]}).json()
    assert body["mode"] == "chains"
    top = body["chains"][0]
    assert top["video_id"] == "V003"
    assert [kf(f) for f in top["frames"]][0] in (10, 11) and kf(top["frames"][1]) in (14, 15)
    assert all(set(f) >= {"row", "frame_idx", "time", "score", "thumb"} for f in top["frames"])


def test_search_videos_mode_and_filter(client):
    body = client.post("/api/search", json={"scenes": [DOOR], "mode": "videos"}).json()
    assert body["mode"] == "videos"
    group = body["videos"][0]
    assert set(group) == {"video_id", "score", "hits"} and group["video_id"] in {"V003", "V004", "V005"}
    assert all("thumb" in h for h in group["hits"])

    filtered = client.post("/api/search", json={"scenes": [DOOR], "filter_videos": ["V006"], "top_k": 20}).json()
    assert filtered["frames"] and {f["video_id"] for f in filtered["frames"]} == {"V006"}
    assert_error(client.post("/api/search", json={"scenes": [DOOR], "filter_videos": ["V999"]}), 404)


@pytest.mark.parametrize("request_body", [
    {"scenes": []},
    {"scenes": ["a", "b", "c", "d", "e"]},
    {"scenes": ["  "]},
    {"scenes": [DOOR], "window": [5, 5]},
    {"scenes": [DOOR], "window": [-1, 5]},
    {"scenes": [DOOR], "top_k": 0},
])
def test_search_rejects_bad_requests(client, request_body):
    assert_error(client.post("/api/search", json=request_body), 400)


def test_search_mode_must_be_known(client):
    assert_error(client.post("/api/search", json={"scenes": [DOOR], "mode": "shots"}), 422)


def test_reformulation_reuses_scene_encodings(client, encoder):
    client.post("/api/search", json={"scenes": [DOOR, DESK]})
    before = encoder.calls
    client.post("/api/search", json={"scenes": [DOOR, BOAT]})
    assert encoder.calls == before + 1  # only the edited scene is encoded


def test_search_reports_timings(client):
    first = client.post("/api/search", json={"scenes": [DOOR, "a scene only this test uses"]}).json()
    timings = first["timings"]
    assert set(timings) == {"encode_ms", "scan_ms", "rank_ms", "cached_scenes"}
    assert all(isinstance(v, int) and v >= 0 for v in timings.values())
    again = client.post("/api/search", json={"scenes": [DOOR, "a scene only this test uses"]}).json()
    assert again["timings"]["cached_scenes"] == 2 and again["timings"]["encode_ms"] == 0
    assert again["timings"]["rank_ms"] <= again["took_ms"]


def test_similar(client):
    body = client.post("/api/similar", json={"video_id": "V003", "frame_idx": 500, "top_k": 4}).json()
    assert set(body) == {"query_id", "took_ms", "frames"}
    got = {(f["video_id"], kf(f)) for f in body["frames"]}
    assert got == {("V003", 11), ("V004", 2), ("V005", 14), ("V005", 15)}
    assert all("thumb" in f for f in body["frames"])
    assert_error(client.post("/api/similar", json={"video_id": "V003", "frame_idx": 7}), 404)
    assert_error(client.post("/api/similar", json={"video_id": "V999", "frame_idx": 0}), 404)


def test_video_timeline_attaches_scene_scores_of_the_query(client):
    query_id = client.post("/api/search", json={"scenes": [DOOR, DESK]}).json()["query_id"]
    body = client.get(f"/api/video/V003?query_id={query_id}").json()
    assert set(body["video"]) >= {"video_id", "fps", "duration", "title", "youtube_id", "file"}
    keyframes = body["keyframes"]
    assert len(keyframes) == 40
    assert set(keyframes[0]) >= {"row", "frame_idx", "time", "thumb", "scores"}
    assert keyframes[0]["scores"] == [None, None]  # invalid first keyframe
    door_scores = [k["scores"][0] for k in keyframes[1:]]
    assert max(range(len(door_scores)), key=door_scores.__getitem__) + 1 in (10, 11)
    assert keyframes[14]["scores"][1] > 0.8

    assert body["query_known"] is True
    plain = client.get("/api/video/V003").json()
    assert all(k["scores"] is None for k in plain["keyframes"]) and plain["query_known"] is None
    evicted = client.get("/api/video/V003?query_id=unknown").json()
    assert all(k["scores"] is None for k in evicted["keyframes"])
    assert evicted["query_known"] is False  # the UI renews the query instead of silently losing curves


def test_similar_query_id_gives_a_similarity_curve(client):
    query_id = client.post("/api/similar", json={"video_id": "V001", "frame_idx": 250}).json()["query_id"]
    keyframes = client.get(f"/api/video/V001?query_id={query_id}").json()["keyframes"]
    assert keyframes[5]["scores"] == [pytest.approx(1.0, abs=1e-4)]


# ---------------------------------------------------------------------------- LLM endpoints
def test_decompose_uses_the_injected_llm(client, llm, tmp_path):
    body = client.post("/api/decompose", json={"text": f"{DOOR}, then {DESK}"}).json()
    assert body == {"scenes": [DOOR, DESK], "source": "llm", "model": "fake-llm"}
    assert list((tmp_path / "cache" / "decompose").glob("*.json"))


def test_decompose_falls_back_to_rules(settings, tmp_path):
    with make_client(settings, tmp_path, llm=FakeLLM(fail=True), providers=[]) as c:
        body = c.post("/api/decompose", json={"text": f"{DOOR}, then {DESK}"}).json()
    assert body["source"] == "rule" and len(body["scenes"]) == 2 and body["model"] is None


def test_verify_answers_within_one_deadline(settings, tmp_path, monkeypatch):
    keyed = Settings(collection_dir=settings.collection_dir, log_dir=settings.log_dir, gemini_key="unused-key")
    with make_client(keyed, tmp_path, llm=FakeLLM()) as c:
        (gemini,) = c.app.state.vbs.verify_providers  # built from the settings, never called here
        assert len(gemini.models) == verify.MAX_MODELS == 2 and gemini._timeout_ms == verify.TIMEOUT_S * 1000

    class Hanging(FakeLLM):
        def generate(self, prompt, parse, images=()):
            time.sleep(1.5)
            return super().generate(prompt, parse, images)

    monkeypatch.setattr(verify, "BUDGET_S", 0.3)
    with make_client(settings, tmp_path, llm=FakeLLM(), providers=[Hanging(), FakeLLM()]) as c:
        started = time.perf_counter()
        body = c.post("/api/verify", json={"scenes": [BOAT], "video_id": "V001", "time": 10.0}).json()
        assert time.perf_counter() - started < 1.2
    assert body["verdict"] == "warn" and "did not answer within 0.3 s" in body["reasons"][0]


def test_models_come_from_settings_so_no_keys_means_rule_fallbacks(settings, tmp_path):
    # the default factories raise in this module, so any use of the process .env would fail here
    with make_client(settings, tmp_path) as c:
        assert c.get("/api/status").json()["llm"] == {"gemini": False, "openai": False}
        assert c.post("/api/decompose", json={"text": f"{DOOR}, then {DESK}"}).json()["source"] == "rule"
        frames = [{"video_id": "V001", "frame_idx": 250}]
        assert_error(c.post("/api/vqa", json={"question": "Colour?", "frames": frames}), 503)
        verdict = c.post("/api/verify", json={"scenes": [BOAT], "video_id": "V001", "time": 10.0}).json()
        assert verdict["verdict"] == "warn" and verdict["model"] is None
        assert c.post("/api/chat", json={"initial": BOAT, "turns": []}).json()["model"] is None


def test_vqa(client, llm):
    frames = [{"video_id": "V001", "frame_idx": 250}, {"video_id": "V001", "frame_idx": 300}]
    body = client.post("/api/vqa", json={"question": "What colour is the boat?", "frames": frames}).json()
    assert body == {"answer": "blue", "evidence": "image 1 shows a blue boat", "model": "fake-llm"}
    assert llm.calls[-1][1] == 2
    assert_error(client.post("/api/vqa", json={"question": "x", "frames": [{"video_id": "V001", "frame_idx": 7}]}), 404)
    assert_error(client.post("/api/vqa", json={"question": "x", "frames": frames * 3}), 400)
    assert_error(client.post("/api/vqa", json={"question": " ", "frames": frames}), 400)


def test_vqa_without_models_is_503(settings, tmp_path):
    with make_client(settings, tmp_path, providers=[]) as c:
        response = c.post("/api/vqa", json={"question": "Why?", "frames": [{"video_id": "V001", "frame_idx": 250}]})
    assert_error(response, 503)


def test_chat_returns_scenes_and_a_question(client, llm):
    query_id = client.post("/api/search", json={"scenes": [DOOR, DESK]}).json()["query_id"]
    turns = [{"role": "team", "text": "What is the desk made of?"}, {"role": "oracle", "text": "Wood."}]
    body = client.post("/api/chat", json={"initial": DOOR, "turns": turns, "scenes": [DOOR, DESK],
                                          "query_id": query_id}).json()
    assert body == {"scenes": [DOOR, "the man sits down at a wooden desk"],
                    "next_question": "Is the desk made of wood?", "model": "fake-llm", "query_known": True}
    prompt, n_images = llm.calls[-1]
    assert "V003" in prompt and "ORACLE: Wood." in prompt and n_images >= 2
    expired = client.post("/api/chat", json={"initial": DOOR, "turns": turns, "query_id": "evicted"}).json()
    assert expired["query_known"] is False
    assert client.post("/api/chat", json={"initial": DOOR, "turns": turns}).json()["query_known"] is None


def test_chat_rule_fallback(settings, tmp_path):
    with make_client(settings, tmp_path, llm=FakeLLM(fail=True), providers=[]) as c:
        turns = [{"role": "team", "text": "What colour is the door?"},
                 {"role": "oracle", "text": "The door is red"}, {"role": "oracle", "text": "yes"}]
        body = c.post("/api/chat", json={"initial": DOOR, "turns": turns, "scenes": [DOOR]}).json()
    assert body["model"] is None
    assert body["scenes"] == [f"{DOOR}. The door is red"]
    assert body["next_question"] and "colour" not in body["next_question"].lower()


# ------------------------------------------------------------------------------------- DRES
def test_submit_without_dres_logs_locally(client, settings):
    response = client.post("/api/submit", json={"kind": "kis", "video_id": "V003", "time": 21.5})
    assert response.json() == {"ok": False, "status": "not connected", "raw": None}
    records = [json.loads(line) for line in
               (settings.log_dir / server.SUBMISSIONS_FILE).read_text(encoding="utf-8").splitlines()]
    assert records[-1]["request"]["video_id"] == "V003"
    assert records[-1]["sent"] == {"kind": "kis", "answers": [["V003", 21500, 21500]]}
    assert records[-1]["response"]["status"] == "not connected"


def test_submit_rejects_bad_requests(client):
    assert_error(client.post("/api/submit", json={"kind": "kis", "video_id": "V999", "time": 1}), 404)
    assert_error(client.post("/api/submit", json={"kind": "kis", "video_id": "V003"}), 400)
    assert_error(client.post("/api/submit", json={"kind": "kis", "video_id": "V003", "start_ms": 5}), 400)
    assert_error(client.post("/api/submit", json={"kind": "text", "text": "  "}), 400)
    assert_error(client.post("/api/submit", json={"kind": "avs", "items": []}), 400)
    assert_error(client.post("/api/submit", json={"kind": "lsc"}), 422)


def test_dres_login_evaluations_and_status(client, dres):
    assert_error(client.get("/api/dres/evaluations"), 409)
    assert_error(client.post("/api/dres/login", json={"username": "team", "password": "bad"}), 401)
    assert client.post("/api/dres/login", json={"username": "team", "password": "pw"}).json() == {
        "connected": True, "user": "team"}
    evaluations = client.get("/api/dres/evaluations").json()["evaluations"]
    assert [e["id"] for e in evaluations] == ["ev1", "ev0"]
    active = evaluations[0]
    assert set(active) == {"id", "name", "status", "type", "task"}
    assert active["task"]["name"] == "kis-t-01" and active["task"]["taskType"] == "KIS-T"
    assert active["task"]["status"] == "RUNNING" and active["task"]["remaining_s"] == 250.0
    assert evaluations[1]["task"] is None
    assert_error(client.post("/api/dres/select", json={"evaluation_id": "nope"}), 404)
    assert client.post("/api/dres/select", json={"evaluation_id": "ev1"}).json() == {"evaluation_id": "ev1"}
    assert client.get("/api/status").json()["dres"] == {"connected": True, "user": "team", "evaluation_id": "ev1"}
    assert all("tok" not in json.dumps(e) for e in evaluations)


def test_login_needs_a_url(settings, tmp_path):
    with make_client(settings, tmp_path, providers=[]) as c:
        assert_error(c.post("/api/dres/login", json={"username": "u", "password": "p"}), 400)
        assert_error(c.post("/api/dres/login", json={"url": "ftp://x", "username": "u", "password": "p"}), 400)


class CapturingDres(DresClient):
    """Records every login attempt (base URL and body) instead of sending it."""

    attempts: list[tuple[str, object]] = []

    def _request(self, method, path, *, body=None, authenticated=True):
        CapturingDres.attempts.append((self.base_url, body))
        raise DresError(401, "Unauthorized request!", method, path)


def test_login_sends_stored_credentials_only_to_dres_url(synthetic_collection_dir, tmp_path, monkeypatch):
    monkeypatch.setattr(server, "DresClient", CapturingDres)
    CapturingDres.attempts = []
    settings = Settings(collection_dir=synthetic_collection_dir, log_dir=tmp_path / "logs",
                        dres_url="http://dres.real.invalid/", dres_user="team7", dres_password="S3CRET-dres-pw")
    with make_client(settings, tmp_path, providers=[]) as c:
        # a caller-chosen host never receives the stored user or password
        response = c.post("/api/dres/login", json={"url": "http://127.0.0.1:9/"})
        assert_error(response, 400)
        assert "S3CRET" not in response.text and CapturingDres.attempts == []
        assert_error(c.post("/api/dres/login", json={"url": "http://127.0.0.1:9", "username": "team7"}), 400)
        assert CapturingDres.attempts == []
        # explicit credentials may go anywhere; the stored ones go to DRES_URL (same after normalising)
        assert_error(c.post("/api/dres/login", json={"url": "http://127.0.0.1:9", "username": "u", "password": "p"}), 401)
        assert_error(c.post("/api/dres/login", json={"url": "http://dres.real.invalid"}), 401)
        assert_error(c.post("/api/dres/login", json={}), 401)
    assert CapturingDres.attempts == [
        ("http://127.0.0.1:9", {"username": "u", "password": "p"}),
        ("http://dres.real.invalid", {"username": "team7", "password": "S3CRET-dres-pw"}),
        ("http://dres.real.invalid", {"username": "team7", "password": "S3CRET-dres-pw"}),
    ]


def test_failed_relogin_keeps_the_working_session(client, dres):
    login(client)
    connected = {"connected": True, "user": "team", "evaluation_id": "ev1"}
    assert client.get("/api/status").json()["dres"] == connected
    assert_error(client.post("/api/dres/login", json={"username": "team", "password": "typo"}), 401)
    assert client.get("/api/status").json()["dres"] == connected
    dres.verdict = "CORRECT"
    body = client.post("/api/submit", json={"kind": "kis", "video_id": "V003", "time": 3}).json()
    assert body["ok"] is True and body["raw"]["submission"] == "CORRECT"


def test_submit_kis_sends_a_point(client, dres):
    login(client)
    dres.verdict = "CORRECT"
    body = client.post("/api/submit", json={"kind": "kis", "video_id": "V003", "time": 21.2346}).json()
    assert body["ok"] is True and body["status"] == "Submission correct!"
    assert body["raw"]["submission"] == "CORRECT"
    (sent,) = dres.submissions()
    assert sent == {"answerSets": [{"answers": [{"mediaItemName": "V003", "start": 21235, "end": 21235}]}]}

    client.post("/api/submit", json={"kind": "kis", "video_id": "V003", "start_ms": 20000, "end_ms": 20500})
    assert dres.submissions()[-1]["answerSets"][0]["answers"][0] == {
        "mediaItemName": "V003", "start": 20000, "end": 20500}


def test_submit_avs_and_text(client, dres):
    login(client)
    dres.verdict = "INDETERMINATE"
    items = [{"video_id": "V001", "time": 10.0}, {"video_id": "V002", "time": 60.5}]
    assert client.post("/api/submit", json={"kind": "avs", "items": items}).json()["ok"] is True
    answer_sets = dres.submissions()[-1]["answerSets"]
    assert [s["answers"][0] for s in answer_sets] == [
        {"mediaItemName": "V001", "start": 10000, "end": 10000},
        {"mediaItemName": "V002", "start": 60500, "end": 60500},
    ]
    client.post("/api/submit", json={"kind": "text", "text": " blue "})
    assert dres.submissions()[-1] == {"answerSets": [{"answers": [{"text": "blue"}]}]}


def test_submit_reports_dres_rejections(client, dres):
    client.post("/api/dres/login", json={"username": "team", "password": "pw"})
    assert client.post("/api/submit", json={"kind": "text", "text": "x"}).json()["status"] == "no DRES evaluation selected"
    client.post("/api/dres/select", json={"evaluation_id": "ev1"})
    dres.submit_error = DresError(412, "Duplicate submission received.", "POST", "/api/v2/submit/ev1")
    body = client.post("/api/submit", json={"kind": "kis", "video_id": "V003", "time": 3}).json()
    assert body == {"ok": False, "status": "Duplicate submission received.",
                    "raw": {"http_status": 412, "description": "Duplicate submission received."}}


def test_verify_fails_near_a_moment_judged_wrong(client, dres, llm):
    login(client)
    request = {"scenes": [DOOR, DESK], "video_id": "V005", "time": 22.0}
    assert client.post("/api/verify", json=request).json()["verdict"] == "ok"
    calls = len(llm.calls)

    dres.verdict = "WRONG"
    assert client.post("/api/submit", json={"kind": "kis", "video_id": "V005", "time": 22.0}).json()["ok"] is True
    body = client.post("/api/verify", json=request).json()
    assert set(body) == {"verdict", "reasons", "model", "took_ms", "rejected", "notes"}
    assert body["verdict"] == "fail" and body["rejected"] is True
    assert "V005 was already judged wrong" in body["reasons"][0] and "0:22.0" in body["reasons"][0]
    nudged = client.post("/api/verify", json={**request, "time": 24.0}).json()
    assert nudged["verdict"] == "fail" and nudged["rejected"] is True  # the next keyframe is the same moment
    basket_item = client.post("/api/verify", json={"scenes": [], "video_id": "V003", "time": 3.0}).json()
    assert basket_item["verdict"] == "ok" and basket_item["rejected"] is False
    assert len(llm.calls) == calls  # no model call for a rejected moment
    assert client.post("/api/verify", json={**request, "video_id": "V003"}).json()["verdict"] == "ok"

    # the same video a minute later: the right video at the wrong moment is common in KIS, so it is checked
    later = client.post("/api/verify", json={**request, "time": 70.0}).json()
    assert later["verdict"] == "ok" and later["rejected"] is False and len(llm.calls) > calls
    assert "V005 already had a wrong submission" in later["notes"][0] and "0:22.0" in later["notes"][0]
    shot = client.post("/api/verify", json={"scenes": [], "video_id": "V005", "time": 60.0}).json()
    assert shot["verdict"] == "ok" and shot["rejected"] is False  # another AVS shot of that video

    dres.task_id = "task-2"  # the next task starts; the poll observes it
    client.get("/api/dres/evaluations")
    assert client.post("/api/verify", json=request).json()["verdict"] == "ok"


def test_verify_warns_with_contradictions(settings, tmp_path):
    fake = FakeLLM(contradictions=["The description says a wooden door; frame 2 shows a boat."])
    with make_client(settings, tmp_path, providers=[fake]) as c:
        body = c.post("/api/verify", json={"scenes": [DOOR], "video_id": "V001", "time": 12.0}).json()
        assert body["verdict"] == "warn" and body["model"] == "fake-llm"
        assert body["reasons"][0] in ("2 of 3 checks listed contradictions.", "3 of 3 checks listed contradictions.")
        assert body["reasons"][1:] == fake.contradictions
        assert fake.calls[-1][1] == 5  # nearest keyframe and two on each side
        assert_error(c.post("/api/verify", json={"scenes": [DOOR], "video_id": "V999", "time": 1}), 404)


def test_verify_votes_come_from_the_app(settings, tmp_path):
    for votes, calls in ((None, verify.VOTES), (1, 1)):
        fake = FakeLLM(contradictions=["The door is blue."])
        extra = {} if votes is None else {"verify_votes": votes}
        with make_client(settings, tmp_path / f"v{votes}", providers=[fake], **extra) as c:
            body = c.post("/api/verify", json={"scenes": [DOOR], "video_id": "V001", "time": 12.0}).json()
        deadline = time.perf_counter() + 2.0  # the last vote may still be running after the majority
        while len(fake.calls) < calls and time.perf_counter() < deadline:
            time.sleep(0.02)
        assert body["verdict"] == "warn" and len(fake.calls) == calls
        assert body["reasons"][-1] == "The door is blue."


def test_log_endpoint_buffers_and_flushes_on_shutdown(settings, tmp_path):
    with make_client(settings, tmp_path, providers=[]) as c:
        events = [
            {"timestamp_ms": 1_700_000_000_000, "category": "TEXT", "type": "textQuery", "value": "boat"},
            {"timestamp_ms": 1_700_000_000_001, "category": "browsing", "type": "openVideo", "value": "{}"},
            {"timestamp_ms": 1_700_000_000_002, "category": "NOPE", "type": "x", "value": "y"},
            {"category": "TEXT", "value": "no type"},
            "not an event",
        ]
        assert c.post("/api/log", json={"events": events}).json() == {"accepted": 2}
        c.post("/api/search", json={"scenes": [BOAT], "top_k": 5})
    lines = (settings.log_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["event"]["type"] for line in lines] == ["textQuery", "openVideo"]
    results = (settings.log_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(json.loads(results[-1])["result_log"]["results"]) == 5
