"""Web UI regressions (scenechain/web/app.js) in headless Chromium against the real FastAPI app.

The server runs in a thread over the synthetic collection with the fake encoder, fake models whose
delays the tests set, and a scripted DRES; the browser may only reach that local server. Skipped when
Playwright or its Chromium is not installed. Set ``VBS_SKIP_UI=1`` to skip on purpose.
"""

from __future__ import annotations

import json
import os
import socket
import threading
import time
from types import SimpleNamespace

import pytest

if os.environ.get("VBS_SKIP_UI") == "1":
    pytest.skip("VBS_SKIP_UI=1", allow_module_level=True)
sync_api = pytest.importorskip("playwright.sync_api")
uvicorn = pytest.importorskip("uvicorn")

from tests.conftest import CONCEPT_PHRASES, FakeEncoder  # noqa: E402
from tests.test_server import ScriptedDres  # noqa: E402
from scenechain import server  # noqa: E402
from scenechain.config import Settings  # noqa: E402
from scenechain.dres import DresError  # noqa: E402

BOAT, DOOR, DESK = CONCEPT_PHRASES["boat"], CONCEPT_PHRASES["door"], CONCEPT_PHRASES["desk"]


class Models:
    """Verify, chat and VQA model with delays set by the tests."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.verify_delay = 0.0
        self.contradictions: list[str] = []
        self.chat_delay = 0.0
        self.calls: list[str] = []

    def generate(self, prompt, parse, images=()):
        if '"contradictions"' in prompt:
            self.calls.append("verify")
            time.sleep(self.verify_delay)
            reply = {"contradictions": list(self.contradictions)}
        elif '"next_question"' in prompt:
            self.calls.append("chat")
            time.sleep(self.chat_delay)
            reply = {"scenes": ["CHAT SCENE: a man opens a wooden door"], "next_question": "Is the door red?"}
        elif '"answer"' in prompt:
            self.calls.append("vqa")
            slow = "boat" in prompt
            time.sleep(1.5 if slow else 0.2)
            answer = "blue" if slow else "red"
            reply = {"answer": answer, "evidence": f"evidence for {answer}"}
        else:
            reply = {"scenes": [prompt[:20]]}
        return parse(json.dumps(reply)), "fake-model"


class Dres(ScriptedDres):
    submit_delay = 0.0

    def _request(self, method, path, *, body=None, authenticated=True):
        if path.startswith("/api/v2/submit/"):
            time.sleep(self.submit_delay)
        return super()._request(method, path, body=body, authenticated=authenticated)


@pytest.fixture(scope="module")
def ui(synthetic_collection_dir, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("web-ui")
    models, dres = Models(), Dres()
    dres.login("team", "pw")
    app = server.create_app(
        Settings(collection_dir=synthetic_collection_dir, log_dir=tmp / "logs", gemini_key="unused"),
        encoder=FakeEncoder(), llm=models, vqa_providers=[models], dres_client=dres,
        cache_dir=None, warm_up=False,
    )
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    uv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=uv.run, daemon=True)
    thread.start()
    deadline = time.time() + 20
    while not uv.started and time.time() < deadline:
        time.sleep(0.05)
    pw = sync_api.sync_playwright().start()
    try:
        browser = pw.chromium.launch()
    except Exception as exc:  # noqa: BLE001 - browser binaries missing
        pw.stop()
        uv.should_exit = True
        pytest.skip(f"Playwright Chromium is not available: {exc}")
    yield SimpleNamespace(app=app, models=models, dres=dres, browser=browser, base=f"http://127.0.0.1:{port}")
    browser.close()
    pw.stop()
    uv.should_exit = True
    thread.join(timeout=10)


class Page:
    """One browser page on a fresh UI, with the /api requests it sent."""

    def __init__(self, ui, *, clock: bool = False) -> None:
        ui.models.reset()
        ui.dres.verdict, ui.dres.submit_error, ui.dres.submit_delay = "WRONG", None, 0.0
        ui.dres.task_id = f"task-{time.time_ns()}"  # a new task: no videos judged wrong yet
        self.ui = ui
        self.context = ui.browser.new_context(viewport={"width": 1600, "height": 900})
        self.context.route("**/*", lambda route: route.continue_() if route.request.url.startswith(ui.base)
                           else route.abort())
        self.page = self.context.new_page()
        self.errors: list[str] = []
        self.requests: list[tuple[str, object]] = []
        self.page.on("pageerror", lambda error: self.errors.append(str(error)))
        self.page.on("request", self._record)
        if clock:
            self.page.clock.install()
        self.page.goto(ui.base + "/")
        self.page.wait_for_function("document.querySelector('#evaluation-select').value === 'ev1'", timeout=15000)

    def _record(self, request) -> None:
        path = request.url[len(self.ui.base):]
        if path.startswith("/api/"):
            data = request.post_data
            self.requests.append((path, json.loads(data) if data else None))

    def sent(self, prefix: str) -> list:
        return [body for path, body in self.requests if path.startswith(prefix)]

    def search(self, text: str) -> None:
        box = self.page.locator("#scene-list textarea").first
        box.fill(text)
        with self.page.expect_response(lambda r: r.url.endswith("/api/search")):
            box.press("Control+Enter")
        self.page.wait_for_selector("#results-content .card")
        self.page.wait_for_timeout(200)

    def open_first_card(self) -> None:
        with self.page.expect_response(lambda r: "/api/video/" in r.url):
            self.page.locator("#results-content .card").first.click()
        self.page.wait_for_function("document.querySelectorAll('.kf-bar').length > 0")

    def log_rows(self) -> list[str]:
        return self.page.eval_on_selector_all("#log-body tr", "rows => rows.map(r => r.innerText.replace(/\\s+/g, ' '))")

    def close(self) -> None:
        assert self.errors == []
        self.context.close()


@pytest.fixture()
def make_page(ui):
    pages: list[Page] = []

    def make(**kwargs) -> Page:
        pages.append(Page(ui, **kwargs))
        return pages[-1]

    yield make
    for page in pages:
        page.close()


# ------------------------------------------------------------------------ submission guard
def test_space_or_enter_after_clicking_submit_never_confirms_a_warning(ui, make_page):
    p = make_page()
    ui.models.contradictions = ["the boat is red, not blue"]
    p.search(BOAT)
    p.open_first_card()
    p.page.click("#video-submit")
    p.page.wait_for_selector("#guard.warn")
    assert p.page.evaluate("document.activeElement.id") != "video-submit"  # a click leaves no focus
    p.page.keyboard.press("Space")  # play / pause, not a click on the submit button
    p.page.focus("#video-submit")  # even when the button has focus (Tab)
    p.page.keyboard.press("Space")
    p.page.keyboard.press("Enter")
    p.page.wait_for_timeout(600)
    assert p.sent("/api/submit") == [] and p.page.is_visible("#guard.warn")

    p.page.keyboard.press("Escape")
    card = p.page.locator("#results-content .card").nth(1)
    card.hover()
    card.locator("[data-action=submit]").click()
    p.page.wait_for_selector("#guard.warn")
    assert "card-action" not in p.page.evaluate("document.activeElement.className")
    p.page.keyboard.press("Space")
    p.page.wait_for_timeout(600)
    assert p.sent("/api/submit") == []
    p.page.keyboard.press("s")  # the second S is still the way to confirm
    p.page.wait_for_function("document.querySelectorAll('#log-body tr').length === 1")
    assert len(p.sent("/api/submit")) == 1 and "warn, confirmed" in p.log_rows()[0]


def test_a_second_press_right_after_an_instant_ok_does_not_submit_again(ui, make_page):
    p = make_page()
    ui.dres.submit_delay = 0.4
    p.search(BOAT)
    p.page.locator("#results-content .card").first.hover()
    p.page.keyboard.press("s")
    p.page.wait_for_timeout(250)
    p.page.keyboard.press("s")
    p.page.wait_for_timeout(1200)
    assert len(p.sent("/api/submit")) == 1

    p.open_first_card()  # a playing video moves between the two presses
    p.page.mouse.move(5, 890)
    p.page.locator("#video-panel").focus()
    p.page.keyboard.press("Space")
    p.page.wait_for_timeout(300)
    p.page.keyboard.press("s")
    p.page.wait_for_timeout(200)
    p.page.keyboard.press("s")
    p.page.wait_for_timeout(1500)
    assert len(p.sent("/api/submit")) == 2


def test_a_hanging_check_times_out_or_can_be_skipped(ui, make_page):
    p = make_page(clock=True)
    ui.models.verify_delay = 12.0
    p.search(BOAT)
    p.page.locator("#results-content .card").first.hover()
    p.page.keyboard.press("s")
    p.page.wait_for_selector("#guard.checking")
    p.page.keyboard.press("s")  # ignored while checking, with a hint
    p.page.wait_for_selector("#toasts .toast:has-text('Still checking')")
    assert p.page.is_visible("#guard .guard-skip")
    p.page.clock.fast_forward(11000)  # the UI stops waiting after 10 s
    p.page.wait_for_selector("#guard.warn")
    assert "did not answer within 10 s" in p.page.inner_text("#guard")
    p.page.keyboard.press("s")
    p.page.wait_for_function("document.querySelectorAll('#log-body tr').length === 1")

    p.page.locator("#results-content .card").nth(3).hover()
    p.page.keyboard.press("s")
    p.page.wait_for_selector("#guard.checking")
    p.page.click("#guard .guard-skip")
    p.page.wait_for_function("document.querySelectorAll('#log-body tr').length === 2")
    assert len(p.sent("/api/submit")) == 2 and "not checked" in p.log_rows()[0]


def test_dres_412_max_correct_is_not_counted_as_correct(ui, make_page):
    p = make_page()
    ui.dres.submit_error = DresError(
        412, "Submission 7c1f was rejected by filter: Maximum number of correct submissions (1) exceeded for the team.",
        "POST", "/api/v2/submit/ev1")
    p.search(BOAT)
    p.page.locator("#results-content .card").first.hover()
    p.page.keyboard.press("s")
    p.page.wait_for_selector("#log-body .verdict:not(.sending)", state="attached")
    assert p.page.inner_text("#count-correct") == "0"
    assert p.page.locator("#log-body .verdict.error").count() == 1


def test_rejected_flags_reset_when_the_task_is_run_again_under_the_same_name(ui, make_page):
    p = make_page(clock=True)
    p.search(BOAT)
    p.page.locator("#results-content .card").first.hover()
    p.page.keyboard.press("s")  # judged WRONG
    p.page.wait_for_selector("#results-content .card.is-rejected")
    ui.dres.task_id = "rerun-of-kis-t-01"  # same task name, new run
    with p.page.expect_response(lambda r: r.url.endswith("/api/dres/evaluations")):
        p.page.clock.fast_forward(10500)
    p.page.wait_for_function("document.querySelectorAll('#results-content .card.is-rejected').length === 0")


# ------------------------------------------------------------------------ stale replies
def test_a_late_vqa_reply_never_overwrites_a_newer_or_typed_answer(ui, make_page):
    p = make_page()
    p.search(BOAT)
    p.page.locator("#results-content .card").first.hover()
    p.page.keyboard.press("a")
    question = p.page.locator("#vqa-question")
    question.fill("What colour is the boat?")  # slow reply: blue
    question.press("Enter")
    p.page.wait_for_timeout(100)
    question.fill("What colour is the car?")  # fast reply: red
    question.press("Enter")
    p.page.wait_for_function("document.querySelector('#vqa-answer').value === 'red'")
    p.page.fill("#vqa-answer", "green")
    p.page.wait_for_timeout(2000)
    assert p.page.input_value("#vqa-answer") == "green"
    assert "blue" not in p.page.inner_text("#vqa-evidence")

    question.fill("What colour is the boat?")
    question.press("Enter")
    p.page.fill("#vqa-answer", "my own answer")  # typed while the model is thinking
    p.page.wait_for_function("document.querySelector('#vqa-evidence').textContent.includes('blue')")
    assert p.page.input_value("#vqa-answer") == "my own answer"


def test_a_chat_reply_after_a_new_search_does_not_replace_it(ui, make_page):
    p = make_page()
    ui.models.chat_delay = 2.0
    p.page.click("#qtab-chat")
    p.page.fill("#chat-initial", "a man opens a wooden door and sits at a desk")
    p.page.click("#chat-update")
    p.page.wait_for_timeout(200)
    p.page.click("#qtab-query")
    p.search(BOAT)
    searches = len(p.sent("/api/search"))
    p.page.wait_for_selector("#chat-apply", state="attached")
    p.page.wait_for_function("document.querySelector('#chat-update-label').textContent === 'Update scenes'", timeout=10000)
    p.page.wait_for_timeout(300)
    assert p.page.input_value("#scene-list textarea >> nth=0") == BOAT
    assert len(p.sent("/api/search")) == searches
    p.page.click("#qtab-chat")
    with p.page.expect_response(lambda r: r.url.endswith("/api/search")):
        p.page.click("#chat-apply")  # the operator decides to use them
    assert p.page.input_value("#scene-list textarea >> nth=0").startswith("CHAT SCENE")


def test_back_to_an_evicted_query_renews_its_score_curves(ui, make_page):
    state = ui.app.state.vbs
    state.queries = server._LRU(2)
    try:
        p = make_page()
        p.search(BOAT)
        p.open_first_card()
        p.search(DOOR)
        p.search(DESK)  # the server now holds only the DOOR and DESK queries
        p.page.click("#results-back")
        p.page.wait_for_timeout(500)
        before = len(p.sent("/api/search"))
        with p.page.expect_response(lambda r: r.url.endswith("/api/search")):
            p.page.click("#results-back")
        p.page.wait_for_function("document.querySelector('#curve-status').textContent.startsWith('Rescored')")
        assert p.page.locator(".kf-bar").count() > 0
        assert p.sent("/api/search")[before:][0]["scenes"] == [BOAT]
    finally:
        state.queries = server._LRU(server.QUERY_CACHE_SIZE)


def test_interaction_log_batches_hold_at_most_1000_events(ui, make_page):
    p = make_page(clock=True)
    p.search(BOAT)
    p.open_first_card()
    p.page.evaluate("""() => {
      for (let i = 0; i < 1300; i += 1) {
        document.dispatchEvent(new KeyboardEvent('keydown', { key: i % 2 ? 'ArrowLeft' : 'ArrowRight', shiftKey: true, bubbles: true }));
      }
    }""")
    responses = []
    p.page.on("response", lambda r: responses.append(r) if r.url.endswith("/api/log") else None)
    p.page.clock.fast_forward(5500)
    p.page.wait_for_timeout(1500)
    batches = [(len(json.loads(r.request.post_data)["events"]), r.json()["accepted"]) for r in responses]
    assert batches and all(sent <= 1000 and accepted == sent for sent, accepted in batches)
    assert sum(sent for sent, _ in batches) >= 1300
