"""FastAPI server of the VBS tool: the HTTP API of ``docs/ARCHITECTURE.md`` §6, the web UI and keyframes.

Run from the repository root::

    python -m scenechain.server --collection collections/aic2026 --port 8787 [--device cuda]

Design points:

* All ranking goes through :mod:`scenechain.search`; the server only validates requests, encodes
  scene texts, keeps state and adds thumbnail URLs.
* Scene similarity vectors of the last :data:`QUERY_CACHE_SIZE` queries are kept under a ``query_id``
  so ``/api/video/{id}?query_id=`` draws score curves without re-encoding, and the similarities of the
  last :data:`TEXT_CACHE_SIZE` scene texts are reused, so editing one scene of a multi-scene query
  (fast reformulation) encodes only that scene. An unknown or evicted ``query_id`` is ignored and
  reported as ``query_known: false``, so the UI can renew it.
* Every error is ``{"error": str}``, including 404/405/422 raised by the framework.
* KIS and AVS answers go to DRES as points (``start_ms == end_ms``): DRES judges a temporal answer
  correct only when the whole range lies inside the target segment (confirmed on DRES 2.0.4).
* KIS moments that DRES judged WRONG are remembered per evaluation and task. ``/api/verify`` fails a
  moment within ``verify.WRONG_RADIUS_S`` of one of them before a second wrong submission is sent;
  other moments of the same video are checked as usual and carry a note (the right video at the wrong
  moment is a common KIS mistake). AVS verdicts come from a human judge and never reach the client,
  so AVS submissions are not recorded, and no AVS shot is blocked because of another shot.
* Startup runs one similarity scan (pages the memory-mapped matrix in; ~3 s on AIC 2026) and one
  encoder call, so the first operator query is not the slow one. The interaction log is flushed
  every :data:`LOG_FLUSH_INTERVAL_S` seconds and on shutdown.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import mimetypes
import re
import secrets
import threading
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional, Sequence

import numpy as np
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.routing import APIRoute
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from starlette.exceptions import HTTPException

from . import conversation, decompose, search, verify, vqa
from .collection import Collection
from .config import VBS_ROOT, Settings, load_settings
from .dres import DresClient, DresError, normalize_base_url
from .interaction_log import InteractionLog
from .llm import GeminiClient, LLMClient, LLMError, OpenAIClient, optional_openai

# app.js is an ES module; some Windows registries map .js to text/plain and the browser refuses it
mimetypes.add_type("text/javascript", ".js")
mimetypes.add_type("text/css", ".css")

_LOG = logging.getLogger("scenechain.server")

WEB_DIR = Path(__file__).resolve().parent / "web"
#: disk caches of the LLM helpers: <cache_dir>/decompose, /verify, /conversation
CACHE_ROOT = VBS_ROOT / "cache"
QUERY_CACHE_SIZE = 32
TEXT_CACHE_SIZE = 16
MAX_SCENES = 4
MAX_TOP_K = 2000
#: frames ranked before grouping in "videos" mode
VIDEO_MODE_POOL = 1000
MAX_VIDEOS = 60
FRAMES_PER_VIDEO = 8
MAX_VQA_FRAMES = vqa.MAX_FRAMES
MAX_LOG_EVENTS = 1000
MAX_AVS_ITEMS = 500
CHAT_CANDIDATES = conversation.MAX_CANDIDATES
LOG_FLUSH_INTERVAL_S = 5.0
SUBMISSIONS_FILE = "submissions.jsonl"
IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"})
VIDEO_EXTENSIONS = frozenset({".mp4", ".webm", ".m4v", ".mov", ".mkv", ".ogv"})
#: one path segment of a frame URL: letters, digits, dot, underscore, hyphen; no leading dot
_SAFE_SEGMENT = re.compile(r"[A-Za-z0-9_-][A-Za-z0-9._-]{0,254}")


# ------------------------------------------------------------------------------ request bodies
class _Body(BaseModel):
    model_config = ConfigDict(extra="ignore")


class SearchBody(_Body):
    scenes: list[str]
    window: list[float] = Field(default_factory=lambda: [0.0, 30.0])
    mode: Optional[Literal["frames", "videos", "chains"]] = None
    top_k: int = 300
    filter_videos: Optional[list[str]] = None


class DecomposeBody(_Body):
    text: str


class SimilarBody(_Body):
    video_id: str
    frame_idx: int
    top_k: int = 300


class FrameRef(_Body):
    video_id: str
    frame_idx: int


class VqaBody(_Body):
    question: str
    frames: list[FrameRef]


class LoginBody(_Body):
    url: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None


class SelectBody(_Body):
    evaluation_id: str


class SubmitItem(_Body):
    video_id: str
    time: Optional[float] = None
    start_ms: Optional[int] = None
    end_ms: Optional[int] = None


class SubmitBody(_Body):
    kind: Literal["kis", "text", "avs"]
    video_id: Optional[str] = None
    time: Optional[float] = None
    start_ms: Optional[int] = None
    end_ms: Optional[int] = None
    text: Optional[str] = None
    items: Optional[list[SubmitItem]] = None


class LogBody(_Body):
    events: list[Any]


class VerifyBody(_Body):
    scenes: list[str]
    video_id: str
    time: float


class ChatTurn(_Body):
    role: str
    text: str


class ChatBody(_Body):
    initial: str = ""
    turns: list[ChatTurn] = Field(default_factory=list)
    scenes: list[str] = Field(default_factory=list)
    query_id: Optional[str] = None


# ------------------------------------------------------------------------------------- state
class NullLLM:
    """The text model when no key is configured: every call fails, so the rule fallbacks run."""

    name = "none"

    def generate(self, prompt, parse, images=()):  # noqa: ANN001 - LLMClient protocol
        raise LLMError("no LLM configured: set GEMINI_API_KEY")


def _fail(status: int, message: str) -> HTTPException:
    return HTTPException(status_code=status, detail=message)


class _TimedEncoder:
    """Wraps a text encoder and adds up the seconds spent in ``encode_chunks``."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.name = getattr(inner, "name", type(inner).__name__)
        self.seconds = 0.0

    def encode_chunks(self, text: str) -> list:
        started = time.perf_counter()
        try:
            return self.inner.encode_chunks(text)
        finally:
            self.seconds += time.perf_counter() - started

    def encode(self, text: str):  # noqa: ANN201 - TextEncoder protocol
        started = time.perf_counter()
        try:
            return self.inner.encode(text)
        finally:
            self.seconds += time.perf_counter() - started


@dataclass
class QueryEntry:
    kind: Literal["text", "similar"]
    scenes: list[str]
    sims: list[np.ndarray]
    window: tuple[float, float]


class _LRU:
    """A small thread-safe LRU map."""

    def __init__(self, size: int) -> None:
        self.size = size
        self._items: OrderedDict[str, Any] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: Optional[str]) -> Any:
        if not key:
            return None
        with self._lock:
            value = self._items.get(key)
            if value is not None:
                self._items.move_to_end(key)
            return value

    def put(self, key: str, value: Any) -> None:
        with self._lock:
            self._items[key] = value
            self._items.move_to_end(key)
            while len(self._items) > self.size:
                self._items.popitem(last=False)

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)


class ServerState:
    """Everything the request handlers share; one instance per app."""

    def __init__(
        self,
        settings: Settings,
        collection: Collection,
        encoder: Any,
        llm: Optional[LLMClient],
        vqa_providers: Optional[Sequence[LLMClient]],
        dres_client: Optional[DresClient],
        cache_dir: Optional[Path] = CACHE_ROOT,
    ) -> None:
        self.settings = settings
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.collection = collection
        self.encoder = encoder
        self.llm = llm
        self.vqa_providers = list(vqa_providers) if vqa_providers is not None else None
        self.verify_providers = self.vqa_providers
        self.queries = _LRU(QUERY_CACHE_SIZE)
        self.texts = _LRU(TEXT_CACHE_SIZE)
        self.encode_lock = threading.Lock()
        self.dres_lock = threading.RLock()
        self.dres_client = dres_client
        self.evaluation_id: Optional[str] = None
        #: last observed DRES task run id per evaluation
        self.task_ids: dict[str, Optional[str]] = {}
        #: evaluation id -> [(task run id, video id, time in seconds)] of KIS submissions judged WRONG;
        #: a moment, not a video, is what DRES rejected (``verify.wrong_history``)
        self.wrong: dict[str, list[tuple[Optional[str], str, float]]] = {}
        #: model calls per pre-submission check; the majority decides (``verify.VOTES``)
        self.verify_votes = verify.VOTES
        self.submissions_lock = threading.Lock()
        self.log = InteractionLog(
            settings.log_dir, dres_client, collection_name=settings.dres_collection
        )

    # -------------------------------------------------------------------------- similarities
    def scene_sims(self, text: str, timings: Optional[dict] = None) -> np.ndarray:
        """Read-only similarity vector of one scene text (cached for the last few texts).

        ``timings`` (optional) accumulates ``encode_ms`` (text model), ``scan_ms`` (similarity scan
        over the embedding matrix) and ``cached_scenes`` (texts served from the cache).
        """
        cached = self.texts.get(text)
        if cached is not None:
            if timings is not None:
                timings["cached_scenes"] = timings.get("cached_scenes", 0) + 1
            return cached
        timed = _TimedEncoder(self.encoder)
        with self.encode_lock:
            started = time.perf_counter()
            sims = search.text_similarities(self.collection, timed, text)
            total_s = time.perf_counter() - started
        sims.setflags(write=False)
        self.texts.put(text, sims)
        if timings is not None:
            timings["encode_ms"] = timings.get("encode_ms", 0.0) + timed.seconds * 1000
            timings["scan_ms"] = timings.get("scan_ms", 0.0) + max(0.0, total_s - timed.seconds) * 1000
        return sims

    def warm_up(self) -> None:
        """Page the embedding matrix in and run the encoder once; failures are only logged."""
        started = time.perf_counter()
        try:
            self.collection.similarities(np.zeros(self.collection.dim, dtype=np.float32))
        except Exception as exc:  # noqa: BLE001
            _LOG.warning("similarity warm-up failed: %s", exc)
        scanned = time.perf_counter()
        try:
            with self.encode_lock:
                self.encoder.encode_chunks("a video keyframe")
        except Exception as exc:  # noqa: BLE001
            _LOG.warning("encoder warm-up failed: %s", exc)
        encoded = time.perf_counter()
        # Build the guard's model clients now (imports google.genai, several seconds on a loaded laptop)
        # so the first pre-submission check does not spend its deadline on the import.
        for provider in self.verify_providers or ():
            prepare = getattr(provider, "_get_client", None)
            if prepare is None:
                continue
            try:
                prepare()
            except Exception as exc:  # noqa: BLE001
                _LOG.warning("verify client warm-up failed: %s", exc)
        _LOG.info("warm-up: similarity scan %.2f s, encoder %.2f s, model clients %.2f s",
                  scanned - started, encoded - scanned, time.perf_counter() - encoded)

    # ---------------------------------------------------------------------------------- DRES
    def dres_snapshot(self) -> tuple[Optional[DresClient], Optional[str], bool]:
        with self.dres_lock:
            client = self.dres_client
            connected = client is not None and bool(getattr(client, "logged_in", False))
            return client, self.evaluation_id, connected

    def require_dres(self) -> DresClient:
        client, _, connected = self.dres_snapshot()
        if not connected or client is None:
            raise _fail(409, "not connected to DRES")
        return client

    def observe_task(self, evaluation_id: str, task_id: Optional[str]) -> None:
        with self.dres_lock:
            self.task_ids[evaluation_id] = task_id

    def fetch_task_id(self, client: DresClient, evaluation_id: str) -> Optional[str]:
        """Current task run id from ``evaluation_state``; the last observed one if that fails."""
        try:
            state = client.evaluation_state(evaluation_id)
            task_id = state.get("taskId") if isinstance(state, dict) else None
        except Exception:  # noqa: BLE001 - best effort
            with self.dres_lock:
                return self.task_ids.get(evaluation_id)
        self.observe_task(evaluation_id, task_id)
        return task_id

    def task_summary(self, client: DresClient, evaluation_id: str) -> Optional[dict]:
        """Name and type from ``task_info``; status, run id and time left from ``evaluation_state``."""
        info = state = None
        try:
            info = client.task_info(evaluation_id)
        except Exception:  # noqa: BLE001 - one failing evaluation must not break the list
            info = None
        try:
            state = client.evaluation_state(evaluation_id)
        except Exception:  # noqa: BLE001
            state = None
        if not isinstance(info, dict) and not isinstance(state, dict):
            return None
        task: dict[str, Any] = {}
        if isinstance(info, dict):
            for key in ("name", "taskGroup", "taskType", "duration"):
                task[key] = info.get(key)
        if isinstance(state, dict):
            status = state.get("taskStatus")
            time_left = state.get("timeLeft")
            running = str(status).upper() == "RUNNING"
            task["status"] = status
            task["task_id"] = state.get("taskId")
            task["remaining_s"] = (
                float(time_left)
                if running and isinstance(time_left, (int, float)) and math.isfinite(time_left)
                else None
            )
            self.observe_task(evaluation_id, state.get("taskId"))
        return task

    def record_wrong(self, client: DresClient, evaluation_id: str, video_id: str, time_s: float) -> None:
        task_id = self.fetch_task_id(client, evaluation_id)
        with self.dres_lock:
            self.wrong.setdefault(evaluation_id, []).append((task_id, video_id, float(time_s)))

    def rejected(self, evaluation_id: Optional[str]) -> dict[str, list[float]]:
        """Videos with a KIS submission judged WRONG in the current task of ``evaluation_id``, mapped to
        the submitted times: :func:`verify.verify_moment` blocks only moments near those times."""
        if not evaluation_id:
            return {}
        with self.dres_lock:
            current = self.task_ids.get(evaluation_id)
            out: dict[str, list[float]] = {}
            for task_id, video_id, time_s in self.wrong.get(evaluation_id, []):
                if task_id == current:
                    out.setdefault(video_id, []).append(time_s)
            return out

    def write_submission(self, record: dict) -> None:
        try:
            self.settings.log_dir.mkdir(parents=True, exist_ok=True)
            with self.submissions_lock, (self.settings.log_dir / SUBMISSIONS_FILE).open(
                "a", encoding="utf-8", newline="\n"
            ) as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as exc:
            _LOG.error("could not write the local submission log: %s", exc)

    # ----------------------------------------------------------------------------------- log
    def flush_log(self) -> None:
        try:
            report = self.log.flush()
            if report.get("dres_error"):
                _LOG.warning("DRES log upload failed: %s", report["dres_error"])
        except Exception as exc:  # noqa: BLE001 - logging must never stop the server
            _LOG.error("interaction log flush failed: %s", exc)

    def flush_loop(self, stop: threading.Event) -> None:
        while not stop.wait(LOG_FLUSH_INTERVAL_S):
            self.flush_log()

    # ---------------------------------------------------------------------------------- chat
    def top_videos(self, entry: Optional[QueryEntry]) -> list[dict]:
        """The best :data:`CHAT_CANDIDATES` videos of a cached query with their best keyframe."""
        if entry is None:
            return []
        coll = self.collection
        best: list[tuple[str, int]] = []
        try:
            if len(entry.sims) == 1:
                hits = search.rank_frames(entry.sims[0], coll, 500)
                groups = search.group_by_video(hits, max_videos=CHAT_CANDIDATES, frames_per_video=1)
                best = [(g["video_id"], g["hits"][0]["row"]) for g in groups]
            else:
                seen: set[str] = set()
                for chain in search.temporal_search(coll, entry.sims, window_s=entry.window, top_k=60):
                    if chain["video_id"] in seen:
                        continue
                    seen.add(chain["video_id"])
                    frame = max(chain["frames"], key=lambda f: f["score"])
                    best.append((chain["video_id"], frame["row"]))
                    if len(best) >= CHAT_CANDIDATES:
                        break
        except Exception as exc:  # noqa: BLE001 - candidates only improve the question
            _LOG.warning("chat candidates failed: %s", exc)
            return []
        return [
            {
                "video_id": vid,
                "caption_hint": str(coll.video(vid).get("title") or ""),
                "image_path": str(coll.frame_path(row)),
            }
            for vid, row in best
        ]


# ----------------------------------------------------------------------------------- helpers
def _with_thumb(collection: Collection, frame: dict) -> dict:
    frame["thumb"] = collection.frame_url(int(frame["row"]))
    return frame


def _took_ms(started: float) -> int:
    return int(round((time.perf_counter() - started) * 1000))


def _require_video(collection: Collection, video_id: str) -> dict:
    try:
        return collection.video(video_id)
    except KeyError:
        raise _fail(404, f"unknown video {video_id!r}") from None


def _check_top_k(top_k: int) -> int:
    if not 1 <= top_k <= MAX_TOP_K:
        raise _fail(400, f"top_k must be between 1 and {MAX_TOP_K}")
    return top_k


def _point(collection: Collection, video_id: str, time_s: Optional[float],
           start_ms: Optional[int], end_ms: Optional[int]) -> tuple[int, int]:
    """``(start_ms, end_ms)`` of one KIS/AVS answer: a point at ``time_s`` unless both ms are given."""
    _require_video(collection, video_id)
    if start_ms is not None or end_ms is not None:
        if start_ms is None or end_ms is None:
            raise _fail(400, "give both start_ms and end_ms, or time")
        if not 0 <= start_ms <= end_ms:
            raise _fail(400, "start_ms and end_ms must satisfy 0 <= start_ms <= end_ms")
        return int(start_ms), int(end_ms)
    if time_s is None or not math.isfinite(time_s):
        raise _fail(400, "a finite time (seconds) or start_ms and end_ms is required")
    return search.submission_segment(collection, video_id, float(time_s))


def _local_video_path(collection: Collection, video: dict) -> Optional[Path]:
    """The local file of a video whose ``file`` is not an http(s) URL, if it exists."""
    file = video.get("file")
    if not file or re.match(r"^https?://", str(file), re.IGNORECASE):
        return None
    p = Path(str(file))
    p = p if p.is_absolute() else collection.root / p
    if p.suffix.lower() not in VIDEO_EXTENSIONS or not p.is_file():
        return None
    return p


def _validation_message(exc: RequestValidationError) -> str:
    parts = []
    for err in exc.errors()[:5]:
        loc = ".".join(str(x) for x in err.get("loc", ()) if x != "body")
        msg = str(err.get("msg") or "invalid value")
        parts.append(f"{loc}: {msg}" if loc else msg)
    return "invalid request: " + ("; ".join(parts) or "malformed body")


# ------------------------------------------------------------------------------------ the app
def create_app(
    settings: Settings,
    collection: Optional[Collection] = None,
    encoder: Any = None,
    llm: Optional[LLMClient] = None,
    vqa_providers: Optional[Sequence[LLMClient]] = None,
    dres_client: Optional[DresClient] = None,
    *,
    warm_up: bool = True,
    web_dir: Optional[Path] = WEB_DIR,
    cache_dir: Optional[Path] = CACHE_ROOT,
    verify_votes: int = verify.VOTES,
) -> FastAPI:
    """Build the FastAPI app.

    ``collection`` defaults to ``Collection.load(settings.collection_dir)`` and ``encoder`` to a
    :class:`~scenechain.encoder.SiglipTextEncoder` (loads the model). ``llm`` is the text model of
    ``/api/decompose`` and ``/api/chat``; ``vqa_providers`` serve ``/api/vqa`` and ``/api/verify``.
    Both default to clients built from ``settings`` (one Gemini client for decompose, chat and VQA,
    plus OpenAI for VQA when available; a Gemini client with a short timeout for verify), so the
    server uses exactly the keys that ``/api/status`` reports; without keys the rule fallbacks run
    and VQA answers 503. ``dres_client`` injects a DRES client (tests); otherwise one is created at
    login. ``warm_up`` runs the startup similarity scan and encoder call in the lifespan.
    ``cache_dir`` is the root of the LLM disk caches (``None`` disables them); ``web_dir`` is served at
    ``/`` (``None`` serves no UI). ``verify_votes`` is the number of parallel model calls of each
    pre-submission check, decided by majority (1: a single call).
    """
    if collection is None:
        collection = Collection.load(settings.collection_dir)
    if encoder is None:
        from .encoder import SiglipTextEncoder  # loads torch; only when no encoder is injected

        encoder = SiglipTextEncoder()
    verify_providers: Optional[list[LLMClient]] = list(vqa_providers) if vqa_providers is not None else None
    if llm is None or vqa_providers is None:
        gemini = GeminiClient(settings.gemini_key) if settings.gemini_key else None
        openai_client = optional_openai(settings.openai_key, settings.openai_base_url, settings.openai_model)
        if llm is None:
            llm = gemini if gemini is not None else NullLLM()
        if vqa_providers is None:
            vqa_providers = [c for c in (gemini, openai_client) if c is not None]
            verify_providers = [c for c in (
                verify.gemini_client(settings.gemini_key) if settings.gemini_key else None,
                openai_client,
            ) if c is not None]
    state = ServerState(settings, collection, encoder, llm, vqa_providers, dres_client, cache_dir)
    state.verify_providers = verify_providers
    state.verify_votes = max(1, int(verify_votes))

    def cache(name: str) -> Optional[Path]:
        return state.cache_dir / name if state.cache_dir is not None else None
    coll = collection

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if warm_up:
            state.warm_up()
        stop = threading.Event()
        flusher = threading.Thread(target=state.flush_loop, args=(stop,), name="vbs-log-flush", daemon=True)
        flusher.start()
        try:
            yield
        finally:
            stop.set()
            flusher.join(timeout=2.0)
            state.flush_log()

    app = FastAPI(title="VBS 2027 search tool", lifespan=lifespan, docs_url=None, redoc_url=None,
                  openapi_url=None)
    app.state.vbs = state

    # ------------------------------------------------------------------------- error shape
    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException) -> JSONResponse:
        return JSONResponse({"error": str(exc.detail)}, status_code=exc.status_code,
                            headers=getattr(exc, "headers", None))

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse({"error": _validation_message(exc)}, status_code=422)

    @app.exception_handler(Exception)
    async def internal_error(request: Request, exc: Exception) -> JSONResponse:
        _LOG.exception("unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse({"error": f"internal error: {type(exc).__name__}: {exc}"}, status_code=500)

    # ------------------------------------------------------------------------------ status
    @app.get("/api/status")
    def api_status() -> dict:
        client, evaluation_id, connected = state.dres_snapshot()
        return {
            "collection": {"name": coll.name, "n_keyframes": coll.n, "n_videos": coll.n_videos,
                           "dim": coll.dim},
            "encoder": str(getattr(encoder, "name", type(encoder).__name__)),
            "dres": {
                "connected": connected,
                "user": getattr(client, "username", None) if connected else None,
                "evaluation_id": evaluation_id if connected else None,
            },
            "llm": {
                "gemini": bool(settings.gemini_key),
                "openai": bool(settings.openai_key) and OpenAIClient.sdk_available(),
            },
        }

    # ------------------------------------------------------------------------------ search
    @app.post("/api/search")
    def api_search(body: SearchBody) -> JSONResponse:
        started = time.perf_counter()
        scenes = [s.strip() for s in body.scenes]
        if not 1 <= len(scenes) <= MAX_SCENES or not all(scenes):
            raise _fail(400, f"scenes must be a list of 1 to {MAX_SCENES} non-empty strings")
        if len(body.window) != 2:
            raise _fail(400, "window must be [min_s, max_s]")
        window = (float(body.window[0]), float(body.window[1]))
        if not (math.isfinite(window[0]) and math.isfinite(window[1]) and 0 <= window[0] < window[1]):
            raise _fail(400, "window must satisfy 0 <= min_s < max_s")
        top_k = _check_top_k(body.top_k)
        mode = body.mode or ("frames" if len(scenes) == 1 else "chains")
        mask = None
        if body.filter_videos is not None:
            if not body.filter_videos:
                raise _fail(400, "filter_videos must not be empty")
            mask = np.zeros(coll.n, dtype=bool)
            for vid in body.filter_videos:
                try:
                    mask[coll.rows_of(vid)] = True
                except KeyError:
                    raise _fail(404, f"unknown video {vid!r} in filter_videos") from None

        timings: dict[str, float] = {"encode_ms": 0.0, "scan_ms": 0.0, "cached_scenes": 0}
        sims = [state.scene_sims(s, timings) for s in scenes]
        ranking_started = time.perf_counter()
        query_id = secrets.token_hex(6)
        state.queries.put(query_id, QueryEntry("text", scenes, sims, window))
        ranked = sims if mask is None else [
            np.where(mask, s, -np.inf).astype(np.float32, copy=False) for s in sims
        ]
        # frames/videos with several scenes: a keyframe counts with the scene it matches best
        combined = ranked[0] if len(ranked) == 1 else np.max(np.stack(ranked), axis=0)

        payload: dict[str, Any]
        if mode == "frames":
            items = [_with_thumb(coll, h) for h in search.rank_frames(combined, coll, top_k)]
            payload = {"frames": items}
        elif mode == "videos":
            hits = search.rank_frames(combined, coll, max(top_k, VIDEO_MODE_POOL))
            items = search.group_by_video(hits, max_videos=min(MAX_VIDEOS, top_k),
                                          frames_per_video=FRAMES_PER_VIDEO)
            for group in items:
                for hit in group["hits"]:
                    _with_thumb(coll, hit)
            payload = {"videos": items}
        else:
            items = search.temporal_search(coll, ranked, window_s=window, top_k=top_k)
            for chain in items:
                for frame in chain["frames"]:
                    _with_thumb(coll, frame)
            payload = {"chains": items}
        rank_ms = _took_ms(ranking_started)
        took = _took_ms(started)
        try:
            state.log.results(body.model_dump(exclude_none=True), items)
        except Exception as exc:  # noqa: BLE001 - logging must not fail a search
            _LOG.warning("result log failed: %s", exc)
        timing_payload = {
            "encode_ms": int(round(timings["encode_ms"])),
            "scan_ms": int(round(timings["scan_ms"])),
            "rank_ms": rank_ms,
            "cached_scenes": int(timings["cached_scenes"]),
        }
        return JSONResponse({"query_id": query_id, "took_ms": took, "mode": mode,
                             "timings": timing_payload, **payload})

    @app.post("/api/decompose")
    def api_decompose(body: DecomposeBody) -> dict:
        return dict(decompose.decompose(body.text, llm=state.llm, cache_dir=cache("decompose")))

    @app.post("/api/similar")
    def api_similar(body: SimilarBody) -> JSONResponse:
        started = time.perf_counter()
        _require_video(coll, body.video_id)
        top_k = _check_top_k(body.top_k)
        row = coll.row(body.video_id, body.frame_idx)
        if row is None:
            raise _fail(404, f"frame {body.frame_idx} of {body.video_id} is not a keyframe")
        sims = coll.similarities(coll.embedding(row))
        sims[~coll.valid] = -np.inf
        ranking = sims.copy()
        ranking[row] = -np.inf  # as search.similar_frames: never return the query keyframe
        sims.setflags(write=False)
        hits = [_with_thumb(coll, h) for h in search.rank_frames(ranking, coll, top_k)]
        query_id = secrets.token_hex(6)
        label = f"similar to {body.video_id} frame {body.frame_idx}"
        state.queries.put(query_id, QueryEntry("similar", [label], [sims], (0.0, 30.0)))
        took = _took_ms(started)
        try:
            state.log.results({"video_id": body.video_id, "frame_idx": body.frame_idx}, hits)
        except Exception as exc:  # noqa: BLE001
            _LOG.warning("result log failed: %s", exc)
        return JSONResponse({"query_id": query_id, "took_ms": took, "frames": hits})

    @app.get("/api/video/{video_id}")
    def api_video(video_id: str, query_id: Optional[str] = None) -> JSONResponse:
        entry: Optional[QueryEntry] = state.queries.get(query_id)
        try:
            timeline = search.video_timeline(coll, video_id, entry.sims if entry else None)
        except KeyError:
            raise _fail(404, f"unknown video {video_id!r}") from None
        for keyframe in timeline["keyframes"]:
            _with_thumb(coll, keyframe)
        video = timeline["video"]
        if video.get("file") and not re.match(r"^https?://", str(video["file"]), re.IGNORECASE):
            video["file"] = f"/media/{video_id}" if _local_video_path(coll, video) else None
        # false: the query id is unknown or was evicted, so the UI can renew it instead of losing curves
        timeline["query_known"] = (entry is not None) if query_id else None
        return JSONResponse(timeline)

    @app.post("/api/vqa")
    def api_vqa(body: VqaBody) -> dict:
        if not body.question.strip():
            raise _fail(400, "the question is empty")
        if not 1 <= len(body.frames) <= MAX_VQA_FRAMES:
            raise _fail(400, f"between 1 and {MAX_VQA_FRAMES} frames are required")
        paths = []
        for ref in body.frames:
            _require_video(coll, ref.video_id)
            row = coll.row(ref.video_id, ref.frame_idx)
            if row is None:
                raise _fail(404, f"frame {ref.frame_idx} of {ref.video_id} is not a keyframe")
            paths.append(str(coll.frame_path(row)))
        try:
            return dict(vqa.answer(body.question, paths, providers=state.vqa_providers))
        except vqa.VQAUnavailable as exc:
            raise _fail(503, str(exc)) from None
        except FileNotFoundError as exc:
            raise _fail(404, str(exc)) from None
        except ValueError as exc:
            raise _fail(400, str(exc)) from None

    # -------------------------------------------------------------------------------- DRES
    @app.post("/api/dres/login")
    def api_dres_login(body: LoginBody) -> dict:
        url = (body.url or "").strip() or None
        with state.dres_lock:
            current = state.dres_client
        try:
            configured = normalize_base_url(settings.dres_url) if settings.dres_url else None
        except ValueError:
            configured = None
        try:
            if url:
                target = normalize_base_url(url)
            elif current is not None:
                target = current.base_url
            elif configured:
                target = configured
            else:
                raise _fail(400, "no DRES URL: pass url or set DRES_URL")
        except ValueError as exc:
            raise _fail(400, str(exc)) from None
        # The stored DRES_USER / DRES_PASSWORD only ever go to DRES_URL, never to a host a caller names.
        stored = configured is not None and target == configured
        username = (body.username or "").strip() or (settings.dres_user if stored else None)
        password = body.password or (settings.dres_password if stored else None)
        if not username or not password:
            raise _fail(400, "username and password are required" if stored or configured is None
                        else "username and password are required (the stored ones are only sent to DRES_URL)")
        # Reusing the connected client keeps its session if this login fails (DresClient.login
        # replaces the session only on success); a new server gets its own client, swapped in below.
        client = current if current is not None and current.base_url == target else DresClient(target)
        try:
            client.login(username, password)
        except DresError as exc:
            status = 401 if exc.status == 401 else 502
            raise _fail(status, f"DRES login failed: {exc.description}") from None
        except ValueError as exc:
            raise _fail(400, str(exc)) from None
        with state.dres_lock:
            state.dres_client = client
            state.evaluation_id = None
            state.log.dres_client = client
            state.log.evaluation_id = None
        return {"connected": True, "user": getattr(client, "username", None) or username}

    @app.get("/api/dres/evaluations")
    def api_dres_evaluations() -> dict:
        client = state.require_dres()
        try:
            evaluations = client.evaluations()
        except DresError as exc:
            raise _fail(502, f"DRES: {exc.description}") from None
        out = []
        for ev in evaluations:
            status = str(ev.get("status") or "")
            task = state.task_summary(client, str(ev["id"])) if status.upper() == "ACTIVE" and ev.get("id") else None
            out.append({"id": ev.get("id"), "name": ev.get("name"), "status": status,
                        "type": ev.get("type"), "task": task})
        return {"evaluations": out}

    @app.post("/api/dres/select")
    def api_dres_select(body: SelectBody) -> dict:
        client = state.require_dres()
        evaluation_id = body.evaluation_id.strip()
        if not evaluation_id:
            raise _fail(400, "evaluation_id is empty")
        try:
            known = {str(ev.get("id")) for ev in client.evaluations()}
        except DresError:
            known = None  # DRES unreachable right now: accept the id, submissions will tell
        if known is not None and evaluation_id not in known:
            raise _fail(404, f"unknown evaluation {evaluation_id!r}")
        with state.dres_lock:
            state.evaluation_id = evaluation_id
            state.log.evaluation_id = evaluation_id
        state.fetch_task_id(client, evaluation_id)
        return {"evaluation_id": evaluation_id}

    @app.post("/api/submit")
    def api_submit(body: SubmitBody) -> dict:
        timestamp_ms = int(time.time() * 1000)
        sent: dict[str, Any] = {"kind": body.kind}
        if body.kind == "kis":
            if not body.video_id:
                raise _fail(400, "video_id is required")
            start_ms, end_ms = _point(coll, body.video_id, body.time, body.start_ms, body.end_ms)
            sent["answers"] = [[body.video_id, start_ms, end_ms]]
        elif body.kind == "avs":
            if not body.items:
                raise _fail(400, "items must be a non-empty list")
            if len(body.items) > MAX_AVS_ITEMS:
                raise _fail(400, f"at most {MAX_AVS_ITEMS} items per submission")
            sent["answers"] = [
                [item.video_id, *_point(coll, item.video_id, item.time, item.start_ms, item.end_ms)]
                for item in body.items
            ]
        else:
            text = (body.text or "").strip()
            if not text:
                raise _fail(400, "text must be a non-empty string")
            sent["text"] = text

        client, evaluation_id, connected = state.dres_snapshot()
        if not connected or client is None:
            result: dict[str, Any] = {"ok": False, "status": "not connected", "raw": None}
        elif not evaluation_id:
            result = {"ok": False, "status": "no DRES evaluation selected", "raw": None}
        else:
            collection_name = settings.dres_collection
            try:
                if body.kind == "kis":
                    vid, start_ms, end_ms = sent["answers"][0]
                    raw = client.submit_kis(evaluation_id, vid, start_ms, end_ms, collection=collection_name)
                elif body.kind == "avs":
                    raw = client.submit_avs(evaluation_id, [tuple(a) for a in sent["answers"]],
                                            collection=collection_name)
                else:
                    raw = client.submit_text(evaluation_id, sent["text"])
                raw = raw if isinstance(raw, dict) else {"response": raw}
                verdict = str(raw.get("submission") or "").upper()
                result = {
                    "ok": bool(raw.get("status", True)),
                    "status": str(raw.get("description") or verdict or "submitted"),
                    "raw": raw,
                }
                if body.kind == "kis" and verdict == "WRONG":
                    vid, start_ms, end_ms = sent["answers"][0]
                    state.record_wrong(client, evaluation_id, vid, (start_ms + end_ms) / 2000.0)
            except DresError as exc:
                result = {"ok": False, "status": exc.description,
                          "raw": {"http_status": exc.status, "description": exc.description}}
            except ValueError as exc:
                result = {"ok": False, "status": str(exc), "raw": None}
        state.write_submission({
            "timestamp_ms": timestamp_ms,
            "evaluation_id": evaluation_id,
            "request": body.model_dump(exclude_none=True),
            "sent": sent,
            "response": result,
        })
        return result

    @app.post("/api/log")
    def api_log(body: LogBody) -> dict:
        accepted = 0
        for event in body.events[:MAX_LOG_EVENTS]:
            if not isinstance(event, dict):
                continue
            kind = str(event.get("type") or "").strip()
            if not kind:
                continue
            category = event.get("category")
            timestamp = event.get("timestamp_ms")
            if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)) \
                    or not math.isfinite(timestamp) or timestamp < 0:
                timestamp = None
            try:
                state.log.event(
                    kind,
                    event.get("value", ""),
                    category=category.upper() if isinstance(category, str) and category else None,
                    timestamp_ms=int(timestamp) if timestamp is not None else None,
                )
            except (ValueError, TypeError):
                continue
            accepted += 1
        return {"accepted": accepted}

    # ------------------------------------------------------------------ guard and KIS-C chat
    @app.post("/api/verify")
    def api_verify(body: VerifyBody) -> dict:
        _require_video(coll, body.video_id)
        if not math.isfinite(body.time):
            raise _fail(400, "time must be a finite number of seconds")
        _, evaluation_id, _ = state.dres_snapshot()
        return dict(verify.verify_moment(
            coll, body.scenes, body.video_id, body.time, state.rejected(evaluation_id),
            providers=state.verify_providers, cache_dir=cache("verify"), budget_s=verify.BUDGET_S,
            votes=state.verify_votes,
        ))

    @app.post("/api/chat")
    def api_chat(body: ChatBody) -> dict:
        entry = state.queries.get(body.query_id)
        result = dict(conversation.update_from_dialogue(
            body.initial, [t.model_dump() for t in body.turns], body.scenes, state.top_videos(entry),
            llm=state.llm, cache_dir=cache("conversation"),
        ))
        result["query_known"] = (entry is not None) if body.query_id else None
        return result

    @app.api_route("/api/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"])
    def api_unknown(path: str, request: Request) -> None:
        target = f"/api/{path}"
        if any(isinstance(r, APIRoute) and r.path == target for r in app.routes):
            raise _fail(405, f"{request.method} is not allowed on {target}")
        raise _fail(404, f"no API endpoint {target}")

    # --------------------------------------------------------------------- files and the UI
    @app.get("/frames/{video_id}/{filename}")
    def frame_file(video_id: str, filename: str) -> FileResponse:
        for segment in (video_id, filename):
            if not _SAFE_SEGMENT.fullmatch(segment) or ".." in segment:
                raise _fail(400, "invalid frame path")
        if Path(filename).suffix.lower() not in IMAGE_EXTENSIONS:
            raise _fail(400, "only keyframe images are served")
        try:
            coll.video_position(video_id)
        except KeyError:
            raise _fail(404, f"unknown video {video_id!r}") from None
        base = coll.frames_dir.resolve()
        path = (base / video_id / filename).resolve()
        if not path.is_relative_to(base) or not path.is_file():
            raise _fail(404, "keyframe not found")
        return FileResponse(path, headers={"Cache-Control": "public, max-age=86400"})

    @app.get("/media/{video_id}")
    def media_file(video_id: str) -> FileResponse:
        video = _require_video(coll, video_id)
        path = _local_video_path(coll, video)
        if path is None:
            raise _fail(404, f"no local video file for {video_id!r}")
        return FileResponse(path)  # Starlette answers Range requests, so the player can seek

    if web_dir is not None and Path(web_dir).is_dir():
        app.mount("/", StaticFiles(directory=str(web_dir), html=True), name="web")
    return app


# ----------------------------------------------------------------------------------------- CLI
def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="SceneChain interactive search server")
    parser.add_argument("--collection", help="collection directory (default: VBS_COLLECTION or collections/default)")
    parser.add_argument("--host", help="bind address (default: VBS_HOST or 127.0.0.1)")
    parser.add_argument("--port", type=int, help="port (default: VBS_PORT or 8787)")
    parser.add_argument("--device", choices=("cpu", "cuda"), help="text encoder device (default: CUDA if available)")
    parser.add_argument("--log-dir", help="interaction log directory (default: VBS_LOG_DIR or logs)")
    parser.add_argument("--verify-votes", type=int, default=verify.VOTES,
                        help=f"model calls per pre-submission check, majority decides "
                             f"(default {verify.VOTES}; 1 = a single call)")
    args = parser.parse_args(argv)
    if args.verify_votes < 1:
        parser.error("--verify-votes must be at least 1")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("urllib3").setLevel(logging.WARNING)  # DEBUG would print DRES session tokens
    settings = load_settings(collection_dir=args.collection, host=args.host, port=args.port,
                             log_dir=args.log_dir)
    started = time.perf_counter()
    collection = Collection.load(settings.collection_dir)
    _LOG.info("collection %s: %d keyframes, %d videos (%.2f s)", collection.name, collection.n,
              collection.n_videos, time.perf_counter() - started)
    from .encoder import SiglipTextEncoder

    started = time.perf_counter()
    encoder = SiglipTextEncoder(device=args.device)
    _LOG.info("encoder %s loaded (%.1f s)", encoder.name, time.perf_counter() - started)
    app = create_app(settings, collection=collection, encoder=encoder, verify_votes=args.verify_votes)

    import uvicorn

    uvicorn.run(app, host=settings.host, port=settings.port, log_level="info")


if __name__ == "__main__":
    main()
