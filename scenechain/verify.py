"""Pre-submission check: does the moment about to be submitted contradict the description?

Why: DRES ``KisTaskScorer`` subtracts points for every wrong submission before the first correct
one, and at VBS 2026 KIS-T saw 264 wrong against 94 correct submissions. A two-second look by a
vision-language model at the frames around the moment, plus the team's own verdict history, catches
some of those before they cost points.

How:

* The keyframe nearest to ``time_s`` and up to two keyframes on each side (valid rows only, time
  order) are sent to the model together with the ordered scene descriptions.
* The model is asked to list *concrete, visible contradictions* between the description and the
  frames, not whether the moment "is correct": a yes/no question invites false agreement, while a
  request for specific mismatches (wrong colour, wrong count, different place) gives the operator
  something to check. Details that are simply not visible in these frames are not contradictions,
  because the other scenes may happen a few seconds away.
* The verdict history blocks a *moment*, not a video. DRES judges a KIS answer by whether it lies
  inside one contiguous target segment, and in KIS the right video at the wrong moment is a common
  mistake, so a WRONG verdict says nothing about the rest of that video. ``"fail"`` (with
  ``rejected: true``) only when the new moment lies within :data:`WRONG_RADIUS_S` of a moment DRES
  already judged WRONG for the current task; that answer skips the model and is instant. A moment
  farther away in the same video is checked as usual and the result carries a non-blocking note
  (``notes``) naming the earlier wrong time. AVS answers are judged per shot (``MAP_TO_SEGMENT``) and
  by a human, so the client never learns their verdicts; the server records KIS verdicts only, and
  because the rule is per moment, other shots of a video are never blocked either way.
* ``"warn"`` when the model lists contradictions or no model could be asked, ``"ok"`` otherwise.
  Without any scene text (a KIS-V task, or an AVS basket item) only the verdict history is checked.
* Verdicts flip between identical calls even at temperature 0 (``docs/BENCHMARK.md`` note 5), so
  the model is asked :data:`VOTES` times in parallel and the majority decides: ``"warn"`` when a
  majority of the calls list contradictions, ``"ok"`` when a majority list none. The check answers as
  soon as the majority is known, so it waits for the second-fastest of three calls, not the slowest.
  ``votes=1`` is the single call of earlier versions.

Answers are cached on disk under ``cache/verify/`` keyed by the collection, the scenes, the
keyframes sent (row, image file with its size and modification time, time) and the number of votes,
so pressing ``S`` twice on the same moment asks the model once and a rebuilt or different collection
never reuses an answer about other frames. Only a majority decided by model answers is cached, never
one caused by failed calls. The whole check has one deadline (:data:`BUDGET_S`): the models are asked
in worker threads, and when the deadline passes the check answers ``"warn"`` at once while the workers
finish in the background (and still cache the majority). :func:`verify_moment` never raises.
"""

from __future__ import annotations

import hashlib
import io
import json
import threading
import time
from collections.abc import Collection as AbstractCollection
from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, Optional, TypedDict, Union

import numpy as np

from .collection import Collection
from .config import VBS_ROOT, load_settings
from .llm import GEMINI_MODELS, GeminiClient, LLMClient, LLMError, extract_json_object, optional_openai

CACHE_DIR = VBS_ROOT / "cache" / "verify"
#: bump when the prompt or the cache key changes so old cached answers are ignored
#: (2: the key includes the collection and the frame files; 3: majority votes, the key includes them)
PROMPT_VERSION = 3
#: keyframes on each side of the nearest one
NEIGHBOURS = 2
#: long side of the images sent; 768 px keeps each image to one Gemini image tile
MAX_SIDE = 768
JPEG_QUALITY = 85
#: a guard in front of a key press must answer quickly; a slow model counts as unavailable.
#: Per-request timeout of the guard's Gemini client. The Gemini API refuses shorter deadlines
#: (HTTP 400 "Minimum allowed deadline is 10s", checked live on 17 Sep), so BUDGET_S bounds the wait.
TIMEOUT_S = 10.0
#: Gemini models the guard tries (the first ones of GEMINI_MODELS); each failure can cost TIMEOUT_S
MAX_MODELS = 2
#: deadline of the whole check (image loading, every model and provider). Measured live on 17 Sep:
#: median 2.1 s, p90 4.4 s, 7.3 s for the first call of a process
BUDGET_S = 8.0
MAX_REASONS = 5
MAX_REASON_CHARS = 200
#: independent model calls per check; the majority decides (1: a single call)
VOTES = 3
#: a moment this close (seconds, either side) to a moment DRES judged WRONG counts as that moment.
#: On AIC 2026 the median keyframe gap is 2.2 s and the five keyframes the model sees span a median
#: 9.3 s, so within +-5 s the model would mostly be shown the frames DRES has already rejected. A KIS
#: answer is judged by containment in one target segment, so such a moment can only be right when a
#: segment boundary lies between the two, which the operator can still claim with a second ``S``.
#: It equals the web UI's "same moment" tolerance (``GUARD_SAME_MOMENT_S``).
WRONG_RADIUS_S = 5.0

PROMPT = """You check a candidate answer in a live known-item video search competition before it is submitted.
A wrong submission costs points, so report real problems, but do not invent any.

The target video moment is described as these scenes, in the order in which they happen:
{scenes}

The {n} attached images are consecutive keyframes of ONE candidate video around the candidate moment, in time order:
{frames}

List every CONCRETE contradiction you can SEE between the description and these frames, for example: a colour, a number of people or objects, clothing, a place or setting, on-screen text, or an action that clearly differs from the description.

Rules:
- The frames cover only a few seconds, so they may show just one of the scenes. A scene or a detail that is simply not visible here is NOT a contradiction.
- Only list what the images clearly show to be different. Do not list uncertainties.
- Write each contradiction as one short English sentence naming what the description says and what the frame shows.
- If nothing contradicts the description, return an empty list.

Return only JSON of the form {{"contradictions": ["...", "..."]}}."""

Verdict = Literal["ok", "warn", "fail"]


class VerifyResult(TypedDict):
    verdict: Verdict
    reasons: list[str]
    model: Optional[str]
    took_ms: int
    rejected: bool
    notes: list[str]


#: videos DRES judged WRONG: ids mapped to the submitted times in seconds, or a set of ids when the
#: times are unknown (then nothing can be blocked, the result only carries a note)
Rejected = Union[AbstractCollection[str], Mapping[str, Sequence[float]]]


def gemini_client(api_key: str, client: Any = None) -> GeminiClient:
    """The guard's Gemini client: the first :data:`MAX_MODELS` models, :data:`TIMEOUT_S` per request."""
    return GeminiClient(api_key, models=GEMINI_MODELS[:MAX_MODELS], client=client, timeout_s=TIMEOUT_S,
                        max_output_tokens=512)


@lru_cache(maxsize=1)
def default_providers() -> tuple[LLMClient, ...]:
    """Gemini (short timeout, two models) then OpenAI when configured; built once per process."""
    settings = load_settings()
    providers: list[LLMClient] = []
    if settings.gemini_key:
        providers.append(gemini_client(settings.gemini_key))
    openai_client = optional_openai(settings.openai_key)
    if openai_client is not None:
        providers.append(openai_client)
    return tuple(providers)


def moment_rows(collection: Collection, video_id: str, time_s: float, neighbours: int = NEIGHBOURS) -> list[int]:
    """Rows of the valid keyframe nearest to ``time_s`` and ``neighbours`` valid ones on each side.

    Raises ``KeyError`` for an unknown video. Falls back to all keyframes when none is valid.
    """
    rows = collection.rows_of(video_id)
    usable = rows[collection.valid[rows]]
    if usable.size == 0:
        usable = rows
    if usable.size == 0:
        return []
    times = collection.times[usable]
    centre = int(np.argmin(np.abs(times - float(time_s))))
    lo, hi = max(0, centre - neighbours), min(usable.size, centre + neighbours + 1)
    return [int(r) for r in usable[lo:hi]]


def load_jpeg(path: Path) -> bytes:
    """JPEG bytes of the image at ``path`` with the long side capped at :data:`MAX_SIDE`."""
    from PIL import Image

    with Image.open(path) as im:
        rgb = im.convert("RGB")
    rgb.thumbnail((MAX_SIDE, MAX_SIDE))
    buf = io.BytesIO()
    rgb.save(buf, "JPEG", quality=JPEG_QUALITY)
    return buf.getvalue()


def _fmt_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    return f"{int(seconds // 60)}:{seconds % 60:04.1f}"


def _parse_contradictions(reply: str) -> list[str]:
    items = extract_json_object(reply).get("contradictions")
    if not isinstance(items, list):
        raise ValueError("'contradictions' is not a list")
    out = []
    for item in items:
        text = str(item).strip() if isinstance(item, (str, int, float)) else ""
        if text:
            out.append(text[:MAX_REASON_CHARS])
    return out[:MAX_REASONS]


def wrong_history(
    rejected: Rejected, video_id: str, time_s: float, radius_s: float = WRONG_RADIUS_S
) -> tuple[Optional[str], Optional[str]]:
    """``(blocking reason, note)`` from the WRONG verdicts of the current task.

    The reason is set when ``time_s`` lies within ``radius_s`` of a moment of ``video_id`` that DRES
    judged WRONG; the note when the video has a WRONG verdict only elsewhere (or at unknown times).
    Both are ``None`` for a video without WRONG verdicts.
    """
    if video_id not in rejected:
        return None, None
    raw = rejected[video_id] if isinstance(rejected, Mapping) else ()
    times = [float(t) for t in (raw or ()) if np.isfinite(float(t))]
    t = float(time_s)
    if times and np.isfinite(t):
        near = [w for w in times if abs(w - t) <= radius_s]
        if near:
            at = ", ".join(_fmt_time(w) for w in near[:4])
            return (f"{video_id} was already judged wrong for this task at {at}; "
                    f"{_fmt_time(t)} is within {radius_s:g} s of it."), None
    if not times:
        return None, (f"{video_id} already had a wrong submission in this task (time unknown); "
                      "this moment was checked as usual.")
    at = ", ".join(_fmt_time(w) for w in times[:4])
    return None, (f"{video_id} already had a wrong submission in this task at {at}; "
                  "this moment is farther away and was checked as usual.")


def _cache_path(
    cache_dir: Path, collection: Collection, scenes: Sequence[str], video_id: str, rows: Sequence[int],
    votes: int = 1,
) -> Path:
    """Cache file of one check. The collection, the scenes, every keyframe sent (row, image file with
    size and modification time, time) and the number of votes are part of the key, so a rebuilt
    collection, or another collection with the same video ids and rows, never gets an answer about
    different frames, and a single call never stands in for a majority."""
    frames = []
    for row in rows:
        path = collection.frame_path(row)
        try:
            stat = path.stat()
            size, mtime = stat.st_size, stat.st_mtime_ns
        except OSError:
            size = mtime = None
        frames.append([int(row), str(path.resolve()), size, mtime, round(float(collection.times[row]), 3)])
    key = json.dumps(
        [PROMPT_VERSION, str(Path(collection.root).resolve()), collection.name, list(scenes), video_id, frames,
         int(votes)],
        ensure_ascii=False,
    )
    return cache_dir / f"{hashlib.sha1(key.encode('utf-8')).hexdigest()}.json"


def _read_cache(path: Optional[Path]) -> Optional[tuple[list[str], Optional[str]]]:
    if path is None:
        return None
    try:
        entry = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(entry, dict) or entry.get("prompt_version") != PROMPT_VERSION:
        return None
    reasons = entry.get("contradictions")
    if not isinstance(reasons, list):
        return None
    return [str(r) for r in reasons], entry.get("model")


def _write_cache(
    path: Optional[Path], reasons: list[str], model: str, video_id: str, ballots: Sequence[list[str]] = ()
) -> None:
    if path is None:
        return
    entry = {"prompt_version": PROMPT_VERSION, "video_id": video_id, "contradictions": reasons, "model": model,
             "ballots": [list(b) for b in ballots]}
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(entry, ensure_ascii=False, indent=1), encoding="utf-8")
    except OSError:
        pass


def _tally(ballots: Sequence[list[str]]) -> tuple[int, int]:
    """``(clean, flagged)``: answers without and with contradictions among ``ballots``."""
    clean = sum(1 for b in ballots if not b)
    return clean, len(ballots) - clean


def _flagged_reasons(ballots: Sequence[list[str]], votes: int) -> list[str]:
    """Reasons of a ``"warn"`` majority: the first flagged answer's, after a count when ``votes > 1``."""
    flagged = [b for b in ballots if b]
    if not flagged:
        return []
    if votes == 1:
        return list(flagged[0])
    head = f"{len(flagged)} of {votes} checks listed contradictions."
    return [head] + list(flagged[0])[:MAX_REASONS]


def verify_moment(
    collection: Collection,
    scenes: Sequence[str],
    video_id: str,
    time_s: float,
    rejected: Rejected = frozenset(),
    *,
    providers: Optional[Sequence[LLMClient]] = None,
    cache_dir: Optional[Path] = CACHE_DIR,
    budget_s: Optional[float] = BUDGET_S,
    votes: int = VOTES,
    radius_s: float = WRONG_RADIUS_S,
) -> VerifyResult:
    """``{"verdict": "ok"|"warn"|"fail", "reasons": [str], "model": str|None, "took_ms": int,
    "rejected": bool, "notes": [str]}``.

    ``rejected`` holds the videos DRES judged WRONG for the current task, mapped to the submitted
    times (a plain set of ids, without times, can only produce a note). ``"fail"`` when ``time_s`` lies
    within ``radius_s`` of such a time (see :func:`wrong_history`). ``providers`` defaults to
    :func:`default_providers`; each of the ``votes`` parallel calls walks the whole provider chain, and
    the majority decides. ``cache_dir=None`` disables the disk cache. ``budget_s`` is the deadline of
    the whole check (``None``: none). Past it the answer is ``"warn"``, while the calls keep running in
    background threads whose majority still goes to the cache. Never raises: any failure becomes
    ``"warn"`` with the reason.
    """
    started = time.perf_counter()

    rejected_moment = False
    notes: list[str] = []

    def done(verdict: Verdict, reasons: list[str], model: Optional[str] = None) -> VerifyResult:
        took = int(round((time.perf_counter() - started) * 1000))
        return {"verdict": verdict, "reasons": reasons, "model": model, "took_ms": took,
                "rejected": rejected_moment, "notes": list(notes)}

    try:
        votes = max(1, int(votes))
        reason, note = wrong_history(rejected, video_id, time_s, radius_s)
        if reason:
            rejected_moment = True
            return done("fail", [reason])
        if note:
            notes.append(note)
        clean = [s.strip() for s in scenes if isinstance(s, str) and s.strip()]
        if not clean:
            return done("ok", [])  # nothing to contradict; the verdict history was checked
        if not np.isfinite(float(time_s)):
            return done("warn", ["The moment has no valid time; it was not verified."])
        try:
            rows = moment_rows(collection, video_id, float(time_s))
        except KeyError:
            return done("warn", [f"Unknown video {video_id!r}; the moment was not verified."])
        rows = [r for r in rows if collection.frame_path(r).is_file()]
        if not rows:
            return done("warn", ["No keyframe images near this moment; it was not verified."])

        path = _cache_path(cache_dir, collection, clean, video_id, rows, votes) if cache_dir is not None else None
        cached = _read_cache(path)
        if cached is not None:
            reasons, model = cached
            return done("warn" if reasons else "ok", reasons, model)

        chain = list(providers) if providers is not None else list(default_providers())
        if not chain:
            return done("warn", ["No vision model configured (GEMINI_API_KEY / OPENAI_API_KEY); "
                                 "the moment was not verified."])
        images = [load_jpeg(collection.frame_path(r)) for r in rows]
        prompt = PROMPT.format(
            scenes="\n".join(f"{i + 1}. {s}" for i, s in enumerate(clean)),
            n=len(images),
            frames="\n".join(
                f"image {i + 1}: {_fmt_time(collection.times[r])}" for i, r in enumerate(rows)
            ),
        )
        need = votes // 2 + 1
        lock = threading.Lock()
        settled = threading.Event()
        ballots: list[list[str]] = []
        models: list[str] = []
        failures: list[str] = []
        errors: list[Exception] = []
        outcome: dict[str, Any] = {}
        state = {"finished": 0}

        def settle() -> None:
            """Under ``lock``: decide as soon as a majority is known, or once every call has ended."""
            if "verdict" in outcome:
                return
            n_clean, n_flagged = _tally(ballots)
            if n_clean >= need or n_flagged >= need:
                reasons = [] if n_clean >= need else _flagged_reasons(ballots, votes)
                model = "+".join(dict.fromkeys(models))
                outcome.update(verdict="warn" if reasons else "ok", reasons=reasons, model=model)
                _write_cache(path, reasons, model, video_id, ballots)
            elif state["finished"] == votes:
                if n_flagged:  # a tie, or too few answers for a majority: show what was found
                    reasons = _flagged_reasons(ballots, votes)
                    reasons.insert(1, "The checks are split; no majority." if len(ballots) == votes
                                   else f"Only {len(ballots)} of {votes} checks answered; no majority.")
                    outcome.update(verdict="warn", reasons=reasons, model="+".join(dict.fromkeys(models)))
                elif ballots:
                    outcome.update(verdict="warn", reasons=[
                        f"Only {len(ballots)} of {votes} checks answered, all without contradictions; "
                        f"no majority ({failures[0] if failures else 'no reply'})."
                    ], model="+".join(dict.fromkeys(models)))
                elif errors:
                    exc = errors[0]
                    outcome.update(verdict="warn", model=None, reasons=[
                        f"The check failed ({type(exc).__name__}: {str(exc)[:120]})."])
                else:
                    detail = failures[0] if failures else "no reply"
                    outcome.update(verdict="warn", model=None, reasons=[
                        f"The vision model could not check this moment ({detail})."])
            else:
                return
            settled.set()

        def ask() -> None:
            mine: list[str] = []
            try:
                for provider in chain:
                    if mine and (settled.is_set() or (
                            budget_s is not None and time.perf_counter() - started >= budget_s)):
                        break  # the operator already has an answer; do not start another provider
                    try:
                        reasons, model = provider.generate(prompt, _parse_contradictions, images)
                    except LLMError as exc:
                        mine.append(str(exc)[:160])
                        continue
                    with lock:
                        ballots.append(reasons)
                        models.append(model)
                    return
                with lock:
                    failures.extend(mine)
            except Exception as exc:  # noqa: BLE001 - reported through settle()
                with lock:
                    errors.append(exc)
            finally:
                with lock:
                    state["finished"] += 1
                    settle()

        # Worker threads, so that the deadline holds whatever a provider does with its own timeouts.
        for i in range(votes):
            threading.Thread(target=ask, name=f"vbs-verify-{i}", daemon=True).start()
        remaining = None if budget_s is None else max(0.0, budget_s - (time.perf_counter() - started))
        if not settled.wait(remaining):
            return done("warn", [f"The vision model did not answer within {budget_s:g} s; "
                                 "the moment was not verified."])
        return done(outcome["verdict"], list(outcome["reasons"]), outcome.get("model"))
    except Exception as exc:  # noqa: BLE001 - the guard must never block a submission by crashing
        return done("warn", [f"The check failed ({type(exc).__name__}: {str(exc)[:120]})."])
