"""DRES v2 REST client for the VBS 2027 interactive search tool.

Specification
-------------
Paths, parameters and payload schemas follow the DRES *client* OpenAPI specification, the file
the DRES team publishes for generating participant clients:

* ``doc/oas-client.json`` in https://github.com/dres-dev/DRES, read at tag ``v2.0.5``
  (commit ``bd600c8d3e586e92356e9362d68634eb365e8a85``, 28 May 2026; ``info.version``
  "2.0.5-SNAPSHOT"). The file last changed in commit ``9059aeb2d6800d92713ac1e1decfbe079780c211``
  (16 Jul 2024) and is byte-identical on ``master`` and ``dev`` as of 17 Sep 2026.
* Every path and request schema used here is identical in release ``2.0.4`` (commit ``932525a0``,
  3 Jun 2024), the newest release with a published distribution. The only client-visible
  difference is that ``ApiClientTaskTemplateInfo.duration`` became optional in 2.0.5.

Behaviour the specification leaves open was checked in the backend sources at the same commit
(paths relative to ``backend/src/main/kotlin/dev/dres/``):

* **Session.** ``POST /api/v2/login`` returns an ``ApiUser`` whose ``sessionId`` is the token. The
  server reads the token from the ``SESSIONID`` cookie and then from the ``session`` query
  parameter, which takes precedence (``api/rest/RestApi.kt``). DRES re-issues the cookie with the
  ``Secure`` flag, so a cookie jar stops sending it over plain HTTP; this client therefore sends
  ``?session=<sessionId>`` on every authenticated call (the log endpoints declare the parameter as
  required anyway) and does not depend on cookies.
* **Answer sets.** Each answer set receives its own verdict. The judged (AVS) validator looks only
  at the first answer of a set (``run/validation/judged/BasicJudgementValidator.kt``) and the
  temporal validator requires *all* answers of a set to be correct
  (``run/validation/TemporalContainmentAnswerSetValidator.kt``), so an AVS submission sends one
  answer set per shot. In synchronous runs the server overwrites ``taskId`` with the running task
  (``run/InteractiveSynchronousRunManager.kt``), so the client never sends task identifiers.
* **Answer type** is inferred from the fields present (``ApiClientAnswer.kt``): ``mediaItemName``
  with ``start`` and ``end`` is temporal, ``mediaItemName`` alone is an item, ``text`` alone is text.
  Without ``mediaItemCollectionName`` the item name must be unique across all collections.
* **No legacy endpoint.** No 2.0.x release (2.0.0 to 2.0.5) registers the v1 GET submission handler,
  and neither ``oas-client.json`` nor ``oas.json`` lists it, so this client offers no fallback.

The client was also run end to end against a locally started DRES 2.0.4 release on 17 Sep 2026
(login, evaluation list, task info, KIS / AVS / text submissions with verdicts, both log endpoints,
logout, error cases); ``docs/DRES_NOTES.md`` records what was verified and how to repeat it.

Secrets: the password is only placed in the login request body and the session token only in the
query string. Neither is logged, and both are removed from every ``DresError`` message.
"""

from __future__ import annotations

import json
import logging
import operator
import re
import time
from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import quote, urlsplit

import requests

__all__ = [
    "DresClient",
    "DresError",
    "QUERY_EVENT_CATEGORIES",
    "SPEC_COMMIT",
    "SPEC_URL",
    "VERDICTS",
    "normalize_base_url",
    "query_event",
    "result_log",
    "temporal_answer",
    "text_answer",
]

SPEC_URL = (
    "https://raw.githubusercontent.com/dres-dev/DRES/"
    "bd600c8d3e586e92356e9362d68634eb365e8a85/doc/oas-client.json"
)
SPEC_COMMIT = "bd600c8d3e586e92356e9362d68634eb365e8a85"

#: ``QueryEventCategory`` enum of the client specification.
QUERY_EVENT_CATEGORIES = ("TEXT", "IMAGE", "SKETCH", "FILTER", "BROWSING", "COOPERATION", "OTHER")
#: ``ApiVerdictStatus`` enum of the client specification.
VERDICTS = ("CORRECT", "WRONG", "INDETERMINATE", "UNDECIDABLE")

_LOG = logging.getLogger(__name__)
_REDACTED = "<redacted>"
_ERROR_BODY_LIMIT = 300
#: 404 description DRES uses for an unknown evaluation id (``utilities/extensions/ContextExtensions.kt``).
_UNKNOWN_EVALUATION = re.compile(r"Evaluation .* not found\.")


class DresError(Exception):
    """A DRES call failed.

    Attributes:
        status: HTTP status code of the response, or ``None`` when the failure is not tied to one
            (connection failure, timeout, call before login, unexpected response shape).
        description: the server's ``ErrorStatus.description`` when it sent one, otherwise a short
            explanation. Never contains the password or the session token.
        method: HTTP method of the failed call.
        path: request path without the query string.
    """

    def __init__(self, status: int | None, description: str, method: str, path: str) -> None:
        self.status = status
        self.description = description
        self.method = method
        self.path = path
        where = f"HTTP {status}" if status is not None else "no HTTP response"
        super().__init__(f"DRES {method} {path} failed ({where}): {description}")


# --------------------------------------------------------------------------------------------------
# Payload builders (pure functions; shapes follow the client specification)
# --------------------------------------------------------------------------------------------------


def _non_negative_int(name: str, value: Any) -> int:
    """Return ``value`` as ``int``; accepts numpy integers, rejects bools, floats and negatives."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer, got bool")
    try:
        result = operator.index(value)
    except TypeError:
        raise ValueError(f"{name} must be an integer, got {type(value).__name__}") from None
    if result < 0:
        raise ValueError(f"{name} must be >= 0, got {result}")
    return result


def _non_empty_str(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _timestamp(timestamp_ms: int | None) -> int:
    """Validate a Unix timestamp in milliseconds, or return the current time when ``None``."""
    if timestamp_ms is None:
        return int(time.time() * 1000)
    return _non_negative_int("timestamp_ms", timestamp_ms)


def temporal_answer(
    media_item: str, start_ms: int, end_ms: int, collection: str | None = None
) -> dict[str, Any]:
    """Build an ``ApiClientAnswer`` for a segment of a media item (times in milliseconds)."""
    answer: dict[str, Any] = {
        "mediaItemName": _non_empty_str("media_item", media_item),
        "start": _non_negative_int("start_ms", start_ms),
        "end": _non_negative_int("end_ms", end_ms),
    }
    if answer["start"] > answer["end"]:
        raise ValueError(f"start_ms ({answer['start']}) must not exceed end_ms ({answer['end']})")
    if collection is not None:
        answer["mediaItemCollectionName"] = _non_empty_str("collection", collection)
    return answer


def text_answer(text: str) -> dict[str, Any]:
    """Build an ``ApiClientAnswer`` carrying free text (VQA / QA tasks); whitespace is trimmed."""
    return {"text": _non_empty_str("text", text).strip()}


def query_event(
    category: str, event_type: str, value: Any, timestamp_ms: int | None = None
) -> dict[str, Any]:
    """Build a ``QueryEvent``.

    Args:
        category: one of :data:`QUERY_EVENT_CATEGORIES` (case-insensitive).
        event_type: free-form type, e.g. ``"jointEmbedding"`` or ``"rankedList"``.
        value: the event value. The specification types it as a string; other values are encoded
            as compact JSON with sorted keys so that logs are deterministic.
        timestamp_ms: client time in Unix milliseconds; defaults to now.
    """
    if not isinstance(category, str) or category.upper() not in QUERY_EVENT_CATEGORIES:
        raise ValueError(f"category must be one of {QUERY_EVENT_CATEGORIES}, got {category!r}")
    if not isinstance(value, str):
        value = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return {
        "timestamp": _timestamp(timestamp_ms),
        "category": category.upper(),
        "type": _non_empty_str("event_type", event_type),
        "value": value,
    }


def result_log(
    answers: Sequence[Mapping[str, Any]],
    *,
    sort_type: str,
    result_set_availability: str,
    events: Sequence[Mapping[str, Any]] = (),
    timestamp_ms: int | None = None,
) -> dict[str, Any]:
    """Build a ``QueryResultLog`` from answers ordered best first.

    Args:
        answers: ``ApiClientAnswer`` dicts (see :func:`temporal_answer`), best first; ranks are
            assigned 1, 2, 3, ... in this order.
        sort_type: how the list is ordered, e.g. ``"score"`` (free string in the specification).
        result_set_availability: which part of the ranking the list represents, e.g. ``"top"``
            (free string in the specification).
        events: the ``QueryEvent``\\ s that produced this result list.
        timestamp_ms: client time in Unix milliseconds; defaults to now.
    """
    log = {
        "timestamp": _timestamp(timestamp_ms),
        "sortType": _non_empty_str("sort_type", sort_type),
        "resultSetAvailability": _non_empty_str("result_set_availability", result_set_availability),
        "results": [{"answer": dict(answer), "rank": rank} for rank, answer in enumerate(answers, start=1)],
        "events": [dict(event) for event in events],
    }
    _check_result_log(log)
    return log


def _check_answer(where: str, answer: Any) -> None:
    """Validate an ``ApiClientAnswer`` against the specification and DRES type inference."""
    allowed = {"text", "mediaItemName", "mediaItemCollectionName", "start", "end"}
    if not isinstance(answer, Mapping):
        raise ValueError(f"{where} must be an object")
    unknown = set(answer) - allowed
    if unknown:
        raise ValueError(f"{where} has unknown fields {sorted(unknown)}")
    for key in ("text", "mediaItemName", "mediaItemCollectionName"):
        if answer.get(key) is not None:
            _non_empty_str(f"{where}.{key}", answer[key])
    for key in ("start", "end"):
        if answer.get(key) is not None:
            _non_negative_int(f"{where}.{key}", answer[key])
    if answer.get("mediaItemName") is None and answer.get("text") is None:
        raise ValueError(f"{where} needs mediaItemName or text")
    if (answer.get("start") is None) != (answer.get("end") is None):
        raise ValueError(f"{where} needs both start and end or neither")


def _check_query_event(where: str, event: Any) -> None:
    """Validate a ``QueryEvent`` (all four fields are required by the specification)."""
    if not isinstance(event, Mapping):
        raise ValueError(f"{where} must be an object")
    if set(event) != {"timestamp", "category", "type", "value"}:
        raise ValueError(f"{where} must have exactly the fields timestamp, category, type, value")
    _non_negative_int(f"{where}.timestamp", event["timestamp"])
    if event["category"] not in QUERY_EVENT_CATEGORIES:
        raise ValueError(f"{where}.category must be one of {QUERY_EVENT_CATEGORIES}")
    if not isinstance(event["type"], str) or not isinstance(event["value"], str):
        raise ValueError(f"{where}.type and {where}.value must be strings")


def _check_result_log(log: Any) -> None:
    """Validate a ``QueryResultLog`` (all five fields are required by the specification)."""
    required = {"timestamp", "sortType", "resultSetAvailability", "results", "events"}
    if not isinstance(log, Mapping) or set(log) != required:
        raise ValueError(f"result_log must have exactly the fields {sorted(required)}")
    _non_negative_int("result_log.timestamp", log["timestamp"])
    _non_empty_str("result_log.sortType", log["sortType"])
    _non_empty_str("result_log.resultSetAvailability", log["resultSetAvailability"])
    if not isinstance(log["results"], (list, tuple)) or not isinstance(log["events"], (list, tuple)):
        raise ValueError("result_log.results and result_log.events must be lists")
    for i, ranked in enumerate(log["results"]):
        if not isinstance(ranked, Mapping) or "answer" not in ranked or set(ranked) - {"answer", "rank"}:
            raise ValueError(f"result_log.results[{i}] must be {{'answer': ..., 'rank': int}}")
        _check_answer(f"result_log.results[{i}].answer", ranked["answer"])
        if ranked.get("rank") is not None:
            _non_negative_int(f"result_log.results[{i}].rank", ranked["rank"])
    for i, event in enumerate(log["events"]):
        _check_query_event(f"result_log.events[{i}]", event)


def normalize_base_url(base_url: str) -> str:
    """The server root as :class:`DresClient` stores it (``scheme://host[:port][/path]``, no trailing
    slash); raises ``ValueError`` for anything but an http(s) URL without query string or fragment."""
    parts = urlsplit(str(base_url).strip())
    if parts.scheme not in ("http", "https") or not parts.netloc:
        raise ValueError(f"base_url must be an http(s) URL, got {base_url!r}")
    if parts.query or parts.fragment:
        raise ValueError("base_url must not contain a query string or fragment")
    return f"{parts.scheme}://{parts.netloc}{parts.path.rstrip('/')}"


# --------------------------------------------------------------------------------------------------
# Client
# --------------------------------------------------------------------------------------------------


class DresClient:
    """Participant client for one DRES server.

    Every public call performs exactly one HTTP request (no retries: a repeated submission could
    count as a second, possibly penalised, submission) and returns the parsed JSON response.
    Failures raise :class:`DresError`; invalid arguments raise ``ValueError`` before any request.
    Once logged in, calls may run concurrently from several threads; ``login`` and ``logout`` must
    not race with other calls.

    Args:
        base_url: server root, e.g. ``"https://vbs.example.org"`` (no ``/api`` suffix).
        timeout: seconds allowed for connecting and for reading each response.
        http: optional ``requests.Session`` (for proxies or custom CA bundles).
    """

    def __init__(self, base_url: str, timeout: float = 10.0, http: requests.Session | None = None) -> None:
        base = normalize_base_url(base_url)
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self.base_url = base
        self.timeout = float(timeout)
        self._http = http if http is not None else requests.Session()
        self._session_id: str | None = None
        self._user: dict[str, Any] | None = None
        self._password: str | None = None  # only set while a login request is in flight

    def __repr__(self) -> str:
        user = self._user.get("username") if self._user else None
        return f"DresClient(base_url={self.base_url!r}, user={user!r})"

    # -- session ------------------------------------------------------------------------------------

    @property
    def logged_in(self) -> bool:
        """Whether a session token from a successful login is held."""
        return self._session_id is not None

    @property
    def username(self) -> str | None:
        """User name of the logged-in account, if any."""
        return self._user.get("username") if self._user else None

    @property
    def role(self) -> str | None:
        """``ApiRole`` of the logged-in account (submitting requires ``PARTICIPANT`` or ``ADMIN``)."""
        return self._user.get("role") if self._user else None

    def login(self, user: str, password: str) -> dict[str, Any]:
        """``POST /api/v2/login``; stores the session token for all later calls.

        Returns the ``ApiUser`` (``id``, ``username``, ``role``, ``sessionId``). Cookies are cleared
        first, so the server issues a fresh token. A previous session is replaced only when the new
        login succeeds: a refused or failed login (typo, timeout) leaves a working session usable.
        """
        _non_empty_str("user", user)
        _non_empty_str("password", password)
        self._http.cookies.clear()
        self._password = password
        try:
            api_user = self._request(
                "POST", "/api/v2/login", body={"username": user, "password": password}, authenticated=False
            )
        finally:
            self._password = None
        session_id = api_user.get("sessionId") if isinstance(api_user, dict) else None
        if not isinstance(session_id, str) or not session_id:
            raise DresError(None, "login response carries no sessionId", "POST", "/api/v2/login")
        self._session_id = session_id
        self._user = api_user
        _LOG.info("DRES login as %r (role %s)", api_user.get("username"), api_user.get("role"))
        return api_user

    def logout(self) -> dict[str, Any]:
        """``GET /api/v2/logout``; the local session is dropped even if the call fails."""
        try:
            return self._request("GET", "/api/v2/logout")
        finally:
            self._forget_session()

    # -- evaluations and tasks ----------------------------------------------------------------------

    def evaluations(self) -> list[dict[str, Any]]:
        """``GET /api/v2/client/evaluation/list``: ``ApiClientEvaluationInfo`` of every evaluation
        whose template includes this user (``id``, ``name``, ``type``, ``status``, ``templateId``,
        ``templateDescription``, ``teams``, ``taskTemplates``)."""
        path = "/api/v2/client/evaluation/list"
        result = self._request("GET", path)
        if not isinstance(result, list):
            raise DresError(None, "expected a JSON array of evaluations", "GET", path)
        return result

    def task_info(self, evaluation_id: str) -> dict[str, Any] | None:
        """``GET /api/v2/client/evaluation/currentTask/{evaluationId}``.

        Returns the ``ApiClientTaskTemplateInfo`` (``name``, ``taskGroup``, ``taskType``,
        ``duration`` in seconds) of the active task, or ``None`` when the evaluation has no active
        task (DRES answers 404 then). An unknown evaluation id still raises ``DresError``.
        """
        path = f"/api/v2/client/evaluation/currentTask/{_path_segment(evaluation_id)}"
        try:
            return self._request("GET", path)
        except DresError as error:
            if error.status == 404 and not _UNKNOWN_EVALUATION.fullmatch(error.description):
                return None
            raise

    def evaluation_state(self, evaluation_id: str) -> dict[str, Any]:
        """``GET /api/v2/evaluation/{evaluationId}/state``: ``ApiEvaluationState`` with
        ``evaluationStatus``, ``taskStatus``, ``taskId``, ``timeLeft`` and ``timeElapsed``
        (both in seconds; ``timeLeft`` may be absent)."""
        return self._request("GET", f"/api/v2/evaluation/{_path_segment(evaluation_id)}/state")

    # -- submission ---------------------------------------------------------------------------------

    def submit(
        self, evaluation_id: str, answer_sets: Sequence[Sequence[Mapping[str, Any]]]
    ) -> dict[str, Any]:
        """``POST /api/v2/submit/{evaluationId}`` with one answer set per inner sequence.

        Returns ``SuccessfulSubmissionsStatus`` (``status``, ``submission`` verdict, ``description``).
        HTTP 200 carries the most frequent verdict over the answer sets; HTTP 202 means
        ``INDETERMINATE`` (awaiting a judge). Rejections (e.g. duplicates, HTTP 412) raise
        ``DresError``.
        """
        if not answer_sets:
            raise ValueError("at least one answer set is required")
        body_sets = []
        for i, answers in enumerate(answer_sets):
            if not answers:
                raise ValueError(f"answer set {i} is empty")
            for j, answer in enumerate(answers):
                _check_answer(f"answer_sets[{i}][{j}]", answer)
            body_sets.append({"answers": [dict(answer) for answer in answers]})
        path = f"/api/v2/submit/{_path_segment(evaluation_id)}"
        result = self._request("POST", path, body={"answerSets": body_sets})
        _LOG.info("DRES submission of %d answer set(s) -> %s", len(body_sets), _verdict_of(result))
        return result

    def submit_kis(
        self, evaluation_id: str, media_item: str, start_ms: int, end_ms: int, collection: str | None = None
    ) -> dict[str, Any]:
        """Submit one temporal segment (KIS-T, KIS-V): one answer set with one temporal answer.

        DRES judges the answer correct only if ``[start_ms, end_ms]`` lies entirely inside the target
        segment (``TemporalContainmentAnswerSetValidator``; confirmed on DRES 2.0.4), so a point
        (``start_ms == end_ms``) or a narrow range is safest near segment boundaries.
        """
        return self.submit(evaluation_id, [[temporal_answer(media_item, start_ms, end_ms, collection)]])

    def submit_text(self, evaluation_id: str, text: str) -> dict[str, Any]:
        """Submit a free-text answer (VQA / QA): one answer set with one text answer."""
        return self.submit(evaluation_id, [[text_answer(text)]])

    def submit_avs(
        self,
        evaluation_id: str,
        items: Sequence[tuple[str, int, int]],
        collection: str | None = None,
    ) -> dict[str, Any]:
        """Submit several shots for an AVS task in one request, one answer set per shot, so that each
        shot is judged on its own. ``items`` are ``(media_item, start_ms, end_ms)`` tuples.

        AVS task types with the ``MAP_TO_SEGMENT`` option (the DRES preset) widen each answer to the
        predefined shot that contains it; an answer whose start and end fall into different shots is
        refused with HTTP 500 (``run/transformer/MapToSegmentTransformer.kt``), so keep ranges inside
        one shot, e.g. ``start_ms == end_ms``.
        """
        if not items:
            raise ValueError("items must not be empty")
        answer_sets = []
        for i, item in enumerate(items):
            if len(item) != 3:
                raise ValueError(f"items[{i}] must be (media_item, start_ms, end_ms)")
            media_item, start_ms, end_ms = item
            answer_sets.append([temporal_answer(media_item, start_ms, end_ms, collection)])
        return self.submit(evaluation_id, answer_sets)

    # -- logging ------------------------------------------------------------------------------------

    def log_query(
        self, evaluation_id: str, events: Sequence[Mapping[str, Any]], timestamp_ms: int | None = None
    ) -> dict[str, Any]:
        """``POST /api/v2/log/query/{evaluationId}`` with a ``QueryEventLog``.

        ``events`` are ``QueryEvent`` dicts (build them with :func:`query_event`);
        ``timestamp_ms`` is the log's client time and defaults to now. Returns ``SuccessStatus``.
        """
        if not events:
            raise ValueError("events must not be empty")
        for i, event in enumerate(events):
            _check_query_event(f"events[{i}]", event)
        body = {"timestamp": _timestamp(timestamp_ms), "events": [dict(event) for event in events]}
        return self._request("POST", f"/api/v2/log/query/{_path_segment(evaluation_id)}", body=body)

    def log_results(self, evaluation_id: str, result_log: Mapping[str, Any]) -> dict[str, Any]:
        """``POST /api/v2/log/result/{evaluationId}`` with a ``QueryResultLog`` (build it with
        :func:`result_log`). Returns ``SuccessStatus``."""
        _check_result_log(result_log)
        path = f"/api/v2/log/result/{_path_segment(evaluation_id)}"
        return self._request("POST", path, body=dict(result_log))

    # -- transport ----------------------------------------------------------------------------------

    def _forget_session(self) -> None:
        self._session_id = None
        self._user = None
        self._http.cookies.clear()

    def _redact(self, text: str) -> str:
        for secret in (self._session_id, self._password):
            if secret:
                text = text.replace(secret, _REDACTED)
        return text

    def _request(
        self, method: str, path: str, *, body: Mapping[str, Any] | None = None, authenticated: bool = True
    ) -> Any:
        """Perform one call and return the parsed JSON body of a 2xx response."""
        params = None
        if authenticated:
            if self._session_id is None:
                raise DresError(None, "not logged in; call login() first", method, path)
            params = {"session": self._session_id}
        headers = {"Accept": "application/json"}
        data = None
        if body is not None:
            headers["Content-Type"] = "application/json"
            data = json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8")
        started = time.perf_counter()
        try:
            response = self._http.request(
                method, self.base_url + path, params=params, data=data, headers=headers, timeout=self.timeout
            )
        except requests.Timeout:
            raise DresError(None, f"timed out after {self.timeout:g} s", method, path) from None
        except requests.RequestException as exc:
            # ``from None``: the chained exception would carry the full URL including the session.
            description = f"connection failed: {type(exc).__name__}: {self._redact(str(exc))}"
            raise DresError(None, description, method, path) from None
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        _LOG.debug("DRES %s %s -> HTTP %d in %.0f ms", method, path, response.status_code, elapsed_ms)

        try:
            payload = response.json() if response.content else None
        except ValueError:
            payload = None
        if 200 <= response.status_code < 300:
            if payload is None:
                raise DresError(response.status_code, "response body is not JSON", method, path)
            return payload
        if isinstance(payload, dict) and isinstance(payload.get("description"), str):
            description = payload["description"]
        else:
            description = response.text.strip()[:_ERROR_BODY_LIMIT] or response.reason or "no description"
        raise DresError(response.status_code, self._redact(description), method, path)


def _path_segment(value: str) -> str:
    """Percent-encode an identifier for use as one URL path segment."""
    return quote(_non_empty_str("evaluation_id", value), safe="")


def _verdict_of(result: Any) -> str:
    return str(result.get("submission")) if isinstance(result, dict) else "unknown"
