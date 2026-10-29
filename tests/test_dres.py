"""Tests for :mod:`scenechain.dres` against a local mock DRES v2 server.

The mock (``http.server`` in a thread) implements exactly the client paths the tool uses. It checks
every request body against schema fragments copied verbatim from the DRES client OpenAPI
specification (``doc/oas-client.json``, tag ``v2.0.5``, commit ``bd600c8d``), and it validates its
own responses against the same fragments so that the mock cannot drift from the specification.

Like DRES, the mock sets a ``Secure`` session cookie on login, but it authenticates requests only
through the ``session`` query parameter; the tests therefore prove that the client propagates the
token explicitly instead of relying on cookies.

Set ``DRES_CLIENT_SPEC=/path/to/oas-client.json`` to also compare the embedded fragments with a
downloaded specification (for example that of a newer DRES release). No network is needed otherwise.
"""

from __future__ import annotations

import json
import logging
import os
import re
import socket
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit

import numpy as np
import pytest
import requests

from scenechain.dres import (
    DresClient,
    DresError,
    query_event,
    result_log,
    temporal_answer,
    text_answer,
)

# --------------------------------------------------------------------------------------------------
# Schema fragments, verbatim from doc/oas-client.json (components.schemas) at DRES tag v2.0.5
# --------------------------------------------------------------------------------------------------

SCHEMAS: dict[str, dict[str, Any]] = {
    "LoginRequest": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "username": {"type": "string"},
            "password": {"type": "string"},
        },
        "required": ["username", "password"],
    },
    "ApiUser": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "id": {"type": "string"},
            "username": {"type": "string"},
            "role": {"$ref": "#/components/schemas/ApiRole"},
            "sessionId": {"type": "string"},
        },
    },
    "ApiRole": {"type": "string", "enum": ["ANYONE", "VIEWER", "PARTICIPANT", "JUDGE", "ADMIN"]},
    "ErrorStatus": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "status": {"type": "boolean"},
            "description": {"type": "string"},
        },
        "required": ["status", "description"],
    },
    "SuccessStatus": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "status": {"type": "boolean"},
            "description": {"type": "string"},
        },
        "required": ["status", "description"],
    },
    "ApiClientEvaluationInfo": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "id": {"type": "string"},
            "name": {"type": "string"},
            "type": {"$ref": "#/components/schemas/ApiEvaluationType"},
            "status": {"$ref": "#/components/schemas/ApiEvaluationStatus"},
            "templateId": {"type": "string"},
            "templateDescription": {"type": "string"},
            "teams": {"type": "array", "items": {"type": "string"}},
            "taskTemplates": {
                "type": "array",
                "items": {"$ref": "#/components/schemas/ApiClientTaskTemplateInfo"},
            },
        },
        "required": ["id", "name", "type", "status", "templateId", "teams", "taskTemplates"],
    },
    "ApiEvaluationType": {"type": "string", "enum": ["SYNCHRONOUS", "ASYNCHRONOUS", "NON_INTERACTIVE"]},
    "ApiEvaluationStatus": {"type": "string", "enum": ["CREATED", "ACTIVE", "TERMINATED"]},
    "ApiClientTaskTemplateInfo": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "name": {"type": "string"},
            "taskGroup": {"type": "string"},
            "taskType": {"type": "string"},
            "duration": {"type": "integer", "format": "int64"},
        },
        "required": ["name", "taskGroup", "taskType"],
    },
    "ApiEvaluationState": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "evaluationId": {"type": "string"},
            "evaluationStatus": {"$ref": "#/components/schemas/ApiEvaluationStatus"},
            "taskId": {"type": "string"},
            "taskStatus": {"$ref": "#/components/schemas/ApiTaskStatus"},
            "taskTemplateId": {"type": "string"},
            "timeLeft": {"type": "integer", "format": "int64"},
            "timeElapsed": {"type": "integer", "format": "int64"},
        },
        "required": ["evaluationId", "evaluationStatus", "taskStatus", "timeElapsed"],
    },
    "ApiTaskStatus": {
        "type": "string",
        "enum": ["NO_TASK", "CREATED", "PREPARING", "RUNNING", "ENDED", "IGNORED"],
    },
    "ApiClientSubmission": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "answerSets": {"type": "array", "items": {"$ref": "#/components/schemas/ApiClientAnswerSet"}},
        },
        "required": ["answerSets"],
    },
    "ApiClientAnswerSet": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "taskId": {"type": "string", "nullable": True},
            "taskName": {"type": "string", "nullable": True},
            "answers": {"type": "array", "items": {"$ref": "#/components/schemas/ApiClientAnswer"}},
        },
        "required": ["answers"],
    },
    "ApiClientAnswer": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "text": {"type": "string", "nullable": True},
            "mediaItemName": {"type": "string", "nullable": True},
            "mediaItemCollectionName": {"type": "string", "nullable": True},
            "start": {"type": "integer", "format": "int64", "nullable": True},
            "end": {"type": "integer", "format": "int64", "nullable": True},
        },
    },
    "SuccessfulSubmissionsStatus": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "status": {"type": "boolean"},
            "submission": {"$ref": "#/components/schemas/ApiVerdictStatus"},
            "description": {"type": "string"},
        },
        "required": ["status", "submission", "description"],
    },
    "ApiVerdictStatus": {"type": "string", "enum": ["CORRECT", "WRONG", "INDETERMINATE", "UNDECIDABLE"]},
    "QueryEventLog": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "timestamp": {"type": "integer", "format": "int64"},
            "events": {"type": "array", "items": {"$ref": "#/components/schemas/QueryEvent"}},
        },
        "required": ["timestamp", "events"],
    },
    "QueryEvent": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "timestamp": {"type": "integer", "format": "int64"},
            "category": {"$ref": "#/components/schemas/QueryEventCategory"},
            "type": {"type": "string"},
            "value": {"type": "string"},
        },
        "required": ["timestamp", "category", "type", "value"],
    },
    "QueryEventCategory": {
        "type": "string",
        "enum": ["TEXT", "IMAGE", "SKETCH", "FILTER", "BROWSING", "COOPERATION", "OTHER"],
    },
    "QueryResultLog": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "timestamp": {"type": "integer", "format": "int64"},
            "sortType": {"type": "string"},
            "resultSetAvailability": {"type": "string"},
            "results": {"type": "array", "items": {"$ref": "#/components/schemas/RankedAnswer"}},
            "events": {"type": "array", "items": {"$ref": "#/components/schemas/QueryEvent"}},
        },
        "required": ["timestamp", "sortType", "resultSetAvailability", "results", "events"],
    },
    "RankedAnswer": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "answer": {"$ref": "#/components/schemas/ApiClientAnswer"},
            "rank": {"type": "integer", "format": "int32"},
        },
        "required": ["answer"],
    },
}


def ref(name: str) -> dict[str, Any]:
    """A ``$ref`` to one of the embedded component schemas."""
    return {"$ref": f"#/components/schemas/{name}"}


def schema_errors(value: Any, schema: dict[str, Any], where: str = "$") -> list[str]:
    """Validate ``value`` against the OpenAPI 3.0 subset used by the DRES client specification."""
    if "$ref" in schema:
        return schema_errors(value, SCHEMAS[schema["$ref"].rsplit("/", 1)[1]], where)
    if value is None:
        return [] if schema.get("nullable") else [f"{where}: null is not allowed"]
    kind = schema.get("type")
    if kind == "object":
        if not isinstance(value, dict):
            return [f"{where}: expected object"]
        errors = [f"{where}: missing {key!r}" for key in schema.get("required", []) if key not in value]
        properties = schema.get("properties", {})
        for key, item in value.items():
            if key in properties:
                errors += schema_errors(item, properties[key], f"{where}.{key}")
            elif schema.get("additionalProperties") is False:
                errors.append(f"{where}: unexpected field {key!r}")
        return errors
    if kind == "array":
        if not isinstance(value, list):
            return [f"{where}: expected array"]
        items = schema["items"]
        return [e for i, item in enumerate(value) for e in schema_errors(item, items, f"{where}[{i}]")]
    if kind == "string":
        if not isinstance(value, str):
            return [f"{where}: expected string"]
        if "enum" in schema and value not in schema["enum"]:
            return [f"{where}: {value!r} not in {schema['enum']}"]
        return []
    if kind == "integer":
        bits = 32 if schema.get("format") == "int32" else 64
        in_range = isinstance(value, int) and -(2 ** (bits - 1)) <= value < 2 ** (bits - 1)
        if not in_range or isinstance(value, bool):
            return [f"{where}: expected int{bits}"]
        return []
    if kind == "boolean":
        return [] if isinstance(value, bool) else [f"{where}: expected boolean"]
    raise AssertionError(f"schema construct not supported by the test validator: {schema}")


# --------------------------------------------------------------------------------------------------
# Mock DRES server
# --------------------------------------------------------------------------------------------------

PASSWORD = "correct-horse-battery-staple"
ACTIVE_ID = "eval-active"
CREATED_ID = "eval-created"


@dataclass(frozen=True)
class Operation:
    """One client operation of the specification: request schema and response schema per status."""

    method: str
    pattern: re.Pattern[str]
    request: str | None
    responses: dict[int, dict[str, Any]]


ERROR = ref("ErrorStatus")
OPERATIONS: dict[str, Operation] = {
    "login": Operation(
        "POST", re.compile(r"/api/v2/login"), "LoginRequest", {200: ref("ApiUser"), 400: ERROR, 401: ERROR}
    ),
    "logout": Operation("GET", re.compile(r"/api/v2/logout"), None, {200: ref("SuccessStatus"), 400: ERROR}),
    "evaluation_list": Operation(
        "GET",
        re.compile(r"/api/v2/client/evaluation/list"),
        None,
        {200: {"type": "array", "items": ref("ApiClientEvaluationInfo")}, 401: ERROR},
    ),
    "current_task": Operation(
        "GET",
        re.compile(r"/api/v2/client/evaluation/currentTask/(?P<evaluation>[^/]+)"),
        None,
        {200: ref("ApiClientTaskTemplateInfo"), 401: ERROR, 404: ERROR},
    ),
    "state": Operation(
        "GET",
        re.compile(r"/api/v2/evaluation/(?P<evaluation>[^/]+)/state"),
        None,
        {200: ref("ApiEvaluationState"), 401: ERROR, 403: ERROR, 404: ERROR},
    ),
    "submit": Operation(
        "POST",
        re.compile(r"/api/v2/submit/(?P<evaluation>[^/]+)"),
        "ApiClientSubmission",
        {
            200: ref("SuccessfulSubmissionsStatus"),
            202: ref("SuccessfulSubmissionsStatus"),
            400: ERROR,
            401: ERROR,
            404: ERROR,
            412: ERROR,
        },
    ),
    "log_query": Operation(
        "POST",
        re.compile(r"/api/v2/log/query/(?P<evaluation>[^/]+)"),
        "QueryEventLog",
        {200: ref("SuccessStatus"), 400: ERROR, 401: ERROR},
    ),
    "log_result": Operation(
        "POST",
        re.compile(r"/api/v2/log/result/(?P<evaluation>[^/]+)"),
        "QueryResultLog",
        {200: ref("SuccessStatus"), 400: ERROR, 401: ERROR},
    ),
}

TASK = {"name": "kis-t-03", "taskGroup": "kis-t", "taskType": "Textual Known Item Search", "duration": 420}
EVALUATIONS = [
    {
        "id": ACTIVE_ID,
        "name": "VBS rehearsal",
        "type": "SYNCHRONOUS",
        "status": "ACTIVE",
        "templateId": "template-1",
        "templateDescription": "Rehearsal with KIS, AVS and QA tasks",
        "teams": ["team-1"],
        "taskTemplates": [
            TASK,
            {"name": "avs-01", "taskGroup": "avs", "taskType": "Ad-hoc Video Search", "duration": 300},
        ],
    },
    {
        "id": CREATED_ID,
        "name": "VBS main run",
        "type": "SYNCHRONOUS",
        "status": "CREATED",
        "templateId": "template-2",
        "teams": ["team-1"],
        "taskTemplates": [],
    },
]
CORRECT = {"status": True, "submission": "CORRECT", "description": "Submission correct, well done!"}


@dataclass
class Recorded:
    """One request as the mock received it."""

    operation: str
    method: str
    path: str
    query: dict[str, list[str]]
    headers: dict[str, str]
    body: Any


@dataclass
class MockDres:
    """State and behaviour of the mock server; tests reconfigure the public attributes."""

    url: str = ""
    requests: list[Recorded] = field(default_factory=list)
    violations: list[str] = field(default_factory=list)
    sessions: dict[str, str] = field(default_factory=dict)
    current_task: dict[str, Any] | None = field(default_factory=lambda: dict(TASK))
    submit_reply: tuple[int, dict[str, Any]] = (200, CORRECT)
    raw_reply: tuple[int, str, bytes] | None = None
    delay_s: float = 0.0
    logins: int = 0

    def handle(self, handler: BaseHTTPRequestHandler) -> None:
        split = urlsplit(handler.path)
        length = int(handler.headers.get("Content-Length") or 0)
        raw_body = handler.rfile.read(length) if length else b""
        name, operation, match = self._route(handler.command, split.path)
        body = json.loads(raw_body) if raw_body else None
        query = parse_qs(split.query)
        headers = dict(handler.headers.items())
        self.requests.append(Recorded(name or "unknown", handler.command, split.path, query, headers, body))
        if self.delay_s:
            time.sleep(self.delay_s)
        if self.raw_reply is not None:
            self._send_raw(handler, *self.raw_reply)
            return
        if operation is None or match is None:
            self._send(handler, None, 404, {"status": False, "description": "Not found."})
            return
        status, payload, headers = self._dispatch(name, operation, match, query, handler, body)
        self._send(handler, operation, status, payload, headers)

    def _route(self, method: str, path: str) -> tuple[str | None, Operation | None, re.Match[str] | None]:
        for name, operation in OPERATIONS.items():
            match = operation.pattern.fullmatch(path)
            if match and operation.method == method:
                return name, operation, match
        return None, None, None

    def _dispatch(
        self,
        name: str,
        operation: Operation,
        match: re.Match[str],
        query: dict[str, list[str]],
        handler: BaseHTTPRequestHandler,
        body: Any,
    ) -> tuple[int, Any, dict[str, str]]:
        if operation.request is not None:
            if handler.headers.get("Content-Type") != "application/json":
                return 400, {"status": False, "description": "Content-Type must be application/json."}, {}
            errors = schema_errors(body, ref(operation.request))
            if errors:
                self.violations += errors
                return 400, {"status": False, "description": "Invalid parameters: " + "; ".join(errors)}, {}
        if name == "login":
            return self._login(body)
        token = query.get("session", [None])[0]
        if token not in self.sessions:
            return 401, {"status": False, "description": "Unauthorized request!"}, {}
        evaluation = unquote(match.groupdict().get("evaluation") or "")
        known = evaluation in {e["id"] for e in EVALUATIONS}
        if name == "logout":
            del self.sessions[token]
            return 200, {"status": True, "description": "User 'team1' logged out successfully."}, {}
        if name == "evaluation_list":
            return 200, EVALUATIONS, {}
        if not known:
            return 404, {"status": False, "description": f"Evaluation {evaluation} not found."}, {}
        if name == "current_task":
            if self.current_task is None:
                return 404, {"status": False, "description": "Specified evaluation has no active task."}, {}
            return 200, self.current_task, {}
        if name == "state":
            state = {
                "evaluationId": evaluation,
                "evaluationStatus": "ACTIVE",
                "taskId": "task-run-7",
                "taskStatus": "RUNNING",
                "taskTemplateId": "task-template-3",
                "timeLeft": 287,
                "timeElapsed": 133,
            }
            return 200, state, {}
        if name == "submit":
            status, payload = self.submit_reply
            return status, payload, {}
        if name == "log_query":
            return 200, {"status": True, "description": "Log received"}, {}
        return 200, {"status": True, "description": "Log entry received."}, {}

    def _login(self, body: dict[str, Any]) -> tuple[int, Any, dict[str, str]]:
        if body["username"] != "team1" or body["password"] != PASSWORD:
            return 401, {"status": False, "description": "Invalid credentials. Please try again!"}, {}
        self.logins += 1
        token = f"mockSessionToken{self.logins:016d}"
        self.sessions[token] = body["username"]
        user = {"id": "user-7", "username": "team1", "role": "PARTICIPANT", "sessionId": token}
        cookie = f"SESSIONID={token}; Max-Age=86400; Path=/; Secure; SameSite=None"
        return 200, user, {"Set-Cookie": cookie}

    def _send(
        self,
        handler: BaseHTTPRequestHandler,
        operation: Operation | None,
        status: int,
        payload: Any,
        headers: dict[str, str] | None = None,
    ) -> None:
        if operation is not None:
            if status not in operation.responses:
                self.violations.append(f"mock sent undeclared status {status}")
            else:
                self.violations += schema_errors(payload, operation.responses[status], f"response {status}")
        self._send_raw(handler, status, "application/json", json.dumps(payload).encode("utf-8"), headers)

    @staticmethod
    def _send_raw(
        handler: BaseHTTPRequestHandler,
        status: int,
        content_type: str,
        data: bytes,
        headers: dict[str, str] | None = None,
    ) -> None:
        try:
            handler.send_response(status)
            handler.send_header("Content-Type", content_type)
            handler.send_header("Content-Length", str(len(data)))
            for key, value in (headers or {}).items():
                handler.send_header(key, value)
            handler.end_headers()
            handler.wfile.write(data)
        except OSError:
            pass  # the client gave up (timeout test)

    def of(self, operation: str) -> list[Recorded]:
        return [r for r in self.requests if r.operation == operation]


@pytest.fixture()
def dres() -> Iterator[MockDres]:
    mock = MockDres()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 (http.server naming)
            mock.handle(self)

        do_POST = do_GET

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            """Keep test output quiet."""

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
    thread.start()
    mock.url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        yield mock
    finally:
        server.shutdown()
        server.server_close()
    assert mock.violations == []


@pytest.fixture()
def client(dres: MockDres) -> Iterator[DresClient]:
    http = requests.Session()
    http.trust_env = False  # never route 127.0.0.1 through a proxy from the environment
    yield DresClient(dres.url + "/", timeout=5.0, http=http)
    http.close()


@pytest.fixture()
def logged_in(client: DresClient) -> DresClient:
    client.login("team1", PASSWORD)
    return client


# --------------------------------------------------------------------------------------------------
# Specification fragments
# --------------------------------------------------------------------------------------------------


def test_embedded_references_resolve_and_validator_is_strict() -> None:
    for name, schema in SCHEMAS.items():
        for target in re.findall(r"#/components/schemas/(\w+)", json.dumps(schema)):
            assert target in SCHEMAS, f"{name} references {target}"
    submission = ref("ApiClientSubmission")

    def one_answer(answer: dict[str, Any]) -> dict[str, Any]:
        return {"answerSets": [{"answers": [answer]}]}

    assert schema_errors(one_answer({"mediaItemName": "V003", "start": 1, "end": 2}), submission) == []
    assert schema_errors(one_answer({"mediaItem": "V003"}), submission)  # wrong field name
    assert schema_errors(one_answer({"text": "x", "start": 1.5}), submission)  # float time
    assert schema_errors({"answers": [{"text": "x"}]}, submission)  # answer set not wrapped
    bad_event = {"timestamp": 1, "category": "QUERY", "type": "t", "value": "v"}  # category not in enum
    assert schema_errors({"timestamp": 1, "events": [bad_event]}, ref("QueryEventLog"))


@pytest.mark.skipif(
    not os.environ.get("DRES_CLIENT_SPEC"), reason="set DRES_CLIENT_SPEC to a downloaded oas-client.json"
)
def test_embedded_fragments_match_specification_file() -> None:
    spec = json.loads(Path(os.environ["DRES_CLIENT_SPEC"]).read_text(encoding="utf-8"))
    for name, schema in SCHEMAS.items():
        assert spec["components"]["schemas"][name] == schema, name
    for operation in OPERATIONS.values():
        template = re.sub(r"\(\?P<evaluation>\[\^/\]\+\)", "{evaluationId}", operation.pattern.pattern)
        declared = spec["paths"][template][operation.method.lower()]
        declared_statuses = {int(code) for code in declared["responses"]}
        assert set(operation.responses) == declared_statuses, template
        if operation.request is not None:
            body = declared["requestBody"]["content"]["application/json"]["schema"]
            assert body == ref(operation.request), template


# --------------------------------------------------------------------------------------------------
# Session handling
# --------------------------------------------------------------------------------------------------


def test_login_success_returns_user_and_holds_session(client: DresClient, dres: MockDres) -> None:
    user = client.login("team1", PASSWORD)

    token = next(iter(dres.sessions))
    assert user == {"id": "user-7", "username": "team1", "role": "PARTICIPANT", "sessionId": token}
    assert client.logged_in and client.username == "team1" and client.role == "PARTICIPANT"
    (request,) = dres.of("login")
    assert request.body == {"username": "team1", "password": PASSWORD}
    assert "session" not in request.query
    assert request.headers["Content-Type"] == "application/json"
    assert client.base_url == dres.url


def test_login_failure_raises_with_server_description(client: DresClient, dres: MockDres) -> None:
    with pytest.raises(DresError) as caught:
        client.login("team1", "wrong-password")

    assert caught.value.status == 401
    assert caught.value.description == "Invalid credentials. Please try again!"
    assert (caught.value.method, caught.value.path) == ("POST", "/api/v2/login")
    assert "wrong-password" not in str(caught.value)
    assert not client.logged_in and client.username is None


def test_failed_relogin_keeps_the_previous_session(logged_in: DresClient, dres: MockDres) -> None:
    logged_in.evaluations()
    token = dres.requests[-1].query["session"]
    with pytest.raises(DresError):
        logged_in.login("team1", "typo")  # a typo in a re-login during a live task
    assert logged_in.logged_in and logged_in.username == "team1"
    logged_in.evaluations()  # the working session is still used
    assert dres.requests[-1].operation != "login" and dres.requests[-1].query["session"] == token


def test_session_token_is_sent_as_query_parameter(logged_in: DresClient, dres: MockDres) -> None:
    token = next(iter(dres.sessions))
    logged_in.evaluations()
    logged_in.task_info(ACTIVE_ID)
    logged_in.evaluation_state(ACTIVE_ID)
    logged_in.submit_kis(ACTIVE_ID, "V003", 20000, 24000)
    event = query_event("TEXT", "jointEmbedding", "a man opens a wooden door", 1)
    logged_in.log_query(ACTIVE_ID, [event])
    answers = [temporal_answer("V003", 0, 1)]
    log = result_log(answers, sort_type="score", result_set_availability="top", events=[event])
    logged_in.log_results(ACTIVE_ID, log)

    after_login = [r for r in dres.requests if r.operation != "login"]
    assert [r.operation for r in after_login] == [
        "evaluation_list", "current_task", "state", "submit", "log_query", "log_result"
    ]
    assert all(r.query == {"session": [token]} for r in after_login)


def test_calls_before_login_fail_without_contacting_the_server(client: DresClient, dres: MockDres) -> None:
    with pytest.raises(DresError) as caught:
        client.evaluations()

    assert caught.value.status is None and "not logged in" in caught.value.description
    assert dres.requests == []


def test_logout_ends_session_and_relogin_uses_new_token(logged_in: DresClient, dres: MockDres) -> None:
    first = next(iter(dres.sessions))
    assert logged_in.logout() == {"status": True, "description": "User 'team1' logged out successfully."}
    assert not logged_in.logged_in and dres.sessions == {}
    with pytest.raises(DresError):
        logged_in.evaluations()

    logged_in.login("team1", PASSWORD)
    logged_in.evaluations()
    second = next(iter(dres.sessions))
    assert second != first
    assert dres.of("evaluation_list")[-1].query == {"session": [second]}


def test_expired_session_surfaces_as_401(logged_in: DresClient, dres: MockDres) -> None:
    dres.sessions.clear()  # e.g. DRES restarted
    with pytest.raises(DresError) as caught:
        logged_in.evaluations()
    assert (caught.value.status, caught.value.description) == (401, "Unauthorized request!")


# --------------------------------------------------------------------------------------------------
# Evaluations and tasks
# --------------------------------------------------------------------------------------------------


def test_evaluations_returns_client_evaluation_info(logged_in: DresClient) -> None:
    evaluations = logged_in.evaluations()

    assert [e["id"] for e in evaluations] == [ACTIVE_ID, CREATED_ID]
    assert schema_errors(evaluations, {"type": "array", "items": ref("ApiClientEvaluationInfo")}) == []


def test_task_info_active_task_no_task_and_unknown_evaluation(logged_in: DresClient, dres: MockDres) -> None:
    assert logged_in.task_info(ACTIVE_ID) == TASK

    dres.current_task = None
    assert logged_in.task_info(ACTIVE_ID) is None

    with pytest.raises(DresError) as caught:
        logged_in.task_info("eval-missing")
    assert (caught.value.status, caught.value.description) == (404, "Evaluation eval-missing not found.")


def test_evaluation_ids_are_percent_encoded(logged_in: DresClient, dres: MockDres) -> None:
    with pytest.raises(DresError):
        logged_in.task_info("run 1/a")
    assert dres.requests[-1].path == "/api/v2/client/evaluation/currentTask/run%201%2Fa"


def test_evaluation_state_reports_remaining_time(logged_in: DresClient, dres: MockDres) -> None:
    state = logged_in.evaluation_state(ACTIVE_ID)

    assert (state["taskStatus"], state["timeLeft"]) == ("RUNNING", 287)
    assert dres.requests[-1].path == f"/api/v2/evaluation/{ACTIVE_ID}/state"


# --------------------------------------------------------------------------------------------------
# Submissions
# --------------------------------------------------------------------------------------------------


def test_submit_kis_sends_one_temporal_answer(logged_in: DresClient, dres: MockDres) -> None:
    reply = logged_in.submit_kis(ACTIVE_ID, "V003", np.int64(20000), 24000)

    assert reply == CORRECT
    (request,) = dres.of("submit")
    assert request.path == f"/api/v2/submit/{ACTIVE_ID}"
    assert request.headers["Content-Type"] == "application/json"
    answer = {"mediaItemName": "V003", "start": 20000, "end": 24000}
    assert request.body == {"answerSets": [{"answers": [answer]}]}
    assert schema_errors(request.body, ref("ApiClientSubmission")) == []


def test_submit_kis_with_collection_name(logged_in: DresClient, dres: MockDres) -> None:
    logged_in.submit_kis(ACTIVE_ID, "00017", 0, 3500, collection="V3C")

    answer = dres.of("submit")[0].body["answerSets"][0]["answers"][0]
    assert answer == {"mediaItemName": "00017", "mediaItemCollectionName": "V3C", "start": 0, "end": 3500}


def test_submit_text_sends_one_text_answer(logged_in: DresClient, dres: MockDres) -> None:
    wrong = {"status": True, "submission": "WRONG", "description": "Submission wrong, try again!"}
    dres.submit_reply = (200, wrong)

    reply = logged_in.submit_text(ACTIVE_ID, "  a blue boat  ")

    assert reply["submission"] == "WRONG"
    assert dres.of("submit")[0].body == {"answerSets": [{"answers": [{"text": "a blue boat"}]}]}


def test_submit_avs_sends_one_answer_set_per_shot(logged_in: DresClient, dres: MockDres) -> None:
    pending = {
        "status": True,
        "submission": "INDETERMINATE",
        "description": "Submission received, awaiting verdict!",
    }
    dres.submit_reply = (202, pending)

    shots = [("V002", 58000, 62000), ("V001", 9000, 13000), ("V002", 60000, 61000)]
    reply = logged_in.submit_avs(ACTIVE_ID, shots)

    assert reply == pending
    body = dres.of("submit")[0].body
    assert body == {
        "answerSets": [
            {"answers": [{"mediaItemName": "V002", "start": 58000, "end": 62000}]},
            {"answers": [{"mediaItemName": "V001", "start": 9000, "end": 13000}]},
            {"answers": [{"mediaItemName": "V002", "start": 60000, "end": 61000}]},
        ]
    }
    assert schema_errors(body, ref("ApiClientSubmission")) == []


def test_generic_submit_accepts_several_answers_per_set(logged_in: DresClient, dres: MockDres) -> None:
    logged_in.submit(ACTIVE_ID, [[temporal_answer("V003", 1, 2), text_answer("door")]])

    assert dres.of("submit")[0].body == {
        "answerSets": [{"answers": [{"mediaItemName": "V003", "start": 1, "end": 2}, {"text": "door"}]}]
    }


# --------------------------------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------------------------------


def test_log_query_sends_query_event_log(logged_in: DresClient, dres: MockDres) -> None:
    events = [
        query_event("text", "jointEmbedding", "a man opens a wooden door", 1_789_000_000_000),
        query_event("BROWSING", "rankedList", {"scenes": 2, "mode": "chains"}, 1_789_000_000_250),
    ]

    reply = logged_in.log_query(ACTIVE_ID, events, timestamp_ms=1_789_000_000_300)

    assert reply == {"status": True, "description": "Log received"}
    (request,) = dres.of("log_query")
    assert request.path == f"/api/v2/log/query/{ACTIVE_ID}"
    assert request.body == {
        "timestamp": 1_789_000_000_300,
        "events": [
            {"timestamp": 1_789_000_000_000, "category": "TEXT", "type": "jointEmbedding",
             "value": "a man opens a wooden door"},
            {"timestamp": 1_789_000_000_250, "category": "BROWSING", "type": "rankedList",
             "value": '{"mode":"chains","scenes":2}'},
        ],
    }
    assert schema_errors(request.body, ref("QueryEventLog")) == []


def test_log_results_sends_ranked_answers(logged_in: DresClient, dres: MockDres) -> None:
    event = query_event("TEXT", "jointEmbedding", "the man sits down at a desk", 1_789_000_000_000)
    log = result_log(
        [
            temporal_answer("V003", 26000, 30000),
            temporal_answer("V005", 18000, 22000),
            {"mediaItemName": "V004"},
        ],
        sort_type="score",
        result_set_availability="top",
        events=[event],
        timestamp_ms=1_789_000_000_100,
    )

    reply = logged_in.log_results(ACTIVE_ID, log)

    assert reply == {"status": True, "description": "Log entry received."}
    (request,) = dres.of("log_result")
    assert request.path == f"/api/v2/log/result/{ACTIVE_ID}"
    assert request.body == {
        "timestamp": 1_789_000_000_100,
        "sortType": "score",
        "resultSetAvailability": "top",
        "results": [
            {"answer": {"mediaItemName": "V003", "start": 26000, "end": 30000}, "rank": 1},
            {"answer": {"mediaItemName": "V005", "start": 18000, "end": 22000}, "rank": 2},
            {"answer": {"mediaItemName": "V004"}, "rank": 3},
        ],
        "events": [event],
    }
    assert schema_errors(request.body, ref("QueryResultLog")) == []


# --------------------------------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("status", "description"),
    [
        (412, "Submission 5f0c was rejected by filter: Duplicate submission received."),
        (400, "Run manager is in wrong state and cannot accept any more submission."),
        (404, "Evaluation with ID 'eval-active' could not be found."),
    ],
)
def test_submission_errors_carry_status_and_server_description(
    logged_in: DresClient, dres: MockDres, status: int, description: str
) -> None:
    dres.submit_reply = (status, {"status": False, "description": description})

    with pytest.raises(DresError) as caught:
        logged_in.submit_kis(ACTIVE_ID, "V003", 20000, 24000)

    assert (caught.value.status, caught.value.description) == (status, description)
    assert str(caught.value) == f"DRES POST /api/v2/submit/{ACTIVE_ID} failed (HTTP {status}): {description}"


def test_non_json_error_body_is_reported(logged_in: DresClient, dres: MockDres) -> None:
    dres.raw_reply = (502, "text/html", b"<html><body>502 Bad Gateway</body></html>")

    with pytest.raises(DresError) as caught:
        logged_in.evaluations()

    assert caught.value.status == 502 and "Bad Gateway" in caught.value.description


def test_success_without_json_body_is_an_error(logged_in: DresClient, dres: MockDres) -> None:
    dres.raw_reply = (200, "text/html", b"<html>DRES frontend</html>")

    with pytest.raises(DresError) as caught:
        logged_in.evaluations()

    assert caught.value.status == 200 and "not JSON" in caught.value.description


def test_session_token_is_redacted_from_server_descriptions(logged_in: DresClient, dres: MockDres) -> None:
    token = next(iter(dres.sessions))
    dres.submit_reply = (401, {"status": False, "description": f"No user registered for session {token}."})

    with pytest.raises(DresError) as caught:
        logged_in.submit_text(ACTIVE_ID, "blue")

    assert token not in str(caught.value)
    assert caught.value.description == "No user registered for session <redacted>."


def test_timeout_raises_dres_error(dres: MockDres) -> None:
    http = requests.Session()
    http.trust_env = False
    slow = DresClient(dres.url, timeout=0.2, http=http)
    dres.delay_s = 0.6

    with pytest.raises(DresError) as caught:
        slow.login("team1", PASSWORD)

    assert caught.value.status is None and "timed out" in caught.value.description
    http.close()


def test_unreachable_server_raises_dres_error() -> None:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    http = requests.Session()
    http.trust_env = False
    unreachable = DresClient(f"http://127.0.0.1:{port}", timeout=0.5, http=http)

    with pytest.raises(DresError) as caught:
        unreachable.login("team1", PASSWORD)

    assert caught.value.status is None
    assert caught.value.__cause__ is None and caught.value.__suppress_context__
    assert PASSWORD not in str(caught.value)
    http.close()


@pytest.mark.parametrize(
    "call",
    [
        lambda c: c.submit_kis(ACTIVE_ID, "V003", 24000, 20000),
        lambda c: c.submit_kis(ACTIVE_ID, "V003", -1, 20000),
        lambda c: c.submit_kis(ACTIVE_ID, "V003", 20.5, 24000),
        lambda c: c.submit_kis(ACTIVE_ID, "V003", True, 24000),
        lambda c: c.submit_kis(ACTIVE_ID, "", 20000, 24000),
        lambda c: c.submit_kis("", "V003", 20000, 24000),
        lambda c: c.submit_text(ACTIVE_ID, "   "),
        lambda c: c.submit_avs(ACTIVE_ID, []),
        lambda c: c.submit_avs(ACTIVE_ID, [("V003", 20000)]),
        lambda c: c.submit(ACTIVE_ID, [[]]),
        lambda c: c.submit(ACTIVE_ID, [[{"mediaItemName": "V003", "start": 5}]]),
        lambda c: c.submit(ACTIVE_ID, [[{"mediaItem": "V003"}]]),
        lambda c: c.log_query(ACTIVE_ID, []),
        lambda c: c.log_query(ACTIVE_ID, [{"timestamp": 1, "category": "QUERY", "type": "t", "value": "v"}]),
        lambda c: c.log_query(ACTIVE_ID, [{"timestamp": 1, "category": "TEXT", "type": "t", "value": 3}]),
        lambda c: c.log_results(ACTIVE_ID, {"timestamp": 1, "sortType": "s", "results": [], "events": []}),
    ],
)
def test_invalid_arguments_are_rejected_before_any_request(
    logged_in: DresClient, dres: MockDres, call: Callable[[DresClient], Any]
) -> None:
    with pytest.raises(ValueError):
        call(logged_in)
    assert [r.operation for r in dres.requests] == ["login"]


@pytest.mark.parametrize(
    "url", ["ftp://dres.example.org", "dres.example.org:8080", "http://host/?session=x", ""]
)
def test_base_url_must_be_plain_http_url(url: str) -> None:
    with pytest.raises(ValueError):
        DresClient(url)


def test_secrets_never_reach_logs_or_repr(
    client: DresClient, dres: MockDres, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG, logger="scenechain.dres")
    with pytest.raises(DresError):
        client.login("team1", "almost-" + PASSWORD)
    client.login("team1", PASSWORD)
    client.submit_kis(ACTIVE_ID, "V003", 20000, 24000)
    token = next(iter(dres.sessions))

    assert "HTTP 200" in caplog.text and "CORRECT" in caplog.text
    for secret in (PASSWORD, token):
        assert secret not in caplog.text
        assert secret not in repr(client)
