"""Buffer interaction events and result lists; flush them to local JSONL files and to DRES.

VBS evaluates interaction logs alongside submissions, so every query, result list, browse and submit
action is recorded. Events and result lists are buffered in memory (calls come from request
handlers and must stay fast) and written by :meth:`InteractionLog.flush`, which the server calls
periodically and on shutdown.

The local files are the source of truth: ``events.jsonl`` and ``results.jsonl`` under ``log_dir``
receive every record, one JSON object per line, before anything is sent to DRES. DRES payloads are
built with the builders of :mod:`scenechain.dres` (``QueryEvent``, ``QueryResultLog``), so they
follow the DRES client specification. A DRES failure never loses local data; the records of that
flush are not re-sent, and the error is returned in the flush report.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence, TypedDict

from .dres import DresClient, query_event, result_log, temporal_answer

EVENTS_FILE = "events.jsonl"
RESULTS_FILE = "results.jsonl"

#: default DRES ``QueryEventCategory`` per event kind; unknown kinds are ``OTHER``
KIND_CATEGORIES = {
    "search": "TEXT",
    "text": "TEXT",
    "decompose": "TEXT",
    "similar": "IMAGE",
    "filter": "FILTER",
    "window": "FILTER",
    "mode": "BROWSING",
    "open_video": "BROWSING",
    "scroll": "BROWSING",
    "seek": "BROWSING",
}


class FlushReport(TypedDict):
    events: int
    results: int
    dres_sent: bool
    dres_error: Optional[str]


def _now_ms() -> int:
    return int(time.time() * 1000)


class InteractionLog:
    """Thread-safe buffer of query events and result logs for one operator session."""

    def __init__(
        self,
        log_dir: str | Path,
        dres_client: Optional[DresClient] = None,
        *,
        evaluation_id: Optional[str] = None,
        collection_name: Optional[str] = None,
        clock: Callable[[], int] = _now_ms,
    ) -> None:
        """``evaluation_id`` may be set later; without it (or a client) nothing goes to DRES."""
        self.log_dir = Path(log_dir)
        self.dres_client = dres_client
        self.evaluation_id = evaluation_id
        self.collection_name = collection_name
        self._clock = clock
        self._events: list[dict] = []
        self._results: list[dict] = []
        self._buffer_lock = threading.Lock()
        self._flush_lock = threading.Lock()

    # ------------------------------------------------------------------------------- record
    def event(
        self,
        kind: str,
        payload: Any,
        *,
        category: Optional[str] = None,
        timestamp_ms: Optional[int] = None,
    ) -> dict:
        """Buffer one ``QueryEvent`` of type ``kind``; non-string payloads are stored as JSON.

        ``category`` defaults to :data:`KIND_CATEGORIES` for ``kind``. Raises ``ValueError`` for an
        unknown category or an empty kind. Returns the buffered event.
        """
        cat = category or KIND_CATEGORIES.get(kind, "OTHER")
        ev = query_event(cat, kind, payload, self._clock() if timestamp_ms is None else timestamp_ms)
        with self._buffer_lock:
            self._events.append(ev)
        return ev

    def results(
        self,
        query_payload: Mapping[str, Any],
        items: Sequence[Mapping[str, Any]],
        *,
        sort_type: str = "score",
        availability: str = "top",
        timestamp_ms: Optional[int] = None,
    ) -> dict:
        """Buffer the ``QueryResultLog`` of one result list shown to the operator.

        ``query_payload`` is the search or similar-frames request (``scenes``/``window``/
        ``filter_videos``, or ``video_id``/``frame_idx``); it becomes the log's query events.
        ``items`` are the displayed results best first: frame hits (``video_id`` + ``time``),
        chains (``video_id`` + ``frames``) or video groups (``video_id`` only).
        """
        ts = self._clock() if timestamp_ms is None else timestamp_ms
        events = self._query_events(query_payload, ts)
        answers = [self._answer(item) for item in items]
        log = result_log(
            answers,
            sort_type=sort_type,
            result_set_availability=availability,
            events=events,
            timestamp_ms=ts,
        )
        with self._buffer_lock:
            self._results.append(log)
        return log

    @staticmethod
    def _query_events(payload: Mapping[str, Any], ts: int) -> list[dict]:
        scenes = [s for s in payload.get("scenes") or [] if isinstance(s, str) and s.strip()]
        if scenes:
            events = [query_event("TEXT", "jointEmbedding", s, ts) for s in scenes]
            if len(scenes) > 1:
                value = {"scenes": len(scenes), "window": payload.get("window")}
                events.append(query_event("TEXT", "temporalQuery", value, ts))
            if payload.get("filter_videos"):
                events.append(query_event("FILTER", "videoFilter", list(payload["filter_videos"]), ts))
            return events
        if "video_id" in payload and "frame_idx" in payload:
            value = {"video_id": payload["video_id"], "frame_idx": payload["frame_idx"]}
            return [query_event("IMAGE", "similarImage", value, ts)]
        return [query_event("OTHER", "query", dict(payload), ts)]

    def _answer(self, item: Mapping[str, Any]) -> dict:
        video_id = str(item["video_id"])
        frames = item.get("frames")
        if frames:
            start, end = float(frames[0]["time"]), float(frames[-1]["time"])
        elif "time" in item:
            start = end = float(item["time"])
        else:
            answer: dict[str, Any] = {"mediaItemName": video_id}
            if self.collection_name:
                answer["mediaItemCollectionName"] = self.collection_name
            return answer
        return temporal_answer(
            video_id, int(round(start * 1000)), int(round(end * 1000)), self.collection_name
        )

    # -------------------------------------------------------------------------------- flush
    def flush(self) -> FlushReport:
        """Write buffered records locally, then send them to DRES when connected.

        Raises ``OSError`` if the local files cannot be written; the records then stay buffered.
        """
        with self._flush_lock:
            with self._buffer_lock:
                events, self._events = self._events, []
                logs, self._results = self._results, []
            report: FlushReport = {
                "events": len(events), "results": len(logs), "dres_sent": False, "dres_error": None
            }
            if not events and not logs:
                return report
            try:
                self._write_local(events, logs)
            except OSError:
                with self._buffer_lock:
                    self._events[:0] = events
                    self._results[:0] = logs
                raise
            if self.dres_client is None or not self.evaluation_id:
                return report
            try:
                if events:
                    self.dres_client.log_query(self.evaluation_id, events)
                for log in logs:
                    self.dres_client.log_results(self.evaluation_id, log)
                report["dres_sent"] = True
            except Exception as exc:  # noqa: BLE001 - DRES trouble must not break the session
                report["dres_error"] = f"{type(exc).__name__}: {exc}"
            return report

    def _write_local(self, events: list[dict], logs: list[dict]) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        if events:
            with (self.log_dir / EVENTS_FILE).open("a", encoding="utf-8", newline="\n") as fh:
                for ev in events:
                    fh.write(json.dumps({"evaluation_id": self.evaluation_id, "event": ev},
                                        ensure_ascii=False) + "\n")
        if logs:
            with (self.log_dir / RESULTS_FILE).open("a", encoding="utf-8", newline="\n") as fh:
                for log in logs:
                    fh.write(json.dumps({"evaluation_id": self.evaluation_id, "result_log": log},
                                        ensure_ascii=False) + "\n")
