"""Interaction log: buffering, DRES payload shapes, local JSONL files, DRES failures."""

from __future__ import annotations

import json

import pytest

from scenechain.dres import DresClient, DresError
from scenechain.interaction_log import EVENTS_FILE, RESULTS_FILE, InteractionLog


class RecordingDres(DresClient):
    """A real DresClient (so payloads go through its validation) whose transport is recorded."""

    def __init__(self, fail: bool = False):
        super().__init__("http://dres.invalid")
        self.fail = fail
        self.requests: list[tuple[str, str, dict]] = []

    def _request(self, method, path, *, body=None, authenticated=True):
        if self.fail:
            raise DresError(None, "connection failed", method, path)
        self.requests.append((method, path, body))
        return {"status": True, "description": "ok"}


def clock() -> int:
    return 1_700_000_000_000


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_events_get_categories_and_string_values(tmp_path):
    log = InteractionLog(tmp_path, clock=clock)
    assert log.event("open_video", {"video_id": "V003"}) == {
        "timestamp": clock(), "category": "BROWSING", "type": "open_video", "value": '{"video_id":"V003"}'
    }
    assert log.event("basket_add", "V003@12.0")["category"] == "OTHER"
    assert log.event("scroll", "page 2", category="browsing", timestamp_ms=5)["timestamp"] == 5
    with pytest.raises(ValueError):
        log.event("search", "x", category="VOICE")


def test_result_logs_for_chains_frames_and_videos(tmp_path):
    log = InteractionLog(tmp_path, collection_name="synthetic", clock=clock)
    chains = log.results(
        {"scenes": ["a man opens a door", "he sits at a desk"], "window": [0, 30],
         "filter_videos": ["V003"]},
        [{"video_id": "V003", "score": 0.7, "frames": [{"time": 20.0}, {"time": 28.5}]}],
    )
    assert chains["sortType"] == "score" and chains["resultSetAvailability"] == "top"
    assert chains["results"] == [{"answer": {"mediaItemName": "V003", "start": 20000, "end": 28500,
                                             "mediaItemCollectionName": "synthetic"}, "rank": 1}]
    assert [(e["category"], e["type"]) for e in chains["events"]] == [
        ("TEXT", "jointEmbedding"), ("TEXT", "jointEmbedding"), ("TEXT", "temporalQuery"),
        ("FILTER", "videoFilter"),
    ]

    frames = log.results({"video_id": "V001", "frame_idx": 150},
                         [{"video_id": "V002", "time": 4.0}, {"video_id": "V005", "time": 6.5}])
    assert [r["rank"] for r in frames["results"]] == [1, 2]
    assert frames["results"][1]["answer"]["start"] == 6500
    assert [(e["category"], e["type"]) for e in frames["events"]] == [("IMAGE", "similarImage")]

    videos = log.results({"scenes": ["a dog"]}, [{"video_id": "V002", "score": 0.5}])
    assert videos["results"][0]["answer"] == {"mediaItemName": "V002", "mediaItemCollectionName": "synthetic"}


def test_flush_writes_locally_and_sends_to_dres(tmp_path):
    dres = RecordingDres()
    log = InteractionLog(tmp_path / "logs", dres, evaluation_id="eval-1", clock=clock)
    log.event("search", {"scenes": ["a red car"]})
    log.event("open_video", {"video_id": "V001"})
    log.results({"scenes": ["a red car"]}, [{"video_id": "V001", "time": 3.0}])

    report = log.flush()
    assert report == {"events": 2, "results": 1, "dres_sent": True, "dres_error": None}
    assert [(m, p) for m, p, _ in dres.requests] == [
        ("POST", "/api/v2/log/query/eval-1"), ("POST", "/api/v2/log/result/eval-1")
    ]
    assert len(dres.requests[0][2]["events"]) == 2
    events = read_jsonl(tmp_path / "logs" / EVENTS_FILE)
    assert [e["event"]["type"] for e in events] == ["search", "open_video"]
    assert all(e["evaluation_id"] == "eval-1" for e in events)
    assert read_jsonl(tmp_path / "logs" / RESULTS_FILE)[0]["result_log"]["results"][0]["rank"] == 1

    assert log.flush() == {"events": 0, "results": 0, "dres_sent": False, "dres_error": None}
    assert len(dres.requests) == 2


def test_without_evaluation_only_local(tmp_path):
    dres = RecordingDres()
    log = InteractionLog(tmp_path, dres, clock=clock)
    log.event("submit", {"kind": "kis", "video_id": "V003", "time": 21.0})
    assert log.flush()["dres_sent"] is False
    assert dres.requests == []
    assert len(read_jsonl(tmp_path / EVENTS_FILE)) == 1


def test_dres_failure_keeps_local_log(tmp_path):
    log = InteractionLog(tmp_path, RecordingDres(fail=True), evaluation_id="eval-1", clock=clock)
    log.event("search", "a blue boat")
    report = log.flush()
    assert report["dres_sent"] is False and "connection failed" in report["dres_error"]
    assert len(read_jsonl(tmp_path / EVENTS_FILE)) == 1
    assert log.flush()["events"] == 0  # not re-sent


def test_local_write_failure_keeps_records_buffered(tmp_path):
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("x", encoding="utf-8")
    log = InteractionLog(blocker, clock=clock)
    log.event("search", "a blue boat")
    with pytest.raises(OSError):
        log.flush()
    log.log_dir = tmp_path / "logs"
    assert log.flush()["events"] == 1
