import json

from fastapi.testclient import TestClient
from sqlalchemy import func, select

from agentseek_api.api import runs as runs_api
from agentseek_api.api import threads as threads_api
from agentseek_api.core.database import db_manager
from agentseek_api.core.orm import Run, RunStreamEvent, Thread
from agentseek_api.models.auth import User
from agentseek_api.models.protocol import ProtocolEventStreamRequest
from agentseek_api.services import run_jobs as run_jobs_module
from agentseek_api.services import stream_persistence as stream_module
from agentseek_api.services.run_state import run_broker
from agentseek_api.services.thread_protocol import publish_values_event, thread_protocol_broker


class FakeRedisCounter:
    def __init__(self) -> None:
        self.counts: dict[str, int] = {}

    async def incr(self, key: str) -> int:
        value = self.counts.get(key, 0) + 1
        self.counts[key] = value
        return value


def _parse_sse(stream_text: str) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    for chunk in stream_text.strip().split("\n\n"):
        event: dict[str, object] = {}
        for line in chunk.splitlines():
            if line.startswith("id: "):
                event["id"] = line.removeprefix("id: ")
            elif line.startswith("event: "):
                event["event"] = line.removeprefix("event: ")
            elif line.startswith("data: "):
                event["data"] = json.loads(line.removeprefix("data: "))
        if event:
            events.append(event)
    return events


async def _seed_running_run(*, user_id: str = "default_user") -> tuple[str, str]:
    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        thread = Thread(user_id=user_id, metadata_json={"case": "redis-run-stream"}, config_json={}, status="busy")
        session.add(thread)
        await session.flush()
        run = Run(thread_id=thread.thread_id, assistant_id="assistant", user_id=user_id, status="running")
        session.add(run)
        await session.commit()
        return thread.thread_id, run.run_id


async def _seed_thread(*, user_id: str = "default_user") -> str:
    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        thread = Thread(user_id=user_id, metadata_json={"case": "redis-thread-stream"}, config_json={}, status="busy")
        session.add(thread)
        await session.commit()
        return thread.thread_id


async def _collect_stream_body(response: object) -> str:
    body_iterator = getattr(response, "body_iterator")
    chunks: list[str] = []
    async for chunk in body_iterator:
        if isinstance(chunk, bytes):
            chunks.append(chunk.decode())
        else:
            chunks.append(str(chunk))
    return "".join(chunks)


def test_run_stream_replays_persisted_events_after_broker_state_is_cleared(client: TestClient) -> None:
    assistant = client.post("/assistants", json={"name": "persisted-run-stream", "graph_id": "default"})
    assert assistant.status_code == 200
    thread = client.post("/threads", json={"metadata": {"case": "persisted-run-stream"}})
    assert thread.status_code == 200

    run = client.post(
        f"/threads/{thread.json()['thread_id']}/runs",
        json={"assistant_id": assistant.json()["assistant_id"], "input": {"message": "persist"}},
    )
    assert run.status_code == 200

    first = client.get(f"/threads/{thread.json()['thread_id']}/runs/{run.json()['run_id']}/stream")
    assert first.status_code == 200
    first_events = _parse_sse(first.text)
    first_event_id = first_events[0]["id"]
    run_broker._events.clear()
    run_broker._signals.clear()
    run_broker._completed_runs.clear()
    run_broker._completed_order.clear()

    replay = client.get(
        f"/threads/{thread.json()['thread_id']}/runs/{run.json()['run_id']}/stream",
        headers={"Last-Event-ID": str(first_event_id)},
    )

    assert replay.status_code == 200
    replay_events = _parse_sse(replay.text)
    assert replay_events
    assert all(int(str(event["id"])) > int(str(first_event_id)) for event in replay_events)
    assert replay_events[-1]["event"] == "end"


def test_run_stream_polls_persisted_events_in_redis_mode(client: TestClient, monkeypatch) -> None:
    thread_id, run_id = client.portal.call(_seed_running_run)
    load_calls = {"count": 0}

    async def fake_load_run_stream_events(requested_run_id: str, *, after_seq: int = 0) -> list[tuple[int, dict[str, object]]]:
        assert requested_run_id == run_id
        load_calls["count"] += 1
        if load_calls["count"] == 1:
            return []
        if load_calls["count"] == 2 and after_seq == 0:
            return [
                (1, {"event": "start"}),
                (2, {"event": "end", "status": "success"}),
            ]
        return []

    async def fake_is_run_terminal(*, run_id: str, thread_id: str, user_id: str) -> bool:
        _ = (run_id, thread_id, user_id)
        return load_calls["count"] >= 2

    def unexpected_stream_records(*args, **kwargs):
        _ = (args, kwargs)

        async def _iter():
            raise AssertionError("Redis stream path should not subscribe to the API-process run broker")
            yield 0, {}

        return _iter()

    monkeypatch.setattr(runs_api.settings, "EXECUTOR_BACKEND", "redis")
    monkeypatch.setattr(runs_api, "REDIS_STREAM_POLL_INTERVAL_SECONDS", 0)
    monkeypatch.setattr(runs_api, "load_run_stream_events", fake_load_run_stream_events)
    monkeypatch.setattr(runs_api, "_is_run_terminal", fake_is_run_terminal)
    monkeypatch.setattr(runs_api.run_broker, "snapshot_records", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(runs_api.run_broker, "stream_records", unexpected_stream_records)

    response = client.portal.call(
        runs_api.stream_run,
        thread_id,
        run_id,
        User(identity="default_user", is_authenticated=True),
        None,
    )

    body = client.portal.call(_collect_stream_body, response)
    events = _parse_sse(body)
    assert [event["event"] for event in events] == ["start", "end"]
    assert events[-1]["data"]["status"] == "success"


def test_protocol_stream_replays_persisted_events_after_broker_state_is_cleared(client: TestClient) -> None:
    assistant = client.post("/assistants", json={"name": "persisted-protocol", "graph_id": "react_agent"})
    assert assistant.status_code == 200
    thread = client.post("/threads", json={"metadata": {"case": "persisted-protocol"}})
    assert thread.status_code == 200
    thread_id = thread.json()["thread_id"]

    command = client.post(
        f"/threads/{thread_id}/commands",
        json={
            "id": 1,
            "method": "run.start",
            "params": {"assistant_id": assistant.json()["assistant_id"], "input": {"message": "persist protocol"}},
        },
    )
    assert command.status_code == 200

    first = client.post(f"/threads/{thread_id}/stream", json={"channels": ["lifecycle", "values"]})
    assert first.status_code == 200
    first_events = _parse_sse(first.text)
    first_event_id = first_events[0]["id"]
    thread_protocol_broker.delete_thread(thread_id)

    replay = client.post(
        f"/threads/{thread_id}/stream",
        json={"channels": ["lifecycle", "values"]},
        headers={"Last-Event-ID": str(first_event_id)},
    )

    assert replay.status_code == 200
    replay_events = _parse_sse(replay.text)
    assert replay_events
    assert all(int(str(event["id"])) > int(str(first_event_id)) for event in replay_events)
    assert "values" in {event["event"] for event in replay_events}


def test_thread_protocol_stream_polls_persisted_events_in_redis_mode(client: TestClient, monkeypatch) -> None:
    thread_id = client.portal.call(_seed_thread)
    load_calls = {"count": 0}

    async def fake_load_thread_stream_events(
        requested_thread_id: str,
        *,
        channels: list[str],
        namespaces: list[list[str]] | None,
        depth: int | None,
        after_seq: int = 0,
    ) -> list[dict[str, object]]:
        assert requested_thread_id == thread_id
        assert channels == ["lifecycle", "values"]
        assert namespaces is None
        assert depth is None
        load_calls["count"] += 1
        if load_calls["count"] == 1:
            return []
        if load_calls["count"] == 2 and after_seq == 0:
            return [
                {
                    "seq": 1,
                    "method": "values",
                    "params": {"namespace": [], "timestamp": 1, "data": {"phase": "mid-run"}},
                },
                {
                    "seq": 2,
                    "method": "lifecycle",
                    "params": {"namespace": [], "timestamp": 2, "data": {"event": "completed"}},
                },
            ]
        return []

    async def fake_thread_has_active_runs(*, thread_id: str, user_id: str) -> bool:
        _ = (thread_id, user_id)
        return load_calls["count"] < 2

    def unexpected_thread_stream(*args, **kwargs):
        _ = (args, kwargs)

        async def _iter():
            raise AssertionError("Redis protocol stream path should not subscribe to the API-process protocol broker")
            yield {}

        return _iter()

    monkeypatch.setattr(threads_api.settings, "EXECUTOR_BACKEND", "redis")
    monkeypatch.setattr(threads_api, "REDIS_STREAM_POLL_INTERVAL_SECONDS", 0)
    monkeypatch.setattr(threads_api, "load_thread_stream_events", fake_load_thread_stream_events)
    monkeypatch.setattr(threads_api, "_thread_has_active_runs", fake_thread_has_active_runs)
    monkeypatch.setattr(threads_api.thread_protocol_broker, "stream", unexpected_thread_stream)

    response = client.portal.call(
        threads_api.stream_thread_protocol_events,
        thread_id,
        ProtocolEventStreamRequest(channels=["lifecycle", "values"]),
        User(identity="default_user", is_authenticated=True),
        None,
    )

    body = client.portal.call(_collect_stream_body, response)
    events = _parse_sse(body)
    assert [event["event"] for event in events] == ["values", "lifecycle"]
    assert events[0]["data"]["params"]["data"] == {"phase": "mid-run"}
    assert events[1]["data"]["params"]["data"] == {"event": "completed"}


def test_run_stream_persistence_uses_shared_seq_after_broker_reset(client: TestClient, monkeypatch) -> None:
    fake_redis = FakeRedisCounter()
    monkeypatch.setattr(stream_module.settings, "EXECUTOR_BACKEND", "redis")
    monkeypatch.setattr(stream_module, "_redis_client", fake_redis)

    client.portal.call(run_jobs_module._publish_run_event, "run-seq-reset", "start")
    run_broker._events.clear()
    run_broker._seqs.clear()
    run_broker._signals.clear()
    run_broker._next_seq.clear()
    run_broker._completed_runs.clear()
    run_broker._completed_order.clear()
    client.portal.call(lambda: run_jobs_module._publish_run_event("run-seq-reset", "end", status="success"))

    persisted = client.portal.call(stream_module.load_run_stream_events, "run-seq-reset")

    assert [(seq, payload["event"]) for seq, payload in persisted] == [(1, "start"), (2, "end")]


def test_thread_stream_persistence_uses_shared_seq_after_broker_reset(client: TestClient, monkeypatch) -> None:
    fake_redis = FakeRedisCounter()
    monkeypatch.setattr(stream_module.settings, "EXECUTOR_BACKEND", "redis")
    monkeypatch.setattr(stream_module, "_redis_client", fake_redis)

    client.portal.call(lambda: run_jobs_module._publish_lifecycle("thread-seq-reset", event="started", graph_name="default"))
    thread_protocol_broker.delete_thread("thread-seq-reset")
    client.portal.call(
        lambda: run_jobs_module._publish_lifecycle(
            "thread-seq-reset",
            event="completed",
            graph_name="default",
        )
    )

    persisted = client.portal.call(
        lambda: stream_module.load_thread_stream_events(
            "thread-seq-reset",
            channels=["lifecycle"],
            namespaces=None,
            depth=None,
        )
    )

    assert [event["seq"] for event in persisted] == [1, 2]
    assert [event["params"]["data"]["event"] for event in persisted] == ["started", "completed"]


def test_protocol_events_are_persisted_when_published(client: TestClient) -> None:
    thread = client.post("/threads", json={"metadata": {"case": "protocol-publish-time"}})
    assert thread.status_code == 200
    thread_id = thread.json()["thread_id"]

    publish_values_event(thread_id, values={"early": True})
    thread_protocol_broker.delete_thread(thread_id)

    replay = client.post(f"/threads/{thread_id}/stream", json={"channels": ["values"]})

    assert replay.status_code == 200
    replay_events = _parse_sse(replay.text)
    assert [event["event"] for event in replay_events] == ["values"]
    assert replay_events[0]["data"]["params"]["data"] == {"early": True}


def test_thread_run_stream_uses_monotonic_ids_across_multiple_runs(client: TestClient) -> None:
    assistant = client.post("/assistants", json={"name": "thread-run-stream", "graph_id": "default"})
    assert assistant.status_code == 200
    thread = client.post("/threads", json={"metadata": {"case": "thread-run-stream"}})
    assert thread.status_code == 200
    thread_id = thread.json()["thread_id"]

    first = client.post(
        f"/threads/{thread_id}/runs",
        json={"assistant_id": assistant.json()["assistant_id"], "input": {"message": "first"}},
    )
    second = client.post(
        f"/threads/{thread_id}/runs",
        json={"assistant_id": assistant.json()["assistant_id"], "input": {"message": "second"}},
    )
    assert first.status_code == 200
    assert second.status_code == 200

    stream = client.get(f"/threads/{thread_id}/stream")
    assert stream.status_code == 200
    events = _parse_sse(stream.text)
    event_ids = [int(str(event["id"])) for event in events]
    assert event_ids == list(range(1, len(events) + 1))

    first_run_last_id = max(
        int(str(event["id"])) for event in events if event["data"]["run_id"] == first.json()["run_id"]
    )
    replay = client.get(f"/threads/{thread_id}/stream", headers={"Last-Event-ID": str(first_run_last_id)})
    assert replay.status_code == 200
    replay_events = _parse_sse(replay.text)
    assert replay_events
    assert {event["data"]["run_id"] for event in replay_events} == {second.json()["run_id"]}
    assert all(int(str(event["id"])) > first_run_last_id for event in replay_events)


async def _run_stream_event_count(run_ids: list[str]) -> int:
    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        return (
            await session.scalar(
                select(func.count()).select_from(RunStreamEvent).where(RunStreamEvent.run_id.in_(run_ids))
            )
            or 0
        )


def test_thread_prune_removes_persisted_run_stream_events(client: TestClient) -> None:
    assistant = client.post("/assistants", json={"name": "prune-stream-events", "graph_id": "default"})
    assert assistant.status_code == 200
    assistant_id = assistant.json()["assistant_id"]
    keep_latest_thread = client.post("/threads", json={"metadata": {"case": "prune-keep-latest-stream-events"}})
    delete_thread = client.post("/threads", json={"metadata": {"case": "prune-delete-stream-events"}})
    assert keep_latest_thread.status_code == 200
    assert delete_thread.status_code == 200

    first = client.post(
        f"/threads/{keep_latest_thread.json()['thread_id']}/runs",
        json={"assistant_id": assistant_id, "input": {"message": "first"}},
    )
    second = client.post(
        f"/threads/{keep_latest_thread.json()['thread_id']}/runs",
        json={"assistant_id": assistant_id, "input": {"message": "second"}},
    )
    deleted = client.post(
        f"/threads/{delete_thread.json()['thread_id']}/runs",
        json={"assistant_id": assistant_id, "input": {"message": "delete"}},
    )
    assert first.status_code == 200
    assert second.status_code == 200
    assert deleted.status_code == 200

    assert client.post(
        "/threads/prune",
        json={"thread_ids": [keep_latest_thread.json()["thread_id"]], "strategy": "keep_latest"},
    ).status_code == 200
    assert client.post(
        "/threads/prune",
        json={"thread_ids": [delete_thread.json()["thread_id"]], "strategy": "delete"},
    ).status_code == 200

    first_run_id = first.json()["run_id"]
    second_run_id = second.json()["run_id"]
    deleted_run_id = deleted.json()["run_id"]
    assert client.portal.call(_run_stream_event_count, [first_run_id]) == 0
    assert client.portal.call(_run_stream_event_count, [deleted_run_id]) == 0
    assert client.portal.call(_run_stream_event_count, [second_run_id]) > 0


def test_stream_rejects_malformed_last_event_id(client: TestClient) -> None:
    thread = client.post("/threads", json={"metadata": {"case": "bad-last-event-id"}})
    assert thread.status_code == 200

    protocol_response = client.post(
        f"/threads/{thread.json()['thread_id']}/stream",
        json={"channels": ["values"]},
        headers={"Last-Event-ID": "not-an-int"},
    )

    run_stream_response = client.get(
        f"/threads/{thread.json()['thread_id']}/stream",
        headers={"Last-Event-ID": "not-an-int"},
    )

    assert protocol_response.status_code == 400
    assert protocol_response.json()["detail"] == "Last-Event-ID must be an integer event sequence."
    assert run_stream_response.status_code == 400
    assert run_stream_response.json()["detail"] == "Last-Event-ID must be an integer event sequence."
