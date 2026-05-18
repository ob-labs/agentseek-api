import json

from fastapi.testclient import TestClient
from sqlalchemy import func, select

from agentseek_api.core.database import db_manager
from agentseek_api.core.orm import RunStreamEvent
from agentseek_api.services.run_state import run_broker
from agentseek_api.services.thread_protocol import publish_values_event, thread_protocol_broker


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
