import json

from fastapi.testclient import TestClient

from agentseek_api.services.run_state import run_broker
from agentseek_api.services.thread_protocol import thread_protocol_broker


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


def test_stream_rejects_malformed_last_event_id(client: TestClient) -> None:
    thread = client.post("/threads", json={"metadata": {"case": "bad-last-event-id"}})
    assert thread.status_code == 200

    response = client.post(
        f"/threads/{thread.json()['thread_id']}/stream",
        json={"channels": ["values"]},
        headers={"Last-Event-ID": "not-an-int"},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Last-Event-ID must be an integer event sequence."
