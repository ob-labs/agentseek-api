from __future__ import annotations

import json
from collections.abc import Awaitable, Callable

from fastapi.testclient import TestClient

from agentseek_api.services.run_jobs import RunExecutionJob
from agentseek_api.services.run_preparation import ActiveThreadRunConflictError


def _parse_sse_events(stream_text: str) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    for chunk in stream_text.strip().split("\n\n"):
        if not chunk.strip():
            continue
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


def test_protocol_run_start_replays_messages_tools_values_and_lifecycle(client: TestClient) -> None:
    assistant = client.post("/assistants", json={"name": "protocol-react", "graph_id": "react_agent"})
    assert assistant.status_code == 200
    assistant_id = assistant.json()["assistant_id"]

    thread = client.post("/threads", json={"metadata": {"case": "protocol-react"}})
    assert thread.status_code == 200
    thread_id = thread.json()["thread_id"]

    command = client.post(
        f"/threads/{thread_id}/commands",
        json={
            "id": 1,
            "method": "run.start",
            "params": {
                "assistant_id": assistant_id,
                "input": {"message": "what is the meaning of life?"},
            },
        },
    )
    assert command.status_code == 200
    command_body = command.json()
    assert command_body["type"] == "success"
    assert command_body["id"] == 1
    assert command_body["result"]["run_id"]

    stream = client.post(
        f"/threads/{thread_id}/stream/events",
        json={"channels": ["lifecycle", "messages", "tools", "values"]},
    )
    assert stream.status_code == 200
    events = _parse_sse_events(stream.text)

    event_methods = [event["event"] for event in events]
    assert "lifecycle" in event_methods
    assert "messages" in event_methods
    assert "tools" in event_methods
    assert "values" in event_methods

    lifecycle_payloads = [event["data"] for event in events if event["event"] == "lifecycle"]
    lifecycle_states = [payload["params"]["data"]["event"] for payload in lifecycle_payloads]
    assert lifecycle_states == ["started", "completed"]

    tool_payloads = [event["data"] for event in events if event["event"] == "tools"]
    assert any(payload["params"]["data"]["event"] == "tool-started" for payload in tool_payloads)
    assert any(payload["params"]["data"]["event"] == "tool-finished" for payload in tool_payloads)

    message_payloads = [event["data"] for event in events if event["event"] == "messages"]
    assert any(payload["params"]["data"]["event"] == "message-start" for payload in message_payloads)
    assert any(payload["params"]["data"]["event"] == "message-finish" for payload in message_payloads)

    values_payloads = [event["data"] for event in events if event["event"] == "values"]
    final_values = values_payloads[-1]["params"]["data"]
    assert "messages" in final_values

    last_seq = int([event for event in events if event["event"] == "values"][-1]["id"])
    replay = client.post(
        f"/threads/{thread_id}/stream/events",
        json={"channels": ["values"], "since": last_seq - 1},
    )
    assert replay.status_code == 200
    replay_events = _parse_sse_events(replay.text)
    assert len(replay_events) == 1
    assert replay_events[0]["event"] == "values"
    assert replay_events[0]["id"] == str(last_seq)


def test_protocol_run_start_rejects_busy_thread(client: TestClient, monkeypatch) -> None:
    class DeferredExecutor:
        def __init__(self) -> None:
            self.submitted: list[Callable[[], Awaitable[None]] | RunExecutionJob] = []

        async def submit(self, job: Callable[[], Awaitable[None]] | RunExecutionJob) -> None:
            self.submitted.append(job)

    monkeypatch.setattr("agentseek_api.services.run_preparation.get_executor", lambda: DeferredExecutor())

    assistant = client.post("/assistants", json={"name": "protocol-busy", "graph_id": "stress_test"})
    assert assistant.status_code == 200
    assistant_id = assistant.json()["assistant_id"]

    thread = client.post("/threads", json={"metadata": {"case": "protocol-busy"}})
    assert thread.status_code == 200
    thread_id = thread.json()["thread_id"]

    first = client.post(
        f"/threads/{thread_id}/commands",
        json={
            "id": 30,
            "method": "run.start",
            "params": {
                "assistant_id": assistant_id,
                "input": {"delay": 0.05, "steps": 20},
            },
        },
    )
    assert first.status_code == 200
    assert first.json()["type"] == "success"

    second = client.post(
        f"/threads/{thread_id}/commands",
        json={
            "id": 31,
            "method": "run.start",
            "params": {
                "assistant_id": assistant_id,
                "input": {"delay": 0.05, "steps": 20},
            },
        },
    )
    assert second.status_code == 409
    assert second.json()["type"] == "error"
    assert second.json()["error"] == "thread_busy"
    assert second.json()["message"] == "Another run is already active for this thread"


def test_protocol_input_respond_resumes_interrupted_run(client: TestClient) -> None:
    assistant = client.post("/assistants", json={"name": "protocol-hitl", "graph_id": "subgraph_hitl_agent"})
    assert assistant.status_code == 200
    assistant_id = assistant.json()["assistant_id"]

    thread = client.post("/threads", json={"metadata": {"case": "protocol-hitl"}})
    assert thread.status_code == 200
    thread_id = thread.json()["thread_id"]

    command = client.post(
        f"/threads/{thread_id}/commands",
        json={
            "id": 10,
            "method": "run.start",
            "params": {
                "assistant_id": assistant_id,
                "input": {"foo": "hello "},
            },
        },
    )
    assert command.status_code == 200
    run_id = command.json()["result"]["run_id"]

    interrupted_stream = client.post(
        f"/threads/{thread_id}/stream/events",
        json={"channels": ["lifecycle", "input", "values"]},
    )
    assert interrupted_stream.status_code == 200
    interrupted_events = _parse_sse_events(interrupted_stream.text)

    input_events = [event for event in interrupted_events if event["event"] == "input.requested"]
    assert len(input_events) == 1
    interrupt_payload = input_events[0]["data"]["params"]["data"]
    assert interrupt_payload["payload"] == "Provide value:"
    interrupt_id = interrupt_payload["interrupt_id"]

    interrupted_lifecycle = [event["data"]["params"]["data"]["event"] for event in interrupted_events if event["event"] == "lifecycle"]
    assert interrupted_lifecycle == ["started", "interrupted"]

    last_seq = int(interrupted_events[-1]["id"])
    respond = client.post(
        f"/threads/{thread_id}/commands",
        json={
            "id": 11,
            "method": "input.respond",
            "params": {
                "namespace": [],
                "interrupt_id": interrupt_id,
                "response": "world",
            },
        },
    )
    assert respond.status_code == 200
    assert respond.json()["type"] == "success"
    assert respond.json()["id"] == 11

    resumed_stream = client.post(
        f"/threads/{thread_id}/stream/events",
        json={"channels": ["lifecycle", "values"], "since": last_seq},
    )
    assert resumed_stream.status_code == 200
    resumed_events = _parse_sse_events(resumed_stream.text)

    resumed_lifecycle = [event["data"]["params"]["data"]["event"] for event in resumed_events if event["event"] == "lifecycle"]
    assert resumed_lifecycle == ["started", "completed"]

    resumed_values = [event["data"]["params"]["data"] for event in resumed_events if event["event"] == "values"]
    assert resumed_values[-1]["foo"].endswith("world")

    waited = client.get(f"/threads/{thread_id}/runs/{run_id}/wait")
    assert waited.status_code == 200
    assert waited.json()["status"] == "success"


def test_protocol_input_respond_maps_resume_conflict_to_thread_busy(client: TestClient, monkeypatch) -> None:
    assistant = client.post("/assistants", json={"name": "protocol-hitl-busy", "graph_id": "subgraph_hitl_agent"})
    assert assistant.status_code == 200
    assistant_id = assistant.json()["assistant_id"]

    thread = client.post("/threads", json={"metadata": {"case": "protocol-hitl-busy"}})
    assert thread.status_code == 200
    thread_id = thread.json()["thread_id"]

    command = client.post(
        f"/threads/{thread_id}/commands",
        json={
            "id": 40,
            "method": "run.start",
            "params": {
                "assistant_id": assistant_id,
                "input": {"foo": "hello "},
            },
        },
    )
    assert command.status_code == 200

    interrupted_stream = client.post(
        f"/threads/{thread_id}/stream/events",
        json={"channels": ["input"]},
    )
    assert interrupted_stream.status_code == 200
    interrupted_events = _parse_sse_events(interrupted_stream.text)
    interrupt_payload = next(event for event in interrupted_events if event["event"] == "input.requested")["data"]["params"]["data"]
    interrupt_id = interrupt_payload["interrupt_id"]

    async def fail_resume(**_kwargs) -> None:
        raise ActiveThreadRunConflictError("Another run is already active for this thread")

    monkeypatch.setattr("agentseek_api.api.streaming.resume_run", fail_resume)

    response = client.post(
        f"/threads/{thread_id}/commands",
        json={
            "id": 41,
            "method": "input.respond",
            "params": {
                "namespace": [],
                "interrupt_id": interrupt_id,
                "response": "world",
            },
        },
    )
    assert response.status_code == 409
    assert response.json()["type"] == "error"
    assert response.json()["error"] == "thread_busy"
    assert response.json()["message"] == "Another run is already active for this thread"


def test_protocol_stream_filters_subgraph_namespace_events(client: TestClient) -> None:
    assistant = client.post("/assistants", json={"name": "protocol-subgraph-namespace", "graph_id": "subgraph_hitl_agent"})
    assert assistant.status_code == 200
    assistant_id = assistant.json()["assistant_id"]

    thread = client.post("/threads", json={"metadata": {"case": "protocol-subgraph-namespace"}})
    assert thread.status_code == 200
    thread_id = thread.json()["thread_id"]

    command = client.post(
        f"/threads/{thread_id}/commands",
        json={
            "id": 20,
            "method": "run.start",
            "params": {
                "assistant_id": assistant_id,
                "input": {"foo": "hello "},
            },
        },
    )
    assert command.status_code == 200
    assert command.json()["type"] == "success"

    unfiltered_stream = client.post(
        f"/threads/{thread_id}/stream/events",
        json={"channels": ["updates", "input"]},
    )
    assert unfiltered_stream.status_code == 200
    unfiltered_events = _parse_sse_events(unfiltered_stream.text)
    update_event = next(event for event in unfiltered_events if event["event"] == "updates")
    namespace_prefix = update_event["data"]["params"]["namespace"][:1]

    stream = client.post(
        f"/threads/{thread_id}/stream/events",
        json={"channels": ["updates", "input"], "namespaces": [namespace_prefix]},
    )
    assert stream.status_code == 200

    events = _parse_sse_events(stream.text)
    assert events
    assert {event["event"] for event in events} <= {"updates", "input.requested"}
    assert all(event["data"]["params"]["namespace"][:1] == namespace_prefix for event in events)
