import asyncio
from collections.abc import Awaitable, Callable
import json

from fastapi.testclient import TestClient
from langchain_core.messages import HumanMessage
from sqlalchemy import select

from agentseek_api.core.database import db_manager
from agentseek_api.core.orm import Run
from agentseek_api.main import app
from agentseek_api.models.api import RunRead
from agentseek_api.services.run_jobs import RunExecutionJob


def _create_assistant(client: TestClient, *, graph_id: str = "default") -> str:
    response = client.post("/assistants", json={"name": f"{graph_id}-assistant", "graph_id": graph_id})
    assert response.status_code == 200
    return response.json()["assistant_id"]


def _create_thread(client: TestClient, *, user_id: str = "default_user") -> str:
    response = client.post("/threads", json={"metadata": {"compat": True}}, headers={"x-user-id": user_id})
    assert response.status_code == 200
    return response.json()["thread_id"]


class DeferredExecutor:
    def __init__(self) -> None:
        self.submitted: list[Callable[[], Awaitable[None]] | RunExecutionJob] = []

    async def submit(self, job: Callable[[], Awaitable[None]] | RunExecutionJob) -> None:
        self.submitted.append(job)


def _parse_sse_events(stream_text: str) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    current: dict[str, object] = {}
    for line in stream_text.splitlines():
        if not line:
            if current:
                events.append(current)
                current = {}
            continue
        if line.startswith("id: "):
            current["id"] = line.removeprefix("id: ")
        elif line.startswith("event: "):
            current["event"] = line.removeprefix("event: ")
        elif line.startswith("data: "):
            current["data"] = json.loads(line.removeprefix("data: "))
    if current:
        events.append(current)
    return events


async def _raw_run_metadata(*, thread_id: str, run_id: str) -> dict[str, object]:
    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        row = await session.scalar(select(Run).where(Run.thread_id == thread_id, Run.run_id == run_id))
        assert row is not None
        return row.metadata_json


def test_thread_run_wait_and_stream_creation_routes(client: TestClient) -> None:
    assistant_id = _create_assistant(client)
    thread_id = _create_thread(client)

    waited = client.post(
        f"/threads/{thread_id}/runs/wait",
        json={"assistant_id": assistant_id, "input": {"message": "wait route"}},
    )
    assert waited.status_code == 200
    waited_body = waited.json()
    assert "run_id" not in waited_body
    assert waited_body["input"] == {"message": "wait route"}
    assert waited_body["output"] == {"echo": {"message": "wait route"}}
    wait_run_id = waited.headers["content-location"].rpartition("/")[2]
    assert waited.headers["content-location"] == f"/threads/{thread_id}/runs/{wait_run_id}"
    assert waited.headers["location"] == f"/threads/{thread_id}/runs/{wait_run_id}/join"
    joined_wait = client.get(waited.headers["location"])
    assert joined_wait.status_code == 200
    assert joined_wait.headers["content-location"] == waited.headers["content-location"]
    assert joined_wait.json() == waited_body

    streamed = client.post(
        f"/threads/{thread_id}/runs/stream",
        json={"assistant_id": assistant_id, "input": {"message": "stream route"}, "stream_mode": "updates"},
    )
    assert streamed.status_code == 200
    assert streamed.headers["content-type"].startswith("text/event-stream")
    events = _parse_sse_events(streamed.text)
    run_id = next(event["data"]["run_id"] for event in events if event["event"] == "metadata")
    assert streamed.headers["location"] == f"/threads/{thread_id}/runs/{run_id}/stream?stream_mode=updates"
    assert streamed.headers["content-location"] == f"/threads/{thread_id}/runs/{run_id}"
    assert "event: metadata" in streamed.text
    assert "event: updates" in streamed.text
    assert "event: start" not in streamed.text
    assert "event: message_chunk" not in streamed.text

    joined = client.get(streamed.headers["location"])
    assert joined.status_code == 200
    joined_events = _parse_sse_events(joined.text)
    assert [event["event"] for event in joined_events] == ["metadata"]
    assert "event: start" not in joined.text
    assert "event: message_chunk" not in joined.text

    replayed = client.get(streamed.headers["location"], headers={"Last-Event-ID": "0"})
    assert replayed.status_code == 200
    replayed_events = _parse_sse_events(replayed.text)
    assert "updates" in {event["event"] for event in replayed_events}


def test_join_run_stays_bound_to_waited_run_after_newer_run_on_same_thread(client: TestClient) -> None:
    assistant_id = _create_assistant(client)
    thread_id = _create_thread(client)

    first = client.post(
        f"/threads/{thread_id}/runs/wait",
        json={"assistant_id": assistant_id, "input": {"message": "first"}},
    )
    assert first.status_code == 200
    first_body = first.json()

    second = client.post(
        f"/threads/{thread_id}/runs",
        json={"assistant_id": assistant_id, "input": {"message": "second"}},
    )
    assert second.status_code == 200
    second_run_id = second.json()["run_id"]

    joined_first = client.get(first.headers["location"])
    joined_second = client.get(f"/threads/{thread_id}/runs/{second_run_id}/join")

    assert joined_first.status_code == 200
    assert joined_first.json() == first_body
    assert joined_second.status_code == 200
    assert joined_second.json()["output"] == {"echo": {"message": "second"}}


def test_run_read_metadata_hides_internal_checkpoint_id(client: TestClient) -> None:
    assistant_id = _create_assistant(client)
    thread_id = _create_thread(client)

    created = client.post(
        f"/threads/{thread_id}/runs",
        json={
            "assistant_id": assistant_id,
            "input": {"message": "metadata visibility"},
            "metadata": {"visible": "yes"},
        },
    )
    assert created.status_code == 200
    run_id = created.json()["run_id"]

    raw_metadata = asyncio.run(_raw_run_metadata(thread_id=thread_id, run_id=run_id))
    assert raw_metadata["visible"] == "yes"
    assert isinstance(raw_metadata["__agentseek_checkpoint_id"], str)

    public_created = created.json()["metadata"]
    assert public_created == {"visible": "yes"}

    fetched = client.get(f"/threads/{thread_id}/runs/{run_id}")
    assert fetched.status_code == 200
    assert fetched.json()["metadata"] == {"visible": "yes"}

    listed = client.get(f"/threads/{thread_id}/runs")
    assert listed.status_code == 200
    listed_run = next(run for run in listed.json() if run["run_id"] == run_id)
    assert listed_run["metadata"] == {"visible": "yes"}


def test_stateless_wait_stream_and_batch_routes(client: TestClient) -> None:
    assistant_id = _create_assistant(client)

    waited = client.post("/runs/wait", json={"assistant_id": assistant_id, "input": {"message": "wait"}})
    assert waited.status_code == 200
    waited_body = waited.json()
    assert "run_id" not in waited_body
    assert waited_body["input"] == {"message": "wait"}
    assert waited_body["output"] == {"echo": {"message": "wait"}}
    stateless_wait_path = waited.headers["content-location"].strip("/").split("/")
    assert stateless_wait_path[0] == "threads"
    assert stateless_wait_path[2] == "runs"
    assert waited.headers["location"] == f"/threads/{stateless_wait_path[1]}/runs/{stateless_wait_path[3]}/join"
    joined_wait = client.get(waited.headers["location"])
    assert joined_wait.status_code == 200
    assert joined_wait.headers["content-location"] == waited.headers["content-location"]
    assert joined_wait.json() == waited_body

    streamed = client.post(
        "/runs/stream",
        json={"assistant_id": assistant_id, "input": {"message": "stream"}, "stream_mode": "updates"},
    )
    assert streamed.status_code == 200
    assert streamed.headers["content-type"].startswith("text/event-stream")
    events = _parse_sse_events(streamed.text)
    run_id = next(event["data"]["run_id"] for event in events if event["event"] == "metadata")
    stateless_stream_path = streamed.headers["content-location"].strip("/").split("/")
    assert streamed.headers["location"] == (
        f"/threads/{stateless_stream_path[1]}/runs/{run_id}/stream?stream_mode=updates"
    )
    assert streamed.headers["content-location"] == f"/threads/{stateless_stream_path[1]}/runs/{run_id}"
    assert "event: metadata" in streamed.text
    assert "event: updates" in streamed.text

    joined = client.get(streamed.headers["location"])
    assert joined.status_code == 200
    joined_events = _parse_sse_events(joined.text)
    assert [event["event"] for event in joined_events] == ["metadata"]
    fetched = client.get(streamed.headers["content-location"])
    assert fetched.status_code == 200
    assert fetched.json()["run_id"] == run_id

    batch = client.post(
        "/runs/batch",
        json=[
            {"assistant_id": assistant_id, "input": {"message": "one"}},
            {"assistant_id": assistant_id, "input": {"message": "two"}},
        ],
    )
    assert batch.status_code == 200
    body = batch.json()
    assert len(body) == 2
    assert body[0]["status"] == "success"
    assert body[1]["status"] == "success"


def test_cancel_routes(client: TestClient, monkeypatch) -> None:
    monkeypatch.setattr("agentseek_api.services.run_preparation.get_executor", lambda: DeferredExecutor())
    assistant_id = _create_assistant(client, graph_id="stress_test")
    thread_id = _create_thread(client)
    created = client.post(
        f"/threads/{thread_id}/runs",
        json={"assistant_id": assistant_id, "input": {"delay": 0.05, "steps": 20}},
    )
    assert created.status_code == 200
    run_id = created.json()["run_id"]

    cancel_one = client.post(f"/threads/{thread_id}/runs/{run_id}/cancel")
    assert cancel_one.status_code == 200
    waited = client.get(f"/threads/{thread_id}/runs/{run_id}/wait")
    assert waited.status_code == 200
    assert waited.json()["status"] == "error"
    thread = client.get(f"/threads/{thread_id}")
    assert thread.status_code == 200
    assert thread.json()["status"] == "error"

    cancel_many = client.post("/runs/cancel", json={"thread_id": thread_id, "run_ids": [run_id]})
    assert cancel_many.status_code == 204


def test_create_run_stream_rejects_unsupported_stream_modes(client: TestClient) -> None:
    assistant_id = _create_assistant(client)
    thread_id = _create_thread(client)
    before_threads = client.post("/threads/search", json={}).json()

    response = client.post(
        f"/threads/{thread_id}/runs/stream",
        json={"assistant_id": assistant_id, "input": {"message": "bad mode"}, "stream_mode": "nonexistent"},
    )

    assert response.status_code == 422
    assert client.get(f"/threads/{thread_id}/runs").json() == []

    stateless_response = client.post(
        "/runs/stream",
        json={"assistant_id": assistant_id, "input": {"message": "bad stateless mode"}, "stream_mode": "nonexistent"},
    )
    assert stateless_response.status_code == 422
    assert client.post("/threads/search", json={}).json() == before_threads


def test_create_run_wait_rejects_unsupported_stream_modes_before_side_effects(client: TestClient) -> None:
    assistant_id = _create_assistant(client)
    thread_id = _create_thread(client)
    before_threads = client.post("/threads/search", json={}).json()

    response = client.post(
        f"/threads/{thread_id}/runs/wait",
        json={"assistant_id": assistant_id, "input": {"message": "bad mode"}, "stream_mode": "nonexistent"},
    )

    assert response.status_code == 422
    assert client.get(f"/threads/{thread_id}/runs").json() == []

    stateless_response = client.post(
        "/runs/wait",
        json={"assistant_id": assistant_id, "input": {"message": "bad stateless mode"}, "stream_mode": "nonexistent"},
    )
    assert stateless_response.status_code == 422
    assert client.post("/threads/search", json={}).json() == before_threads


def test_create_run_stream_messages_mode_emits_message_events(client: TestClient) -> None:
    assistant_id = _create_assistant(client, graph_id="react_agent")
    thread_id = _create_thread(client)

    response = client.post(
        f"/threads/{thread_id}/runs/stream",
        json={"assistant_id": assistant_id, "input": {"message": "stream route"}, "stream_mode": "messages"},
    )

    assert response.status_code == 200
    events = _parse_sse_events(response.text)
    assert any(event["event"] == "metadata" for event in events)


def test_create_run_stream_messages_tuple_mode_aliases_to_messages(client: TestClient) -> None:
    assistant_id = _create_assistant(client, graph_id="react_agent")
    thread_id = _create_thread(client)

    response = client.post(
        f"/threads/{thread_id}/runs/stream",
        json={"assistant_id": assistant_id, "input": {"message": "tuple route"}, "stream_mode": "messages-tuple"},
    )

    assert response.status_code == 200
    events = _parse_sse_events(response.text)
    assert any(event["event"] == "metadata" for event in events)


def test_join_stream_accepts_official_json_array_stream_mode_query_without_replay(client: TestClient) -> None:
    assistant_id = _create_assistant(client, graph_id="react_agent")
    thread_id = _create_thread(client)
    created = client.post(
        f"/threads/{thread_id}/runs",
        json={"assistant_id": assistant_id, "input": {"message": "json array query"}},
    )
    assert created.status_code == 200
    run_id = created.json()["run_id"]

    streamed = client.get(
        f"/threads/{thread_id}/runs/{run_id}/stream",
        params={"stream_mode": json.dumps(["messages-tuple"])},
    )

    assert streamed.status_code == 200
    events = _parse_sse_events(streamed.text)
    assert [event["event"] for event in events] == ["metadata"]


def test_join_stream_rejects_blank_stream_mode_query(client: TestClient) -> None:
    assistant_id = _create_assistant(client)
    thread_id = _create_thread(client)
    created = client.post(
        f"/threads/{thread_id}/runs",
        json={"assistant_id": assistant_id, "input": {"message": "blank query"}},
    )
    assert created.status_code == 200
    run_id = created.json()["run_id"]

    streamed = client.get(
        f"/threads/{thread_id}/runs/{run_id}/stream",
        params={"stream_mode": ""},
    )

    assert streamed.status_code == 422
    assert "Unsupported stream_mode value(s)" in streamed.json()["detail"]


def test_create_run_wait_and_stream_accept_official_contract_fields(client: TestClient) -> None:
    assistant_id = _create_assistant(client)
    thread_id = _create_thread(client)

    thread_wait = client.post(
        f"/threads/{thread_id}/runs/wait",
        json={
            "assistant_id": assistant_id,
            "on_disconnect": "continue",
            "durability": "async",
            "input": {"message": "official stateful wait"},
        },
    )
    assert thread_wait.status_code == 200
    assert thread_wait.json()["output"] == {"echo": {"message": "official stateful wait"}}

    stateless_wait = client.post(
        "/runs/wait",
        json={
            "assistant_id": assistant_id,
            "on_disconnect": "continue",
            "on_completion": "keep",
            "durability": "async",
            "input": {"message": "official stateless wait"},
        },
    )
    assert stateless_wait.status_code == 200
    assert stateless_wait.json()["output"] == {"echo": {"message": "official stateless wait"}}


def test_create_run_rejects_unsupported_official_control_fields_before_side_effects(client: TestClient) -> None:
    assistant_id = _create_assistant(client)
    thread_id = _create_thread(client)
    before_threads = client.post("/threads/search", json={}).json()

    response = client.post(
        f"/threads/{thread_id}/runs",
        json={
            "assistant_id": assistant_id,
            "webhook": "https://example.com/hook",
            "feedback_keys": ["thumbs-up"],
            "input": {"message": "bad controls"},
        },
    )

    assert response.status_code == 422
    assert "Unsupported run control field(s)" in response.json()["detail"]
    assert "webhook" in response.json()["detail"]
    assert "feedback_keys" in response.json()["detail"]
    assert client.get(f"/threads/{thread_id}/runs").json() == []

    stateless_response = client.post(
        "/runs",
        json={
            "assistant_id": assistant_id,
            "on_completion": "delete",
            "input": {"message": "bad stateless controls"},
        },
    )
    assert stateless_response.status_code == 422
    assert "Unsupported run control field(s)" in stateless_response.json()["detail"]
    assert "on_completion" in stateless_response.json()["detail"]
    assert client.post("/threads/search", json={}).json() == before_threads


def test_create_run_wait_rejects_unsupported_official_control_fields_before_side_effects(client: TestClient) -> None:
    assistant_id = _create_assistant(client)
    thread_id = _create_thread(client)
    before_threads = client.post("/threads/search", json={}).json()

    response = client.post(
        f"/threads/{thread_id}/runs/wait",
        json={
            "assistant_id": assistant_id,
            "feedback_keys": ["thumbs-up"],
            "input": {"message": "bad wait controls"},
        },
    )

    assert response.status_code == 422
    assert "Unsupported run control field(s)" in response.json()["detail"]
    assert "feedback_keys" in response.json()["detail"]
    assert client.get(f"/threads/{thread_id}/runs").json() == []

    stateless_response = client.post(
        "/runs/wait",
        json={
            "assistant_id": assistant_id,
            "after_seconds": 5,
            "input": {"message": "bad stateless wait controls"},
        },
    )
    assert stateless_response.status_code == 422
    assert "Unsupported run control field(s)" in stateless_response.json()["detail"]
    assert "after_seconds" in stateless_response.json()["detail"]
    assert client.post("/threads/search", json={}).json() == before_threads


def test_create_run_stream_rejects_unsupported_official_control_fields_before_side_effects(client: TestClient) -> None:
    assistant_id = _create_assistant(client)
    thread_id = _create_thread(client)
    before_threads = client.post("/threads/search", json={}).json()

    response = client.post(
        f"/threads/{thread_id}/runs/stream",
        json={
            "assistant_id": assistant_id,
            "after_seconds": 1,
            "input": {"message": "bad stream controls"},
        },
    )

    assert response.status_code == 422
    assert "Unsupported run control field(s)" in response.json()["detail"]
    assert "after_seconds" in response.json()["detail"]
    assert client.get(f"/threads/{thread_id}/runs").json() == []

    stateless_response = client.post(
        "/runs/stream",
        json={
            "assistant_id": assistant_id,
            "on_completion": "delete",
            "input": {"message": "bad stateless stream controls"},
        },
    )
    assert stateless_response.status_code == 422
    assert "Unsupported run control field(s)" in stateless_response.json()["detail"]
    assert "on_completion" in stateless_response.json()["detail"]
    assert client.post("/threads/search", json={}).json() == before_threads


def test_create_run_rejects_unknown_control_fields_before_side_effects(client: TestClient) -> None:
    assistant_id = _create_assistant(client)
    thread_id = _create_thread(client)
    before_threads = client.post("/threads/search", json={}).json()

    response = client.post(
        f"/threads/{thread_id}/runs",
        json={
            "assistant_id": assistant_id,
            "stream_mod": "updates",
            "input": {"message": "bad unknown create control"},
        },
    )

    assert response.status_code == 422
    assert client.get(f"/threads/{thread_id}/runs").json() == []

    stateless_response = client.post(
        "/runs",
        json={
            "assistant_id": assistant_id,
            "onDisconect": "cancel",
            "input": {"message": "bad unknown stateless create control"},
        },
    )

    assert stateless_response.status_code == 422
    assert client.post("/threads/search", json={}).json() == before_threads


def test_create_run_wait_rejects_unknown_control_fields_before_side_effects(client: TestClient) -> None:
    assistant_id = _create_assistant(client)
    thread_id = _create_thread(client)
    before_threads = client.post("/threads/search", json={}).json()

    response = client.post(
        f"/threads/{thread_id}/runs/wait",
        json={
            "assistant_id": assistant_id,
            "stream_mod": "updates",
            "input": {"message": "bad unknown wait control"},
        },
    )

    assert response.status_code == 422
    assert client.get(f"/threads/{thread_id}/runs").json() == []

    stateless_response = client.post(
        "/runs/wait",
        json={
            "assistant_id": assistant_id,
            "onDisconect": "cancel",
            "input": {"message": "bad unknown stateless wait control"},
        },
    )

    assert stateless_response.status_code == 422
    assert client.post("/threads/search", json={}).json() == before_threads


def test_create_run_stream_rejects_unknown_control_fields_before_side_effects(client: TestClient) -> None:
    assistant_id = _create_assistant(client)
    thread_id = _create_thread(client)
    before_threads = client.post("/threads/search", json={}).json()

    response = client.post(
        f"/threads/{thread_id}/runs/stream",
        json={
            "assistant_id": assistant_id,
            "stream_mod": "updates",
            "input": {"message": "bad unknown stream control"},
        },
    )

    assert response.status_code == 422
    assert client.get(f"/threads/{thread_id}/runs").json() == []

    stateless_response = client.post(
        "/runs/stream",
        json={
            "assistant_id": assistant_id,
            "onDisconect": "cancel",
            "input": {"message": "bad unknown stateless stream control"},
        },
    )

    assert stateless_response.status_code == 422
    assert client.post("/threads/search", json={}).json() == before_threads


def test_create_run_stream_filters_protocol_events_to_created_run(client: TestClient, monkeypatch) -> None:
    assistant_id = _create_assistant(client)
    thread_id = _create_thread(client)
    created = RunRead.model_validate(
        {
            "run_id": "created-run",
            "thread_id": thread_id,
            "assistant_id": assistant_id,
            "status": "success",
            "output": {"ok": True},
            "metadata": {},
            "kwargs": {},
            "multitask_strategy": "enqueue",
        }
    )

    async def fake_create_run(*args, **kwargs):
        return created

    async def fake_stream(*args, **kwargs):
        yield {
            "seq": 1,
            "method": "updates",
            "params": {"run_id": "foreign-run", "data": {"output": {"echo": {"message": "foreign"}}}},
        }
        yield {
            "seq": 2,
            "method": "updates",
            "params": {"run_id": "created-run", "data": {"output": {"echo": {"message": "created"}}}},
        }

    monkeypatch.setattr("agentseek_api.api.runs.create_run", fake_create_run)
    monkeypatch.setattr("agentseek_api.api.runs.thread_protocol_broker.latest_seq", lambda _thread_id: 0)
    monkeypatch.setattr("agentseek_api.api.runs.thread_protocol_broker.stream", fake_stream)

    response = client.post(
        f"/threads/{thread_id}/runs/stream",
        json={"assistant_id": assistant_id, "input": {"message": "filtered"}, "stream_mode": "updates"},
    )

    assert response.status_code == 200
    events = _parse_sse_events(response.text)
    update_events = [event["data"] for event in events if event["event"] == "updates"]
    assert update_events == [{"output": {"echo": {"message": "created"}}}]


def test_create_run_stream_values_mode_surfaces_interrupt_payload(client: TestClient) -> None:
    assistant_id = _create_assistant(client, graph_id="subgraph_hitl_agent")
    thread_id = _create_thread(client)

    response = client.post(
        f"/threads/{thread_id}/runs/stream",
        json={"assistant_id": assistant_id, "input": {"foo": "hello "}, "stream_mode": "values"},
    )

    assert response.status_code == 200
    events = _parse_sse_events(response.text)
    values_events = [event for event in events if event["event"] == "values"]
    assert values_events
    assert values_events[-1]["data"]["__interrupt__"][0]["value"] == "Provide value:"


def test_create_run_stream_updates_mode_surfaces_interrupt_payload(client: TestClient) -> None:
    assistant_id = _create_assistant(client, graph_id="subgraph_hitl_agent")
    thread_id = _create_thread(client)

    response = client.post(
        f"/threads/{thread_id}/runs/stream",
        json={"assistant_id": assistant_id, "input": {"foo": "hello "}, "stream_mode": "updates"},
    )

    assert response.status_code == 200
    events = _parse_sse_events(response.text)
    update_events = [event for event in events if event["event"] == "updates"]
    assert update_events
    assert update_events[-1]["data"]["__interrupt__"][0]["value"] == "Provide value:"


def test_create_run_stream_messages_mode_surfaces_input_requested_interrupt(client: TestClient) -> None:
    assistant_id = _create_assistant(client, graph_id="subgraph_hitl_agent")
    thread_id = _create_thread(client)

    response = client.post(
        f"/threads/{thread_id}/runs/stream",
        json={"assistant_id": assistant_id, "input": {"foo": "hello "}, "stream_mode": "messages"},
    )

    assert response.status_code == 200
    events = _parse_sse_events(response.text)
    input_events = [event for event in events if event["event"] == "input.requested"]
    assert input_events
    assert input_events[-1]["data"]["payload"] == "Provide value:"
    assert input_events[-1]["data"]["interrupt_id"]


def test_create_run_wait_preserves_non_dict_values(client: TestClient, monkeypatch) -> None:
    assistant_id = _create_assistant(client)
    thread_id = _create_thread(client)
    created = RunRead.model_validate(
        {
            "run_id": "non-dict-run",
            "thread_id": thread_id,
            "assistant_id": assistant_id,
            "status": "success",
            "output": None,
            "metadata": {},
            "kwargs": {},
            "multitask_strategy": "enqueue",
        }
    )

    async def fake_create_run(*args, **kwargs):
        return created

    async def fake_get_run_state(*args, **kwargs):
        return {"values": ["one", "two"]}

    monkeypatch.setattr("agentseek_api.api.runs.create_run", fake_create_run)
    monkeypatch.setattr("agentseek_api.api.runs._get_run_state", fake_get_run_state)

    response = client.post(
        f"/threads/{thread_id}/runs/wait",
        json={"assistant_id": assistant_id, "input": {"message": "non-dict"}},
    )

    assert response.status_code == 200
    assert response.json() == ["one", "two"]


def test_create_run_wait_preserves_empty_run_scoped_values(client: TestClient, monkeypatch) -> None:
    assistant_id = _create_assistant(client)
    thread_id = _create_thread(client)
    created = RunRead.model_validate(
        {
            "run_id": "empty-run",
            "thread_id": thread_id,
            "assistant_id": assistant_id,
            "status": "success",
            "output": None,
            "metadata": {},
            "kwargs": {},
            "multitask_strategy": "enqueue",
        }
    )

    async def fake_create_run(*args, **kwargs):
        return created

    async def fake_get_run_state(*args, **kwargs):
        return {"values": {}}

    monkeypatch.setattr("agentseek_api.api.runs.create_run", fake_create_run)
    monkeypatch.setattr("agentseek_api.api.runs._get_run_state", fake_get_run_state)

    response = client.post(
        f"/threads/{thread_id}/runs/wait",
        json={"assistant_id": assistant_id, "input": {"message": "empty"}},
    )

    assert response.status_code == 200
    assert response.json() == {}


def test_create_run_wait_json_encodes_langchain_messages(client: TestClient, monkeypatch) -> None:
    assistant_id = _create_assistant(client)
    thread_id = _create_thread(client)
    created = RunRead.model_validate(
        {
            "run_id": "message-run",
            "thread_id": thread_id,
            "assistant_id": assistant_id,
            "status": "success",
            "output": None,
            "metadata": {},
            "kwargs": {},
            "multitask_strategy": "enqueue",
        }
    )

    async def fake_create_run(*args, **kwargs):
        return created

    async def fake_get_run_state(*args, **kwargs):
        return {"values": {"messages": [HumanMessage(content="hello from state")]}}

    monkeypatch.setattr("agentseek_api.api.runs.create_run", fake_create_run)
    monkeypatch.setattr("agentseek_api.api.stateless_runs.create_run", fake_create_run)
    monkeypatch.setattr("agentseek_api.api.runs._get_run_state", fake_get_run_state)

    response = client.post(
        f"/threads/{thread_id}/runs/wait",
        json={"assistant_id": assistant_id, "input": {"message": "serialize message"}},
    )

    assert response.status_code == 200
    assert response.json()["messages"][0]["content"] == "hello from state"
    assert response.json()["messages"][0]["type"] == "human"

    stateless_response = client.post(
        "/runs/wait",
        json={"assistant_id": assistant_id, "input": {"message": "serialize stateless message"}},
    )

    assert stateless_response.status_code == 200
    assert stateless_response.json()["messages"][0]["content"] == "hello from state"
    assert stateless_response.json()["messages"][0]["type"] == "human"


def test_create_run_stream_surfaces_terminal_error_event(client: TestClient) -> None:
    assistant_id = _create_assistant(client, graph_id="stress_test")
    thread_id = _create_thread(client)

    response = client.post(
        f"/threads/{thread_id}/runs/stream",
        json={"assistant_id": assistant_id, "input": {"delay": 0.0, "steps": 1, "fail": True}, "stream_mode": "updates"},
    )

    assert response.status_code == 200
    events = _parse_sse_events(response.text)
    assert events[-1]["event"] == "error"
    assert "Intentional failure" in events[-1]["data"]["message"]


def test_create_run_compat_openapi_documents_wait_and_stream_routes() -> None:
    openapi = app.openapi()

    thread_wait = openapi["paths"]["/threads/{thread_id}/runs/wait"]["post"]
    assert thread_wait["requestBody"]["content"]["application/json"]["schema"]["$ref"].endswith(
        "RunCreateStreamingStateful"
    )
    assert "Location" in thread_wait["responses"]["200"]["headers"]
    assert "Content-Location" in thread_wait["responses"]["200"]["headers"]

    stateless_wait = openapi["paths"]["/runs/wait"]["post"]
    assert stateless_wait["requestBody"]["content"]["application/json"]["schema"]["$ref"].endswith(
        "RunCreateStreamingStateless"
    )
    assert "Location" in stateless_wait["responses"]["200"]["headers"]
    assert "Content-Location" in stateless_wait["responses"]["200"]["headers"]

    thread_stream = openapi["paths"]["/threads/{thread_id}/runs/stream"]["post"]
    assert thread_stream["requestBody"]["content"]["application/json"]["schema"]["$ref"].endswith(
        "RunCreateStreamingStateful"
    )
    assert "text/event-stream" in thread_stream["responses"]["200"]["content"]
    assert "Location" in thread_stream["responses"]["200"]["headers"]
    assert "Content-Location" in thread_stream["responses"]["200"]["headers"]

    stateless_stream = openapi["paths"]["/runs/stream"]["post"]
    assert stateless_stream["requestBody"]["content"]["application/json"]["schema"]["$ref"].endswith(
        "RunCreateStreamingStateless"
    )
    assert "text/event-stream" in stateless_stream["responses"]["200"]["content"]
    assert "Location" in stateless_stream["responses"]["200"]["headers"]
    assert "Content-Location" in stateless_stream["responses"]["200"]["headers"]

    stateful_schema = openapi["components"]["schemas"]["RunCreateStreamingStateful"]
    assert "on_disconnect" in stateful_schema["properties"]
    assert "interrupt_before" in stateful_schema["properties"]
    assert "interrupt_after" in stateful_schema["properties"]
    assert "stream_subgraphs" in stateful_schema["properties"]
    assert "stream_resumable" in stateful_schema["properties"]
    assert "feedback_keys" in stateful_schema["properties"]
    assert "durability" in stateful_schema["properties"]
    stateful_stream_mode_variants = stateful_schema["properties"]["stream_mode"]["anyOf"]
    stateful_stream_mode_enums = [
        variant["enum"]
        for variant in stateful_stream_mode_variants
        if isinstance(variant, dict) and "enum" in variant
    ]
    stateful_stream_mode_enums.extend(
        [
            variant["items"]["enum"]
            for variant in stateful_stream_mode_variants
            if isinstance(variant, dict) and "items" in variant and "enum" in variant["items"]
        ]
    )
    assert any("messages-tuple" in enum_values for enum_values in stateful_stream_mode_enums)

    stateless_schema = openapi["components"]["schemas"]["RunCreateStreamingStateless"]
    assert "on_disconnect" in stateless_schema["properties"]
    assert "on_completion" in stateless_schema["properties"]
    assert "stream_subgraphs" in stateless_schema["properties"]
    assert "stream_resumable" in stateless_schema["properties"]
    assert "feedback_keys" in stateless_schema["properties"]
    assert "durability" in stateless_schema["properties"]
