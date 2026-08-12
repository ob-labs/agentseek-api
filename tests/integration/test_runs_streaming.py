import asyncio
import json

from fastapi.testclient import TestClient

from agentseek_api.services.run_state import run_broker


def _stream_payloads(stream_text: str) -> list[dict[str, object]]:
    return [
        json.loads(line.replace("data: ", "", 1))
        for line in stream_text.splitlines()
        if line.startswith("data: ")
    ]


def test_run_stream_returns_start_and_end_events(client: TestClient) -> None:
    assistant = client.post("/assistants", json={"name": "streaming", "graph_id": "default"})
    assert assistant.status_code == 200
    assistant_id = assistant.json()["assistant_id"]

    thread = client.post("/threads", json={"metadata": {"case": "stream"}})
    assert thread.status_code == 200
    thread_id = thread.json()["thread_id"]

    run = client.post(
        f"/threads/{thread_id}/runs",
        json={"assistant_id": assistant_id, "input": {"message": "stream"}},
    )
    assert run.status_code == 200
    run_id = run.json()["run_id"]

    stream_response = client.get(f"/threads/{thread_id}/runs/{run_id}/stream")
    assert stream_response.status_code == 200
    body = stream_response.text
    assert "event: start" in body
    assert "event: end" in body


def test_react_agent_stream_includes_tool_and_message_events(client: TestClient) -> None:
    assistant = client.post("/assistants", json={"name": "streaming-react", "graph_id": "react_agent"})
    assert assistant.status_code == 200
    assistant_id = assistant.json()["assistant_id"]

    thread = client.post("/threads", json={"metadata": {"case": "react-stream"}})
    assert thread.status_code == 200
    thread_id = thread.json()["thread_id"]

    run = client.post(
        f"/threads/{thread_id}/runs",
        json={"assistant_id": assistant_id, "input": {"message": "stream"}},
    )
    assert run.status_code == 200
    run_id = run.json()["run_id"]

    stream_response = client.get(f"/threads/{thread_id}/runs/{run_id}/stream")
    assert stream_response.status_code == 200
    payloads = _stream_payloads(stream_response.text)

    assert any(isinstance(payload, dict) and payload.get("event") == "tool-started" and payload.get("tool_name") == "lookup" for payload in payloads)
    assert any(isinstance(payload, dict) and payload.get("event") == "tool-finished" and payload.get("tool_name") == "lookup" for payload in payloads)
    assert any(
        isinstance(payload, dict) and "Final answer:" in json.dumps(payload, ensure_ascii=False)
        for payload in payloads
    )


def test_stress_tool_agent_stream_includes_multiple_tool_cycles(client: TestClient) -> None:
    assistant = client.post("/assistants", json={"name": "streaming-stress-tools", "graph_id": "stress_tool_agent"})
    assert assistant.status_code == 200
    assistant_id = assistant.json()["assistant_id"]

    thread = client.post("/threads", json={"metadata": {"case": "stress-tool-stream"}})
    assert thread.status_code == 200
    thread_id = thread.json()["thread_id"]

    run = client.post(
        f"/threads/{thread_id}/runs",
        json={"assistant_id": assistant_id, "input": {"delay": 0.0, "steps": 3}},
    )
    assert run.status_code == 200
    run_id = run.json()["run_id"]

    stream_response = client.get(f"/threads/{thread_id}/runs/{run_id}/stream")
    assert stream_response.status_code == 200
    payloads = _stream_payloads(stream_response.text)

    tool_starts = [
        payload for payload in payloads if isinstance(payload, dict) and payload.get("event") == "tool-started" and payload.get("tool_name") == "slow_process"
    ]
    tool_ends = [
        payload for payload in payloads if isinstance(payload, dict) and payload.get("event") == "tool-finished" and payload.get("tool_name") == "slow_process"
    ]
    assert len(tool_starts) == 3
    assert len(tool_ends) == 3
    assert any(
        isinstance(payload, dict) and "steps_completed" in json.dumps(payload, ensure_ascii=False)
        for payload in payloads
    )


def test_interrupted_run_stream_payload_includes_terminal_status(client: TestClient) -> None:
    assistant = client.post("/assistants", json={"name": "streaming-hitl", "graph_id": "subgraph_hitl_agent"})
    assert assistant.status_code == 200
    assistant_id = assistant.json()["assistant_id"]

    thread = client.post("/threads", json={"metadata": {"case": "interrupt-stream"}})
    assert thread.status_code == 200
    thread_id = thread.json()["thread_id"]

    run = client.post(
        f"/threads/{thread_id}/runs",
        json={"assistant_id": assistant_id, "input": {"foo": "hello "}},
    )
    assert run.status_code == 200
    run_id = run.json()["run_id"]

    stream_response = client.get(f"/threads/{thread_id}/runs/{run_id}/stream")
    assert stream_response.status_code == 200
    payload = _stream_payloads(stream_response.text)[-1]
    assert payload["status"] == "interrupted"


def test_resumed_run_stream_preserves_each_terminal_status(client: TestClient) -> None:
    assistant = client.post("/assistants", json={"name": "streaming-resume", "graph_id": "subgraph_hitl_agent"})
    assert assistant.status_code == 200
    assistant_id = assistant.json()["assistant_id"]

    thread = client.post("/threads", json={"metadata": {"case": "resume-stream"}})
    assert thread.status_code == 200
    thread_id = thread.json()["thread_id"]

    run = client.post(
        f"/threads/{thread_id}/runs",
        json={"assistant_id": assistant_id, "input": {"foo": "hello "}},
    )
    assert run.status_code == 200
    run_id = run.json()["run_id"]

    resumed = client.post(
        f"/threads/{thread_id}/runs/{run_id}/resume",
        json={"resume": "world"},
    )
    assert resumed.status_code == 200

    stream_response = client.get(f"/threads/{thread_id}/runs/{run_id}/stream")
    assert stream_response.status_code == 200
    payloads = _stream_payloads(stream_response.text)
    end_statuses = [payload["status"] for payload in payloads if payload.get("event") == "end"]
    assert end_statuses == ["interrupted", "success"]

    # Every SSE id across the whole (interrupted + resumed) log must be
    # strictly monotonic. The historical interrupted run's terminal "end" must
    # keep its original seq and not be deferred past the resumed run's frames
    # (a resumed stream previously re-ordered the earlier end after newer
    # frames, producing a non-monotonic cursor like 1..9, 11..17, 10, 18).
    ids = _sse_ids(stream_response.text)
    assert ids == sorted(ids), f"resumed run stream ids not monotonic: {ids}"
    assert len(set(ids)) == len(ids), f"resumed run stream ids not unique: {ids}"


def test_create_run_rejects_configurable_and_context_together(client: TestClient) -> None:
    assistant = client.post("/assistants", json={"name": "reject-both", "graph_id": "default"})
    assert assistant.status_code == 200
    assistant_id = assistant.json()["assistant_id"]

    thread = client.post("/threads", json={"metadata": {"case": "reject-both"}})
    assert thread.status_code == 200
    thread_id = thread.json()["thread_id"]

    run = client.post(
        f"/threads/{thread_id}/runs",
        json={
            "assistant_id": assistant_id,
            "input": {"message": "hello"},
            "config": {"configurable": {"client_param": "x"}},
            "context": {"tenant": "acme"},
        },
    )
    assert run.status_code == 400


def test_run_with_assistant_context_and_client_configurable_succeeds(client: TestClient) -> None:
    """Assistant-level default context merged with client config.configurable must not error."""
    assistant = client.post(
        "/assistants",
        json={"name": "ctx-config", "graph_id": "default", "context": {"tenant": "acme"}},
    )
    assert assistant.status_code == 200
    assistant_id = assistant.json()["assistant_id"]

    thread = client.post("/threads", json={"metadata": {"case": "ctx-config"}})
    assert thread.status_code == 200
    thread_id = thread.json()["thread_id"]

    run = client.post(
        f"/threads/{thread_id}/runs",
        json={
            "assistant_id": assistant_id,
            "input": {"message": "hello"},
            "config": {"configurable": {"client_param": "x"}},
        },
    )
    assert run.status_code == 200
    run_id = run.json()["run_id"]

    run_read = client.get(f"/threads/{thread_id}/runs/{run_id}")
    assert run_read.status_code == 200
    assert run_read.json()["status"] == "success"


def test_run_context_params_persist_in_checkpoint_metadata(client: TestClient) -> None:
    """Scalar context/configurable params are persisted in checkpoint metadata."""
    import asyncio

    from agentseek_api.services.thread_checkpoint_store import get_latest_checkpoint

    assistant = client.post(
        "/assistants",
        json={"name": "ctx-meta", "graph_id": "default", "context": {"tenant": "acme"}},
    )
    assert assistant.status_code == 200
    assistant_id = assistant.json()["assistant_id"]

    thread = client.post("/threads", json={"metadata": {"case": "ctx-meta"}})
    assert thread.status_code == 200
    thread_id = thread.json()["thread_id"]

    run = client.post(
        f"/threads/{thread_id}/runs",
        json={
            "assistant_id": assistant_id,
            "input": {"message": "hello"},
            "config": {"configurable": {"client_param": "x"}},
        },
    )
    assert run.status_code == 200
    run_id = run.json()["run_id"]
    run_read = client.get(f"/threads/{thread_id}/runs/{run_id}")
    assert run_read.json()["status"] == "success"

    checkpoint = asyncio.run(get_latest_checkpoint(thread_id))
    assert checkpoint is not None
    metadata = checkpoint.metadata or {}
    assert metadata.get("tenant") == "acme"
    assert metadata.get("client_param") == "x"

def test_run_uses_assistant_config_defaults(client: TestClient) -> None:
    """Assistant-level config.configurable is carried into the run (aegra parity)."""
    assistant = client.post(
        "/assistants",
        json={
            "name": "cfg-defaults",
            "graph_id": "default",
            "config": {"configurable": {"client_param": "assistant-default"}},
        },
    )
    assert assistant.status_code == 200
    assistant_id = assistant.json()["assistant_id"]

    thread = client.post("/threads", json={"metadata": {"case": "cfg-defaults"}})
    assert thread.status_code == 200
    thread_id = thread.json()["thread_id"]

    run = client.post(
        f"/threads/{thread_id}/runs",
        json={"assistant_id": assistant_id, "input": {"message": "hello"}},
    )
    assert run.status_code == 200
    run_id = run.json()["run_id"]

    run_read = client.get(f"/threads/{thread_id}/runs/{run_id}")
    assert run_read.status_code == 200
    assert run_read.json()["status"] == "success"

    # The persisted run kwargs carry the assistant config as defaults.
    kwargs = run_read.json().get("kwargs") or {}
    config = kwargs.get("config") or {}
    assert config.get("configurable", {}).get("client_param") == "assistant-default"


def test_run_client_config_overrides_assistant_config_defaults(client: TestClient) -> None:
    """Client config.configurable wins over the assistant default without wiping it."""
    assistant = client.post(
        "/assistants",
        json={
            "name": "cfg-override",
            "graph_id": "default",
            "config": {"configurable": {"client_param": "assistant-default", "model": "assistant-model"}},
        },
    )
    assert assistant.status_code == 200
    assistant_id = assistant.json()["assistant_id"]

    thread = client.post("/threads", json={"metadata": {"case": "cfg-override"}})
    assert thread.status_code == 200
    thread_id = thread.json()["thread_id"]

    run = client.post(
        f"/threads/{thread_id}/runs",
        json={
            "assistant_id": assistant_id,
            "input": {"message": "hello"},
            "config": {"configurable": {"model": "client-model"}},
        },
    )
    assert run.status_code == 200
    run_id = run.json()["run_id"]

    run_read = client.get(f"/threads/{thread_id}/runs/{run_id}")
    assert run_read.status_code == 200
    assert run_read.json()["status"] == "success"

    kwargs = run_read.json().get("kwargs") or {}
    config = kwargs.get("config") or {}
    configurable = config.get("configurable") or {}
    assert configurable.get("client_param") == "assistant-default"
    assert configurable.get("model") == "client-model"


def _sse_ids(stream_text: str) -> list[int]:
    ids: list[int] = []
    for line in stream_text.splitlines():
        if line.startswith("id: "):
            ids.append(int(line[len("id: "):].strip()))
    return ids


def test_run_stream_sse_ids_are_monotonic(client: TestClient) -> None:
    """Every SSE ``id`` in the default run stream shares one monotonic cursor.

    The replay merges run-scoped lifecycle events (start/end) with
    thread-protocol events that carry their own independent sequence domains.
    A non-monotonic cursor (e.g. ``1, 3, ...25, 2``) breaks Last-Event-ID
    resume, so all emitted ids must be strictly increasing.
    """
    assistant = client.post("/assistants", json={"name": "streaming-monotonic", "graph_id": "react_agent"})
    assert assistant.status_code == 200
    assistant_id = assistant.json()["assistant_id"]

    thread = client.post("/threads", json={"metadata": {"case": "monotonic"}})
    assert thread.status_code == 200
    thread_id = thread.json()["thread_id"]

    run = client.post(
        f"/threads/{thread_id}/runs",
        json={"assistant_id": assistant_id, "input": {"message": "stream"}},
    )
    assert run.status_code == 200
    run_id = run.json()["run_id"]

    stream_response = client.get(f"/threads/{thread_id}/runs/{run_id}/stream")
    assert stream_response.status_code == 200
    ids = _sse_ids(stream_response.text)
    assert len(ids) >= 2, "expected at least start + end frames"
    assert ids == sorted(ids), f"SSE ids are not monotonic: {ids}"
    assert len(set(ids)) == len(ids), f"SSE ids are not unique: {ids}"


def test_run_stream_resume_after_terminal_end_does_not_replay(client: TestClient) -> None:
    """Reconnecting with the terminal frame's Last-Event-ID must not replay
    already-delivered events."""
    assistant = client.post("/assistants", json={"name": "streaming-resume", "graph_id": "react_agent"})
    assert assistant.status_code == 200
    assistant_id = assistant.json()["assistant_id"]

    thread = client.post("/threads", json={"metadata": {"case": "resume"}})
    assert thread.status_code == 200
    thread_id = thread.json()["thread_id"]

    run = client.post(
        f"/threads/{thread_id}/runs",
        json={"assistant_id": assistant_id, "input": {"message": "stream"}},
    )
    assert run.status_code == 200
    run_id = run.json()["run_id"]

    full = client.get(f"/threads/{thread_id}/runs/{run_id}/stream")
    assert full.status_code == 200
    ids = _sse_ids(full.text)
    assert ids, "stream produced no id frames"
    last_id = ids[-1]

    resumed = client.get(
        f"/threads/{thread_id}/runs/{run_id}/stream",
        headers={"Last-Event-ID": str(last_id)},
    )
    assert resumed.status_code == 200
    resumed_ids = _sse_ids(resumed.text)
    assert resumed_ids == [], f"resume after terminal end should replay nothing, got: {resumed_ids}"


def test_run_stream_midrun_reconnect_is_exactly_once(midrun_app_factory, midrun_reconnect_flow) -> None:
    """HTTP-level regression: a client that connects mid-run, disconnects, and
    reconnects with Last-Event-ID must not replay delivered frames and must not
    lose frames produced while disconnected (inline executor path)."""
    app = midrun_app_factory(executor_backend="inline")
    asyncio.run(midrun_reconnect_flow(app))


def test_run_stream_cold_broker_resume_keeps_monotonic_ids(client: TestClient, monkeypatch) -> None:
    """A resume after the in-memory broker state is cleared must keep allocating
    from the persisted seq watermark, not restart at 1.

    Regression for a process-local allocation bug: clearing the broker's event
    state *and* ``_next_seq`` before resuming a persisted run caused new frames
    to reuse seqs already in the DB (e.g. ``1..7, 9, 8, 10``) and returned
    contradictory terminal statuses. Lifecycle and protocol publication must
    share one persistent run-scoped sequence.
    """
    assistant = client.post("/assistants", json={"name": "cold-broker", "graph_id": "subgraph_hitl_agent"})
    assert assistant.status_code == 200
    assistant_id = assistant.json()["assistant_id"]

    thread = client.post("/threads", json={"metadata": {"case": "cold-broker"}})
    assert thread.status_code == 200
    thread_id = thread.json()["thread_id"]

    run = client.post(
        f"/threads/{thread_id}/runs",
        json={"assistant_id": assistant_id, "input": {"foo": "hello "}},
    )
    assert run.status_code == 200
    run_id = run.json()["run_id"]

    # First run interrupted; its protocol frames + end are persisted.
    waited = client.get(f"/threads/{thread_id}/runs/{run_id}/wait")
    assert waited.json()["status"] == "interrupted"

    # Simulate a cold broker (e.g. process restart): drop every in-memory
    # trace of this run, including the seq counter.
    run_broker._events.pop(run_id, None)
    run_broker._seqs.pop(run_id, None)
    run_broker._signals.pop(run_id, None)
    run_broker._next_seq.pop(run_id, None)
    run_broker._completed_runs.discard(run_id)
    try:
        run_broker._completed_order.remove(run_id)
    except ValueError:
        pass

    resumed = client.post(
        f"/threads/{thread_id}/runs/{run_id}/resume",
        json={"resume": "world"},
    )
    assert resumed.status_code == 200

    stream_response = client.get(f"/threads/{thread_id}/runs/{run_id}/stream")
    assert stream_response.status_code == 200
    ids = _sse_ids(stream_response.text)
    assert ids, "expected frames across the interrupted + resumed log"
    # ids must be strictly increasing and unique (no reuse of persisted seqs).
    assert ids == sorted(ids), f"cold-broker resume ids not monotonic: {ids}"
    assert len(set(ids)) == len(ids), f"cold-broker resume ids not unique: {ids}"

    # Both runs' terminal statuses must be present and ordered by seq.
    payloads = _stream_payloads(stream_response.text)
    end_statuses = [payload["status"] for payload in payloads if payload.get("event") == "end"]
    assert end_statuses == ["interrupted", "success"]