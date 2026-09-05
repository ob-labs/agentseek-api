from fastapi.testclient import TestClient
import json


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

    assert any(payload["event"] == "tool_start" and payload["name"] == "lookup" for payload in payloads)
    assert any(payload["event"] == "tool_end" and payload["name"] == "lookup" for payload in payloads)
    assert any(
        payload["event"] == "message_chunk" and "Final answer:" in str(payload.get("content", ""))
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

    tool_starts = [payload for payload in payloads if payload["event"] == "tool_start" and payload["name"] == "slow_process"]
    tool_ends = [payload for payload in payloads if payload["event"] == "tool_end" and payload["name"] == "slow_process"]
    assert len(tool_starts) == 3
    assert len(tool_ends) == 3
    assert any(
        payload["event"] == "message_chunk" and '"steps_completed": 3' in str(payload.get("content", ""))
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
    end_statuses = [payload["status"] for payload in payloads if payload["event"] == "end"]
    assert end_statuses == ["interrupted", "success"]


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