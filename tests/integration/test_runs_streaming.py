from fastapi.testclient import TestClient
import json


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
    lines = [line for line in stream_response.text.splitlines() if line.startswith("data: ")]
    payload = json.loads(lines[-1].replace("data: ", "", 1))
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
    payloads = [
        json.loads(line.replace("data: ", "", 1))
        for line in stream_response.text.splitlines()
        if line.startswith("data: ")
    ]
    end_statuses = [payload["status"] for payload in payloads if payload["event"] == "end"]
    assert end_statuses == ["interrupted", "success"]
