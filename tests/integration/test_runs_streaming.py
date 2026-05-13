from fastapi.testclient import TestClient


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
