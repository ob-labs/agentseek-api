from fastapi.testclient import TestClient


def _seed_stream_run(client: TestClient, *, user_id: str = "default_user") -> tuple[str, str]:
    assistant = client.post("/assistants", json={"name": "stream-owner", "graph_id": "default"})
    assert assistant.status_code == 200
    assistant_id = assistant.json()["assistant_id"]

    thread = client.post("/threads", json={"metadata": {"kind": "stream"}}, headers={"x-user-id": user_id})
    assert thread.status_code == 200
    thread_id = thread.json()["thread_id"]

    run = client.post(
        f"/threads/{thread_id}/runs",
        json={"assistant_id": assistant_id, "input": {"message": "stream"}},
        headers={"x-user-id": user_id},
    )
    assert run.status_code == 200
    return thread_id, run.json()["run_id"]


def test_stream_not_found_returns_404(client: TestClient) -> None:
    thread = client.post("/threads", json={"metadata": {}})
    assert thread.status_code == 200
    thread_id = thread.json()["thread_id"]
    response = client.get(f"/threads/{thread_id}/runs/missing/stream")
    assert response.status_code == 404


def test_stream_hidden_for_wrong_user(client: TestClient) -> None:
    thread_id, run_id = _seed_stream_run(client, user_id="owner")
    response = client.get(f"/threads/{thread_id}/runs/{run_id}/stream", headers={"x-user-id": "other"})
    assert response.status_code == 404


def test_stream_success_content_type(client: TestClient) -> None:
    thread_id, run_id = _seed_stream_run(client)
    response = client.get(f"/threads/{thread_id}/runs/{run_id}/stream")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
