from fastapi.testclient import TestClient


def _seed_run(client: TestClient, *, user_id: str = "default_user") -> tuple[str, str]:
    assistant = client.post("/assistants", json={"name": "wait-assistant", "graph_id": "default"})
    assert assistant.status_code == 200
    assistant_id = assistant.json()["assistant_id"]
    thread = client.post("/threads", json={"metadata": {"k": "v"}}, headers={"x-user-id": user_id})
    assert thread.status_code == 200
    thread_id = thread.json()["thread_id"]
    run = client.post(
        f"/threads/{thread_id}/runs",
        json={"assistant_id": assistant_id, "input": {"message": "hello"}},
        headers={"x-user-id": user_id},
    )
    assert run.status_code == 200
    return thread_id, run.json()["run_id"]


def test_wait_not_found_returns_404(client: TestClient) -> None:
    thread = client.post("/threads", json={"metadata": {}}, headers={"x-user-id": "u1"})
    assert thread.status_code == 200
    thread_id = thread.json()["thread_id"]
    response = client.get(f"/threads/{thread_id}/runs/missing/wait", headers={"x-user-id": "u1"})
    assert response.status_code == 404


def test_get_run_visible_without_auth_on(client: TestClient) -> None:
    """Without @auth.on handlers, runs are visible to all authenticated users."""
    thread_id, run_id = _seed_run(client, user_id="owner")
    response = client.get(f"/threads/{thread_id}/runs/{run_id}", headers={"x-user-id": "other"})
    assert response.status_code == 200


def test_wait_run_visible_without_auth_on(client: TestClient) -> None:
    """Without @auth.on handlers, runs are visible to all authenticated users."""
    thread_id, run_id = _seed_run(client, user_id="owner")
    response = client.get(f"/threads/{thread_id}/runs/{run_id}/wait", headers={"x-user-id": "other"})
    assert response.status_code == 200


def test_get_and_wait_run_success(client: TestClient) -> None:
    thread_id, run_id = _seed_run(client)
    get_response = client.get(f"/threads/{thread_id}/runs/{run_id}")
    assert get_response.status_code == 200
    wait_response = client.get(f"/threads/{thread_id}/runs/{run_id}/wait")
    assert wait_response.status_code == 200
    assert wait_response.json()["status"] == "success"
