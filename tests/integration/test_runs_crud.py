from fastapi.testclient import TestClient


def _create_assistant(client: TestClient) -> str:
    response = client.post("/assistants", json={"name": "run-assistant", "graph_id": "default"})
    assert response.status_code == 200
    return response.json()["assistant_id"]


def _create_thread(client: TestClient, user_id: str = "default_user") -> str:
    response = client.post("/threads", json={"metadata": {"scope": "run"}}, headers={"x-user-id": user_id})
    assert response.status_code == 200
    return response.json()["thread_id"]


def test_create_run_happy_path(client: TestClient) -> None:
    assistant_id = _create_assistant(client)
    thread_id = _create_thread(client)

    created = client.post(
        f"/threads/{thread_id}/runs",
        json={"assistant_id": assistant_id, "input": {"message": "hello"}},
    )
    assert created.status_code == 200
    body = created.json()
    assert body["thread_id"] == thread_id
    assert body["assistant_id"] == assistant_id


def test_create_run_missing_thread(client: TestClient) -> None:
    assistant_id = _create_assistant(client)
    response = client.post(
        "/threads/does-not-exist/runs",
        json={"assistant_id": assistant_id, "input": {"message": "hello"}},
    )
    assert response.status_code == 404


def test_create_run_missing_assistant(client: TestClient) -> None:
    thread_id = _create_thread(client)
    response = client.post(
        f"/threads/{thread_id}/runs",
        json={"assistant_id": "does-not-exist", "input": {"message": "hello"}},
    )
    assert response.status_code == 404


def test_get_run_not_found(client: TestClient) -> None:
    thread_id = _create_thread(client)
    response = client.get(f"/threads/{thread_id}/runs/not-found")
    assert response.status_code == 404


def test_run_visibility_is_user_scoped(client: TestClient) -> None:
    assistant_id = _create_assistant(client)
    owner_thread = _create_thread(client, user_id="owner")
    run = client.post(
        f"/threads/{owner_thread}/runs",
        json={"assistant_id": assistant_id, "input": {"message": "secret"}},
        headers={"x-user-id": "owner"},
    )
    assert run.status_code == 200
    run_id = run.json()["run_id"]

    hidden = client.get(f"/threads/{owner_thread}/runs/{run_id}", headers={"x-user-id": "other"})
    assert hidden.status_code == 404


def test_wait_run_returns_terminal_status(client: TestClient) -> None:
    assistant_id = _create_assistant(client)
    thread_id = _create_thread(client)
    created = client.post(
        f"/threads/{thread_id}/runs",
        json={"assistant_id": assistant_id, "input": {"message": "wait"}},
    )
    assert created.status_code == 200
    run_id = created.json()["run_id"]

    waited = client.get(f"/threads/{thread_id}/runs/{run_id}/wait")
    assert waited.status_code == 200
    assert waited.json()["status"] == "success"
