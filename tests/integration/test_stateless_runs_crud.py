from fastapi.testclient import TestClient


def _create_assistant(client: TestClient, name: str = "stateless-a") -> str:
    response = client.post("/assistants", json={"name": name, "graph_id": "default"})
    assert response.status_code == 200
    return response.json()["assistant_id"]


def test_stateless_run_happy_path(client: TestClient) -> None:
    assistant_id = _create_assistant(client)
    response = client.post("/runs", json={"assistant_id": assistant_id, "input": {"kind": "stateless"}})
    assert response.status_code == 200
    body = response.json()
    assert body["assistant_id"] == assistant_id
    assert body["thread_id"]


def test_stateless_run_missing_assistant(client: TestClient) -> None:
    response = client.post("/runs", json={"assistant_id": "missing", "input": {"kind": "stateless"}})
    assert response.status_code == 404


def test_stateless_run_missing_input_validation(client: TestClient) -> None:
    assistant_id = _create_assistant(client)
    response = client.post("/runs", json={"assistant_id": assistant_id})
    assert response.status_code == 422


def test_stateless_run_missing_assistant_validation(client: TestClient) -> None:
    response = client.post("/runs", json={"input": {"hello": "world"}})
    assert response.status_code == 422


def test_stateless_run_respects_user_scope(client: TestClient) -> None:
    assistant_id = _create_assistant(client, name="user-scope")
    owner = client.post(
        "/runs",
        json={"assistant_id": assistant_id, "input": {"secret": True}},
        headers={"x-user-id": "owner"},
    )
    assert owner.status_code == 200
    owner_thread_id = owner.json()["thread_id"]
    owner_run_id = owner.json()["run_id"]

    hidden = client.get(
        f"/threads/{owner_thread_id}/runs/{owner_run_id}",
        headers={"x-user-id": "other"},
    )
    assert hidden.status_code == 404


def test_stateless_run_creates_thread_visible_to_creator(client: TestClient) -> None:
    assistant_id = _create_assistant(client)
    response = client.post(
        "/runs",
        json={"assistant_id": assistant_id, "input": {"creator": True}},
        headers={"x-user-id": "creator"},
    )
    assert response.status_code == 200
    thread_id = response.json()["thread_id"]

    thread_get = client.get(f"/threads/{thread_id}", headers={"x-user-id": "creator"})
    assert thread_get.status_code == 200


def test_stateless_runs_can_be_created_multiple_times(client: TestClient) -> None:
    assistant_id = _create_assistant(client)
    first = client.post("/runs", json={"assistant_id": assistant_id, "input": {"n": 1}})
    second = client.post("/runs", json={"assistant_id": assistant_id, "input": {"n": 2}})
    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["run_id"] != second.json()["run_id"]
