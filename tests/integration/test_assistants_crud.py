from fastapi.testclient import TestClient


def test_list_assistants_empty(client: TestClient) -> None:
    response = client.post("/assistants/search", json={})
    assert response.status_code == 200
    assert response.json() == []


def test_create_and_get_assistant(client: TestClient) -> None:
    created = client.post("/assistants", json={"name": "assistant-1", "graph_id": "default"})
    assert created.status_code == 200
    body = created.json()
    assert body["name"] == "assistant-1"
    assert body["graph_id"] == "default"

    fetched = client.get(f"/assistants/{body['assistant_id']}")
    assert fetched.status_code == 200
    assert fetched.json()["assistant_id"] == body["assistant_id"]


def test_get_assistant_not_found(client: TestClient) -> None:
    response = client.get("/assistants/does-not-exist")
    assert response.status_code == 404


def test_create_assistant_validation_error(client: TestClient) -> None:
    response = client.post("/assistants", json={})
    assert response.status_code == 422


def test_list_assistants_returns_multiple_items(client: TestClient) -> None:
    first = client.post("/assistants", json={"name": "a1", "graph_id": "default"})
    second = client.post("/assistants", json={"name": "a2", "graph_id": "default"})
    assert first.status_code == 200
    assert second.status_code == 200

    listed = client.post("/assistants/search", json={})
    assert listed.status_code == 200
    items = listed.json()
    assert len(items) == 2
