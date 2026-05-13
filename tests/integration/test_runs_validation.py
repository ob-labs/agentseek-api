from fastapi.testclient import TestClient


def test_create_run_missing_input_returns_422(client: TestClient) -> None:
    assistant = client.post("/assistants", json={"name": "v-assistant", "graph_id": "default"})
    thread = client.post("/threads", json={"metadata": {}})
    assert assistant.status_code == 200
    assert thread.status_code == 200

    response = client.post(
        f"/threads/{thread.json()['thread_id']}/runs",
        json={"assistant_id": assistant.json()["assistant_id"]},
    )
    assert response.status_code == 422


def test_create_run_missing_assistant_id_returns_422(client: TestClient) -> None:
    thread = client.post("/threads", json={"metadata": {}})
    assert thread.status_code == 200
    response = client.post(
        f"/threads/{thread.json()['thread_id']}/runs",
        json={"input": {"m": "x"}},
    )
    assert response.status_code == 422
