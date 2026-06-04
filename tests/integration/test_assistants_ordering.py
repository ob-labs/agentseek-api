from fastapi.testclient import TestClient


def test_list_assistants_latest_first(client: TestClient) -> None:
    first = client.post("/assistants", json={"name": "first", "graph_id": "default"})
    second = client.post("/assistants", json={"name": "second", "graph_id": "default"})
    assert first.status_code == 200
    assert second.status_code == 200

    listed = client.post("/assistants/search", json={})
    assert listed.status_code == 200
    body = listed.json()
    assert body[0]["name"] == "second"
    assert body[1]["name"] == "first"
