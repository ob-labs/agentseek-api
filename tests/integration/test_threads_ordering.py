from fastapi.testclient import TestClient


def test_search_threads_latest_first(client: TestClient) -> None:
    first = client.post("/threads", json={"metadata": {"index": 1}}, headers={"x-user-id": "u1"})
    second = client.post("/threads", json={"metadata": {"index": 2}}, headers={"x-user-id": "u1"})
    assert first.status_code == 200
    assert second.status_code == 200

    listed = client.post("/threads/search", json={}, headers={"x-user-id": "u1"})
    assert listed.status_code == 200
    body = listed.json()
    assert len(body) == 2
    assert body[0]["thread_id"] == second.json()["thread_id"]
    assert body[1]["thread_id"] == first.json()["thread_id"]
