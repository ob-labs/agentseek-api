from fastapi.testclient import TestClient


def test_search_threads_empty_for_user(client: TestClient) -> None:
    response = client.post("/threads/search", json={}, headers={"x-user-id": "u1"})
    assert response.status_code == 200
    assert response.json() == []


def test_create_thread_and_get_thread(client: TestClient) -> None:
    created = client.post("/threads", json={"metadata": {"topic": "t1"}}, headers={"x-user-id": "u1"})
    assert created.status_code == 200
    thread_id = created.json()["thread_id"]

    fetched = client.get(f"/threads/{thread_id}", headers={"x-user-id": "u1"})
    assert fetched.status_code == 200
    assert fetched.json()["metadata"]["topic"] == "t1"


def test_get_thread_visible_without_auth_on(client: TestClient) -> None:
    """Without @auth.on handlers, threads are visible to all authenticated users."""
    created = client.post("/threads", json={"metadata": {"topic": "private"}}, headers={"x-user-id": "owner"})
    assert created.status_code == 200
    thread_id = created.json()["thread_id"]

    visible = client.get(f"/threads/{thread_id}", headers={"x-user-id": "other"})
    assert visible.status_code == 200


def test_search_threads_visible_to_all_without_auth_on(client: TestClient) -> None:
    """Without @auth.on handlers, all threads are visible to all authenticated users."""
    t1 = client.post("/threads", json={"metadata": {"id": 1}}, headers={"x-user-id": "u1"})
    t2 = client.post("/threads", json={"metadata": {"id": 2}}, headers={"x-user-id": "u2"})
    assert t1.status_code == 200
    assert t2.status_code == 200

    u1_list = client.post("/threads/search", json={}, headers={"x-user-id": "u1"})
    assert u1_list.status_code == 200
    assert len(u1_list.json()) == 2
