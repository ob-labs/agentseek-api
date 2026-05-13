from fastapi.testclient import TestClient


def test_list_threads_empty_for_user(client: TestClient) -> None:
    response = client.get("/threads", headers={"x-user-id": "u1"})
    assert response.status_code == 200
    assert response.json() == []


def test_create_thread_and_get_thread(client: TestClient) -> None:
    created = client.post("/threads", json={"metadata": {"topic": "t1"}}, headers={"x-user-id": "u1"})
    assert created.status_code == 200
    thread_id = created.json()["thread_id"]

    fetched = client.get(f"/threads/{thread_id}", headers={"x-user-id": "u1"})
    assert fetched.status_code == 200
    assert fetched.json()["metadata"]["topic"] == "t1"


def test_get_thread_not_found_for_other_user(client: TestClient) -> None:
    created = client.post("/threads", json={"metadata": {"topic": "private"}}, headers={"x-user-id": "owner"})
    assert created.status_code == 200
    thread_id = created.json()["thread_id"]

    forbidden = client.get(f"/threads/{thread_id}", headers={"x-user-id": "other"})
    assert forbidden.status_code == 404


def test_list_threads_is_user_scoped(client: TestClient) -> None:
    t1 = client.post("/threads", json={"metadata": {"id": 1}}, headers={"x-user-id": "u1"})
    t2 = client.post("/threads", json={"metadata": {"id": 2}}, headers={"x-user-id": "u2"})
    assert t1.status_code == 200
    assert t2.status_code == 200

    u1_list = client.get("/threads", headers={"x-user-id": "u1"})
    u2_list = client.get("/threads", headers={"x-user-id": "u2"})
    assert u1_list.status_code == 200
    assert u2_list.status_code == 200
    assert len(u1_list.json()) == 1
    assert len(u2_list.json()) == 1
