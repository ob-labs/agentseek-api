from fastapi.testclient import TestClient


def test_thread_and_run_not_visible_cross_user_matrix(client: TestClient) -> None:
    assistant = client.post("/assistants", json={"name": "scope-matrix", "graph_id": "default"})
    assert assistant.status_code == 200
    assistant_id = assistant.json()["assistant_id"]

    owner_thread = client.post("/threads", json={"metadata": {"scope": "matrix"}}, headers={"x-user-id": "owner"})
    assert owner_thread.status_code == 200
    thread_id = owner_thread.json()["thread_id"]

    owner_run = client.post(
        f"/threads/{thread_id}/runs",
        json={"assistant_id": assistant_id, "input": {"sensitive": "yes"}},
        headers={"x-user-id": "owner"},
    )
    assert owner_run.status_code == 200
    run_id = owner_run.json()["run_id"]

    # Other user cannot list owner's thread
    other_threads = client.get("/threads", headers={"x-user-id": "other"})
    assert other_threads.status_code == 200
    assert all(item["thread_id"] != thread_id for item in other_threads.json())

    # Other user cannot list or fetch owner's run
    other_runs = client.get(f"/threads/{thread_id}/runs", headers={"x-user-id": "other"})
    assert other_runs.status_code == 200
    assert other_runs.json() == []

    other_run_get = client.get(f"/threads/{thread_id}/runs/{run_id}", headers={"x-user-id": "other"})
    assert other_run_get.status_code == 404
