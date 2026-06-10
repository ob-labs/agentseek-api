from fastapi.testclient import TestClient


def test_resources_visible_to_all_users_without_auth_on(client: TestClient) -> None:
    """Without @auth.on handlers, all authenticated users share all resources.

    This matches LangSmith behavior: authentication verifies who you are,
    but without authorization handlers, no per-user filtering is applied.
    """
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

    other_threads = client.post("/threads/search", json={}, headers={"x-user-id": "other"})
    assert other_threads.status_code == 200
    assert any(item["thread_id"] == thread_id for item in other_threads.json())

    other_runs = client.get(f"/threads/{thread_id}/runs", headers={"x-user-id": "other"})
    assert other_runs.status_code == 200
    assert any(item["run_id"] == run_id for item in other_runs.json())

    other_run_get = client.get(f"/threads/{thread_id}/runs/{run_id}", headers={"x-user-id": "other"})
    assert other_run_get.status_code == 200
