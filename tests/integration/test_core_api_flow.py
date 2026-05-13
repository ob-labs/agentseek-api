from fastapi.testclient import TestClient

def test_assistants_threads_runs_flow(client: TestClient) -> None:
    assistant_resp = client.post("/assistants", json={"name": "demo", "graph_id": "default"})
    assert assistant_resp.status_code == 200
    assistant_id = assistant_resp.json()["assistant_id"]

    thread_resp = client.post("/threads", json={"metadata": {"topic": "test"}})
    assert thread_resp.status_code == 200
    thread_id = thread_resp.json()["thread_id"]

    run_resp = client.post(
        f"/threads/{thread_id}/runs",
        json={"assistant_id": assistant_id, "input": {"message": "hello"}},
    )
    assert run_resp.status_code == 200
    run_id = run_resp.json()["run_id"]

    wait_resp = client.get(f"/threads/{thread_id}/runs/{run_id}/wait")
    assert wait_resp.status_code == 200
    assert wait_resp.json()["status"] == "success"

    list_resp = client.get(f"/threads/{thread_id}/runs")
    assert list_resp.status_code == 200
    assert len(list_resp.json()) == 1


def test_stateless_run_creates_thread_and_run(client: TestClient) -> None:
    assistant_resp = client.post("/assistants", json={"name": "stateless", "graph_id": "default"})
    assert assistant_resp.status_code == 200
    assistant_id = assistant_resp.json()["assistant_id"]

    run_resp = client.post("/runs", json={"assistant_id": assistant_id, "input": {"foo": "bar"}})
    assert run_resp.status_code == 200
    assert run_resp.json()["assistant_id"] == assistant_id
