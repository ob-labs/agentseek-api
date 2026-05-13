from fastapi.testclient import TestClient


def test_list_runs_latest_first(client: TestClient) -> None:
    assistant = client.post("/assistants", json={"name": "order-assistant", "graph_id": "default"})
    assert assistant.status_code == 200
    assistant_id = assistant.json()["assistant_id"]

    thread = client.post("/threads", json={"metadata": {"order": True}})
    assert thread.status_code == 200
    thread_id = thread.json()["thread_id"]

    first = client.post(
        f"/threads/{thread_id}/runs",
        json={"assistant_id": assistant_id, "input": {"index": 1}},
    )
    second = client.post(
        f"/threads/{thread_id}/runs",
        json={"assistant_id": assistant_id, "input": {"index": 2}},
    )
    assert first.status_code == 200
    assert second.status_code == 200

    listed = client.get(f"/threads/{thread_id}/runs")
    assert listed.status_code == 200
    body = listed.json()
    assert len(body) == 2
    assert body[0]["run_id"] == second.json()["run_id"]
    assert body[1]["run_id"] == first.json()["run_id"]
