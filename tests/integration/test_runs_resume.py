from fastapi.testclient import TestClient


def _create_assistant(client: TestClient, *, graph_id: str) -> str:
    response = client.post("/assistants", json={"name": f"{graph_id}-assistant", "graph_id": graph_id})
    assert response.status_code == 200
    return response.json()["assistant_id"]


def _create_thread(client: TestClient) -> str:
    response = client.post("/threads", json={"metadata": {"suite": "resume"}})
    assert response.status_code == 200
    return response.json()["thread_id"]


def test_hitl_run_interrupts_and_can_resume(client: TestClient) -> None:
    assistant_id = _create_assistant(client, graph_id="subgraph_hitl_agent")
    thread_id = _create_thread(client)

    created = client.post(
        f"/threads/{thread_id}/runs",
        json={"assistant_id": assistant_id, "input": {"foo": "hello "}},
    )
    assert created.status_code == 200
    created_body = created.json()
    assert created_body["status"] == "interrupted"
    assert created_body["interrupts"][0]["value"] == "Provide value:"

    waited = client.get(f"/threads/{thread_id}/runs/{created_body['run_id']}/wait")
    assert waited.status_code == 200
    assert waited.json()["status"] == "interrupted"

    resumed = client.post(
        f"/threads/{thread_id}/runs/{created_body['run_id']}/resume",
        json={"resume": "world"},
    )
    assert resumed.status_code == 200
    resumed_body = resumed.json()
    assert resumed_body["run_id"] == created_body["run_id"]
    assert resumed_body["status"] == "success"
    assert resumed_body["output"]["state"]["foo"].endswith("world")
    assert resumed_body["interrupts"] == []


def test_resume_rejects_non_interrupted_runs(client: TestClient) -> None:
    assistant_id = _create_assistant(client, graph_id="default")
    thread_id = _create_thread(client)

    created = client.post(
        f"/threads/{thread_id}/runs",
        json={"assistant_id": assistant_id, "input": {"message": "done"}},
    )
    assert created.status_code == 200
    run_id = created.json()["run_id"]

    resumed = client.post(f"/threads/{thread_id}/runs/{run_id}/resume", json={"resume": "ignored"})
    assert resumed.status_code == 409
