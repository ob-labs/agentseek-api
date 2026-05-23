from collections.abc import Awaitable, Callable

from fastapi.testclient import TestClient

from agentseek_api.services.run_jobs import RunExecutionJob


def _create_assistant(client: TestClient, *, graph_id: str = "default") -> str:
    response = client.post("/assistants", json={"name": f"{graph_id}-assistant", "graph_id": graph_id})
    assert response.status_code == 200
    return response.json()["assistant_id"]


def _create_thread(client: TestClient, *, user_id: str = "default_user") -> str:
    response = client.post("/threads", json={"metadata": {"compat": True}}, headers={"x-user-id": user_id})
    assert response.status_code == 200
    return response.json()["thread_id"]


class DeferredExecutor:
    def __init__(self) -> None:
        self.submitted: list[Callable[[], Awaitable[None]] | RunExecutionJob] = []

    async def submit(self, job: Callable[[], Awaitable[None]] | RunExecutionJob) -> None:
        self.submitted.append(job)


def test_thread_run_wait_and_stream_creation_routes(client: TestClient) -> None:
    assistant_id = _create_assistant(client)
    thread_id = _create_thread(client)

    waited = client.post(
        f"/threads/{thread_id}/runs/wait",
        json={"assistant_id": assistant_id, "input": {"message": "wait route"}},
    )
    assert waited.status_code == 200
    assert waited.json()["status"] == "success"

    streamed = client.post(
        f"/threads/{thread_id}/runs/stream",
        json={"assistant_id": assistant_id, "input": {"message": "stream route"}},
    )
    assert streamed.status_code == 200
    assert streamed.headers["content-type"].startswith("text/event-stream")
    assert "event: start" in streamed.text
    assert "event: end" in streamed.text


def test_stateless_wait_stream_and_batch_routes(client: TestClient) -> None:
    assistant_id = _create_assistant(client)

    waited = client.post("/runs/wait", json={"assistant_id": assistant_id, "input": {"message": "wait"}})
    assert waited.status_code == 200
    assert waited.json()["status"] == "success"

    streamed = client.post("/runs/stream", json={"assistant_id": assistant_id, "input": {"message": "stream"}})
    assert streamed.status_code == 200
    assert streamed.headers["content-type"].startswith("text/event-stream")

    batch = client.post(
        "/runs/batch",
        json=[
            {"assistant_id": assistant_id, "input": {"message": "one"}},
            {"assistant_id": assistant_id, "input": {"message": "two"}},
        ],
    )
    assert batch.status_code == 200
    body = batch.json()
    assert len(body) == 2
    assert body[0]["status"] == "success"
    assert body[1]["status"] == "success"


def test_cancel_routes(client: TestClient, monkeypatch) -> None:
    monkeypatch.setattr("agentseek_api.services.run_preparation.get_executor", lambda: DeferredExecutor())
    assistant_id = _create_assistant(client, graph_id="stress_test")
    thread_id = _create_thread(client)
    created = client.post(
        f"/threads/{thread_id}/runs",
        json={"assistant_id": assistant_id, "input": {"delay": 0.05, "steps": 20}},
    )
    assert created.status_code == 200
    run_id = created.json()["run_id"]

    cancel_one = client.post(f"/threads/{thread_id}/runs/{run_id}/cancel")
    assert cancel_one.status_code == 200
    waited = client.get(f"/threads/{thread_id}/runs/{run_id}/wait")
    assert waited.status_code == 200
    assert waited.json()["status"] == "error"
    thread = client.get(f"/threads/{thread_id}")
    assert thread.status_code == 200
    assert thread.json()["status"] == "error"

    cancel_many = client.post("/runs/cancel", json={"thread_id": thread_id, "run_ids": [run_id]})
    assert cancel_many.status_code == 204
