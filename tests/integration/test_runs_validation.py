from collections.abc import Awaitable, Callable

from fastapi.testclient import TestClient

from agentseek_api.services.run_jobs import RunExecutionJob


def test_create_run_missing_input_returns_422(client: TestClient) -> None:
    assistant = client.post("/assistants", json={"name": "v-assistant", "graph_id": "default"})
    thread = client.post("/threads", json={"metadata": {}})
    assert assistant.status_code == 200
    assert thread.status_code == 200

    response = client.post(
        f"/threads/{thread.json()['thread_id']}/runs",
        json={"assistant_id": assistant.json()["assistant_id"]},
    )
    assert response.status_code == 422


def test_create_run_missing_assistant_id_returns_422(client: TestClient) -> None:
    thread = client.post("/threads", json={"metadata": {}})
    assert thread.status_code == 200
    response = client.post(
        f"/threads/{thread.json()['thread_id']}/runs",
        json={"input": {"m": "x"}},
    )
    assert response.status_code == 422


def test_create_run_on_busy_thread_returns_409(client: TestClient, monkeypatch) -> None:
    class DeferredExecutor:
        def __init__(self) -> None:
            self.submitted: list[Callable[[], Awaitable[None]] | RunExecutionJob] = []

        async def submit(self, job: Callable[[], Awaitable[None]] | RunExecutionJob) -> None:
            self.submitted.append(job)

    monkeypatch.setattr("agentseek_api.services.run_preparation.get_executor", lambda: DeferredExecutor())

    assistant = client.post("/assistants", json={"name": "busy-check", "graph_id": "stress_test"})
    thread = client.post("/threads", json={"metadata": {"busy": True}})
    assert assistant.status_code == 200
    assert thread.status_code == 200

    assistant_id = assistant.json()["assistant_id"]
    thread_id = thread.json()["thread_id"]

    first = client.post(
        f"/threads/{thread_id}/runs",
        json={"assistant_id": assistant_id, "input": {"delay": 0.05, "steps": 20}},
    )
    assert first.status_code == 200
    assert first.json()["status"] == "pending"

    second = client.post(
        f"/threads/{thread_id}/runs",
        json={"assistant_id": assistant_id, "input": {"delay": 0.05, "steps": 20}},
    )
    assert second.status_code == 409
    assert second.json()["detail"] == "Another run is already active for this thread"
