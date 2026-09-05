from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from agentseek_api.core import auth_middleware
from agentseek_api.main import create_app
from agentseek_api.services.run_jobs import RunExecutionJob
from agentseek_api.settings import settings


class FakeCheckpointer:
    def __init__(self, connection_args: dict[str, str]) -> None:
        self.connection_args = connection_args

    def setup(self) -> None:
        return None

    def save_checkpoint(self, *, thread_id: str, run_id: str, payload: dict[str, Any]) -> None:
        _ = (thread_id, run_id, payload)


class InlineExecutor:
    async def submit(self, job: Callable[[], Awaitable[None]] | RunExecutionJob) -> None:
        if callable(job):
            await job()
            return
        from agentseek_api.services.run_preparation import _execute_and_persist

        await _execute_and_persist(
            run_id=job.run_id,
            thread_id=job.thread_id,
            user_id=job.user_id,
            payload=job.payload,
            graph_id=job.graph_id,
            kwargs=job.kwargs,
            resume=job.resume,
            is_resume=job.is_resume,
        )


def _write_auth_file(tmp_path: Path) -> str:
    auth_file = tmp_path / "test_auth.py"
    auth_file.write_text(
        """
from langgraph_sdk import Auth

auth = Auth()

@auth.authenticate
async def get_current_user(authorization: str | None):
    if not authorization:
        raise Auth.exceptions.HTTPException(status_code=401, detail="Missing token")
    return {"identity": "test-user"}
""".strip(),
        encoding="utf-8",
    )
    return f"{auth_file}:auth"


@pytest.fixture
def auth_client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> TestClient:
    monkeypatch.setattr("agentseek_api.core.database.OceanBaseCheckpointSaver", FakeCheckpointer)
    monkeypatch.setattr("agentseek_api.services.run_preparation.get_executor", lambda: InlineExecutor())
    monkeypatch.setattr(settings, "SEEKDB_URL", f"sqlite+aiosqlite:///{tmp_path}/test.db")
    monkeypatch.setattr(settings, "AUTH_MODULE_PATH", _write_auth_file(tmp_path))
    monkeypatch.setattr(settings, "AGENTSEEK_GRAPHS", None)
    auth_middleware._backend = None

    app = create_app()
    with TestClient(app, client=("127.0.0.1", 50000)) as test_client:
        yield test_client
    auth_middleware._backend = None


@pytest.fixture
def local_dev_auth_client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> TestClient:
    monkeypatch.setattr("agentseek_api.core.database.OceanBaseCheckpointSaver", FakeCheckpointer)
    monkeypatch.setattr("agentseek_api.services.run_preparation.get_executor", lambda: InlineExecutor())
    monkeypatch.setattr(settings, "SEEKDB_URL", f"sqlite+aiosqlite:///{tmp_path}/test.db")
    monkeypatch.setattr(settings, "AUTH_MODULE_PATH", _write_auth_file(tmp_path))
    monkeypatch.setattr(settings, "STUDIO_AUTH_LOCAL_DEV", True)
    monkeypatch.setattr(settings, "AGENTSEEK_GRAPHS", None)

    app = create_app()
    with TestClient(app, client=("127.0.0.1", 50000)) as test_client:
        yield test_client
    auth_middleware._backend = None


@pytest.fixture
def studio_auth_disabled_client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> TestClient:
    monkeypatch.setattr("agentseek_api.core.database.OceanBaseCheckpointSaver", FakeCheckpointer)
    monkeypatch.setattr("agentseek_api.services.run_preparation.get_executor", lambda: InlineExecutor())
    monkeypatch.setattr(settings, "SEEKDB_URL", f"sqlite+aiosqlite:///{tmp_path}/test.db")
    monkeypatch.setattr(settings, "AUTH_MODULE_PATH", _write_auth_file(tmp_path))
    monkeypatch.setattr(settings, "STUDIO_AUTH_LOCAL_DEV", True)
    config_path = tmp_path / "agentseek.json"
    config_path.write_text(
        """
{
  "graphs": {
    "chat": "chat.graph:graph"
  },
  "auth": {
    "disable_studio_auth": true
  }
}
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setattr(settings, "AGENTSEEK_GRAPHS", str(config_path))
    auth_middleware._backend = None

    app = create_app()
    with TestClient(app, client=("127.0.0.1", 50000)) as test_client:
        yield test_client
    auth_middleware._backend = None


def test_auth_protects_assistant_thread_and_run_routes(auth_client: TestClient) -> None:
    missing_assistant = auth_client.post("/assistants", json={"name": "protected-assistant", "graph_id": "default"})
    assert missing_assistant.status_code == 401

    assistant = auth_client.post(
        "/assistants",
        json={"name": "protected-assistant", "graph_id": "default"},
        headers={"Authorization": "Bearer test-token"},
    )
    assert assistant.status_code == 200
    assistant_id = assistant.json()["assistant_id"]

    missing_assistant_list = auth_client.post("/assistants/search", json={})
    assert missing_assistant_list.status_code == 401

    assistant_list = auth_client.post("/assistants/search", json={}, headers={"Authorization": "Bearer test-token"})
    assert assistant_list.status_code == 200
    assert any(item["assistant_id"] == assistant_id for item in assistant_list.json())

    missing_thread = auth_client.post("/threads", json={"metadata": {"scope": "auth"}})
    assert missing_thread.status_code == 401

    valid_thread = auth_client.post(
        "/threads",
        json={"metadata": {"scope": "auth"}},
        headers={"Authorization": "Bearer test-token"},
    )
    assert valid_thread.status_code == 200
    assert "thread_id" in valid_thread.json()
    thread_id = valid_thread.json()["thread_id"]

    missing_thread_run = auth_client.post(
        f"/threads/{thread_id}/runs",
        json={"assistant_id": assistant_id, "input": {"message": "hello"}},
    )
    assert missing_thread_run.status_code == 401

    missing_stateless_run = auth_client.post(
        "/runs",
        json={"assistant_id": assistant_id, "input": {"message": "hello"}},
    )
    assert missing_stateless_run.status_code == 401

    valid_stateless_run = auth_client.post(
        "/runs",
        json={"assistant_id": assistant_id, "input": {"message": "hello"}},
        headers={"Authorization": "Bearer test-token"},
    )
    assert valid_stateless_run.status_code == 200
    assert valid_stateless_run.json()["assistant_id"] == assistant_id


def test_studio_requests_bypass_auth_in_local_dev(local_dev_auth_client: TestClient) -> None:
    assistant = local_dev_auth_client.post(
        "/assistants",
        json={"name": "studio-assistant", "graph_id": "default"},
        headers={"x-auth-scheme": "langsmith"},
    )
    assert assistant.status_code == 200
    assistant_id = assistant.json()["assistant_id"]

    thread = local_dev_auth_client.post(
        "/threads",
        json={"metadata": {"scope": "studio"}},
        headers={"x-auth-scheme": "langsmith"},
    )
    assert thread.status_code == 200
    assert "thread_id" in thread.json()
    thread_id = thread.json()["thread_id"]

    run = local_dev_auth_client.post(
        f"/threads/{thread_id}/runs",
        json={"assistant_id": assistant_id, "input": {"message": "studio"}},
        headers={"x-auth-scheme": "langsmith"},
    )
    assert run.status_code == 200


def test_studio_requests_respect_disable_studio_auth_flag(studio_auth_disabled_client: TestClient) -> None:
    response = studio_auth_disabled_client.post(
        "/assistants",
        json={"name": "blocked-studio", "graph_id": "default"},
        headers={"x-auth-scheme": "langsmith"},
    )

    assert response.status_code == 401


def test_studio_header_does_not_bypass_auth_outside_local_dev(auth_client: TestClient) -> None:
    response = auth_client.post(
        "/assistants",
        json={"name": "blocked-studio", "graph_id": "default"},
        headers={"x-auth-scheme": "langsmith"},
    )

    assert response.status_code == 401
