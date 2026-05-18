from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from agentseek_api.core import auth_middleware
from agentseek_api.main import create_app
from agentseek_api.settings import settings


class FakeCheckpointer:
    def __init__(self, connection_args: dict[str, str]) -> None:
        self.connection_args = connection_args

    def setup(self) -> None:
        return None

    def save_checkpoint(self, *, thread_id: str, run_id: str, payload: dict[str, Any]) -> None:
        _ = (thread_id, run_id, payload)


class InlineExecutor:
    async def submit(self, func: Callable[[], Awaitable[None]]) -> None:
        await func()


@pytest.fixture
def auth_client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> TestClient:
    monkeypatch.setattr("agentseek_api.core.database.OceanBaseCheckpointSaver", FakeCheckpointer)
    monkeypatch.setattr("agentseek_api.services.run_preparation.get_executor", lambda: InlineExecutor())
    monkeypatch.setattr(settings, "SEEKDB_URL", f"sqlite+aiosqlite:///{tmp_path}/test.db")
    monkeypatch.setattr(settings, "AUTH_TYPE", "api_key")
    monkeypatch.setattr(settings, "AUTH_API_KEYS", "secret=api-user")
    monkeypatch.setattr(settings, "AUTH_MODULE_PATH", None)
    monkeypatch.setattr(settings, "AGENTSEEK_GRAPHS", None)
    auth_middleware._backend = None

    app = create_app()
    with TestClient(app) as test_client:
        yield test_client
    auth_middleware._backend = None


def test_api_key_auth_protects_user_scoped_routes_and_leaves_assistants_public(auth_client: TestClient) -> None:
    assistant = auth_client.post("/assistants", json={"name": "public-assistant", "graph_id": "default"})
    assert assistant.status_code == 200
    assistant_id = assistant.json()["assistant_id"]

    assistant_list = auth_client.get("/assistants")
    assert assistant_list.status_code == 200
    assert any(item["assistant_id"] == assistant_id for item in assistant_list.json())

    missing_thread = auth_client.post("/threads", json={"metadata": {"scope": "auth"}})
    assert missing_thread.status_code == 401

    valid_thread = auth_client.post(
        "/threads",
        json={"metadata": {"scope": "auth"}},
        headers={"X-API-Key": "secret"},
    )
    assert valid_thread.status_code == 200
    assert valid_thread.json()["user_id"] == "api-user"
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
        headers={"X-API-Key": "secret"},
    )
    assert valid_stateless_run.status_code == 200
    assert valid_stateless_run.json()["assistant_id"] == assistant_id
