import json
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import pytest
from fastapi import Request
from fastapi.testclient import TestClient

from agentseek_api.core import auth_middleware
from agentseek_api.core.auth_deps import get_current_user
from agentseek_api.main import create_app
from agentseek_api.models.auth import User
from agentseek_api.services import langgraph_service as langgraph_service_module
from agentseek_api.services.run_jobs import RunExecutionJob
from agentseek_api.settings import settings


class FakeCheckpointer:
    def __init__(self, connection_args: dict[str, str]) -> None:
        self.connection_args = connection_args

    def setup(self) -> None:
        return None

    def save_checkpoint(self, *, thread_id: str, run_id: str, payload: dict[str, object]) -> None:
        _ = (thread_id, run_id, payload)


class InlineExecutor:
    async def submit(self, job):
        if callable(job):
            await job()
            return
        assert isinstance(job, RunExecutionJob)
        from agentseek_api.services.run_preparation import _execute_and_persist

        await _execute_and_persist(
            run_id=job.run_id,
            thread_id=job.thread_id,
            user_id=job.user_id,
            payload=job.payload,
            graph_id=job.graph_id,
            resume=job.resume,
            is_resume=job.is_resume,
        )


async def header_user_override(request: Request) -> User:
    identity = request.headers.get("x-user-id", "default_user")
    return User(identity=identity, is_authenticated=True)


def _write_agent_config(*, config_path: Path, stress_graph_path: Path) -> None:
    config_path.write_text(
        json.dumps(
            {
                "graphs": {
                    "stress_test": {
                        "graph": f"{stress_graph_path}:build_graph",
                        "name": "Manifest Stress Graph",
                        "description": "A2A integration graph",
                        "input_schema": {
                            "type": "object",
                            "properties": {"messages": {"type": "array"}},
                            "required": ["messages"],
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )


@contextmanager
def _a2a_client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[TestClient]:
    monkeypatch.setattr("agentseek_api.core.database.OceanBaseCheckpointSaver", FakeCheckpointer)
    monkeypatch.setattr("agentseek_api.services.run_preparation.get_executor", lambda: InlineExecutor())
    monkeypatch.setattr(settings, "SEEKDB_URL", f"sqlite+aiosqlite:///{tmp_path}/test.db")
    monkeypatch.setattr(settings, "AUTH_TYPE", "api_key")
    monkeypatch.setattr(settings, "AUTH_API_KEYS", "secret=api-user")
    monkeypatch.setattr(settings, "AUTH_MODULE_PATH", None)

    stress_graph_path = Path(__file__).resolve().parents[2] / "examples" / "graphs" / "stress_test" / "graph.py"
    config_path = tmp_path / "agentseek.json"
    _write_agent_config(config_path=config_path, stress_graph_path=stress_graph_path)
    monkeypatch.setattr(settings, "AGENTSEEK_GRAPHS", str(config_path))
    auth_middleware._backend = None
    langgraph_service_module._langgraph_service = None

    try:
        app = create_app()
        app.dependency_overrides[get_current_user] = header_user_override
        with TestClient(app) as client:
            yield client
    finally:
        auth_middleware._backend = None
        langgraph_service_module._langgraph_service = None


def _create_stress_assistant(client: TestClient) -> dict[str, object]:
    response = client.post(
        "/assistants",
        headers={"X-API-Key": "secret"},
        json={"name": "Messages Echo", "graph_id": "stress_test", "description": "Echoes text"},
    )
    response.raise_for_status()
    return response.json()


def test_message_send_returns_completed_task(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    with _a2a_client(monkeypatch, tmp_path) as client:
        assistant = _create_stress_assistant(client)

        response = client.post(
            f"/a2a/{assistant['assistant_id']}",
            headers={"X-API-Key": "secret", "Accept": "application/json"},
            json={
                "jsonrpc": "2.0",
                "id": "1",
                "method": "message/send",
                "params": {
                    "message": {
                        "role": "user",
                        "parts": [{"kind": "text", "text": '{"delay":0.0,"steps":1,"note":"hello from a2a"}'}],
                        "messageId": "msg-1",
                    }
                },
            },
        )

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["kind"] == "task"
    assert result["status"]["state"] == "completed"
    assert result["contextId"]
    assert "hello from a2a" in result["artifacts"][0]["parts"][0]["text"]


def test_tasks_get_returns_saved_snapshot(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    with _a2a_client(monkeypatch, tmp_path) as client:
        assistant = _create_stress_assistant(client)
        send_response = client.post(
            f"/a2a/{assistant['assistant_id']}",
            headers={"X-API-Key": "secret", "Accept": "application/json"},
            json={
                "jsonrpc": "2.0",
                "id": "2",
                "method": "message/send",
                "params": {
                    "message": {
                        "role": "user",
                        "parts": [{"kind": "text", "text": '{"delay":0.0,"steps":1,"note":"lookup me later"}'}],
                        "messageId": "msg-2",
                    }
                },
            },
        )
        task_id = send_response.json()["result"]["id"]

        get_response = client.post(
            f"/a2a/{assistant['assistant_id']}",
            headers={"X-API-Key": "secret", "Accept": "application/json"},
            json={
                "jsonrpc": "2.0",
                "id": "3",
                "method": "tasks/get",
                "params": {"id": task_id},
            },
        )

    assert get_response.status_code == 200
    assert get_response.json()["result"]["id"] == task_id
    assert get_response.json()["result"]["status"]["state"] == "completed"
