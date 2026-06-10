import json
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

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
            kwargs=job.kwargs,
            resume=job.resume,
            is_resume=job.is_resume,
        )


async def header_user_override(request: Request) -> User:
    identity = request.headers.get("x-user-id", "default_user")
    return User(identity=identity, is_authenticated=True)


def _write_agent_config(*, config_path: Path, stress_graph_path: Path, disable_a2a: bool) -> None:
    payload: dict[str, object] = {
        "graphs": {
            "stress_test": {
                "graph": f"{stress_graph_path}:build_graph",
                "name": "Manifest Stress Graph",
                "description": "Manifest graph description should lose to assistant metadata",
                "input_schema": {
                    "type": "object",
                    "properties": {"messages": {"type": "array"}},
                    "required": ["messages"],
                },
            }
        }
    }
    if disable_a2a:
        payload["http"] = {"disable_a2a": True}
    config_path.write_text(json.dumps(payload), encoding="utf-8")


@contextmanager
def _agent_card_client(
    monkeypatch,
    tmp_path: Path,
    *,
    disable_a2a: bool = False,
) -> Iterator[TestClient]:
    monkeypatch.setattr("agentseek_api.core.database.OceanBaseCheckpointSaver", FakeCheckpointer)
    monkeypatch.setattr("agentseek_api.services.run_preparation.get_executor", lambda: InlineExecutor())
    monkeypatch.setattr(settings, "SEEKDB_URL", f"sqlite+aiosqlite:///{tmp_path}/test.db")
    monkeypatch.setattr(settings, "AUTH_MODULE_PATH", None)

    stress_graph_path = Path(__file__).resolve().parents[2] / "examples" / "graphs" / "stress_test" / "graph.py"
    config_path = tmp_path / "agentseek.json"
    _write_agent_config(config_path=config_path, stress_graph_path=stress_graph_path, disable_a2a=disable_a2a)
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


def test_agent_card_endpoint_returns_assistant_shaped_card(monkeypatch, tmp_path: Path) -> None:
    with _agent_card_client(monkeypatch, tmp_path) as client:
        assistant = client.post(
            "/assistants",
            json={
                "name": "Stress Agent",
                "description": "Deterministic agent card coverage",
                "graph_id": "stress_test",
            },
        )
        assistant.raise_for_status()
        assistant_id = assistant.json()["assistant_id"]

        response = client.get(f"/.well-known/agent-card.json?assistant_id={assistant_id}")

    assert response.status_code == 200
    body = response.json()
    assert body["name"] == "Stress Agent"
    assert body["description"] == "Deterministic agent card coverage"
    assert body["capabilities"] == {"streaming": True, "pushNotifications": False}
    assert body["name"] != "Manifest Stress Graph"
    assert body["description"] != "Manifest graph description should lose to assistant metadata"
    assert body["supportedInterfaces"][0]["url"].endswith(f"/a2a/{assistant_id}")


def test_agent_card_endpoint_is_not_mounted_when_a2a_is_disabled(monkeypatch, tmp_path: Path) -> None:
    with _agent_card_client(monkeypatch, tmp_path, disable_a2a=True) as client:
        response = client.get("/.well-known/agent-card.json?assistant_id=assistant-123")

    assert response.status_code == 404


    assert body["securityRequirements"] == [{"apiKeyAuth": []}]
