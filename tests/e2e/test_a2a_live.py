from __future__ import annotations

import json
import socket
import threading
import time
from pathlib import Path
from typing import Generator

import httpx
import pytest
import uvicorn
from a2a.client import ClientConfig, create_client
from a2a.types import GetTaskRequest, SendMessageRequest, TaskState
from google.protobuf.json_format import ParseDict

from agentseek_api.core import auth_middleware
from agentseek_api.main import create_app
from agentseek_api.services import langgraph_service as langgraph_service_module
from agentseek_api.settings import settings


class FakeCheckpointer:
    def __init__(self, connection_args: dict[str, str]) -> None:
        self.connection_args = connection_args

    def setup(self) -> None:
        return None

    def save_checkpoint(self, *, thread_id: str, run_id: str, payload: dict[str, object]) -> None:
        _ = (thread_id, run_id, payload)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_health(base_url: str, timeout_seconds: float) -> bool:
    deadline = time.time() + timeout_seconds
    with httpx.Client(timeout=1.0, trust_env=False) as client:
        while time.time() < deadline:
            try:
                response = client.get(f"{base_url}/health")
                if response.status_code == 200:
                    return True
            except Exception:  # noqa: BLE001
                pass
            time.sleep(0.1)
    return False


@pytest.fixture
def a2a_live_base_url(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Generator[str, None, None]:
    monkeypatch.setattr("agentseek_api.core.database.OceanBaseCheckpointSaver", FakeCheckpointer)
    monkeypatch.setattr(settings, "SEEKDB_URL", f"sqlite+aiosqlite:///{tmp_path}/test.db")
    monkeypatch.setattr(settings, "AUTH_TYPE", "custom")
    monkeypatch.setattr(settings, "AUTH_API_KEYS", "")
    monkeypatch.setattr(settings, "AUTH_MODULE_PATH", "examples/auth/custom_backend.py:backend")

    stress_graph_path = Path(__file__).resolve().parents[2] / "examples" / "graphs" / "stress_test" / "graph.py"
    config_path = tmp_path / "agentseek.json"
    config_path.write_text(
        json.dumps(
            {
                "graphs": {
                    "stress_test": {
                        "graph": f"{stress_graph_path}:build_graph",
                        "name": "Manifest Stress Graph",
                        "description": "A2A live interoperability graph",
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
    monkeypatch.setattr(settings, "AGENTSEEK_GRAPHS", str(config_path))
    auth_middleware._backend = None
    langgraph_service_module._langgraph_service = None

    app = create_app()
    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        if not _wait_for_health(base_url, 15.0):
            pytest.fail("Local A2A live server failed to become healthy")
        yield base_url
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        auth_middleware._backend = None
        langgraph_service_module._langgraph_service = None


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_a2a_sdk_client_can_fetch_card_stream_and_get_task(a2a_live_base_url: str) -> None:
    async with httpx.AsyncClient(
        headers={"x-user-id": "a2a-e2e-user"},
        timeout=10.0,
        trust_env=False,
    ) as http_client:
        assistant_response = await http_client.post(
            f"{a2a_live_base_url}/assistants",
            json={
                "name": "Live A2A",
                "graph_id": "stress_test",
                "description": "Live A2A test assistant",
            },
        )
        assert assistant_response.status_code == 200
        assistant_id = assistant_response.json()["assistant_id"]

        client = await create_client(
            a2a_live_base_url,
            client_config=ClientConfig(httpx_client=http_client),
            relative_card_path=f"/.well-known/agent-card.json?assistant_id={assistant_id}",
        )
        request = ParseDict(
            {
                "message": {
                    "messageId": "live-msg-1",
                    "role": "ROLE_USER",
                    "parts": [{"text": '{"delay":0.0,"steps":1,"note":"hello from a2a sdk"}'}],
                }
            },
            SendMessageRequest(),
        )

        observed_payloads: list[str] = []
        task_id: str | None = None
        async for event in client.send_message(request):
            payload = event.WhichOneof("payload")
            observed_payloads.append(payload or "")
            if payload in {"status_update", "artifact_update"}:
                task_id = getattr(getattr(event, payload), "task_id", task_id)

        assert "status_update" in observed_payloads
        assert "artifact_update" in observed_payloads
        assert task_id

        fetched = await client.get_task(GetTaskRequest(id=task_id))
        assert fetched.id == task_id
        assert TaskState.Name(fetched.status.state) == "TASK_STATE_COMPLETED"
        assert fetched.artifacts
        assert "hello from a2a sdk" in fetched.artifacts[0].parts[0].text
