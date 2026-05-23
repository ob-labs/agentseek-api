from __future__ import annotations

import socket
import threading
import time
from pathlib import Path
from typing import Generator

import httpx
import pytest
import uvicorn
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

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
def mcp_live_base_url(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Generator[str, None, None]:
    monkeypatch.setattr("agentseek_api.core.database.OceanBaseCheckpointSaver", FakeCheckpointer)
    monkeypatch.setattr(settings, "SEEKDB_URL", f"sqlite+aiosqlite:///{tmp_path}/test.db")
    monkeypatch.setattr(settings, "AUTH_TYPE", "custom")
    monkeypatch.setattr(settings, "AUTH_API_KEYS", "")
    monkeypatch.setattr(settings, "AUTH_MODULE_PATH", "examples/auth/custom_backend.py:backend")
    monkeypatch.setattr(settings, "AGENTSEEK_GRAPHS", None)
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
            pytest.fail("Local MCP live server failed to become healthy")
        yield base_url
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        auth_middleware._backend = None
        langgraph_service_module._langgraph_service = None


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_mcp_client_can_list_tools_and_call_default(mcp_live_base_url: str) -> None:
    async with httpx.AsyncClient(
        headers={"x-user-id": "mcp-e2e-user"},
        timeout=10.0,
        trust_env=False,
    ) as http_client:
        async with streamable_http_client(
            url=f"{mcp_live_base_url}/mcp",
            http_client=http_client,
        ) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()

                listed = await session.list_tools()
                assert any(tool.name == "default" for tool in listed.tools)

                result = await session.call_tool("default", {"message": "hello from mcp"})
                assert result.structuredContent["echo"] == {"message": "hello from mcp"}
