from pathlib import Path

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

    def save_checkpoint(self, *, thread_id: str, run_id: str, payload: dict[str, object]) -> None:
        _ = (thread_id, run_id, payload)


def _mcp_headers(api_key: str | None = None) -> dict[str, str]:
    headers = {"Accept": "application/json"}
    if api_key is not None:
        headers["X-API-Key"] = api_key
    return headers


@pytest.fixture
def auth_client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> TestClient:
    monkeypatch.setattr("agentseek_api.core.database.OceanBaseCheckpointSaver", FakeCheckpointer)
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


def test_mcp_endpoint_requires_auth(auth_client: TestClient) -> None:
    response = auth_client.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"})

    assert response.status_code == 401


def test_mcp_endpoint_lists_tools_for_authenticated_client(auth_client: TestClient) -> None:
    response = auth_client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
        headers=_mcp_headers("secret"),
    )

    assert response.status_code == 200
    body = response.json()
    assert body["result"]["tools"]
    assert any(tool["name"] == "default" for tool in body["result"]["tools"])


def test_mcp_endpoint_can_call_graph_tool(auth_client: TestClient) -> None:
    response = auth_client.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": "default", "arguments": {"message": "hello"}},
        },
        headers=_mcp_headers("secret"),
    )

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["structuredContent"]["echo"] == {"message": "hello"}


def test_mcp_route_is_not_mounted_when_disabled(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("agentseek_api.core.database.OceanBaseCheckpointSaver", FakeCheckpointer)
    monkeypatch.setattr(settings, "SEEKDB_URL", f"sqlite+aiosqlite:///{tmp_path}/test.db")
    monkeypatch.setattr(settings, "AUTH_TYPE", "noop")
    monkeypatch.setattr(settings, "AUTH_API_KEYS", "")
    monkeypatch.setattr(settings, "AUTH_MODULE_PATH", None)
    config_path = tmp_path / "agentseek.json"
    config_path.write_text(
        """
{
  "graphs": {
    "chat": "chat.graph:graph"
  },
  "http": {
    "disable_mcp": true
  }
}
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setattr(settings, "AGENTSEEK_GRAPHS", str(config_path))
    auth_middleware._backend = None

    with TestClient(create_app()) as client:
        response = client.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"})

    assert response.status_code == 404
