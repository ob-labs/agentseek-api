from pathlib import Path

from fastapi.testclient import TestClient

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


def test_health_endpoint(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_ok_endpoint(client: TestClient) -> None:
    response = client.get("/ok")
    assert response.status_code == 200
    assert response.json() == {"ok": True}


def test_docs_and_openapi_endpoints(client: TestClient) -> None:
    docs = client.get("/docs")
    redoc = client.get("/redoc")
    openapi = client.get("/openapi.json")

    assert docs.status_code == 200
    assert docs.headers["content-type"].startswith("text/html")
    assert redoc.status_code == 200
    assert redoc.headers["content-type"].startswith("text/html")
    assert openapi.status_code == 200
    assert openapi.headers["content-type"].startswith("application/json")
    assert openapi.json()["openapi"].startswith("3.")


def test_info_endpoint(client: TestClient) -> None:
    response = client.get("/info")
    assert response.status_code == 200
    body = response.json()
    assert body["version"]
    assert body["langgraph_py_version"]
    assert body["flags"]["agents"] is True
    assert body["flags"]["assistants"] is True
    assert body["flags"]["crons"] is True
    assert body["flags"]["mcp"] is True
    assert body["flags"]["protocol_v2"] is True
    assert isinstance(body["metadata"], dict)
    assert body["metadata"]["compatibility_tier"] == "oss-core"
    assert body["metadata"]["unsupported_features"] == [
        "distributed_runtime",
        "assistant_version_promotion",
    ]


def test_info_endpoint_reports_crons_supported(client: TestClient) -> None:
    response = client.get("/info")
    assert response.status_code == 200
    body = response.json()
    assert body["flags"]["crons"] is True
    assert "crons" not in body["metadata"]["unsupported_features"]


def test_info_endpoint_reports_mcp_runtime_state_from_startup(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("agentseek_api.core.database.OceanBaseCheckpointSaver", FakeCheckpointer)
    monkeypatch.setattr(settings, "SEEKDB_URL", f"sqlite+aiosqlite:///{tmp_path}/test.db")
    monkeypatch.setattr(settings, "AUTH_MODULE_PATH", None)
    graph_path = tmp_path / "graph.py"
    graph_path.write_text(
        """
from langgraph.graph import END, START, StateGraph

builder = StateGraph(dict)
builder.add_node("node", lambda state: state)
builder.add_edge(START, "node")
builder.add_edge("node", END)
graph = builder.compile()
""".strip(),
        encoding="utf-8",
    )
    config_path = tmp_path / "agentseek.json"
    config_path.write_text(
        """
{
  "graphs": {
    "chat": {
      "graph": "./graph.py:graph"
    }
  }
}
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setattr(settings, "AGENTSEEK_GRAPHS", str(config_path))
    auth_middleware._backend = None
    langgraph_service_module._langgraph_service = None

    app = create_app()
    config_path.write_text(
        """
{
  "graphs": {
    "chat": {
      "graph": "./graph.py:graph"
    }
  },
  "http": {
    "disable_mcp": true
  }
}
""".strip(),
        encoding="utf-8",
    )

    with TestClient(app) as test_client:
        response = test_client.get("/info")

    assert response.status_code == 200
    assert response.json()["flags"]["mcp"] is True
    auth_middleware._backend = None
    langgraph_service_module._langgraph_service = None


def test_info_endpoint_reports_a2a_runtime_state_from_startup(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("agentseek_api.core.database.OceanBaseCheckpointSaver", FakeCheckpointer)
    monkeypatch.setattr(settings, "SEEKDB_URL", f"sqlite+aiosqlite:///{tmp_path}/test.db")
    monkeypatch.setattr(settings, "AUTH_MODULE_PATH", None)
    graph_path = tmp_path / "graph.py"
    graph_path.write_text(
        """
from langgraph.graph import END, START, StateGraph

builder = StateGraph(dict)
builder.add_node("node", lambda state: state)
builder.add_edge(START, "node")
builder.add_edge("node", END)
graph = builder.compile()
""".strip(),
        encoding="utf-8",
    )
    config_path = tmp_path / "agentseek.json"
    config_path.write_text(
        """
{
  "graphs": {
    "chat": {
      "graph": "./graph.py:graph"
    }
  }
}
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setattr(settings, "AGENTSEEK_GRAPHS", str(config_path))
    auth_middleware._backend = None
    langgraph_service_module._langgraph_service = None

    app = create_app()
    config_path.write_text(
        """
{
  "graphs": {
    "chat": {
      "graph": "./graph.py:graph"
    }
  },
  "http": {
    "disable_a2a": true
  }
}
""".strip(),
        encoding="utf-8",
    )

    with TestClient(app) as test_client:
        response = test_client.get("/info")

    assert response.status_code == 200
    assert response.json()["flags"]["a2a"] is True
    auth_middleware._backend = None
    langgraph_service_module._langgraph_service = None


def test_metrics_endpoint_prometheus_default(client: TestClient) -> None:
    response = client.get("/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert "agentseek_api_info" in response.text


def test_metrics_endpoint_json_format(client: TestClient) -> None:
    response = client.get("/metrics?format=json")
    assert response.status_code == 200
    body = response.json()
    assert body["app_name"] == "AgentSeek API"
    assert body["checks"]["database"] == "ok"
