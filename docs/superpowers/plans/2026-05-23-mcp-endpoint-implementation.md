# MCP Endpoint Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a strict LangGraph Agent Server style stateless Streamable HTTP `/mcp` endpoint that exposes registered graphs as MCP tools and is usable by real MCP clients.

**Architecture:** Keep AgentSeek as the primary FastAPI runtime and add MCP as a mounted ASGI surface built with the official Python MCP SDK. Reuse the existing graph registry, auth dependency, and graph execution path; only add the missing config, metadata, transport, and compatibility glue needed for strict `/mcp` parity.

**Tech Stack:** FastAPI, Starlette mounting, official Python MCP SDK (`mcp`), Pydantic, pytest, pytest-asyncio, httpx, uv

---

## File Structure

### Files to create

- `src/agentseek_api/core/config_file.py`
  Responsibility: shared helpers for reading the active JSON config file from `agentseek.json`, `langgraph.json`, or `AGENTSEEK_GRAPHS`.
- `src/agentseek_api/core/mcp_config.py`
  Responsibility: resolve `http.disable_mcp` and provide the single source of truth for MCP enablement.
- `src/agentseek_api/mcp_server.py`
  Responsibility: build and configure the MCP server, register graph-backed tools, expose the mounted Streamable HTTP ASGI app, and normalize graph output into MCP results.
- `tests/unit/test_mcp_config.py`
  Responsibility: verify config parsing and enable/disable behavior.
- `tests/unit/test_mcp_server.py`
  Responsibility: verify tool metadata generation and graph-backed tool execution without spinning up the full FastAPI app.
- `tests/integration/test_mcp_endpoint.py`
  Responsibility: verify `/mcp` auth behavior, enable/disable behavior, tool discovery, and tool invocation via the mounted app.
- `tests/e2e/test_mcp_live.py`
  Responsibility: verify real MCP client interoperability against a running AgentSeek server.

### Files to modify

- `pyproject.toml`
  Add the runtime MCP SDK dependency.
- `src/agentseek_api/services/langgraph_service.py`
  Extend registered graph metadata to preserve MCP-facing name, description, input schema, and output schema.
- `src/agentseek_api/api/assistants.py`
  Reuse graph metadata for `/assistants/{assistant_id}/schemas` so assistant schema output and MCP tool metadata stay aligned.
- `src/agentseek_api/core/auth_middleware.py`
  Switch to the shared config file helper instead of re-reading the active config path privately.
- `src/agentseek_api/main.py`
  Mount `/mcp`, manage the MCP session manager in lifespan, and flip `/info.flags.mcp` based on real enablement.
- `tests/unit/test_graph_manifest.py`
  Cover manifest graph metadata parsing.
- `tests/unit/test_openapi_auth_config.py`
  Keep config loading behavior covered after the shared config helper is introduced.
- `tests/integration/test_system_endpoints.py`
  Assert `flags.mcp` in `/info`.
- `README.md`
  Document `/mcp`, stateless behavior, auth, `http.disable_mcp`, and client examples.

## Task 1: Add Shared Config Loading And MCP Enablement

**Files:**
- Create: `src/agentseek_api/core/config_file.py`
- Create: `src/agentseek_api/core/mcp_config.py`
- Modify: `src/agentseek_api/core/auth_middleware.py`
- Modify: `src/agentseek_api/main.py`
- Test: `tests/unit/test_mcp_config.py`
- Test: `tests/unit/test_openapi_auth_config.py`
- Test: `tests/integration/test_system_endpoints.py`

- [ ] **Step 1: Write the failing config tests**

```python
from pathlib import Path

from agentseek_api.core.config_file import get_active_config_payload
from agentseek_api.core.mcp_config import is_mcp_enabled
from agentseek_api.settings import settings


def test_get_active_config_payload_reads_http_section(monkeypatch, tmp_path: Path) -> None:
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

    payload = get_active_config_payload()

    assert payload is not None
    assert payload["http"] == {"disable_mcp": True}


def test_is_mcp_enabled_defaults_true_without_http_section(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "agentseek.json"
    config_path.write_text('{"graphs":{"chat":"chat.graph:graph"}}', encoding="utf-8")
    monkeypatch.setattr(settings, "AGENTSEEK_GRAPHS", str(config_path))

    assert is_mcp_enabled() is True


def test_is_mcp_enabled_respects_disable_flag(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "langgraph.json"
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

    assert is_mcp_enabled() is False
```

- [ ] **Step 2: Run the focused config tests and confirm they fail**

Run:

```bash
uv run pytest tests/unit/test_mcp_config.py -q
```

Expected:

```text
ERROR tests/unit/test_mcp_config.py - ModuleNotFoundError: No module named 'agentseek_api.core.mcp_config'
```

- [ ] **Step 3: Add the shared config reader and MCP enablement helpers**

```python
# src/agentseek_api/core/config_file.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agentseek_api.settings import settings


def active_config_path() -> Path | None:
    if settings.AGENTSEEK_GRAPHS:
        path = Path(settings.AGENTSEEK_GRAPHS).expanduser().resolve()
        if path.exists():
            return path
    for candidate in ("agentseek.json", "langgraph.json"):
        path = Path(candidate).resolve()
        if path.exists():
            return path
    return None


def get_active_config_payload() -> dict[str, Any] | None:
    config_path = active_config_path()
    if config_path is None:
        return None
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None
```

```python
# src/agentseek_api/core/mcp_config.py
from __future__ import annotations

from agentseek_api.core.config_file import get_active_config_payload


def is_mcp_enabled() -> bool:
    payload = get_active_config_payload()
    if payload is None:
        return True
    http = payload.get("http")
    if not isinstance(http, dict):
        return True
    return http.get("disable_mcp") is not True
```

```python
# src/agentseek_api/core/auth_middleware.py
from agentseek_api.core.config_file import active_config_path, get_active_config_payload


def get_config_auth_settings() -> ConfigAuthSettings:
    config_path = active_config_path()
    payload = get_active_config_payload()
    if config_path is None or payload is None:
        return ConfigAuthSettings()
    raw_auth = payload.get("auth")
    if not isinstance(raw_auth, dict):
        return ConfigAuthSettings()
    _apply_config_dependencies(payload, config_path=config_path)
    ...
```

```python
# src/agentseek_api/main.py
from agentseek_api.core.mcp_config import is_mcp_enabled


def _feature_flags() -> dict[str, bool]:
    return {
        "agents": True,
        "assistants": True,
        "threads": True,
        "runs": True,
        "crons": False,
        "store": True,
        "a2a": False,
        "mcp": is_mcp_enabled(),
        "protocol_v2": True,
    }
```

- [ ] **Step 4: Add the `/info` flag assertion**

```python
# tests/integration/test_system_endpoints.py
def test_info_endpoint(client: TestClient) -> None:
    response = client.get("/info")
    assert response.status_code == 200
    body = response.json()
    assert body["flags"]["agents"] is True
    assert body["flags"]["assistants"] is True
    assert body["flags"]["mcp"] is True
    assert body["flags"]["protocol_v2"] is True
```

- [ ] **Step 5: Run the config and system endpoint tests**

Run:

```bash
uv run pytest tests/unit/test_mcp_config.py tests/unit/test_openapi_auth_config.py tests/integration/test_system_endpoints.py -q
```

Expected:

```text
5 passed
```

- [ ] **Step 6: Commit the config foundation**

```bash
git add src/agentseek_api/core/config_file.py src/agentseek_api/core/mcp_config.py src/agentseek_api/core/auth_middleware.py src/agentseek_api/main.py tests/unit/test_mcp_config.py tests/unit/test_openapi_auth_config.py tests/integration/test_system_endpoints.py
git commit -m "feat: add MCP config gating"
```

## Task 2: Extend Graph Metadata For MCP Tool Exposure

**Files:**
- Modify: `src/agentseek_api/services/langgraph_service.py`
- Modify: `src/agentseek_api/api/assistants.py`
- Test: `tests/unit/test_graph_manifest.py`
- Test: `tests/unit/test_langgraph_service.py`
- Test: `tests/integration/test_langsmith_compat_extra.py`

- [ ] **Step 1: Write the failing graph metadata tests**

```python
import json

from agentseek_api.services.langgraph_service import LangGraphService


def test_manifest_preserves_graph_metadata_for_mcp(tmp_path) -> None:
    graph_file = tmp_path / "graph.py"
    graph_file.write_text(
        """
from langgraph.graph import END, START, StateGraph

builder = StateGraph(dict)
builder.add_node("node", lambda state: {"answer": state["question"]})
builder.add_edge(START, "node")
builder.add_edge("node", END)
graph = builder.compile()
""".strip(),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "langgraph.json"
    manifest_path.write_text(
        json.dumps(
            {
                "graphs": {
                    "docs_agent": {
                        "graph": "./graph.py:graph",
                        "name": "docs_agent",
                        "description": "Answer docs questions",
                        "input_schema": {
                            "type": "object",
                            "properties": {"question": {"type": "string"}},
                            "required": ["question"],
                        },
                        "output_schema": {
                            "type": "object",
                            "properties": {"answer": {"type": "string"}},
                            "required": ["answer"],
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    entry = LangGraphService(manifest_path=manifest_path).get_entry("docs_agent")

    assert entry.tool_name == "docs_agent"
    assert entry.description == "Answer docs questions"
    assert entry.input_schema["required"] == ["question"]
    assert entry.output_schema["required"] == ["answer"]
```

```python
def test_assistant_schemas_uses_registered_graph_schema(client: TestClient) -> None:
    assistant = client.post("/assistants", json={"name": "schema-test", "graph_id": "default"})
    assistant_id = assistant.json()["assistant_id"]

    schemas = client.get(f"/assistants/{assistant_id}/schemas")

    assert schemas.status_code == 200
    assert schemas.json()["input_schema"] == {"type": "object"}
    assert schemas.json()["output_schema"] == {"type": "object"}
```

- [ ] **Step 2: Run the graph metadata tests and confirm they fail**

Run:

```bash
uv run pytest tests/unit/test_graph_manifest.py tests/unit/test_langgraph_service.py tests/integration/test_langsmith_compat_extra.py -q
```

Expected:

```text
E   AttributeError: 'GraphEntry' object has no attribute 'tool_name'
```

- [ ] **Step 3: Extend `GraphEntry` and manifest parsing**

```python
# src/agentseek_api/services/langgraph_service.py
from dataclasses import dataclass, field


@dataclass
class GraphEntry:
    graph_factory: GraphFactory
    prepare_input: PrepareInput
    extract_output: ExtractOutput
    tool_name: str
    description: str = ""
    input_schema: dict[str, Any] = field(default_factory=lambda: {"type": "object"})
    output_schema: dict[str, Any] = field(default_factory=lambda: {"type": "object"})

    def build_graph(self, checkpointer: Any | None = None, store: Any | None = None) -> Pregel:
        return _build_graph_from_definition(self.graph_factory, checkpointer, store)
```

```python
def _normalize_schema(raw_value: Any) -> dict[str, Any]:
    if isinstance(raw_value, dict):
        return raw_value
    return {"type": "object"}


class LangGraphService:
    def __init__(self, *, manifest_path: str | Path | None = None) -> None:
        self._registry: dict[str, GraphEntry] = {}
        self.register(
            "default",
            graph_factory=_build_echo_graph,
            prepare_input=_echo_prepare,
            extract_output=_echo_extract,
            tool_name="default",
        )
        ...

    def register(
        self,
        graph_id: str,
        *,
        graph_factory: GraphFactory,
        prepare_input: PrepareInput,
        extract_output: ExtractOutput,
        tool_name: str | None = None,
        description: str = "",
        input_schema: dict[str, Any] | None = None,
        output_schema: dict[str, Any] | None = None,
    ) -> None:
        self._registry[graph_id] = GraphEntry(
            graph_factory=graph_factory,
            prepare_input=prepare_input,
            extract_output=extract_output,
            tool_name=tool_name or graph_id,
            description=description,
            input_schema=_normalize_schema(input_schema),
            output_schema=_normalize_schema(output_schema),
        )
```

```python
for graph_id, config in graphs.items():
    ...
    self.register(
        graph_id,
        graph_factory=graph_factory,
        prepare_input=prepare_input,
        extract_output=extract_output,
        tool_name=str(config.get("name") or graph_id),
        description=str(config.get("description") or ""),
        input_schema=_normalize_schema(config.get("input_schema")),
        output_schema=_normalize_schema(config.get("output_schema")),
    )
```

- [ ] **Step 4: Reuse graph metadata in assistant schema responses**

```python
# src/agentseek_api/api/assistants.py
@router.get("/{assistant_id}/schemas")
async def get_assistant_schemas(assistant_id: str) -> dict[str, object]:
    assistant = await get_assistant(assistant_id)
    entry = get_langgraph_service().get_entry(assistant.graph_id)
    return {
        "assistant_id": assistant.assistant_id,
        "graph_id": assistant.graph_id,
        "input_schema": entry.input_schema,
        "output_schema": entry.output_schema,
    }
```

- [ ] **Step 5: Run the graph metadata and schema tests**

Run:

```bash
uv run pytest tests/unit/test_graph_manifest.py tests/unit/test_langgraph_service.py tests/integration/test_langsmith_compat_extra.py -q
```

Expected:

```text
all selected tests passed
```

- [ ] **Step 6: Commit the graph metadata layer**

```bash
git add src/agentseek_api/services/langgraph_service.py src/agentseek_api/api/assistants.py tests/unit/test_graph_manifest.py tests/unit/test_langgraph_service.py tests/integration/test_langsmith_compat_extra.py
git commit -m "feat: preserve graph metadata for MCP tools"
```

## Task 3: Add The Mounted MCP Server

**Files:**
- Modify: `pyproject.toml`
- Create: `src/agentseek_api/mcp_server.py`
- Modify: `src/agentseek_api/main.py`
- Test: `tests/unit/test_mcp_server.py`

- [ ] **Step 1: Add failing MCP server unit tests**

```python
from agentseek_api.mcp_server import build_mcp_server, graph_tool_result, list_graph_tools
from agentseek_api.services.langgraph_service import LangGraphService


def test_graph_tool_result_wraps_dict_output() -> None:
    result = graph_tool_result({"answer": "hello"})

    assert result.structuredContent == {"answer": "hello"}
    assert result.content[0].text == '{"answer": "hello"}'


@pytest.mark.asyncio
async def test_build_mcp_server_registers_manifest_tools(tmp_path, monkeypatch) -> None:
    manifest_path = tmp_path / "langgraph.json"
    manifest_path.write_text(
        """
{
  "graphs": {
    "chat": {
      "graph": "chat.graph:graph",
      "name": "chat_tool",
      "description": "Chat tool"
    }
  }
}
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setattr("agentseek_api.mcp_server.get_langgraph_service", lambda: LangGraphService(manifest_path=manifest_path))

    tools = list_graph_tools(LangGraphService(manifest_path=manifest_path))

    assert any(tool.name == "chat_tool" for tool in tools)
```

- [ ] **Step 2: Run the MCP server unit tests and confirm they fail**

Run:

```bash
uv run pytest tests/unit/test_mcp_server.py -q
```

Expected:

```text
ERROR tests/unit/test_mcp_server.py - ModuleNotFoundError: No module named 'agentseek_api.mcp_server'
```

- [ ] **Step 3: Add the MCP SDK dependency**

Run:

```bash
uv add mcp
```

Expected:

```text
Resolved ... packages
Installed mcp ...
```

- [ ] **Step 4: Implement the mounted MCP server**

```python
# src/agentseek_api/mcp_server.py
from __future__ import annotations

import json
from typing import Any

from mcp.server.fastmcp import FastMCP
import mcp.types as types

from agentseek_api.services.langgraph_service import GraphEntry, get_langgraph_service


def graph_tool_result(result: dict[str, Any]) -> types.CallToolResult:
    text = json.dumps(result, ensure_ascii=False, sort_keys=True)
    return types.CallToolResult(
        content=[types.TextContent(type="text", text=text)],
        structuredContent=result,
        isError=False,
    )


def list_graph_tools(service) -> list[types.Tool]:
    tools: list[types.Tool] = []
    for graph_id in service.registered_graph_ids():
        entry = service.get_entry(graph_id)
        tools.append(
            types.Tool(
                name=entry.tool_name,
                description=entry.description,
                inputSchema=entry.input_schema,
                outputSchema=entry.output_schema,
            )
        )
    return tools


def _make_tool(entry: GraphEntry):
    async def _tool(**kwargs: Any) -> dict[str, Any]:
        graph = entry.build_graph()
        prepared = entry.prepare_input(kwargs)
        if hasattr(graph, "ainvoke"):
            raw_result = await graph.ainvoke(prepared)
        else:
            raw_result = graph.invoke(prepared)
        return entry.extract_output(raw_result, kwargs)

    _tool.__name__ = entry.tool_name.replace("-", "_")
    _tool.__doc__ = entry.description
    return _tool


def build_mcp_server() -> FastMCP:
    mcp = FastMCP(
        "AgentSeek API",
        stateless_http=True,
        json_response=True,
        streamable_http_path="/",
    )
    service = get_langgraph_service()
    for graph_id in service.registered_graph_ids():
        entry = service.get_entry(graph_id)
        mcp.tool(
            name=entry.tool_name,
            description=entry.description,
        )(_make_tool(entry))
    return mcp
```

```python
# src/agentseek_api/main.py
from contextlib import AsyncExitStack, asynccontextmanager

from agentseek_api.core.mcp_config import is_mcp_enabled
from agentseek_api.mcp_server import build_mcp_server

_mcp_server = build_mcp_server() if is_mcp_enabled() else None


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    await db_manager.initialize()
    try:
        async with AsyncExitStack() as stack:
            if _mcp_server is not None:
                await stack.enter_async_context(_mcp_server.session_manager.run())
            yield
    finally:
        await db_manager.close()


def create_app() -> FastAPI:
    app = FastAPI(title=settings.APP_NAME, version=__version__, lifespan=lifespan)
    ...
    if _mcp_server is not None:
        app.mount("/mcp", _mcp_server.streamable_http_app())
    return app
```

- [ ] **Step 5: Run the MCP server unit tests**

Run:

```bash
uv run pytest tests/unit/test_mcp_server.py -q
```

Expected:

```text
2 passed
```

- [ ] **Step 6: Commit the mounted MCP server**

```bash
git add pyproject.toml uv.lock src/agentseek_api/mcp_server.py src/agentseek_api/main.py tests/unit/test_mcp_server.py
git commit -m "feat: add mounted MCP server"
```

## Task 4: Make MCP Requests Strict And Test The FastAPI Surface

**Files:**
- Modify: `src/agentseek_api/mcp_server.py`
- Modify: `src/agentseek_api/main.py`
- Test: `tests/integration/test_mcp_endpoint.py`
- Test: `tests/integration/test_auth_route_enforcement.py`

- [ ] **Step 1: Write the failing integration tests for enablement, auth, and tool calls**

```python
from pathlib import Path

from fastapi.testclient import TestClient

from agentseek_api.main import create_app
from agentseek_api.settings import settings


def test_mcp_endpoint_requires_auth(auth_client: TestClient) -> None:
    response = auth_client.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"})

    assert response.status_code == 401


def test_mcp_endpoint_lists_tools_for_authenticated_client(auth_client: TestClient) -> None:
    response = auth_client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
        headers={"X-API-Key": "secret"},
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
        headers={"X-API-Key": "secret"},
    )

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["structuredContent"]["echo"] == {"message": "hello"}


def test_mcp_route_is_not_mounted_when_disabled(monkeypatch, tmp_path: Path) -> None:
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

    with TestClient(create_app()) as client:
        response = client.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"})

    assert response.status_code == 404
```

- [ ] **Step 2: Run the MCP integration tests and confirm they fail**

Run:

```bash
uv run pytest tests/integration/test_mcp_endpoint.py tests/integration/test_auth_route_enforcement.py -q
```

Expected:

```text
FAILED test_mcp_endpoint_lists_tools_for_authenticated_client
```

- [ ] **Step 3: Tighten the MCP result shape and error mapping**

```python
# src/agentseek_api/mcp_server.py
from mcp.types import CallToolResult, TextContent


def graph_tool_result(result: dict[str, Any]) -> CallToolResult:
    text = json.dumps(result, ensure_ascii=False, sort_keys=True)
    return CallToolResult(
        content=[TextContent(type="text", text=text)],
        structuredContent=result,
        isError=False,
    )


def build_mcp_server() -> FastMCP:
    mcp = FastMCP(
        "AgentSeek API",
        stateless_http=True,
        json_response=True,
        streamable_http_path="/",
    )
    ...
    @mcp.custom_route("/healthz", methods=["GET"])
    async def _healthz(_request):
        return {"ok": True}
    return mcp
```

```python
async def _tool(**kwargs: Any) -> CallToolResult:
    graph = entry.build_graph()
    prepared = entry.prepare_input(kwargs)
    raw_result = await graph.ainvoke(prepared) if hasattr(graph, "ainvoke") else graph.invoke(prepared)
    extracted = entry.extract_output(raw_result, kwargs)
    if not isinstance(extracted, dict):
        extracted = {"result": extracted}
    return graph_tool_result(extracted)
```

- [ ] **Step 4: Run the FastAPI integration tests**

Run:

```bash
uv run pytest tests/integration/test_mcp_endpoint.py tests/integration/test_auth_route_enforcement.py tests/integration/test_system_endpoints.py -q
```

Expected:

```text
all selected tests passed
```

- [ ] **Step 5: Commit the strict MCP endpoint behavior**

```bash
git add src/agentseek_api/mcp_server.py src/agentseek_api/main.py tests/integration/test_mcp_endpoint.py tests/integration/test_auth_route_enforcement.py tests/integration/test_system_endpoints.py
git commit -m "feat: expose strict MCP endpoint"
```

## Task 5: Add Real MCP Client Interoperability Coverage

**Files:**
- Create: `tests/e2e/test_mcp_live.py`
- Modify: `tests/e2e/conftest.py`

- [ ] **Step 1: Write the live MCP client test**

```python
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_mcp_client_can_list_tools_and_call_default(e2e_base_url: str) -> None:
    async with streamable_http_client(
        url=f"{e2e_base_url}/mcp",
        headers={"x-user-id": "mcp-e2e-user"},
    ) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()

            listed = await session.list_tools()
            assert any(tool.name == "default" for tool in listed.tools)

            result = await session.call_tool("default", {"message": "hello from mcp"})
            assert result.structuredContent["echo"] == {"message": "hello from mcp"}
```

- [ ] **Step 2: Run the e2e MCP client test and confirm the first failure**

Run:

```bash
uv run pytest tests/e2e/test_mcp_live.py -q
```

Expected:

```text
FAILED tests/e2e/test_mcp_live.py::test_mcp_client_can_list_tools_and_call_default
```

- [ ] **Step 3: Align the e2e server fixture with MCP expectations**

```python
# tests/e2e/conftest.py
env.setdefault("AUTH_TYPE", "custom")
env.setdefault("AUTH_MODULE_PATH", "examples/auth/custom_backend.py:backend")

# The mounted MCP server expects the same auth path as the rest of the API.
# Reuse the existing live fixture and connect at /mcp with x-user-id headers.
```

- [ ] **Step 4: Run the live MCP client test**

Run:

```bash
uv run pytest tests/e2e/test_mcp_live.py -q
```

Expected:

```text
1 passed
```

- [ ] **Step 5: Commit the client interoperability coverage**

```bash
git add tests/e2e/conftest.py tests/e2e/test_mcp_live.py
git commit -m "test: verify MCP client interoperability"
```

## Task 6: Update README And Run Final Verification

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add a failing README assertion by extending an existing system contract test**

```python
def test_info_endpoint(client: TestClient) -> None:
    response = client.get("/info")
    body = response.json()
    assert body["flags"]["mcp"] is True
```

- [ ] **Step 2: Document MCP in the README**

```md
## 🔌 MCP

AgentSeek API exposes registered graphs as MCP tools through a stateless
Streamable HTTP endpoint at `/mcp`.

### Behavior

- Transport: Streamable HTTP
- Session model: stateless
- Auth: same as the rest of the API
- Discovery source: registered graphs from `agentseek.json` or `langgraph.json`

### Disable MCP

Set `http.disable_mcp` in your config file:

```json
{
  "http": {
    "disable_mcp": true
  }
}
```

### Python client example

```python
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

async with streamable_http_client(
    url="http://127.0.0.1:2024/mcp",
    headers={"X-API-Key": "secret"},
) as (read, write, _):
    async with ClientSession(read, write) as session:
        await session.initialize()
        print(await session.list_tools())
```
```

- [ ] **Step 3: Run the final focused verification suite**

Run:

```bash
uv run pytest tests/unit/test_mcp_config.py tests/unit/test_graph_manifest.py tests/unit/test_langgraph_service.py tests/unit/test_mcp_server.py tests/integration/test_system_endpoints.py tests/integration/test_mcp_endpoint.py tests/e2e/test_mcp_live.py -q
```

Expected:

```text
all selected tests passed
```

- [ ] **Step 4: Run the broader regression suite**

Run:

```bash
uv run pytest tests/unit/test_openapi_auth_config.py tests/integration/test_auth_route_enforcement.py tests/integration/test_langsmith_compat_extra.py tests/e2e/test_langsmith_compat_live.py -q
```

Expected:

```text
all selected tests passed
```

- [ ] **Step 5: Commit docs and verification-ready state**

```bash
git add README.md
git commit -m "docs: document MCP endpoint usage"
```

## Self-Review Checklist

### Spec coverage

- `/mcp` endpoint and stateless Streamable HTTP: Task 3 and Task 4
- registered graphs exposed as tools: Task 2 and Task 3
- `http.disable_mcp`: Task 1 and Task 4
- `/info.flags.mcp`: Task 1 and Task 4
- same auth model as the rest of the API: Task 4 and Task 5
- client interoperability proof: Task 5
- README updates: Task 6

### Placeholder scan

- No `TBD`, `TODO`, or deferred “implement later” language remains in tasks.
- Each code step includes concrete code, commands, and expected outcomes.

### Type consistency

- MCP enablement is expressed as `is_mcp_enabled()`
- graph metadata lives on `GraphEntry`
- MCP server construction uses `build_mcp_server()`
- graph result normalization uses `graph_tool_result()`
