from __future__ import annotations

from contextvars import ContextVar
import json
from dataclasses import dataclass
from typing import Any

from starlette.requests import Request
from starlette.responses import JSONResponse
from mcp.server.lowlevel.server import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
import mcp.types as types

from agentseek_api import __version__
from agentseek_api.core.auth_middleware import get_auth_backend
from agentseek_api.models.auth import User
from agentseek_api.services.langgraph_service import GraphEntry, LangGraphService, get_langgraph_service

_current_mcp_user: ContextVar[User | None] = ContextVar("current_mcp_user", default=None)


class _StreamableHTTPASGIApp:
    def __init__(self, session_manager: StreamableHTTPSessionManager) -> None:
        self.session_manager = session_manager

    async def __call__(self, scope, receive, send) -> None:  # pragma: no cover
        await self.session_manager.handle_request(scope, receive, send)


class _AuthenticatedMCPASGIApp:
    def __init__(self, inner_app: _StreamableHTTPASGIApp) -> None:
        self.inner_app = inner_app

    async def __call__(self, scope, receive, send) -> None:  # pragma: no cover
        if scope["type"] != "http":
            await self.inner_app(scope, receive, send)
            return

        request = Request(scope, receive=receive)
        user = await get_auth_backend().authenticate(request)
        if not user.is_authenticated:
            response = JSONResponse({"detail": "Not authenticated"}, status_code=401)
            await response(scope, receive, send)
            return

        token = _current_mcp_user.set(user)
        try:
            await self.inner_app(scope, receive, send)
        finally:
            _current_mcp_user.reset(token)


@dataclass
class MCPMount:
    server: Server
    session_manager: StreamableHTTPSessionManager
    app: _AuthenticatedMCPASGIApp


def graph_tool_result(result: dict[str, Any]) -> types.CallToolResult:
    text = json.dumps(result, ensure_ascii=False, sort_keys=True)
    return types.CallToolResult(
        content=[types.TextContent(type="text", text=text)],
        structuredContent=result,
        isError=False,
    )


def list_graph_tools(service: LangGraphService) -> list[types.Tool]:
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


def _entry_for_tool(service: LangGraphService, tool_name: str) -> GraphEntry:
    for graph_id in service.registered_graph_ids():
        entry = service.get_entry(graph_id)
        if entry.tool_name == tool_name:
            return entry
    raise ValueError(f"Unknown tool: {tool_name}")


async def _invoke_graph_tool(entry: GraphEntry, arguments: dict[str, Any]) -> types.CallToolResult:
    graph = entry.build_graph()
    prepared = entry.prepare_input(arguments)
    if hasattr(graph, "ainvoke"):
        raw_result = await graph.ainvoke(prepared)
    else:  # pragma: no cover
        raw_result = graph.invoke(prepared)
    extracted = entry.extract_output(raw_result, arguments)
    if not isinstance(extracted, dict):
        extracted = {"result": extracted}
    return graph_tool_result(extracted)


def build_mcp_server(service: LangGraphService | None = None) -> Server:
    resolved_service = service or get_langgraph_service()
    server: Server = Server("AgentSeek API", version=__version__)

    @server.list_tools()
    async def _list_tools() -> list[types.Tool]:
        return list_graph_tools(resolved_service)

    @server.call_tool()
    async def _call_tool(name: str, arguments: dict[str, Any]) -> types.CallToolResult:
        entry = _entry_for_tool(resolved_service, name)
        return await _invoke_graph_tool(entry, arguments)

    return server


def build_mcp_mount(service: LangGraphService | None = None) -> MCPMount:
    server = build_mcp_server(service=service)
    session_manager = StreamableHTTPSessionManager(
        app=server,
        json_response=True,
        stateless=True,
    )
    http_app = _StreamableHTTPASGIApp(session_manager)
    return MCPMount(
        server=server,
        session_manager=session_manager,
        app=_AuthenticatedMCPASGIApp(http_app),
    )
