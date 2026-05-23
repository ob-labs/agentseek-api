from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
import inspect
from typing import Any
from uuid import uuid4

from fastapi import HTTPException
from langgraph.constants import CONF, CONFIG_KEY_CHECKPOINTER
from starlette.requests import Request
from starlette.responses import JSONResponse
from mcp.server.lowlevel.server import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
import mcp.types as types

from agentseek_api import __version__
from agentseek_api.core.auth_middleware import get_auth_backend
from agentseek_api.core.database import db_manager
from agentseek_api.core.runtime_store import UserScopedStore
from agentseek_api.models.auth import User
from agentseek_api.services.langgraph_service import (
    GraphEntry,
    LangGraphService,
    ensure_sync_checkpoint_mode,
    get_langgraph_service,
)

_current_mcp_user: ContextVar[User | None] = ContextVar("current_mcp_user", default=None)


class _StreamableHTTPASGIApp:
    def __init__(self, session_manager: StreamableHTTPSessionManager) -> None:
        self.session_manager = session_manager

    async def __call__(self, scope, receive, send) -> None:  # pragma: no cover
        await self.session_manager.handle_request(scope, receive, send)


class _AuthenticatedMCPASGIApp:
    def __init__(self, inner_app: _StreamableHTTPASGIApp, *, user_resolver) -> None:
        self.inner_app = inner_app
        self.user_resolver = user_resolver

    @staticmethod
    async def _buffer_request_messages(receive) -> list[dict[str, Any]]:
        buffered: list[dict[str, Any]] = []
        while True:
            message = await receive()
            buffered.append(dict(message))
            if message["type"] != "http.request" or not message.get("more_body", False):
                return buffered

    @staticmethod
    def _replay_receive(messages: list[dict[str, Any]]):
        remaining = [dict(message) for message in messages]

        async def _receive() -> dict[str, Any]:
            if remaining:
                return remaining.pop(0)
            return {"type": "http.disconnect"}

        return _receive

    async def __call__(self, scope, receive, send) -> None:  # pragma: no cover
        if scope["type"] != "http":
            await self.inner_app(scope, receive, send)
            return

        buffered_messages = await self._buffer_request_messages(receive)
        request = Request(scope, receive=self._replay_receive(buffered_messages))
        try:
            resolved = self.user_resolver(request)
            user = await resolved if inspect.isawaitable(resolved) else resolved
        except HTTPException as exc:
            response = JSONResponse({"detail": exc.detail}, status_code=exc.status_code)
            await response(scope, self._replay_receive(buffered_messages), send)
            return

        if not isinstance(user, User) or not user.is_authenticated:
            response = JSONResponse({"detail": "Not authenticated"}, status_code=401)
            await response(scope, self._replay_receive(buffered_messages), send)
            return

        token = _current_mcp_user.set(user)
        try:
            await self.inner_app(scope, self._replay_receive(buffered_messages), send)
        finally:
            _current_mcp_user.reset(token)


@dataclass
class MCPMount:
    server: Server
    session_manager: StreamableHTTPSessionManager
    app: _AuthenticatedMCPASGIApp


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


def _current_authenticated_user() -> User:
    user = _current_mcp_user.get()
    if user is None or not user.is_authenticated:
        raise RuntimeError("MCP tool execution requires an authenticated user.")
    return user


def _graph_invocation_config(user: User) -> tuple[UserScopedStore, dict[str, Any]]:
    thread_id = str(uuid4())
    checkpoint_ns = f"mcp:{uuid4()}"
    runtime_store = UserScopedStore(db_manager.get_store(), user_id=user.identity)
    checkpointer = db_manager.get_langgraph_checkpointer()
    return runtime_store, {
        CONF: {
            "thread_id": thread_id,
            "checkpoint_ns": checkpoint_ns,
            CONFIG_KEY_CHECKPOINTER: checkpointer,
            "store": runtime_store,
            "langgraph_auth_user": user.model_dump(),
        }
    }


async def _invoke_graph_tool(entry: GraphEntry, arguments: dict[str, Any]) -> dict[str, Any]:
    ensure_sync_checkpoint_mode(requested_async=False)
    user = _current_authenticated_user()
    runtime_store, config = _graph_invocation_config(user)
    graph = entry.build_graph(
        checkpointer=db_manager.get_langgraph_checkpointer(),
        store=runtime_store,
    )
    prepared = entry.prepare_input(arguments)
    if hasattr(graph, "ainvoke"):
        raw_result = await graph.ainvoke(prepared, config)
    else:  # pragma: no cover
        raw_result = graph.invoke(prepared, config)
    extracted = entry.extract_output(raw_result, arguments)
    if not isinstance(extracted, dict):
        extracted = {"result": extracted}
    return extracted


def build_mcp_server(service: LangGraphService | None = None) -> Server:
    resolved_service = service or get_langgraph_service()
    server: Server = Server("AgentSeek API", version=__version__)

    @server.list_tools()
    async def _list_tools() -> list[types.Tool]:
        return list_graph_tools(resolved_service)

    @server.call_tool()
    async def _call_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        entry = _entry_for_tool(resolved_service, name)
        return await _invoke_graph_tool(entry, arguments)

    return server


def build_mcp_mount(service: LangGraphService | None = None, *, user_resolver=None) -> MCPMount:
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
        app=_AuthenticatedMCPASGIApp(http_app, user_resolver=user_resolver or get_auth_backend().authenticate),
    )
