from contextlib import AsyncExitStack, asynccontextmanager
from collections.abc import AsyncIterator
from importlib.metadata import PackageNotFoundError, version as package_version

from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse, PlainTextResponse
from starlette.routing import Route

from agentseek_api import __version__
from agentseek_api.api.assistants import router as assistants_api_router
from agentseek_api.api.runs import router as runs_router
from agentseek_api.api.stateless_runs import router as stateless_runs_router
from agentseek_api.api.store import router as store_router
from agentseek_api.api.threads import router as threads_router
from agentseek_api.core.auth_middleware import get_config_auth_openapi
from agentseek_api.core.database import db_manager
from agentseek_api.core.mcp_config import is_mcp_enabled
from agentseek_api.mcp_server import MCPMount, build_mcp_mount
from agentseek_api.settings import settings


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    await db_manager.initialize()
    try:
        async with AsyncExitStack() as stack:
            mcp_mount: MCPMount | None = getattr(_app.state, "mcp_mount", None)
            if mcp_mount is not None:
                await stack.enter_async_context(mcp_mount.session_manager.run())
            yield
    finally:
        await db_manager.close()


def _langgraph_py_version() -> str:
    try:
        return package_version("langgraph")
    except PackageNotFoundError:
        return "unknown"


def _langchain_oceanbase_version() -> str:
    try:
        return package_version("langchain-oceanbase")
    except PackageNotFoundError:
        return "unknown"


def _feature_flags(*, mcp_enabled: bool) -> dict[str, bool]:
    return {
        "agents": True,
        "assistants": True,
        "threads": True,
        "runs": True,
        "crons": False,
        "store": True,
        "a2a": False,
        "mcp": mcp_enabled,
        "protocol_v2": True,
    }


def _server_metadata() -> dict[str, str]:
    return {
        "app_name": settings.APP_NAME,
        "auth_type": settings.AUTH_TYPE,
        "checkpoint_backend": "langchain-oceanbase",
        "checkpoint_backend_version": _langchain_oceanbase_version(),
    }


def _apply_auth_openapi(app: FastAPI) -> None:
    auth_openapi = get_config_auth_openapi()
    if not auth_openapi:
        return

    original_openapi = app.openapi

    def custom_openapi() -> dict[str, object]:
        if app.openapi_schema:
            return app.openapi_schema
        schema = original_openapi()
        security_schemes = auth_openapi.get("securitySchemes")
        if isinstance(security_schemes, dict):
            components = schema.setdefault("components", {})
            existing_schemes = components.setdefault("securitySchemes", {})
            if isinstance(existing_schemes, dict):
                existing_schemes.update(security_schemes)
        security = auth_openapi.get("security")
        if isinstance(security, list):
            schema["security"] = security
        app.openapi_schema = schema
        return app.openapi_schema

    app.openapi = custom_openapi  # type: ignore[method-assign]


def _register_mcp_routes(app: FastAPI, mcp_mount: MCPMount) -> None:
    for path in ("/mcp", "/mcp/"):
        app.router.routes.append(
            Route(
                path,
                endpoint=mcp_mount.app,
                methods=["GET", "POST", "DELETE"],
                include_in_schema=False,
            )
        )


def create_app() -> FastAPI:
    app = FastAPI(title=settings.APP_NAME, version=__version__, lifespan=lifespan)
    _apply_auth_openapi(app)
    app.state.mcp_enabled = is_mcp_enabled()
    if app.state.mcp_enabled:
        app.state.mcp_mount = build_mcp_mount()
        _register_mcp_routes(app, app.state.mcp_mount)

    @app.get("/health")
    async def health() -> dict[str, str]:
        db_manager.get_engine()
        db_manager.get_checkpointer()
        return {"status": "healthy"}

    @app.get("/ok")
    async def ok() -> dict[str, bool]:
        return {"ok": True}

    @app.get("/info")
    async def info() -> dict[str, object]:
        return {
            "version": __version__,
            "langgraph_py_version": _langgraph_py_version(),
            "flags": _feature_flags(mcp_enabled=app.state.mcp_enabled),
            "metadata": _server_metadata(),
        }

    @app.get("/metrics", response_model=None)
    async def metrics(format: str = Query(default="prometheus")) -> PlainTextResponse | JSONResponse:
        db_manager.get_engine()
        db_manager.get_checkpointer()
        if format == "json":
            return JSONResponse(
                {
                    "app_name": settings.APP_NAME,
                    "version": __version__,
                    "langgraph_py_version": _langgraph_py_version(),
                    "checks": {
                        "database": "ok",
                        "checkpointer": "ok",
                    },
                    "flags": _feature_flags(mcp_enabled=app.state.mcp_enabled),
                }
            )

        body = "\n".join(
            [
                "# HELP agentseek_api_info AgentSeek API build information.",
                "# TYPE agentseek_api_info gauge",
                (
                    f'agentseek_api_info{{version="{__version__}",'
                    f'langgraph_py_version="{_langgraph_py_version()}"}} 1'
                ),
                "# HELP agentseek_api_database_up Database connectivity status.",
                "# TYPE agentseek_api_database_up gauge",
                "agentseek_api_database_up 1",
            ]
        )
        return PlainTextResponse(body + "\n")

    app.include_router(assistants_api_router, prefix="/assistants", tags=["Assistants"])
    app.include_router(assistants_api_router, prefix="/agents", tags=["Agents"])
    app.include_router(threads_router)
    app.include_router(runs_router)
    app.include_router(stateless_runs_router)
    app.include_router(store_router)
    return app


app = create_app()
