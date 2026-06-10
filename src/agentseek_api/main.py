import inspect
import logging
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from importlib.metadata import PackageNotFoundError, version as package_version
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from scalar_fastapi import get_scalar_api_reference
from starlette.routing import Route

from agentseek_api import __version__
from agentseek_api.a2a_server import (
    A2ATaskRegistry,
    build_agent_card,
    handle_a2a_request,
    is_a2a_compatible_entry,
    load_assistant,
)
from agentseek_api.api.assistants import router as assistants_api_router
from agentseek_api.api.crons import router as crons_router
from agentseek_api.api.runs import router as runs_router
from agentseek_api.api.stateless_runs import router as stateless_runs_router
from agentseek_api.api.store import router as store_router
from agentseek_api.api.streaming import router as streaming_router
from agentseek_api.api.threads import router as threads_router
from agentseek_api.core.app_loader import load_custom_app
from agentseek_api.core.auth_deps import get_current_user
from agentseek_api.core.auth_middleware import get_config_auth_openapi
from agentseek_api.core.a2a_config import is_a2a_enabled
from agentseek_api.core.content_type_fix import ContentTypeFixMiddleware
from agentseek_api.core.cors_config import (
    DEFAULT_EXPOSE_HEADERS,
    CorsConfig,
    get_cors_config,
)
from agentseek_api.core.database import db_manager
from agentseek_api.core.http_config import get_config_dir, get_http_config
from agentseek_api.core.mcp_config import is_mcp_enabled
from agentseek_api.mcp_server import MCPMount, build_mcp_mount
from agentseek_api.services.default_assistants import ensure_default_assistants
from agentseek_api.services.langgraph_service import get_langgraph_service
from agentseek_api.settings import settings

_FASTAPI_DEFAULT_PATHS = frozenset({
    "/openapi.json",
    "/docs",
    "/docs/oauth2-redirect",
    "/redoc",
})


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    await db_manager.initialize()
    await ensure_default_assistants()
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


def _feature_flags(*, a2a_enabled: bool, mcp_enabled: bool) -> dict[str, bool]:
    return {
        "agents": True,
        "assistants": True,
        "threads": True,
        "runs": True,
        "crons": True,
        "store": True,
        "a2a": a2a_enabled,
        "mcp": mcp_enabled,
        "protocol_v2": True,
    }


def _server_metadata() -> dict[str, object]:
    return {
        "app_name": settings.APP_NAME,
        "auth_type": "langgraph_sdk" if settings.AUTH_MODULE_PATH else "noop",
        "checkpoint_backend": "langchain-oceanbase",
        "checkpoint_backend_version": _langchain_oceanbase_version(),
        "compatibility_tier": "oss-core",
        "unsupported_features": [
            "distributed_runtime",
            "assistant_version_promotion",
        ],
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


async def _resolve_mcp_user(app: FastAPI, request) -> object:
    resolver = app.dependency_overrides.get(get_current_user, get_current_user)
    resolved = resolver(request)
    if inspect.isawaitable(resolved):
        return await resolved
    return resolved


def _add_cors_middleware(app: FastAPI, cors_config: CorsConfig | None) -> None:
    if not cors_config:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=False,
            allow_methods=["*"],
            allow_headers=["*"],
            expose_headers=DEFAULT_EXPOSE_HEADERS,
        )
        return

    origins = cors_config.get("allow_origins", ["*"])
    credentials = cors_config.get(
        "allow_credentials",
        origins not in (["*"], "*"),
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=credentials,
        allow_methods=cors_config.get("allow_methods", ["*"]),
        allow_headers=cors_config.get("allow_headers", ["*"]),
        allow_origin_regex=cors_config.get("allow_origin_regex"),
        expose_headers=cors_config.get("expose_headers", DEFAULT_EXPOSE_HEADERS),
        max_age=cors_config.get("max_age", 600),
    )


def _merge_custom_app(app: FastAPI) -> None:
    """Load custom FastAPI app from config and merge its routes, lifespan, and middleware."""
    http_config = get_http_config()
    if not http_config or not http_config.get("app"):
        return

    logger = logging.getLogger(__name__)
    config_dir = get_config_dir()
    try:
        custom_app = load_custom_app(http_config["app"], base_dir=config_dir)
    except Exception as exc:
        logger.error("Failed to load custom app '%s': %s", http_config["app"], exc)
        raise

    # --- Merge lifespan ---
    user_lifespan = custom_app.router.lifespan_context
    core_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def combined_lifespan(a):
        async with core_lifespan(a):
            async with user_lifespan(a):
                yield

    app.router.lifespan_context = combined_lifespan

    # --- Merge middleware ---
    app.user_middleware.extend(custom_app.user_middleware)
    if custom_app.user_middleware:
        logger.info(
            "Merged %d custom middleware(s): %s",
            len(custom_app.user_middleware),
            [mw.cls.__name__ for mw in custom_app.user_middleware],
        )

    # --- Merge routes ---
    existing_paths = {getattr(r, "path", None) for r in app.router.routes}

    for route in custom_app.routes:
        path = getattr(route, "path", None)
        if path in _FASTAPI_DEFAULT_PATHS:
            continue
        if path in existing_paths:
            logger.warning(
                "Custom route path '%s' conflicts with a built-in route, skipping", path
            )
            continue
        app.router.routes.append(route)


def create_app() -> FastAPI:
    app = FastAPI(title=settings.APP_NAME, version=__version__, lifespan=lifespan)
    _add_cors_middleware(app, get_cors_config())
    app.add_middleware(ContentTypeFixMiddleware)
    _apply_auth_openapi(app)
    app.state.a2a_enabled = is_a2a_enabled()
    app.state.a2a_registry = A2ATaskRegistry()
    app.state.mcp_enabled = is_mcp_enabled()
    if app.state.mcp_enabled:
        app.state.mcp_mount = build_mcp_mount(
            user_resolver=lambda request: _resolve_mcp_user(app, request),
        )
        _register_mcp_routes(app, app.state.mcp_mount)

    @app.get("/health", tags=["System"])
    async def health_check() -> dict[str, str]:
        db_manager.get_engine()
        db_manager.get_checkpointer()
        return {"status": "healthy"}

    @app.get("/ok", tags=["System"])
    async def ok() -> dict[str, bool]:
        return {"ok": True}

    @app.get("/scalar-docs", include_in_schema=False)
    async def scalar_docs() -> HTMLResponse:
        return get_scalar_api_reference(
            openapi_url=app.openapi_url,
            title=f"{settings.APP_NAME} - Scalar",
            persist_auth=True,
        )

    @app.get("/info", tags=["System"])
    async def server_information() -> dict[str, object]:
        return {
            "version": __version__,
            "langgraph_py_version": _langgraph_py_version(),
            "flags": _feature_flags(
                a2a_enabled=app.state.a2a_enabled,
                mcp_enabled=app.state.mcp_enabled,
            ),
            "metadata": _server_metadata(),
        }

    @app.get("/metrics", response_model=None, tags=["System"])
    async def system_metrics(
        format: str = Query(default="prometheus"),
    ) -> PlainTextResponse | JSONResponse:
        db_manager.get_engine()
        db_manager.get_checkpointer()
        if format == "json":
            return JSONResponse({
                "app_name": settings.APP_NAME,
                "version": __version__,
                "langgraph_py_version": _langgraph_py_version(),
                "checks": {
                    "database": "ok",
                    "checkpointer": "ok",
                },
                "flags": _feature_flags(
                    a2a_enabled=app.state.a2a_enabled,
                    mcp_enabled=app.state.mcp_enabled,
                ),
            })

        body = "\n".join([
            "# HELP agentseek_api_info AgentSeek API build information.",
            "# TYPE agentseek_api_info gauge",
            (
                f'agentseek_api_info{{version="{__version__}",'
                f'langgraph_py_version="{_langgraph_py_version()}"}} 1'
            ),
            "# HELP agentseek_api_database_up Database connectivity status.",
            "# TYPE agentseek_api_database_up gauge",
            "agentseek_api_database_up 1",
        ])
        return PlainTextResponse(body + "\n")

    if app.state.a2a_enabled:

        @app.get("/.well-known/agent-card.json", include_in_schema=False)
        async def agent_card(
            request: Request,
            assistant_id: str,
            _user=Depends(get_current_user),
        ) -> dict[str, object]:
            assistant = await load_assistant(assistant_id)
            entry = get_langgraph_service().get_entry(assistant.graph_id)
            if not is_a2a_compatible_entry(entry):
                raise HTTPException(
                    status_code=400, detail="Assistant graph is not A2A-compatible"
                )
            return build_agent_card(
                base_url=str(request.base_url).rstrip("/"),
                assistant=assistant,
                entry=entry,
            )

        @app.post("/a2a/{assistant_id}", include_in_schema=False, response_model=None)
        async def a2a_jsonrpc(
            assistant_id: str,
            payload: dict[str, Any],
            user=Depends(get_current_user),
        ) -> Response | dict[str, Any]:
            return await handle_a2a_request(
                assistant_id=assistant_id,
                payload=payload,
                user=user,
                service=get_langgraph_service(),
                registry=app.state.a2a_registry,
            )

    app.include_router(assistants_api_router, prefix="/assistants", tags=["Assistants"])
    app.include_router(assistants_api_router, prefix="/agents", tags=["Agents"])
    app.include_router(threads_router)
    app.include_router(streaming_router)
    app.include_router(crons_router)
    app.include_router(runs_router)
    app.include_router(stateless_runs_router)
    app.include_router(store_router)

    _merge_custom_app(app)

    return app


app = create_app()
