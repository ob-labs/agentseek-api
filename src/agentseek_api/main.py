from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from importlib.metadata import PackageNotFoundError, version as package_version

from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse, PlainTextResponse

from agentseek_api import __version__
from agentseek_api.api.assistants import router as assistants_router
from agentseek_api.api.runs import router as runs_router
from agentseek_api.api.stateless_runs import router as stateless_runs_router
from agentseek_api.api.store import router as store_router
from agentseek_api.api.threads import router as threads_router
from agentseek_api.core.database import db_manager
from agentseek_api.settings import settings


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    await db_manager.initialize()
    try:
        yield
    finally:
        await db_manager.close()


def _langgraph_py_version() -> str:
    try:
        return package_version("langgraph")
    except PackageNotFoundError:
        return "unknown"


def _feature_flags() -> dict[str, bool]:
    return {
        "assistants": True,
        "threads": True,
        "runs": True,
        "crons": False,
        "store": True,
        "a2a": False,
        "mcp": False,
        "protocol_v2": True,
    }


def _server_metadata() -> dict[str, str]:
    return {
        "app_name": settings.APP_NAME,
        "auth_type": settings.AUTH_TYPE,
        "checkpoint_backend": "langchain-oceanbase",
        "checkpoint_backend_version": "0.4.0",
    }


def create_app() -> FastAPI:
    app = FastAPI(title=settings.APP_NAME, version=__version__, lifespan=lifespan)

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
            "flags": _feature_flags(),
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
                    "flags": _feature_flags(),
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

    app.include_router(assistants_router)
    app.include_router(threads_router)
    app.include_router(runs_router)
    app.include_router(stateless_runs_router)
    app.include_router(store_router)
    return app


app = create_app()
