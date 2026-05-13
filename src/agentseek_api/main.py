from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from fastapi import FastAPI

from agentseek_api import __version__
from agentseek_api.api.assistants import router as assistants_router
from agentseek_api.api.runs import router as runs_router
from agentseek_api.api.stateless_runs import router as stateless_runs_router
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


def create_app() -> FastAPI:
    app = FastAPI(title=settings.APP_NAME, version=__version__, lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict[str, str]:
        db_manager.get_engine()
        db_manager.get_checkpointer()
        return {"status": "healthy"}

    @app.get("/info")
    async def info() -> dict[str, object]:
        return {
            "name": settings.APP_NAME,
            "version": __version__,
            "flags": {"assistants": True, "threads": True, "runs": True, "crons": False},
        }

    app.include_router(assistants_router)
    app.include_router(threads_router)
    app.include_router(runs_router)
    app.include_router(stateless_runs_router)
    return app


app = create_app()
