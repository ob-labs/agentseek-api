import asyncio
from collections.abc import Callable
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langchain_oceanbase.checkpointer import OceanBaseCheckpointSaver as LangGraphOceanBaseCheckpointSaver
from langchain_oceanbase.store import OceanBaseStore
from sqlalchemy.engine import URL, make_url
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from agentseek_api.core.oceanbase_checkpointer import OceanBaseCheckpointSaver
from agentseek_api.core.orm import Base
from agentseek_api.core.runtime_store import SqliteStore
from agentseek_api.core.store_config import load_store_config
from agentseek_api.settings import settings

DEFAULT_SEEKDB_URL = "mysql+aiomysql://root%40test:@localhost:2881/seekdb"


class NullStore:
    async def aget(self, _namespace: tuple[str, ...], _key: str) -> None:
        return None


def _resolve_metadata_backend(*, configured_backend: str, url_drivername: str) -> str:
    normalized_backend = configured_backend.strip().lower()
    if normalized_backend in {"postgres", "postgresql"}:
        return "postgresql"
    if normalized_backend in {"mysql", "seekdb", "oceanbase"}:
        return "mysql"
    if normalized_backend == "sqlite":
        return normalized_backend
    if normalized_backend != "auto":
        raise ValueError(f"Unsupported METADATA_DB_BACKEND: {configured_backend}")

    base_drivername = url_drivername.split("+", maxsplit=1)[0].lower()
    if base_drivername in {"postgres", "postgresql"}:
        return "postgresql"
    if base_drivername in {"mysql", "mariadb"}:
        return "mysql"
    if base_drivername == "sqlite":
        return "sqlite"
    raise ValueError(f"Cannot infer metadata backend from URL scheme: {url_drivername}")


def _ensure_async_driver(*, url: URL, backend: str) -> URL:
    if backend == "postgresql":
        return url.set(drivername="postgresql+asyncpg")
    if backend == "mysql":
        return url.set(drivername="mysql+aiomysql")
    if backend == "sqlite":
        if url.drivername == "sqlite":
            return url.set(drivername="sqlite+aiosqlite")
        return url
    raise ValueError(f"Unsupported metadata backend: {backend}")


def _resolve_seekdb_url() -> str:
    if settings.SEEKDB_URL != DEFAULT_SEEKDB_URL:
        return settings.SEEKDB_URL
    return URL.create(
        drivername="mysql+aiomysql",
        username=settings.OCEANBASE_USER,
        password=settings.OCEANBASE_PASSWORD,
        host=settings.OCEANBASE_HOST,
        port=int(settings.OCEANBASE_PORT),
        database=settings.OCEANBASE_DB_NAME,
    ).render_as_string(hide_password=False)


def _resolve_base_metadata_url() -> str:
    return settings.METADATA_DB_URL or _resolve_seekdb_url()


def resolve_metadata_db_url() -> str:
    raw_url = _resolve_base_metadata_url()
    parsed_url = make_url(raw_url)
    backend = _resolve_metadata_backend(
        configured_backend=settings.METADATA_DB_BACKEND,
        url_drivername=parsed_url.drivername,
    )
    return _ensure_async_driver(url=parsed_url, backend=backend).render_as_string(hide_password=False)


class DatabaseManager:
    def __init__(self) -> None:
        self.engine: AsyncEngine | None = None
        self.session_factory: async_sessionmaker[AsyncSession] | None = None
        self._checkpointer: OceanBaseCheckpointSaver | None = None
        self._langgraph_checkpointer: Any | None = None
        self._store: Any | None = None
        self._setup_lock: asyncio.Lock = asyncio.Lock()
        self._checkpointer_setup_done: bool = False
        self._langgraph_checkpointer_setup_done: bool = False
        self._store_setup_done: bool = False

    async def initialize(self) -> None:
        if self.engine is not None:
            return
        metadata_db_url = resolve_metadata_db_url()
        parsed_url = make_url(_resolve_base_metadata_url())
        metadata_backend = _resolve_metadata_backend(
            configured_backend=settings.METADATA_DB_BACKEND,
            url_drivername=parsed_url.drivername,
        )
        store_config = load_store_config(agentseek_graphs=settings.AGENTSEEK_GRAPHS)
        runtime_index = store_config.index.to_runtime_config()
        runtime_ttl = store_config.ttl.to_runtime_config()
        self.engine = create_async_engine(metadata_db_url, pool_pre_ping=metadata_backend != "mysql")
        self.session_factory = async_sessionmaker(self.engine, expire_on_commit=False)
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        self._checkpointer = OceanBaseCheckpointSaver(
            connection_args={
                "host": settings.OCEANBASE_HOST,
                "port": settings.OCEANBASE_PORT,
                "user": settings.OCEANBASE_USER,
                "password": settings.OCEANBASE_PASSWORD,
                "db_name": settings.OCEANBASE_DB_NAME,
            }
        )
        if metadata_backend == "sqlite":
            self._langgraph_checkpointer = InMemorySaver()
            self._store = SqliteStore(
                url=metadata_db_url,
                index=runtime_index,
                ttl_config=runtime_ttl,
            )
        else:
            self._langgraph_checkpointer = LangGraphOceanBaseCheckpointSaver(
                connection_args={
                    "host": settings.OCEANBASE_HOST,
                    "port": settings.OCEANBASE_PORT,
                    "user": settings.OCEANBASE_USER,
                    "password": settings.OCEANBASE_PASSWORD,
                    "db_name": settings.OCEANBASE_DB_NAME,
                }
            )
            self._store = OceanBaseStore(
                connection_args={
                    "host": settings.OCEANBASE_HOST,
                    "port": settings.OCEANBASE_PORT,
                    "user": settings.OCEANBASE_USER,
                    "password": settings.OCEANBASE_PASSWORD,
                    "db_name": settings.OCEANBASE_DB_NAME,
                },
                index=runtime_index,
                ttl_config=runtime_ttl,
            )
        await self._setup_checkpointer_once()
        await self._setup_langgraph_checkpointer_once()
        await self._setup_store_once()

    async def _setup_checkpointer_once(self) -> None:
        if self._checkpointer_setup_done:
            return
        async with self._setup_lock:
            if self._checkpointer_setup_done:
                return
            if self._checkpointer is None:
                raise RuntimeError("Checkpointer not initialized")
            await asyncio.to_thread(self._checkpointer.setup)
            self._checkpointer_setup_done = True

    async def _setup_langgraph_checkpointer_once(self) -> None:
        if self._langgraph_checkpointer_setup_done:
            return
        async with self._setup_lock:
            if self._langgraph_checkpointer_setup_done:
                return
            if self._langgraph_checkpointer is None:
                raise RuntimeError("LangGraph checkpointer not initialized")
            setup = getattr(self._langgraph_checkpointer, "setup", None)
            if callable(setup):
                await asyncio.to_thread(setup)
            self._langgraph_checkpointer_setup_done = True

    async def _setup_store_once(self) -> None:
        if self._store_setup_done:
            return
        async with self._setup_lock:
            if self._store_setup_done:
                return
            if self._store is None:
                raise RuntimeError("Store not initialized")
            setup = getattr(self._store, "setup", None)
            if callable(setup):
                await asyncio.to_thread(setup)
            self._store_setup_done = True

    async def close(self) -> None:
        if self.engine is not None:
            await self.engine.dispose()
        store_engine = getattr(getattr(self._store, "obvector", None), "engine", None)
        dispose_store_engine = getattr(store_engine, "dispose", None)
        if callable(dispose_store_engine):
            await asyncio.to_thread(dispose_store_engine)
        self.engine = None
        self.session_factory = None
        self._checkpointer = None
        self._langgraph_checkpointer = None
        self._store = None
        self._checkpointer_setup_done = False
        self._langgraph_checkpointer_setup_done = False
        self._store_setup_done = False

    def get_engine(self) -> AsyncEngine:
        if self.engine is None:
            raise RuntimeError("Database not initialized")
        return self.engine

    def get_session_factory(self) -> async_sessionmaker[AsyncSession]:
        if self.session_factory is None:
            raise RuntimeError("Database not initialized")
        return self.session_factory

    def get_checkpointer(self) -> OceanBaseCheckpointSaver:
        if self._checkpointer is None:
            raise RuntimeError("Database not initialized")
        return self._checkpointer

    def get_langgraph_checkpointer(self) -> Any:
        if self._langgraph_checkpointer is None:
            raise RuntimeError("Database not initialized")
        return self._langgraph_checkpointer

    def get_store(self) -> Any:
        if self._store is None:
            raise RuntimeError("Database not initialized")
        return self._store

    async def run_checkpointer_call(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        return await asyncio.to_thread(func, *args, **kwargs)


db_manager = DatabaseManager()
