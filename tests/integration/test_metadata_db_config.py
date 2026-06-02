from collections.abc import Callable
from typing import Any

import pytest
from fastapi.testclient import TestClient

from agentseek_api.core.database import DatabaseManager
from agentseek_api.main import create_app
from agentseek_api.settings import settings


class FakeConnection:
    async def run_sync(self, _fn: Callable[..., Any]) -> None:
        return None


class FakeBeginContext:
    async def __aenter__(self) -> FakeConnection:
        return FakeConnection()

    async def __aexit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> None:
        return None


class FakeEngine:
    def begin(self) -> FakeBeginContext:
        return FakeBeginContext()

    async def dispose(self) -> None:
        return None


class FakeCheckpointer:
    def __init__(self, connection_args: dict[str, str]) -> None:
        self.connection_args = connection_args

    def setup(self) -> None:
        return None


class FakeStore:
    def __init__(self, connection_args: dict[str, str], **_kwargs: Any) -> None:
        self.connection_args = connection_args

    def setup(self) -> None:
        return None


async def _noop_ensure_default_assistants() -> None:
    return None


def test_health_uses_postgresql_async_driver(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_calls: list[tuple[str, bool]] = []

    def fake_create_async_engine(url: str, *, pool_pre_ping: bool) -> FakeEngine:
        captured_calls.append((url, pool_pre_ping))
        return FakeEngine()

    monkeypatch.setattr("agentseek_api.core.database.create_async_engine", fake_create_async_engine)
    monkeypatch.setattr("agentseek_api.core.database.async_sessionmaker", lambda *_args, **_kwargs: object())
    monkeypatch.setattr("agentseek_api.core.database.OceanBaseCheckpointSaver", FakeCheckpointer)
    monkeypatch.setattr("agentseek_api.core.database.LangGraphOceanBaseCheckpointSaver", FakeCheckpointer)
    monkeypatch.setattr("agentseek_api.core.database.OceanBaseStore", FakeStore)
    monkeypatch.setattr("agentseek_api.main.ensure_default_assistants", _noop_ensure_default_assistants)
    monkeypatch.setattr(
        settings,
        "METADATA_DB_URL",
        "postgresql://postgres:postgres@localhost:5432/agentseek",
    )
    monkeypatch.setattr(settings, "METADATA_DB_BACKEND", "auto")

    manager = DatabaseManager()
    monkeypatch.setattr("agentseek_api.main.db_manager", manager)

    with TestClient(create_app()) as client:
        response = client.get("/health")
        assert response.status_code == 200

    assert captured_calls == [("postgresql+asyncpg://postgres:postgres@localhost:5432/agentseek", True)]


def test_health_uses_mysql_async_driver(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_calls: list[tuple[str, bool]] = []

    def fake_create_async_engine(url: str, *, pool_pre_ping: bool) -> FakeEngine:
        captured_calls.append((url, pool_pre_ping))
        return FakeEngine()

    monkeypatch.setattr("agentseek_api.core.database.create_async_engine", fake_create_async_engine)
    monkeypatch.setattr("agentseek_api.core.database.async_sessionmaker", lambda *_args, **_kwargs: object())
    monkeypatch.setattr("agentseek_api.core.database.OceanBaseCheckpointSaver", FakeCheckpointer)
    monkeypatch.setattr("agentseek_api.core.database.LangGraphOceanBaseCheckpointSaver", FakeCheckpointer)
    monkeypatch.setattr("agentseek_api.core.database.OceanBaseStore", FakeStore)
    monkeypatch.setattr("agentseek_api.main.ensure_default_assistants", _noop_ensure_default_assistants)
    monkeypatch.setattr(
        settings,
        "METADATA_DB_URL",
        "mysql://root%40test:@localhost:2881/seekdb",
    )
    monkeypatch.setattr(settings, "METADATA_DB_BACKEND", "auto")

    manager = DatabaseManager()
    monkeypatch.setattr("agentseek_api.main.db_manager", manager)

    with TestClient(create_app()) as client:
        response = client.get("/health")
        assert response.status_code == 200

    assert captured_calls == [("mysql+aiomysql://root%40test:@localhost:2881/seekdb", False)]
