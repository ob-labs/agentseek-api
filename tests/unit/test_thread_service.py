import pytest

from agentseek_api.core.database import DatabaseManager
from agentseek_api.models.api import ThreadCreate
from agentseek_api.models.auth import User
from agentseek_api.services.thread_service import create_thread_for_user
from agentseek_api.settings import settings


async def _init_manager(monkeypatch: pytest.MonkeyPatch) -> DatabaseManager:
    monkeypatch.setattr(settings, "SEEKDB_URL", "sqlite+aiosqlite:///:memory:")

    class FakeCheckpointer:
        def __init__(self, connection_args: dict[str, str]) -> None:
            self.connection_args = connection_args

        def setup(self) -> None:
            return None

    monkeypatch.setattr("agentseek_api.core.database.OceanBaseCheckpointSaver", FakeCheckpointer)
    manager = DatabaseManager()
    await manager.initialize()
    return manager


@pytest.mark.asyncio
async def test_create_thread_for_user_persists_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = await _init_manager(monkeypatch)
    try:
        monkeypatch.setattr("agentseek_api.services.thread_service.db_manager", manager)
        result = await create_thread_for_user(
            payload=ThreadCreate(metadata={"purpose": "unit"}),
            user=User(identity="u1", is_authenticated=True),
        )
        assert result.thread_id is not None
        assert result.metadata["purpose"] == "unit"
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_create_thread_with_explicit_thread_id(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = await _init_manager(monkeypatch)
    try:
        monkeypatch.setattr("agentseek_api.services.thread_service.db_manager", manager)
        result = await create_thread_for_user(
            payload=ThreadCreate(thread_id="custom-id-1", metadata={}),
            user=User(identity="u1", is_authenticated=True),
        )
        assert result.thread_id == "custom-id-1"
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_create_thread_if_exists_raise_returns_409(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = await _init_manager(monkeypatch)
    try:
        monkeypatch.setattr("agentseek_api.services.thread_service.db_manager", manager)
        await create_thread_for_user(
            payload=ThreadCreate(thread_id="dup-1", metadata={"v": 1}),
            user=User(identity="u1", is_authenticated=True),
        )
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc:
            await create_thread_for_user(
                payload=ThreadCreate(thread_id="dup-1", metadata={"v": 2}, if_exists="raise"),
                user=User(identity="u1", is_authenticated=True),
            )
        assert exc.value.status_code == 409
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_create_thread_if_exists_do_nothing_returns_existing(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = await _init_manager(monkeypatch)
    try:
        monkeypatch.setattr("agentseek_api.services.thread_service.db_manager", manager)
        first = await create_thread_for_user(
            payload=ThreadCreate(thread_id="dup-2", metadata={"v": 1}),
            user=User(identity="u1", is_authenticated=True),
        )
        second = await create_thread_for_user(
            payload=ThreadCreate(thread_id="dup-2", metadata={"v": 2}, if_exists="do_nothing"),
            user=User(identity="u1", is_authenticated=True),
        )
        assert second.thread_id == first.thread_id
        assert second.metadata == {"v": 1}
    finally:
        await manager.close()
