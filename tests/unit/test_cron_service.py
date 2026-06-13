from datetime import timedelta

import pytest
from sqlalchemy import select

from agentseek_api.core.database import DatabaseManager
from agentseek_api.core.orm import CronJob
from agentseek_api.models.api import CronCreate, CronPatch, CronSearchRequest
from agentseek_api.models.auth import User
from agentseek_api.services.cron_service import create_cron, patch_cron, search_crons
from agentseek_api.settings import settings


@pytest.mark.asyncio
async def test_create_cron_persists_webhook_timezone_and_runtime_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "SEEKDB_URL", "sqlite+aiosqlite:///:memory:")

    class FakeCheckpointer:
        def __init__(self, connection_args: dict[str, str]) -> None:
            self.connection_args = connection_args

        def setup(self) -> None:
            return None

    monkeypatch.setattr("agentseek_api.core.database.OceanBaseCheckpointSaver", FakeCheckpointer)

    manager = DatabaseManager()
    await manager.initialize()
    try:
        monkeypatch.setattr("agentseek_api.services.cron_service.db_manager", manager)
        user = User(identity="user-1", is_authenticated=True)

        created = await create_cron(
            assistant_id="assistant-1",
            thread_id=None,
            payload=CronCreate(
                assistant_id="assistant-1",
                schedule="FREQ=DAILY;INTERVAL=1;BYHOUR=9;BYMINUTE=15",
                timezone="Asia/Shanghai",
                input={"kind": "extended"},
                metadata={"source": "cron-test"},
                config={"model": "gpt-test"},
                context={"tenant": "acme"},
                webhook="https://example.com/hook",
                enabled=True,
            ),
            user=user,
        )

        session_factory = manager.get_session_factory()
        async with session_factory() as session:
            row = await session.scalar(select(CronJob).where(CronJob.cron_id == created.cron_id))

        assert row is not None
        assert created.timezone == "Asia/Shanghai"
        assert created.webhook == "https://example.com/hook"
        assert row.timezone == "Asia/Shanghai"
        assert row.webhook == "https://example.com/hook"
        assert row.metadata_json == {"source": "cron-test"}
        assert row.kwargs_json == {"config": {"model": "gpt-test"}, "context": {"tenant": "acme"}}
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_search_crons_filters_by_assistant_and_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "SEEKDB_URL", "sqlite+aiosqlite:///:memory:")

    class FakeCheckpointer:
        def __init__(self, connection_args: dict[str, str]) -> None:
            self.connection_args = connection_args

        def setup(self) -> None:
            return None

    monkeypatch.setattr("agentseek_api.core.database.OceanBaseCheckpointSaver", FakeCheckpointer)

    manager = DatabaseManager()
    await manager.initialize()
    try:
        monkeypatch.setattr("agentseek_api.services.cron_service.db_manager", manager)
        user = User(identity="user-1", is_authenticated=True)

        matching = await create_cron(
            assistant_id="assistant-1",
            thread_id=None,
            payload=CronCreate(
                assistant_id="assistant-1",
                schedule="FREQ=MINUTELY;INTERVAL=5",
                input={"kind": "match"},
                enabled=True,
            ),
            user=user,
        )
        await create_cron(
            assistant_id="assistant-1",
            thread_id=None,
            payload=CronCreate(
                assistant_id="assistant-1",
                schedule="FREQ=MINUTELY;INTERVAL=5",
                input={"kind": "disabled"},
                enabled=False,
            ),
            user=user,
        )
        await create_cron(
            assistant_id="assistant-2",
            thread_id=None,
            payload=CronCreate(
                assistant_id="assistant-2",
                schedule="FREQ=MINUTELY;INTERVAL=5",
                input={"kind": "other-assistant"},
                enabled=True,
            ),
            user=user,
        )

        result = await search_crons(
            payload=CronSearchRequest(assistant_id="assistant-1", enabled=True, limit=10, offset=0),
            user=user,
        )

        assert [item.cron_id for item in result.items] == [matching.cron_id]
        assert result.items[0].assistant_id == "assistant-1"
        assert result.items[0].enabled is True
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_patch_cron_recomputes_next_run_at(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "SEEKDB_URL", "sqlite+aiosqlite:///:memory:")

    class FakeCheckpointer:
        def __init__(self, connection_args: dict[str, str]) -> None:
            self.connection_args = connection_args

        def setup(self) -> None:
            return None

    monkeypatch.setattr("agentseek_api.core.database.OceanBaseCheckpointSaver", FakeCheckpointer)

    manager = DatabaseManager()
    await manager.initialize()
    try:
        monkeypatch.setattr("agentseek_api.services.cron_service.db_manager", manager)
        user = User(identity="user-1", is_authenticated=True)

        created = await create_cron(
            assistant_id="assistant-1",
            thread_id=None,
            payload=CronCreate(
                assistant_id="assistant-1",
                schedule="FREQ=MINUTELY;INTERVAL=15",
                input={"kind": "original"},
                enabled=True,
            ),
            user=user,
        )

        updated = await patch_cron(
            cron_id=created.cron_id,
            payload=CronPatch(schedule="FREQ=MINUTELY;INTERVAL=5", input={"kind": "updated"}),
            user=user,
        )

        assert updated.schedule == "FREQ=MINUTELY;INTERVAL=5"
        assert updated.next_run_at <= created.next_run_at
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_patch_cron_updates_webhook_timezone_and_runtime_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "SEEKDB_URL", "sqlite+aiosqlite:///:memory:")

    class FakeCheckpointer:
        def __init__(self, connection_args: dict[str, str]) -> None:
            self.connection_args = connection_args

        def setup(self) -> None:
            return None

    monkeypatch.setattr("agentseek_api.core.database.OceanBaseCheckpointSaver", FakeCheckpointer)

    manager = DatabaseManager()
    await manager.initialize()
    try:
        monkeypatch.setattr("agentseek_api.services.cron_service.db_manager", manager)
        user = User(identity="user-1", is_authenticated=True)

        created = await create_cron(
            assistant_id="assistant-1",
            thread_id=None,
            payload=CronCreate(
                assistant_id="assistant-1",
                schedule="FREQ=MINUTELY;INTERVAL=15",
                input={"kind": "original"},
                enabled=True,
            ),
            user=user,
        )

        updated = await patch_cron(
            cron_id=created.cron_id,
            payload=CronPatch(
                timezone="America/Los_Angeles",
                metadata={"source": "patched"},
                config={"temperature": 0.1},
                context={"workspace": "west"},
                webhook="https://example.com/patched",
            ),
            user=user,
        )

        session_factory = manager.get_session_factory()
        async with session_factory() as session:
            row = await session.scalar(select(CronJob).where(CronJob.cron_id == created.cron_id))

        assert row is not None
        assert updated.timezone == "America/Los_Angeles"
        assert updated.webhook == "https://example.com/patched"
        assert row.timezone == "America/Los_Angeles"
        assert row.webhook == "https://example.com/patched"
        assert row.metadata_json == {"source": "patched"}
        assert row.kwargs_json == {"config": {"temperature": 0.1}, "context": {"workspace": "west"}}
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_patch_cron_reenabling_recomputes_next_run_at(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "SEEKDB_URL", "sqlite+aiosqlite:///:memory:")

    class FakeCheckpointer:
        def __init__(self, connection_args: dict[str, str]) -> None:
            self.connection_args = connection_args

        def setup(self) -> None:
            return None

    monkeypatch.setattr("agentseek_api.core.database.OceanBaseCheckpointSaver", FakeCheckpointer)

    manager = DatabaseManager()
    await manager.initialize()
    try:
        monkeypatch.setattr("agentseek_api.services.cron_service.db_manager", manager)
        user = User(identity="user-1", is_authenticated=True)

        created = await create_cron(
            assistant_id="assistant-1",
            thread_id=None,
            payload=CronCreate(
                assistant_id="assistant-1",
                schedule="FREQ=MINUTELY;INTERVAL=1",
                input={"kind": "original"},
                enabled=True,
            ),
            user=user,
        )

        disabled = await patch_cron(
            cron_id=created.cron_id,
            payload=CronPatch(enabled=False),
            user=user,
        )
        stale_next_run_at = disabled.next_run_at
        session_factory = manager.get_session_factory()
        async with session_factory() as session:
            row = await session.scalar(select(CronJob).where(CronJob.cron_id == created.cron_id))
            assert row is not None
            row.next_run_at = row.next_run_at - timedelta(hours=1)
            stale_next_run_at = row.next_run_at
            await session.commit()
        reenabled = await patch_cron(
            cron_id=created.cron_id,
            payload=CronPatch(enabled=True),
            user=user,
        )

        assert disabled.enabled is False
        assert reenabled.enabled is True
        assert reenabled.next_run_at > stale_next_run_at
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_patch_cron_rejects_invalid_timezone_even_while_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "SEEKDB_URL", "sqlite+aiosqlite:///:memory:")

    class FakeCheckpointer:
        def __init__(self, connection_args: dict[str, str]) -> None:
            self.connection_args = connection_args

        def setup(self) -> None:
            return None

    monkeypatch.setattr("agentseek_api.core.database.OceanBaseCheckpointSaver", FakeCheckpointer)

    manager = DatabaseManager()
    await manager.initialize()
    try:
        monkeypatch.setattr("agentseek_api.services.cron_service.db_manager", manager)
        user = User(identity="user-1", is_authenticated=True)

        created = await create_cron(
            assistant_id="assistant-1",
            thread_id=None,
            payload=CronCreate(
                assistant_id="assistant-1",
                schedule="FREQ=MINUTELY;INTERVAL=1",
                input={"kind": "original"},
                enabled=False,
            ),
            user=user,
        )

        with pytest.raises(ValueError, match="Invalid timezone: Mars/Olympus"):
            await patch_cron(
                cron_id=created.cron_id,
                payload=CronPatch(timezone="Mars/Olympus"),
                user=user,
            )
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_to_read_model_exposes_spec_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "SEEKDB_URL", "sqlite+aiosqlite:///:memory:")

    class FakeCheckpointer:
        def __init__(self, connection_args: dict[str, str]) -> None:
            self.connection_args = connection_args

        def setup(self) -> None:
            return None

    monkeypatch.setattr("agentseek_api.core.database.OceanBaseCheckpointSaver", FakeCheckpointer)

    manager = DatabaseManager()
    await manager.initialize()
    try:
        monkeypatch.setattr("agentseek_api.services.cron_service.db_manager", manager)
        user = User(identity="user-42", is_authenticated=True)

        created = await create_cron(
            assistant_id="assistant-1",
            thread_id=None,
            payload=CronCreate(
                assistant_id="assistant-1",
                schedule="FREQ=MINUTELY;INTERVAL=5",
                input={"kind": "read-model"},
                metadata={"source": "unit"},
                config={"model": "gpt-test"},
                context={"tenant": "acme"},
            ),
            user=user,
        )

        assert created.user_id == "user-42"
        assert created.metadata == {"source": "unit"}
        assert created.next_run_date == created.next_run_at
        assert created.end_time is None
        assert created.payload == {
            "input": {"kind": "read-model"},
            "config": {"model": "gpt-test"},
            "context": {"tenant": "acme"},
        }
    finally:
        await manager.close()
