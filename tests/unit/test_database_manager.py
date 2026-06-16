import pytest

from agentseek_api.core.database import DatabaseManager, resolve_metadata_db_url
from agentseek_api.core.runtime_store import SqliteStore
from agentseek_api.settings import settings


@pytest.mark.asyncio
async def test_checkpointer_setup_called_once(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeCheckpointer:
        setup_calls = 0
        latest_connection_args: dict[str, str] | None = None

        def __init__(self, connection_args: dict[str, str]) -> None:
            FakeCheckpointer.latest_connection_args = connection_args

        def setup(self) -> None:
            FakeCheckpointer.setup_calls += 1

    monkeypatch.setattr("agentseek_api.core.database.OceanBaseCheckpointSaver", FakeCheckpointer)
    monkeypatch.setattr(settings, "SEEKDB_URL", "sqlite+aiosqlite:///:memory:")
    monkeypatch.setattr(settings, "OCEANBASE_HOST", "127.0.0.1")
    monkeypatch.setattr(settings, "OCEANBASE_PORT", "2881")
    monkeypatch.setattr(settings, "OCEANBASE_USER", "root@test")
    monkeypatch.setattr(settings, "OCEANBASE_PASSWORD", "")
    monkeypatch.setattr(settings, "OCEANBASE_DB_NAME", "test")

    manager = DatabaseManager()
    await manager.initialize()
    await manager.initialize()

    assert FakeCheckpointer.setup_calls == 1
    assert FakeCheckpointer.latest_connection_args is not None
    assert FakeCheckpointer.latest_connection_args["host"] == "127.0.0.1"
    assert FakeCheckpointer.latest_connection_args["db_name"] == "test"
    assert isinstance(manager.get_store(), SqliteStore)

    await manager.close()


@pytest.mark.asyncio
async def test_initialize_builds_oceanbase_store_for_mysql_backends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeCheckpointer:
        def __init__(self, connection_args: dict[str, str]) -> None:
            self.connection_args = connection_args

        def setup(self) -> None:
            return None

    class FakeStore:
        setup_calls = 0
        latest_connection_args: dict[str, str] | None = None

        def __init__(self, connection_args: dict[str, str], **_kwargs) -> None:
            FakeStore.latest_connection_args = connection_args

        def setup(self) -> None:
            FakeStore.setup_calls += 1

    class FakeConnection:
        async def run_sync(self, _fn) -> None:
            return None

    class FakeBeginContext:
        async def __aenter__(self) -> FakeConnection:
            return FakeConnection()

        async def __aexit__(self, _exc_type, _exc, _tb) -> None:
            return None

    class FakeEngine:
        def begin(self) -> FakeBeginContext:
            return FakeBeginContext()

        async def dispose(self) -> None:
            return None

    monkeypatch.setattr(
        "agentseek_api.core.database.create_async_engine",
        lambda *_args, **_kwargs: FakeEngine(),
    )
    monkeypatch.setattr(
        "agentseek_api.core.database.async_sessionmaker",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr("agentseek_api.core.database.OceanBaseCheckpointSaver", FakeCheckpointer)
    monkeypatch.setattr("agentseek_api.core.database.LangGraphOceanBaseCheckpointSaver", FakeCheckpointer)
    monkeypatch.setattr("agentseek_api.core.database.OceanBaseStore", FakeStore)
    monkeypatch.setattr(settings, "METADATA_DB_URL", "mysql://root%40test:@localhost:2881/seekdb")
    monkeypatch.setattr(settings, "METADATA_DB_BACKEND", "auto")
    monkeypatch.setattr(settings, "OCEANBASE_HOST", "127.0.0.1")
    monkeypatch.setattr(settings, "OCEANBASE_PORT", "2881")
    monkeypatch.setattr(settings, "OCEANBASE_USER", "root@test")
    monkeypatch.setattr(settings, "OCEANBASE_PASSWORD", "")
    monkeypatch.setattr(settings, "OCEANBASE_DB_NAME", "seekdb")

    manager = DatabaseManager()
    await manager.initialize()

    assert FakeStore.setup_calls == 1
    assert isinstance(manager.get_store(), FakeStore)
    assert FakeStore.latest_connection_args is not None
    assert FakeStore.latest_connection_args["db_name"] == "seekdb"

    await manager.close()


@pytest.mark.asyncio
async def test_initialize_tolerates_store_setup_already_exists_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeCheckpointer:
        def __init__(self, connection_args: dict[str, str]) -> None:
            self.connection_args = connection_args

        def setup(self) -> None:
            return None

    class FakeStore:
        setup_calls = 0

        def __init__(self, connection_args: dict[str, str], **_kwargs) -> None:
            self.connection_args = connection_args

        def setup(self) -> None:
            FakeStore.setup_calls += 1
            raise RuntimeError("Table 'store_items' already exists")

    class FakeConnection:
        async def run_sync(self, _fn) -> None:
            return None

    class FakeBeginContext:
        async def __aenter__(self) -> FakeConnection:
            return FakeConnection()

        async def __aexit__(self, _exc_type, _exc, _tb) -> None:
            return None

    class FakeEngine:
        def begin(self) -> FakeBeginContext:
            return FakeBeginContext()

        async def dispose(self) -> None:
            return None

    monkeypatch.setattr(
        "agentseek_api.core.database.create_async_engine",
        lambda *_args, **_kwargs: FakeEngine(),
    )
    monkeypatch.setattr(
        "agentseek_api.core.database.async_sessionmaker",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr("agentseek_api.core.database.OceanBaseCheckpointSaver", FakeCheckpointer)
    monkeypatch.setattr("agentseek_api.core.database.LangGraphOceanBaseCheckpointSaver", FakeCheckpointer)
    monkeypatch.setattr("agentseek_api.core.database.OceanBaseStore", FakeStore)
    monkeypatch.setattr(settings, "METADATA_DB_URL", "mysql://root%40test:@localhost:2881/seekdb")
    monkeypatch.setattr(settings, "METADATA_DB_BACKEND", "auto")
    monkeypatch.setattr(settings, "OCEANBASE_HOST", "127.0.0.1")
    monkeypatch.setattr(settings, "OCEANBASE_PORT", "2881")
    monkeypatch.setattr(settings, "OCEANBASE_USER", "root@test")
    monkeypatch.setattr(settings, "OCEANBASE_PASSWORD", "")
    monkeypatch.setattr(settings, "OCEANBASE_DB_NAME", "seekdb")

    manager = DatabaseManager()
    await manager.initialize()

    assert FakeStore.setup_calls == 1
    assert isinstance(manager.get_store(), FakeStore)

    await manager.close()


@pytest.mark.parametrize(
    ("metadata_db_url", "backend", "expected_url"),
    [
        (
            "postgresql://postgres:postgres@localhost:5432/agentseek",
            "auto",
            "postgresql+asyncpg://postgres:postgres@localhost:5432/agentseek",
        ),
        (
            "mysql://root%40test:@localhost:2881/seekdb",
            "auto",
            "mysql+aiomysql://root%40test:@localhost:2881/seekdb",
        ),
        (
            "mysql://root%40test:@localhost:2881/seekdb",
            "seekdb",
            "mysql+aiomysql://root%40test:@localhost:2881/seekdb",
        ),
        (
            "mysql://root%40test:@localhost:2881/seekdb",
            "oceanbase",
            "mysql+aiomysql://root%40test:@localhost:2881/seekdb",
        ),
        (
            "postgresql://postgres:postgres@localhost:5432/agentseek",
            "postgresql",
            "postgresql+asyncpg://postgres:postgres@localhost:5432/agentseek",
        ),
    ],
)
def test_resolve_metadata_db_url_normalizes_driver(
    monkeypatch: pytest.MonkeyPatch,
    metadata_db_url: str,
    backend: str,
    expected_url: str,
) -> None:
    monkeypatch.setattr(settings, "METADATA_DB_URL", metadata_db_url)
    monkeypatch.setattr(settings, "METADATA_DB_BACKEND", backend)

    resolved_url = resolve_metadata_db_url()

    assert resolved_url == expected_url


def test_resolve_metadata_db_url_prefers_explicit_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "METADATA_DB_URL", "mysql://root%40test:@localhost:2881/seekdb")
    monkeypatch.setattr(settings, "METADATA_DB_BACKEND", "postgresql")

    resolved_url = resolve_metadata_db_url()

    assert resolved_url.startswith("postgresql+asyncpg://")


def test_resolve_metadata_db_url_treats_blank_backend_as_auto(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "METADATA_DB_URL", "mysql://root%40test:@localhost:2881/seekdb")
    monkeypatch.setattr(settings, "METADATA_DB_BACKEND", "")

    resolved_url = resolve_metadata_db_url()

    assert resolved_url == "mysql+aiomysql://root%40test:@localhost:2881/seekdb"


def test_resolve_metadata_db_url_builds_seekdb_url_from_oceanbase_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "METADATA_DB_URL", None)
    monkeypatch.setattr(settings, "SEEKDB_URL", "mysql+aiomysql://root%40test:@localhost:2881/seekdb")
    monkeypatch.setattr(settings, "METADATA_DB_BACKEND", "auto")
    monkeypatch.setattr(settings, "OCEANBASE_HOST", "host.docker.internal")
    monkeypatch.setattr(settings, "OCEANBASE_PORT", "3306")
    monkeypatch.setattr(settings, "OCEANBASE_USER", "root")
    monkeypatch.setattr(settings, "OCEANBASE_PASSWORD", "")
    monkeypatch.setattr(settings, "OCEANBASE_DB_NAME", "seekdb")

    resolved_url = resolve_metadata_db_url()

    assert resolved_url == "mysql+aiomysql://root:@host.docker.internal:3306/seekdb"


def test_resolve_metadata_db_url_raises_on_unknown_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "METADATA_DB_URL", "postgresql://postgres:postgres@localhost:5432/agentseek")
    monkeypatch.setattr(settings, "METADATA_DB_BACKEND", "oracle")

    with pytest.raises(ValueError, match="Unsupported METADATA_DB_BACKEND"):
        resolve_metadata_db_url()


@pytest.mark.asyncio
async def test_initialize_uses_embedded_seekdb_when_seekdb_embed_enabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    captured_checkpointer_args: list[dict] = []
    captured_store_args: list[dict] = []

    class FakeCheckpointer:
        def __init__(self, connection_args: dict[str, str]) -> None:
            captured_checkpointer_args.append(connection_args)

        def setup(self) -> None:
            return None

    class FakeStore:
        def __init__(self, connection_args: dict[str, str], **_kwargs) -> None:
            captured_store_args.append(connection_args)

        def setup(self) -> None:
            return None

    monkeypatch.setattr("agentseek_api.core.database.OceanBaseCheckpointSaver", FakeCheckpointer)
    monkeypatch.setattr("agentseek_api.core.database.LangGraphOceanBaseCheckpointSaver", FakeCheckpointer)
    monkeypatch.setattr("agentseek_api.core.database.OceanBaseStore", FakeStore)
    monkeypatch.setattr("agentseek_api.core.database._ensure_embed_database", lambda *_args: None)
    monkeypatch.setattr(settings, "SEEKDB_EMBED", True)
    monkeypatch.setattr(settings, "SEEKDB_EMBED_DIR", str(tmp_path / "embed_data"))
    monkeypatch.setattr(settings, "OCEANBASE_DB_NAME", "testdb")

    manager = DatabaseManager()
    await manager.initialize()

    assert len(captured_checkpointer_args) == 2
    assert captured_checkpointer_args[0]["path"] == str(tmp_path / "embed_data")
    assert captured_checkpointer_args[0]["db_name"] == "testdb"
    assert "host" not in captured_checkpointer_args[0]

    assert len(captured_store_args) == 1
    assert captured_store_args[0]["path"] == str(tmp_path / "embed_data")
    assert captured_store_args[0]["db_name"] == "testdb"

    assert (tmp_path / "embed_data").is_dir()
    assert (tmp_path / "embed_data" / "metadata.db").exists()

    await manager.close()


@pytest.mark.asyncio
async def test_apply_additive_migrations_adds_missing_columns_to_legacy_table() -> None:
    """A cron_jobs table created before end_time/on_run_completed existed must
    gain those columns on startup (create_all never alters existing tables)."""
    from sqlalchemy import inspect, text
    from sqlalchemy.ext.asyncio import create_async_engine

    from agentseek_api.core.orm import Base

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    try:
        # Legacy schema: cron_jobs WITHOUT end_time / on_run_completed, with a row.
        async with engine.begin() as conn:
            await conn.execute(
                text(
                    """
                    CREATE TABLE cron_jobs (
                        cron_id VARCHAR(36) PRIMARY KEY,
                        assistant_id VARCHAR(36) NOT NULL,
                        thread_id VARCHAR(36),
                        user_id VARCHAR(255) NOT NULL,
                        schedule VARCHAR(255) NOT NULL,
                        timezone VARCHAR(128) NOT NULL DEFAULT 'UTC',
                        enabled BOOLEAN NOT NULL DEFAULT 1,
                        input JSON NOT NULL,
                        metadata JSON NOT NULL,
                        kwargs JSON NOT NULL,
                        webhook TEXT,
                        max_webhook_attempts INTEGER NOT NULL DEFAULT 3,
                        next_run_at DATETIME NOT NULL,
                        last_run_at DATETIME,
                        last_tick_status VARCHAR(32),
                        last_error TEXT,
                        created_at DATETIME NOT NULL,
                        updated_at DATETIME NOT NULL
                    )
                    """
                )
            )
            await conn.execute(
                text(
                    """
                    INSERT INTO cron_jobs
                        (cron_id, assistant_id, user_id, schedule, timezone, enabled,
                         input, metadata, kwargs, max_webhook_attempts, next_run_at,
                         created_at, updated_at)
                    VALUES
                        ('c1', 'a1', 'u1', 'FREQ=MINUTELY;INTERVAL=1', 'UTC', 1,
                         '{}', '{}', '{}', 3, '2030-01-01 00:00:00',
                         '2026-01-01 00:00:00', '2026-01-01 00:00:00')
                    """
                )
            )

        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            await conn.run_sync(DatabaseManager._apply_additive_migrations)

        async with engine.connect() as conn:
            columns = await conn.run_sync(
                lambda c: {col["name"] for col in inspect(c).get_columns("cron_jobs")}
            )
            assert "end_time" in columns
            assert "on_run_completed" in columns
            # Legacy row backfilled via server_default on the NOT NULL column.
            value = (
                await conn.execute(
                    text("SELECT on_run_completed FROM cron_jobs WHERE cron_id = :i"),
                    {"i": "c1"},
                )
            ).scalar()
            assert value == "delete"

        # Idempotent: a second pass must not error.
        async with engine.begin() as conn:
            await conn.run_sync(DatabaseManager._apply_additive_migrations)
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_apply_additive_migrations_skips_unsafe_not_null_column(caplog) -> None:
    """A NOT NULL column without a server_default cannot be added to a populated
    table; the real migration must skip it with a logged error rather than emit
    failing DDL or crash startup."""
    import logging

    from sqlalchemy import Column, Integer, String, Table, inspect, text
    from sqlalchemy.ext.asyncio import create_async_engine

    from agentseek_api.core.orm import Base

    # Temporarily register an unsafe table on the real metadata that
    # _apply_additive_migrations iterates, then remove it afterward.
    unsafe = Table(
        "widget_unsafe_migration",
        Base.metadata,
        Column("id", Integer, primary_key=True),
        Column("name", String(32), nullable=False),  # NOT NULL, no server_default
    )
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    try:
        # Legacy table predates the `name` column and already has a row.
        async with engine.begin() as conn:
            await conn.execute(text("CREATE TABLE widget_unsafe_migration (id INTEGER PRIMARY KEY)"))
            await conn.execute(text("INSERT INTO widget_unsafe_migration (id) VALUES (1)"))

        with caplog.at_level(logging.ERROR):
            async with engine.begin() as conn:
                await conn.run_sync(DatabaseManager._apply_additive_migrations)

        # Column was NOT added (skipped), and the legacy row is intact.
        async with engine.connect() as conn:
            columns = await conn.run_sync(
                lambda c: {col["name"] for col in inspect(c).get_columns("widget_unsafe_migration")}
            )
            assert "name" not in columns
            count = (await conn.execute(text("SELECT COUNT(*) FROM widget_unsafe_migration"))).scalar()
            assert count == 1
        assert any("without a server_default" in r.message for r in caplog.records)
    finally:
        await engine.dispose()
        Base.metadata.remove(unsafe)
