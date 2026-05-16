import pytest

from agentseek_api.core.database import DatabaseManager, resolve_metadata_db_url
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
