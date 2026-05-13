import pytest

from agentseek_api.core.database import DatabaseManager
from agentseek_api.models.api import ThreadCreate
from agentseek_api.models.auth import User
from agentseek_api.services.thread_service import create_thread_for_user
from agentseek_api.settings import settings


@pytest.mark.asyncio
async def test_create_thread_for_user_persists_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
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
        monkeypatch.setattr("agentseek_api.services.thread_service.db_manager", manager)
        result = await create_thread_for_user(
            payload=ThreadCreate(metadata={"purpose": "unit"}),
            user=User(identity="u1", is_authenticated=True),
        )
        assert result.user_id == "u1"
        assert result.metadata["purpose"] == "unit"
    finally:
        await manager.close()
