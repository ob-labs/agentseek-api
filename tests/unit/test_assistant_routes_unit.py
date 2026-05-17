from datetime import UTC, datetime

import pytest
from fastapi import HTTPException

from agentseek_api.api import assistants as assistants_module
from agentseek_api.core.orm import Assistant
from agentseek_api.models.api import AssistantCreate, AssistantPatch, AssistantSearchRequest


class FakeScalarResult:
    def __init__(self, rows: list[Assistant]) -> None:
        self._rows = rows

    def all(self) -> list[Assistant]:
        return list(self._rows)


class FakeSession:
    def __init__(
        self,
        *,
        scalar_rows: list[Assistant | None] | None = None,
        scalars_rows: list[list[Assistant]] | None = None,
    ) -> None:
        self.scalar_rows = list(scalar_rows or [])
        self.scalars_rows = list(scalars_rows or [])
        self.deleted: list[Assistant] = []

    async def scalar(self, _query) -> Assistant | None:
        return self.scalar_rows.pop(0) if self.scalar_rows else None

    async def scalars(self, _query) -> FakeScalarResult:
        return FakeScalarResult(self.scalars_rows.pop(0) if self.scalars_rows else [])

    def add(self, _obj: object) -> None:
        return None

    async def commit(self) -> None:
        return None

    async def refresh(self, obj: Assistant) -> None:
        now = datetime.now(UTC)
        if not obj.assistant_id:
            obj.assistant_id = "assistant-1"
        if not obj.created_at:
            obj.created_at = now
        if obj.metadata_json is None:
            obj.metadata_json = {}
        if obj.config_json is None:
            obj.config_json = {}
        if obj.context_json is None:
            obj.context_json = {}
        if obj.version is None:
            obj.version = 1
        obj.updated_at = now

    async def delete(self, obj: Assistant) -> None:
        self.deleted.append(obj)


class FakeSessionContext:
    def __init__(self, session: FakeSession) -> None:
        self.session = session

    async def __aenter__(self) -> FakeSession:
        return self.session

    async def __aexit__(self, _exc_type, _exc, _tb) -> None:
        return None


class FakeSessionFactory:
    def __init__(self, sessions: list[FakeSession]) -> None:
        self.sessions = sessions

    def __call__(self) -> FakeSessionContext:
        return FakeSessionContext(self.sessions.pop(0))


def _assistant(*, assistant_id: str, name: str = "assistant", graph_id: str = "default") -> Assistant:
    row = Assistant(name=name, graph_id=graph_id)
    row.assistant_id = assistant_id
    row.created_at = datetime.now(UTC)
    row.updated_at = row.created_at
    row.metadata_json = {}
    row.config_json = {}
    row.context_json = {}
    row.version = 1
    return row


@pytest.mark.asyncio
async def test_assistant_route_handlers_cover_crud_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    existing = _assistant(assistant_id="assistant-existing", name="before")
    delete_target = _assistant(assistant_id="assistant-delete", name="delete-me")
    create_session = FakeSession()
    list_session = FakeSession(scalars_rows=[[existing]])
    patch_session = FakeSession(scalar_rows=[existing])
    delete_session = FakeSession(scalar_rows=[delete_target])
    session_factory = FakeSessionFactory([create_session, list_session, patch_session, delete_session])

    monkeypatch.setattr(
        "agentseek_api.api.assistants.db_manager.get_session_factory",
        lambda: session_factory,
    )

    created = await assistants_module.create_assistant(AssistantCreate(name="created", graph_id="react_agent"))
    assert created.assistant_id == "assistant-1"
    assert created.graph_id == "react_agent"

    listed = await assistants_module.list_assistants()
    assert [item.assistant_id for item in listed] == ["assistant-existing"]

    patched = await assistants_module.patch_assistant(
        "assistant-existing",
        AssistantPatch(
            name="after",
            graph_id="stress_test",
            metadata={"team": "compat"},
            config={"temperature": 0},
            context={"tenant": "unit"},
            description="patched",
        ),
    )
    assert patched.name == "after"
    assert patched.graph_id == "stress_test"
    assert patched.metadata == {"team": "compat"}
    assert patched.config == {"temperature": 0}
    assert patched.context == {"tenant": "unit"}
    assert patched.description == "patched"

    deleted = await assistants_module.delete_assistant("assistant-delete")
    assert deleted.status_code == 204
    assert delete_session.deleted == [delete_target]


@pytest.mark.asyncio
async def test_assistant_route_handlers_raise_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    session_factory = FakeSessionFactory([FakeSession(scalar_rows=[None]), FakeSession(scalar_rows=[None]), FakeSession(scalar_rows=[None])])
    monkeypatch.setattr(
        "agentseek_api.api.assistants.db_manager.get_session_factory",
        lambda: session_factory,
    )

    with pytest.raises(HTTPException, match="Assistant not found") as get_error:
        await assistants_module.get_assistant("missing")
    assert get_error.value.status_code == 404

    with pytest.raises(HTTPException, match="Assistant not found") as patch_error:
        await assistants_module.patch_assistant("missing", AssistantPatch(name="nope"))
    assert patch_error.value.status_code == 404

    with pytest.raises(HTTPException, match="Assistant not found") as delete_error:
        await assistants_module.delete_assistant("missing")
    assert delete_error.value.status_code == 404


@pytest.mark.asyncio
async def test_delete_assistant_rejects_delete_threads_until_supported(monkeypatch: pytest.MonkeyPatch) -> None:
    session_factory = FakeSessionFactory([])
    monkeypatch.setattr(
        "agentseek_api.api.assistants.db_manager.get_session_factory",
        lambda: session_factory,
    )

    with pytest.raises(HTTPException, match="delete_threads=true is not supported") as error:
        await assistants_module.delete_assistant("assistant-1", delete_threads=True)
    assert error.value.status_code == 400


@pytest.mark.asyncio
async def test_count_assistants_returns_exact_count_beyond_page_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    assistants = [_assistant(assistant_id=f"assistant-{index}") for index in range(10_001)]
    session_factory = FakeSessionFactory([FakeSession(scalars_rows=[assistants])])
    monkeypatch.setattr(
        "agentseek_api.api.assistants.db_manager.get_session_factory",
        lambda: session_factory,
    )

    count = await assistants_module.count_assistants(AssistantSearchRequest())

    assert count == 10_001
