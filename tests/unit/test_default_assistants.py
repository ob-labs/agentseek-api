import pytest

from agentseek_api.services.default_assistants import (
    derive_assistant_id,
    ensure_default_assistants,
    resolve_assistant_id,
)


def test_derive_assistant_id_is_deterministic() -> None:
    id1 = derive_assistant_id("react_agent")
    id2 = derive_assistant_id("react_agent")
    assert id1 == id2
    assert derive_assistant_id("other") != id1


def test_resolve_assistant_id_returns_derived_id_for_known_graph() -> None:
    graphs = {"react_agent", "default"}
    resolved = resolve_assistant_id("react_agent", available_graphs=graphs)
    assert resolved == derive_assistant_id("react_agent")


def test_resolve_assistant_id_returns_input_unchanged_for_unknown_graph() -> None:
    graphs = {"react_agent", "default"}
    raw_id = "some-uuid-value"
    resolved = resolve_assistant_id(raw_id, available_graphs=graphs)
    assert resolved == raw_id


def test_resolve_assistant_id_uses_langgraph_service_when_graphs_not_provided(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeLangGraphService:
        def registered_graph_ids(self):
            return ["react_agent", "default"]

    monkeypatch.setattr(
        "agentseek_api.services.default_assistants.get_langgraph_service",
        FakeLangGraphService,
    )
    resolved = resolve_assistant_id("react_agent")
    assert resolved == derive_assistant_id("react_agent")


@pytest.mark.asyncio
async def test_ensure_default_assistants_creates_missing_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "agentseek_api.services.default_assistants.get_active_config_payload",
        lambda: {"graphs": {"react_agent": {}, "default": {}}},
    )

    created_ids: list[str] = []

    class FakeScalarResult:
        def __init__(self, val):
            self.val = val

        def __await__(self):
            async def _inner():
                return self.val
            return _inner().__await__()

    class FakeSession:
        async def scalar(self, _query):
            return None

        def add(self, obj):
            created_ids.append(obj.assistant_id)

        async def commit(self):
            pass

    class FakeSessionContext:
        async def __aenter__(self):
            return FakeSession()

        async def __aexit__(self, *_):
            pass

    class FakeSessionFactory:
        def __call__(self):
            return FakeSessionContext()

    class FakeDBManager:
        def get_session_factory(self):
            return FakeSessionFactory()

    monkeypatch.setattr(
        "agentseek_api.services.default_assistants.db_manager",
        FakeDBManager(),
    )

    await ensure_default_assistants()
    assert len(created_ids) == 2
    assert derive_assistant_id("react_agent") in created_ids
    assert derive_assistant_id("default") in created_ids


@pytest.mark.asyncio
async def test_ensure_default_assistants_skips_existing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "agentseek_api.services.default_assistants.get_active_config_payload",
        lambda: {"graphs": {"react_agent": {}}},
    )

    class FakeSession:
        async def scalar(self, _query):
            return object()

        def add(self, obj):
            raise AssertionError("Should not add existing assistant")

        async def commit(self):
            pass

    class FakeSessionContext:
        async def __aenter__(self):
            return FakeSession()

        async def __aexit__(self, *_):
            pass

    class FakeSessionFactory:
        def __call__(self):
            return FakeSessionContext()

    class FakeDBManager:
        def get_session_factory(self):
            return FakeSessionFactory()

    monkeypatch.setattr(
        "agentseek_api.services.default_assistants.db_manager",
        FakeDBManager(),
    )

    await ensure_default_assistants()


@pytest.mark.asyncio
async def test_ensure_default_assistants_noop_when_no_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "agentseek_api.services.default_assistants.get_active_config_payload",
        lambda: None,
    )
    await ensure_default_assistants()


@pytest.mark.asyncio
async def test_ensure_default_assistants_noop_when_no_graphs_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "agentseek_api.services.default_assistants.get_active_config_payload",
        lambda: {"not_graphs": {}},
    )
    await ensure_default_assistants()


@pytest.mark.asyncio
async def test_ensure_default_assistants_noop_when_graphs_not_dict(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "agentseek_api.services.default_assistants.get_active_config_payload",
        lambda: {"graphs": "not-a-dict"},
    )
    await ensure_default_assistants()
