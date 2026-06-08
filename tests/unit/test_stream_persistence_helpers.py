from types import SimpleNamespace
from typing import Any

import pytest

from agentseek_api.services import stream_persistence as stream_module


class FakeScalarResult:
    def __init__(self, rows: list[object]) -> None:
        self._rows = rows

    def all(self) -> list[object]:
        return self._rows


class FakeSession:
    def __init__(
        self,
        *,
        scalar_values: list[object | None] | None = None,
        scalars_rows: list[object] | None = None,
        execute_error: Exception | None = None,
        commit_error: Exception | None = None,
    ) -> None:
        self.scalar_values = list(scalar_values or [])
        self.scalars_rows = list(scalars_rows or [])
        self.execute_error = execute_error
        self.commit_error = commit_error
        self.added: list[object] = []
        self.commits = 0

    async def scalar(self, _query: Any) -> object | None:
        return self.scalar_values.pop(0) if self.scalar_values else None

    async def scalars(self, _query: Any) -> FakeScalarResult:
        return FakeScalarResult(self.scalars_rows)

    async def execute(self, _query: Any) -> None:
        if self.execute_error is not None:
            raise self.execute_error

    async def commit(self) -> None:
        self.commits += 1
        if self.commit_error is not None:
            raise self.commit_error

    def add(self, obj: object) -> None:
        self.added.append(obj)


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


def test_parse_last_event_id_handles_empty_and_invalid_values() -> None:
    assert stream_module.parse_last_event_id(None) is None
    assert stream_module.parse_last_event_id("") is None
    assert stream_module.parse_last_event_id("7") == 7
    assert stream_module.parse_last_event_id("-1") is None
    assert stream_module.parse_last_event_id("not-an-int") is None


@pytest.mark.asyncio
async def test_add_stream_event_helpers_skip_existing_records() -> None:
    session = FakeSession(scalar_values=[1, 1])

    await stream_module.add_run_stream_event_to_session(
        session,
        "run-1",
        seq=1,
        payload={"event": "start"},
    )
    await stream_module.add_thread_stream_event_to_session(
        session,
        "thread-1",
        seq=1,
        payload={"method": "lifecycle"},
    )

    assert session.added == []


@pytest.mark.asyncio
async def test_load_helpers_return_empty_when_metadata_db_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(stream_module, "_metadata_db_ready", lambda: False)

    assert await stream_module.load_run_stream_events("run-1") == []
    assert await stream_module.load_thread_stream_events(
        "thread-1",
        channels=["values"],
        namespaces=None,
        depth=None,
    ) == []

    await stream_module.delete_run_stream_events([])


@pytest.mark.asyncio
async def test_load_helpers_return_empty_when_session_factory_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(stream_module, "_metadata_db_ready", lambda: True)

    def raise_runtime_error() -> object:
        raise RuntimeError("not initialized")

    monkeypatch.setattr(stream_module.db_manager, "get_session_factory", raise_runtime_error)

    assert await stream_module.load_run_stream_events("run-1") == []
    assert await stream_module.load_thread_stream_events(
        "thread-1",
        channels=["values"],
        namespaces=None,
        depth=None,
    ) == []


@pytest.mark.asyncio
async def test_persist_helpers_swallow_commit_errors_and_invalid_thread_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_session = FakeSession(scalar_values=[None], commit_error=RuntimeError("boom"))
    thread_session = FakeSession(scalar_values=[None], commit_error=RuntimeError("boom"))
    session_factory = FakeSessionFactory([run_session, thread_session])

    monkeypatch.setattr(stream_module, "_metadata_db_ready", lambda: True)
    monkeypatch.setattr(stream_module.db_manager, "get_session_factory", lambda: session_factory)

    await stream_module.persist_run_stream_event("run-1", seq=1, payload={"event": "start"})
    await stream_module.persist_thread_stream_event("thread-1", {"seq": 0, "method": "lifecycle"})
    await stream_module.persist_thread_stream_event("thread-1", {"seq": 2, "method": "lifecycle"})

    assert len(run_session.added) == 1
    assert len(thread_session.added) == 1


@pytest.mark.asyncio
async def test_delete_helpers_swallow_session_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    session_factory = FakeSessionFactory(
        [
            FakeSession(execute_error=RuntimeError("boom")),
            FakeSession(execute_error=RuntimeError("boom")),
        ]
    )

    monkeypatch.setattr(stream_module, "_metadata_db_ready", lambda: True)
    monkeypatch.setattr(stream_module.db_manager, "get_session_factory", lambda: session_factory)

    await stream_module.delete_run_stream_events(["run-1"])
    await stream_module.delete_thread_stream_events("thread-1")


@pytest.mark.asyncio
async def test_load_thread_stream_events_normalizes_non_list_namespaces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = SimpleNamespace(
        payload_json={
            "seq": 1,
            "method": "values",
            "params": {"namespace": "not-a-list", "timestamp": 1, "data": {"ok": True}},
        }
    )
    session_factory = FakeSessionFactory([FakeSession(scalars_rows=[row])])

    monkeypatch.setattr(stream_module, "_metadata_db_ready", lambda: True)
    monkeypatch.setattr(stream_module.db_manager, "get_session_factory", lambda: session_factory)

    events = await stream_module.load_thread_stream_events(
        "thread-1",
        channels=["values"],
        namespaces=None,
        depth=None,
    )

    assert events == [row.payload_json]
