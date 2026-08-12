from datetime import UTC, datetime
from typing import Any

import pytest

from agentseek_api.settings import settings
from agentseek_api.services import run_jobs as run_jobs_module
from agentseek_api.services.thread_protocol import ThreadProtocolEventBroker


class FakeSession:
    def __init__(self, scalar_values: list[object | None], operations: list[str]) -> None:
        self.scalar_values = scalar_values
        self.operations = operations
        self.commits = 0

    async def scalar(self, _query: Any) -> object | None:
        return self.scalar_values.pop(0) if self.scalar_values else None

    async def commit(self) -> None:
        self.operations.append("commit")
        self.commits += 1

    async def refresh(self, _obj: object) -> None:
        return None

    def add(self, _obj: object) -> None:
        return None


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


def _job(*, run_id: str = "r1", thread_id: str = "t1") -> run_jobs_module.RunExecutionJob:
    return run_jobs_module.RunExecutionJob(
        run_id=run_id,
        thread_id=thread_id,
        user_id="u1",
        payload={"message": "hello"},
        graph_id="default",
    )


@pytest.mark.asyncio
async def test_publish_run_event_uses_atomic_redis_append(monkeypatch: pytest.MonkeyPatch) -> None:
    published: list[tuple[str, str, int | None, dict[str, Any]]] = []
    monkeypatch.setattr(settings, "EXECUTOR_BACKEND", "redis")

    async def fake_append(run_id: str, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        assert run_id == "run-1"
        return 7, payload

    async def unexpected_next_seq(_run_id: str) -> int:
        raise AssertionError("Redis sequence allocation must be part of the append")

    monkeypatch.setattr(run_jobs_module, "append_redis_run_stream_event", fake_append, raising=False)
    monkeypatch.setattr(run_jobs_module, "next_run_stream_seq", unexpected_next_seq)
    monkeypatch.setattr(
        run_jobs_module.run_broker,
        "publish",
        lambda run_id, event, *, seq=None, **payload: (
            published.append((run_id, event, seq, payload)) or (seq, {"event": event, **payload})
        ),
    )

    result = await run_jobs_module._publish_run_event("run-1", "message", data="hello")

    assert result == (7, {"event": "message", "data": "hello"})
    assert published == [("run-1", "message", 7, {"data": "hello"})]


@pytest.mark.asyncio
async def test_publish_lifecycle_uses_atomic_redis_append(monkeypatch: pytest.MonkeyPatch) -> None:
    published: list[tuple[str, str]] = []
    monkeypatch.setattr(settings, "EXECUTOR_BACKEND", "redis")

    async def fake_apublish(thread_id: str, **payload: Any) -> dict[str, Any]:
        published.append((thread_id, payload["event"]))
        return {"seq": 3, "method": "lifecycle"}

    async def unexpected_next_seq(_thread_id: str) -> int:
        raise AssertionError("Redis lifecycle sequence must be allocated atomically")

    monkeypatch.setattr(run_jobs_module, "apublish_lifecycle_event", fake_apublish, raising=False)
    monkeypatch.setattr(run_jobs_module, "next_thread_stream_seq", unexpected_next_seq)

    await run_jobs_module._publish_lifecycle("thread-1", event="completed", session=FakeSession([], []))

    assert published == [("thread-1", "completed")]


@pytest.mark.asyncio
async def test_terminal_run_event_uses_atomic_redis_append_without_sql_helper(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, str, bool, dict[str, Any]]] = []
    monkeypatch.setattr(settings, "EXECUTOR_BACKEND", "redis")
    publish_terminal = getattr(run_jobs_module, "_publish_terminal_run_event", None)

    async def fake_publish(run_id: str, event: str, *, persist: bool = True, **payload: Any):
        calls.append((run_id, event, persist, payload))
        return 4, {"event": event, **payload}

    async def unexpected_sql_helper(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("Redis terminal events must not use the SQL session helper")

    monkeypatch.setattr(run_jobs_module, "_publish_run_event", fake_publish)
    monkeypatch.setattr(run_jobs_module, "add_run_stream_event_to_session", unexpected_sql_helper)

    assert callable(publish_terminal)
    await publish_terminal(FakeSession([], []), "run-1", status="success")

    assert calls == [("run-1", "end", True, {"status": "success"})]


@pytest.mark.asyncio
async def test_execute_run_job_skips_terminal_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    operations: list[str] = []
    db_run = type("DbRun", (), {"run_id": "r1", "status": "success", "last_error": None})()
    session_factory = FakeSessionFactory([FakeSession([db_run], operations)])

    async def unexpected_execute_run(**_kwargs: Any) -> run_jobs_module.RunExecutionResult:
        raise AssertionError("Terminal runs should not be re-executed")

    monkeypatch.setattr(run_jobs_module.db_manager, "get_session_factory", lambda: session_factory)
    monkeypatch.setattr(run_jobs_module, "execute_run", unexpected_execute_run)

    await run_jobs_module.execute_run_job(_job())

    assert operations == []


@pytest.mark.asyncio
async def test_execute_run_job_persists_terminal_lifecycle_before_final_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operations: list[str] = []
    db_run = type(
        "DbRun",
        (),
        {
            "run_id": "r1",
            "status": "pending",
            "output_json": None,
            "last_error": None,
            "updated_at": datetime.now(UTC),
        },
    )()
    fake_thread = type("FakeThread", (), {"thread_id": "t1", "status": "idle", "state_updated_at": None})()
    session_factory = FakeSessionFactory([FakeSession([db_run, fake_thread, fake_thread], operations)])

    async def successful_execute_run(**_kwargs: Any) -> run_jobs_module.RunExecutionResult:
        return run_jobs_module.RunExecutionResult(output={"ok": True}, interrupted=False, interrupts=[])

    async def fake_publish_run_event(_run_id: str, event: str, *, persist: bool = True, **payload: Any) -> tuple[int, dict[str, Any]]:
        _ = persist
        operations.append(f"publish:{event}")
        return (1 if event == "start" else 2), {"event": event, **payload}


    async def fake_add_run_stream_event_to_session(
        _session: FakeSession,
        _run_id: str,
        *,
        seq: int | None = None,
        payload: dict[str, Any],
    ) -> tuple[int | None, dict[str, Any]]:
        operations.append(f"persist:run:{payload['event']}:{seq}")
        return seq or 0, payload

    def fake_publish_lifecycle_event(
        _thread_id: str,
        *,
        event: str,
        graph_name: str | None = None,
        error: str | None = None,
        namespace: list[str] | None = None,
        persist: bool = True,
        seq: int | None = None,
    ) -> dict[str, Any]:
        _ = (graph_name, error, namespace, seq)
        operations.append(f"publish:lifecycle:{event}:{persist}")
        return {
            "seq": 3,
            "method": "lifecycle",
            "params": {"namespace": [], "timestamp": 1, "data": {"event": event}},
        }

    async def fake_add_thread_stream_event_to_session(
        _session: FakeSession,
        _thread_id: str,
        *,
        seq: int | None = None,
        payload: dict[str, Any],
    ) -> tuple[int | None, dict[str, Any]]:
        operations.append(f"persist:thread:{payload['params']['data']['event']}:{seq}")
        return seq or 0, payload

    monkeypatch.setattr(run_jobs_module.db_manager, "get_session_factory", lambda: session_factory)
    monkeypatch.setattr(run_jobs_module, "execute_run", successful_execute_run)
    monkeypatch.setattr(run_jobs_module, "_publish_run_event", fake_publish_run_event)
    monkeypatch.setattr(run_jobs_module, "add_run_stream_event_to_session", fake_add_run_stream_event_to_session)
    monkeypatch.setattr(run_jobs_module, "publish_lifecycle_event", fake_publish_lifecycle_event)
    monkeypatch.setattr(run_jobs_module, "add_thread_stream_event_to_session", fake_add_thread_stream_event_to_session)

    await run_jobs_module.execute_run_job(_job())

    assert operations == [
        "commit",
        "publish:start",
        "persist:run:end:None",
        "persist:thread:completed:None",
        "commit",
    ]


@pytest.mark.asyncio
async def test_execute_run_job_publishes_failed_lifecycle_when_run_deleted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operations: list[str] = []
    session_factory = FakeSessionFactory([FakeSession([None], operations)])
    protocol_broker = ThreadProtocolEventBroker()

    async def fake_add_thread_stream_event_to_session(_session: Any, _thread_id: str, *, seq: int | None = None, payload: dict[str, Any]) -> None:
        pass

    monkeypatch.setattr(run_jobs_module.db_manager, "get_session_factory", lambda: session_factory)
    monkeypatch.setattr(run_jobs_module, "thread_protocol_broker", protocol_broker)
    monkeypatch.setattr(run_jobs_module, "add_thread_stream_event_to_session", fake_add_thread_stream_event_to_session)

    await run_jobs_module.execute_run_job(_job())

    lifecycle_events = [
        event["params"]["data"]
        for event in protocol_broker.snapshot_records("t1")
    ]
    assert any(
        event.get("event") == "failed" and "Run was deleted" in (event.get("error") or "")
        for event in lifecycle_events
    )


def test_from_payload_rejects_unsupported_kind() -> None:
    with pytest.raises(ValueError, match="Unsupported run job kind"):
        run_jobs_module.RunExecutionJob.from_payload({"kind": "unknown"})


@pytest.mark.asyncio
async def test_execute_run_job_publishes_interrupted_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operations: list[str] = []
    db_run = type(
        "DbRun",
        (),
        {
            "run_id": "r1",
            "status": "pending",
            "output_json": None,
            "last_error": None,
            "updated_at": datetime.now(UTC),
            "metadata_json": {},
        },
    )()
    fake_thread = type("FakeThread", (), {"thread_id": "t1", "status": "idle", "state_updated_at": None})()
    session_factory = FakeSessionFactory([FakeSession([db_run, fake_thread, fake_thread], operations)])

    async def interrupted_execute_run(**_kwargs: Any) -> run_jobs_module.RunExecutionResult:
        return run_jobs_module.RunExecutionResult(output={"__interrupt__": []}, interrupted=True, interrupts=[{"id": "i1"}])

    async def fake_publish_run_event(_run_id: str, event: str, *, persist: bool = True, **payload: Any) -> tuple[int, dict[str, Any]]:
        _ = persist
        operations.append(f"publish:{event}")
        return (1 if event == "start" else 2), {"event": event, **payload}


    async def fake_add_run_stream_event_to_session(
        _session: FakeSession,
        _run_id: str,
        *,
        seq: int | None = None,
        payload: dict[str, Any],
    ) -> tuple[int | None, dict[str, Any]]:
        operations.append(f"persist:run:{payload['event']}:{seq}")
        return seq or 0, payload

    def fake_publish_lifecycle_event(
        _thread_id: str,
        *,
        event: str,
        graph_name: str | None = None,
        error: str | None = None,
        namespace: list[str] | None = None,
        persist: bool = True,
        seq: int | None = None,
    ) -> dict[str, Any]:
        _ = (graph_name, error, namespace, seq)
        operations.append(f"publish:lifecycle:{event}:{persist}")
        return {
            "seq": 3,
            "method": "lifecycle",
            "params": {"namespace": [], "timestamp": 1, "data": {"event": event}},
        }

    async def fake_add_thread_stream_event_to_session(
        _session: FakeSession,
        _thread_id: str,
        *,
        seq: int | None = None,
        payload: dict[str, Any],
    ) -> tuple[int | None, dict[str, Any]]:
        operations.append(f"persist:thread:{payload['params']['data']['event']}:{seq}")
        return seq or 0, payload

    monkeypatch.setattr(run_jobs_module.db_manager, "get_session_factory", lambda: session_factory)
    monkeypatch.setattr(run_jobs_module, "execute_run", interrupted_execute_run)
    monkeypatch.setattr(run_jobs_module, "_publish_run_event", fake_publish_run_event)
    monkeypatch.setattr(run_jobs_module, "add_run_stream_event_to_session", fake_add_run_stream_event_to_session)
    monkeypatch.setattr(run_jobs_module, "publish_lifecycle_event", fake_publish_lifecycle_event)
    monkeypatch.setattr(run_jobs_module, "add_thread_stream_event_to_session", fake_add_thread_stream_event_to_session)

    await run_jobs_module.execute_run_job(_job())

    assert db_run.status == "interrupted"
    assert "persist:thread:interrupted:None" in operations
