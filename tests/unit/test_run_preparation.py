from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import pytest

from agentseek_api.models.auth import User
from agentseek_api.services import run_preparation as run_prep_module
from agentseek_api.services.run_jobs import RunExecutionJob
from agentseek_api.services.thread_protocol import ThreadProtocolEventBroker


class FakeSession:
    def __init__(self, scalar_values: list[object | None], execute_rowcounts: list[int] | None = None) -> None:
        self.scalar_values = scalar_values
        self.execute_rowcounts = list(execute_rowcounts or [])
        self.added: list[object] = []
        self.commits = 0

    async def scalar(self, _query: Any) -> object | None:
        return self.scalar_values.pop(0) if self.scalar_values else None

    async def execute(self, _statement: Any):
        rowcount = self.execute_rowcounts.pop(0) if self.execute_rowcounts else 1
        return type("FakeExecuteResult", (), {"rowcount": rowcount})()

    def add(self, obj: object) -> None:
        self.added.append(obj)

    async def commit(self) -> None:
        self.commits += 1

    async def refresh(self, _obj: object) -> None:
        return None

    async def flush(self) -> None:
        return None


class TrackingSession(FakeSession):
    def __init__(self, scalar_values: list[object | None], operations: list[str]) -> None:
        super().__init__(scalar_values)
        self.operations = operations

    async def commit(self) -> None:
        self.operations.append("commit")
        await super().commit()


class CallbackSession(FakeSession):
    def __init__(self, scalar_values: list[Callable[[], object | None] | object | None]) -> None:
        super().__init__([])
        self.scalar_values = scalar_values

    async def scalar(self, _query: Any) -> object | None:
        value = self.scalar_values.pop(0) if self.scalar_values else None
        return value() if callable(value) else value


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


class InlineExecutor:
    async def submit(self, job: RunExecutionJob) -> None:
        await run_prep_module._execute_and_persist(
            run_id=job.run_id,
            thread_id=job.thread_id,
            user_id=job.user_id,
            payload=job.payload,
            graph_id=job.graph_id,
            kwargs=job.kwargs,
            resume=job.resume,
            is_resume=job.is_resume,
        )


class DeferredExecutor:
    def __init__(self) -> None:
        self.submitted: list[RunExecutionJob] = []

    async def submit(self, job: RunExecutionJob) -> None:
        self.submitted.append(job)


class RaisingExecutor:
    async def submit(self, _job: RunExecutionJob) -> None:
        raise RuntimeError("submit failed")


@pytest.mark.asyncio
async def test_prepare_run_raises_when_thread_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    session_factory = FakeSessionFactory([FakeSession([None])])
    monkeypatch.setattr("agentseek_api.services.run_preparation.db_manager.get_session_factory", lambda: session_factory)

    with pytest.raises(ValueError, match="Thread not found"):
        await run_prep_module.prepare_and_submit_run(
            thread_id="t1",
            assistant_id="a1",
            payload={"x": 1},
            user=User(identity="u1", is_authenticated=True),
        )


@pytest.mark.asyncio
async def test_prepare_run_raises_when_assistant_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    session_factory = FakeSessionFactory([FakeSession([object(), None])])
    monkeypatch.setattr("agentseek_api.services.run_preparation.db_manager.get_session_factory", lambda: session_factory)

    with pytest.raises(ValueError, match="Assistant not found"):
        await run_prep_module.prepare_and_submit_run(
            thread_id="t1",
            assistant_id="a1",
            payload={"x": 1},
            user=User(identity="u1", is_authenticated=True),
        )


@pytest.mark.asyncio
async def test_prepare_run_sets_error_status_when_execute_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_assistant = type("FakeAssistant", (), {"graph_id": "stress_test"})()
    fake_thread = type("FakeThread", (), {"thread_id": "t1", "user_id": "u1", "status": "idle", "state_updated_at": None})()
    create_session = FakeSession([fake_thread, fake_assistant], execute_rowcounts=[1])
    db_run = type("DbRun", (), {"run_id": "r1", "status": "pending", "output_json": None, "last_error": None})()
    exec_session = FakeSession([db_run])
    reload_session = FakeSession([db_run])
    session_factory = FakeSessionFactory([create_session, exec_session, reload_session])

    monkeypatch.setattr("agentseek_api.services.run_preparation.db_manager.get_session_factory", lambda: session_factory)
    monkeypatch.setattr("agentseek_api.services.run_preparation.get_executor", lambda: InlineExecutor())

    captured: dict[str, Any] = {}

    async def failing_execute_run(
        *,
        thread_id: str,
        run_id: str,
        payload: dict,
        user_id: str,
        graph_id: str | None = None,
        resume: Any = None,
    ) -> dict:
        captured["graph_id"] = graph_id
        captured["user_id"] = user_id
        _ = (thread_id, run_id, payload, resume)
        raise RuntimeError("boom")

    events: list[tuple[str, str, dict[str, Any]]] = []
    monkeypatch.setattr("agentseek_api.services.run_preparation.execute_run", failing_execute_run)
    monkeypatch.setattr(
        "agentseek_api.services.run_preparation.run_broker.publish",
        lambda run_id, event, **payload: events.append((run_id, event, payload)),
    )

    run = await run_prep_module.prepare_and_submit_run(
        thread_id="t1",
        assistant_id="a1",
        payload={"x": 1},
        user=User(identity="u1", is_authenticated=True),
    )

    assert run.status == "error"
    assert db_run.status == "error"
    assert db_run.last_error == "boom"
    assert events[-1][1] == "end"
    assert events[-1][2]["status"] == "error"
    assert captured["graph_id"] == "stress_test"
    assert captured["user_id"] == "u1"


@pytest.mark.asyncio
async def test_execute_and_persist_publishes_terminal_run_event_before_terminal_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    operations: list[str] = []
    exec_session = TrackingSession([db_run, fake_thread, fake_thread], operations)
    session_factory = FakeSessionFactory([exec_session])
    published_run_events: list[tuple[str, int, dict[str, Any]]] = []

    async def successful_execute_run(**_kwargs: Any) -> run_prep_module.RunExecutionResult:
        return run_prep_module.RunExecutionResult(output={"ok": True}, interrupted=False, interrupts=[])

    async def fake_publish_run_event(_run_id: str, event: str, *, persist: bool = True, **payload: Any) -> tuple[int, dict[str, Any]]:
        _ = persist
        operations.append(f"publish:{event}")
        published_run_events.append((event, exec_session.commits, payload))
        return len(published_run_events), {"event": event, **payload}

    async def fake_publish_lifecycle(*_args: Any, **_kwargs: Any) -> None:
        return None

    async def fake_persist_thread_snapshot(_thread_id: str) -> None:
        return None

    async def fake_add_run_stream_event_to_session(
        _session: FakeSession,
        _run_id: str,
        *,
        seq: int,
        payload: dict[str, Any],
    ) -> None:
        operations.append(f"persist:{payload['event']}:{seq}")

    monkeypatch.setattr("agentseek_api.services.run_preparation.db_manager.get_session_factory", lambda: session_factory)
    monkeypatch.setattr("agentseek_api.services.run_preparation.execute_run", successful_execute_run)
    monkeypatch.setattr("agentseek_api.services.run_preparation._publish_run_event", fake_publish_run_event)
    monkeypatch.setattr("agentseek_api.services.run_preparation._publish_lifecycle", fake_publish_lifecycle)
    monkeypatch.setattr("agentseek_api.services.run_preparation._persist_thread_snapshot", fake_persist_thread_snapshot)
    monkeypatch.setattr(
        "agentseek_api.services.run_preparation.add_run_stream_event_to_session",
        fake_add_run_stream_event_to_session,
    )

    await run_prep_module._execute_and_persist(
        run_id="r1",
        thread_id="t1",
        user_id="u1",
        payload={"x": 1},
        graph_id="default",
    )

    assert exec_session.commits == 2
    assert published_run_events == [
        ("start", 1, {}),
        ("end", 1, {"status": "success"}),
    ]
    assert operations == [
        "commit",
        "publish:start",
        "publish:end",
        "persist:end:2",
        "commit",
    ]


@pytest.mark.asyncio
async def test_prepare_run_marks_thread_busy_before_background_execution(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_thread = type("FakeThread", (), {"thread_id": "t1", "user_id": "u1", "status": "idle", "state_updated_at": None})()
    fake_assistant = type("FakeAssistant", (), {"graph_id": "default"})()
    db_run = type(
        "DbRun",
        (),
        {
            "run_id": "r1",
            "thread_id": "t1",
            "assistant_id": "a1",
            "user_id": "u1",
            "status": "pending",
            "input_json": {"x": 1},
            "output_json": None,
            "last_error": None,
        },
    )()
    create_session = FakeSession([fake_thread, fake_assistant], execute_rowcounts=[1])
    reload_session = FakeSession([db_run])
    session_factory = FakeSessionFactory([create_session, reload_session])
    executor = DeferredExecutor()

    monkeypatch.setattr("agentseek_api.services.run_preparation.db_manager.get_session_factory", lambda: session_factory)
    monkeypatch.setattr("agentseek_api.services.run_preparation.get_executor", lambda: executor)

    run = await run_prep_module.prepare_and_submit_run(
        thread_id="t1",
        assistant_id="a1",
        payload={"x": 1},
        user=User(identity="u1", is_authenticated=True),
    )

    assert run.status == "pending"
    assert fake_thread.status == "busy"
    assert fake_thread.state_updated_at is not None
    assert len(executor.submitted) == 1
    submitted = executor.submitted[0]
    assert submitted.thread_id == "t1"
    assert submitted.user_id == "u1"
    assert submitted.payload == {"x": 1}
    assert submitted.graph_id == "default"
    assert submitted.is_resume is False


@pytest.mark.asyncio
async def test_prepare_run_cleans_protocol_state_when_submit_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_thread = type("FakeThread", (), {"thread_id": "t1", "user_id": "u1", "status": "idle", "state_updated_at": None})()
    fake_assistant = type("FakeAssistant", (), {"graph_id": "default"})()
    create_session = FakeSession([fake_thread, fake_assistant], execute_rowcounts=[1])
    persist_session = CallbackSession([lambda: create_session.added[-1], fake_thread])
    session_factory = FakeSessionFactory([create_session, persist_session])
    protocol_broker = ThreadProtocolEventBroker()
    published_lifecycle: list[dict[str, Any]] = []

    monkeypatch.setattr("agentseek_api.services.run_preparation.db_manager.get_session_factory", lambda: session_factory)
    monkeypatch.setattr("agentseek_api.services.run_preparation.get_executor", lambda: RaisingExecutor())
    monkeypatch.setattr("agentseek_api.services.run_preparation.thread_protocol_broker", protocol_broker)
    monkeypatch.setattr(
        "agentseek_api.services.run_preparation.publish_lifecycle_event",
        lambda thread_id, **payload: published_lifecycle.append({"thread_id": thread_id, **payload}),
    )

    with pytest.raises(RuntimeError, match="submit failed"):
        await run_prep_module.prepare_and_submit_run(
            thread_id="t1",
            assistant_id="a1",
            payload={"x": 1},
            user=User(identity="u1", is_authenticated=True),
        )

    created_run = create_session.added[-1]
    assert created_run.status == "error"
    assert created_run.last_error == "submit failed"
    assert fake_thread.status == "error"
    assert protocol_broker._active_runs["t1"] == 0
    assert published_lifecycle == [
        {"thread_id": "t1", "event": "started", "graph_name": "default", "persist": False, "seq": None},
        {
            "thread_id": "t1",
            "event": "failed",
            "graph_name": "default",
            "error": "submit failed",
            "persist": False,
            "seq": None,
        },
    ]


@pytest.mark.asyncio
async def test_execute_and_persist_cleans_protocol_state_for_cancelled_run(monkeypatch: pytest.MonkeyPatch) -> None:
    cancelled_run = type("DbRun", (), {"run_id": "r1", "status": "error", "last_error": "Run cancelled"})()
    session_factory = FakeSessionFactory([FakeSession([cancelled_run])])
    protocol_broker = ThreadProtocolEventBroker()
    published_lifecycle: list[dict[str, Any]] = []

    monkeypatch.setattr("agentseek_api.services.run_preparation.db_manager.get_session_factory", lambda: session_factory)
    monkeypatch.setattr("agentseek_api.services.run_preparation.thread_protocol_broker", protocol_broker)
    monkeypatch.setattr(
        "agentseek_api.services.run_preparation.publish_lifecycle_event",
        lambda thread_id, **payload: published_lifecycle.append({"thread_id": thread_id, **payload}),
    )

    protocol_broker.run_started("t1")
    await run_prep_module._execute_and_persist(
        run_id="r1",
        thread_id="t1",
        user_id="u1",
        payload={"x": 1},
        graph_id="default",
    )

    assert protocol_broker._active_runs["t1"] == 0
    assert published_lifecycle == [
        {
            "thread_id": "t1",
            "event": "failed",
            "graph_name": "default",
            "error": "Run cancelled",
            "persist": False,
            "seq": None,
        }
    ]


@pytest.mark.asyncio
async def test_resume_run_marks_row_pending_before_background_execution(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_assistant = type("FakeAssistant", (), {"assistant_id": "a1", "graph_id": "subgraph_hitl_agent"})()
    fake_thread = type("FakeThread", (), {"thread_id": "t1", "user_id": "u1", "status": "idle", "state_updated_at": None})()
    db_run = type(
        "DbRun",
        (),
        {
            "run_id": "r1",
            "thread_id": "t1",
            "assistant_id": "a1",
            "user_id": "u1",
            "status": "interrupted",
            "input_json": {"foo": "hello "},
            "output_json": {"interrupts": [{"value": "Provide value:"}], "interrupted": True},
            "last_error": None,
        },
    )()
    load_session = FakeSession([fake_thread, db_run, fake_assistant], execute_rowcounts=[1])
    reload_session = FakeSession([db_run])
    session_factory = FakeSessionFactory([load_session, reload_session])
    executor = DeferredExecutor()

    monkeypatch.setattr("agentseek_api.services.run_preparation.db_manager.get_session_factory", lambda: session_factory)
    monkeypatch.setattr("agentseek_api.services.run_preparation.get_executor", lambda: executor)

    run = await run_prep_module.resume_run(
        thread_id="t1",
        run_id="r1",
        resume="world",
        user=User(identity="u1", is_authenticated=True),
    )

    assert run.status == "pending"
    assert db_run.status == "pending"
    assert fake_thread.status == "busy"
    assert fake_thread.state_updated_at is not None
    assert load_session.commits == 1
    assert len(executor.submitted) == 1
    submitted = executor.submitted[0]
    assert submitted.run_id == "r1"
    assert submitted.thread_id == "t1"
    assert submitted.payload == {"foo": "hello "}
    assert submitted.graph_id == "subgraph_hitl_agent"
    assert submitted.resume == "world"
    assert submitted.is_resume is True


@pytest.mark.asyncio
async def test_resume_run_restores_interrupted_state_when_submit_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_assistant = type("FakeAssistant", (), {"assistant_id": "a1", "graph_id": "subgraph_hitl_agent"})()
    fake_thread = type("FakeThread", (), {"thread_id": "t1", "user_id": "u1", "status": "idle", "state_updated_at": None})()
    db_run = type(
        "DbRun",
        (),
        {
            "run_id": "r1",
            "thread_id": "t1",
            "assistant_id": "a1",
            "user_id": "u1",
            "status": "interrupted",
            "input_json": {"foo": "hello "},
            "output_json": {"interrupts": [{"value": "Provide value:"}], "interrupted": True},
            "last_error": None,
        },
    )()
    load_session = FakeSession([fake_thread, db_run, fake_assistant], execute_rowcounts=[1])
    persist_session = CallbackSession([db_run, fake_thread])
    session_factory = FakeSessionFactory([load_session, persist_session])
    protocol_broker = ThreadProtocolEventBroker()
    published_lifecycle: list[dict[str, Any]] = []

    monkeypatch.setattr("agentseek_api.services.run_preparation.db_manager.get_session_factory", lambda: session_factory)
    monkeypatch.setattr("agentseek_api.services.run_preparation.get_executor", lambda: RaisingExecutor())
    monkeypatch.setattr("agentseek_api.services.run_preparation.thread_protocol_broker", protocol_broker)
    monkeypatch.setattr(
        "agentseek_api.services.run_preparation.publish_lifecycle_event",
        lambda thread_id, **payload: published_lifecycle.append({"thread_id": thread_id, **payload}),
    )

    with pytest.raises(RuntimeError, match="submit failed"):
        await run_prep_module.resume_run(
            thread_id="t1",
            run_id="r1",
            resume="world",
            user=User(identity="u1", is_authenticated=True),
        )

    assert db_run.status == "interrupted"
    assert db_run.last_error == "submit failed"
    assert fake_thread.status == "interrupted"
    assert protocol_broker._active_runs["t1"] == 0
    assert published_lifecycle == [
        {"thread_id": "t1", "event": "started", "graph_name": "subgraph_hitl_agent", "persist": False, "seq": None},
        {
            "thread_id": "t1",
            "event": "failed",
            "graph_name": "subgraph_hitl_agent",
            "error": "submit failed",
            "persist": False,
            "seq": None,
        },
    ]


@pytest.mark.asyncio
async def test_resume_run_rejects_when_thread_already_has_active_run(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_thread = type("FakeThread", (), {"thread_id": "t1", "user_id": "u1", "status": "busy", "state_updated_at": None})()
    db_run = type(
        "DbRun",
        (),
        {
            "run_id": "r1",
            "thread_id": "t1",
            "assistant_id": "a1",
            "user_id": "u1",
            "status": "interrupted",
            "input_json": {"foo": "hello "},
            "output_json": {"interrupts": [{"value": "Provide value:"}], "interrupted": True},
            "last_error": None,
        },
    )()
    fake_assistant = type("FakeAssistant", (), {"assistant_id": "a1", "graph_id": "subgraph_hitl_agent"})()
    active_run_id = "r1"
    session_factory = FakeSessionFactory([FakeSession([fake_thread, db_run, fake_assistant, active_run_id], execute_rowcounts=[0])])

    monkeypatch.setattr("agentseek_api.services.run_preparation.db_manager.get_session_factory", lambda: session_factory)

    with pytest.raises(run_prep_module.ActiveThreadRunConflictError, match=run_prep_module.ACTIVE_THREAD_RUN_CONFLICT):
        await run_prep_module.resume_run(
            thread_id="t1",
            run_id="r1",
            resume="world",
            user=User(identity="u1", is_authenticated=True),
        )


@pytest.mark.asyncio
async def test_resume_run_reports_not_interrupted_when_claim_fails_without_active_run(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_thread = type("FakeThread", (), {"thread_id": "t1", "user_id": "u1", "status": "idle", "state_updated_at": None})()
    db_run = type(
        "DbRun",
        (),
        {
            "run_id": "r1",
            "thread_id": "t1",
            "assistant_id": "a1",
            "user_id": "u1",
            "status": "success",
            "input_json": {"foo": "hello "},
            "output_json": {"foo": "hello world"},
            "last_error": None,
        },
    )()
    fake_assistant = type("FakeAssistant", (), {"assistant_id": "a1", "graph_id": "subgraph_hitl_agent"})()
    session_factory = FakeSessionFactory([FakeSession([fake_thread, db_run, fake_assistant, None], execute_rowcounts=[0])])

    monkeypatch.setattr("agentseek_api.services.run_preparation.db_manager.get_session_factory", lambda: session_factory)

    with pytest.raises(RuntimeError, match="Run is not interrupted"):
        await run_prep_module.resume_run(
            thread_id="t1",
            run_id="r1",
            resume="world",
            user=User(identity="u1", is_authenticated=True),
        )
