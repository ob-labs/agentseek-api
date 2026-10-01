from collections.abc import Callable
from typing import Any

import pytest

from agentseek_api.models.auth import User
from agentseek_api.services import run_preparation as run_prep_module
from agentseek_api.services.run_jobs import RunExecutionJob


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
            execution_id=job.execution_id,
            owns_accounting=job.owns_accounting,
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


@pytest.fixture(autouse=True)
def real_storage_for_orchestration(run_storage):
    return run_storage


async def _seed_run(factory, *, status="pending", error=None):
    from agentseek_api.core.orm import Run
    async with factory() as session:
        session.add(Run(run_id="r1", thread_id="t1", assistant_id="a1", user_id="u1",
                        status=status, last_error=error, input_json={"foo": "hello "},
                        output_json={"interrupts": [{"value": "Provide value:"}]}))
        await session.commit()


async def test_prepare_run_sets_error_status_when_execute_fails(run_storage, monkeypatch):
    async def execute(**kwargs):
        assert kwargs["user_id"] == "u1"
        raise RuntimeError("boom")
    monkeypatch.setattr(run_prep_module, "execute_run", execute)
    monkeypatch.setattr(run_prep_module, "get_executor", lambda: InlineExecutor())
    run = await run_prep_module.prepare_and_submit_run(
        thread_id="t1", assistant_id="a1", payload={"x": 1},
        user=User(identity="u1", is_authenticated=True))
    assert run.status == "error" and run.last_error == "RuntimeError: boom"
    assert run_prep_module.run_broker.snapshot_records(run.run_id)[-1][1]["event"] == "end"


async def test_prepare_run_marks_thread_busy_before_background_execution(run_storage, monkeypatch):
    from agentseek_api.core.orm import Thread
    executor = DeferredExecutor()
    monkeypatch.setattr(run_prep_module, "get_executor", lambda: executor)
    run = await run_prep_module.prepare_and_submit_run(
        thread_id="t1", assistant_id="a1", payload={"x": 1},
        user=User(identity="u1", is_authenticated=True))
    assert run.status == "pending"
    async with run_storage() as session:
        assert (await session.get(Thread, "t1")).status == "busy"
    assert len(executor.submitted) == 1
    assert executor.submitted[0].payload == {"x": 1}
    assert executor.submitted[0].execution_id == run.execution_id


async def test_prepare_run_cleans_protocol_state_when_submit_fails(run_storage, monkeypatch):
    from sqlalchemy import select
    from agentseek_api.core.orm import Run, Thread
    monkeypatch.setattr(run_prep_module, "get_executor", lambda: RaisingExecutor())
    with pytest.raises(RuntimeError, match="submit failed"):
        await run_prep_module.prepare_and_submit_run(
            thread_id="t1", assistant_id="a1", payload={"x": 1},
            user=User(identity="u1", is_authenticated=True))
    async with run_storage() as session:
        row = await session.scalar(select(Run))
        assert row.status == "error" and row.last_error == "submit failed"
        assert (await session.get(Thread, "t1")).status == "error"
    assert run_prep_module.thread_protocol_broker._active_runs["t1"] == 0
    assert [e["params"]["data"]["event"] for e in run_prep_module.thread_protocol_broker.snapshot_records("t1")] == ["started", "failed"]


async def test_execute_and_persist_cleans_protocol_state_for_cancelled_run(run_storage):
    await _seed_run(run_storage, status="error", error="Run cancelled")
    run_prep_module.thread_protocol_broker.run_started("t1")
    await run_prep_module._execute_and_persist(
        run_id="r1", thread_id="t1", user_id="u1", payload={}, graph_id="default", owns_accounting=True)
    assert run_prep_module.thread_protocol_broker._active_runs["t1"] == 0
    assert run_prep_module.run_broker.snapshot_records("r1") == []


async def test_resume_run_marks_row_pending_before_background_execution(run_storage, monkeypatch):
    from agentseek_api.core.orm import Thread
    await _seed_run(run_storage, status="interrupted")
    executor = DeferredExecutor()
    monkeypatch.setattr(run_prep_module, "get_executor", lambda: executor)
    run = await run_prep_module.resume_run(
        thread_id="t1", run_id="r1", resume="world", user=User(identity="u1", is_authenticated=True))
    assert run.status == "pending"
    async with run_storage() as session:
        assert (await session.get(Thread, "t1")).status == "busy"
    submitted = executor.submitted[0]
    assert submitted.is_resume and submitted.resume == "world"
    assert submitted.payload == {"foo": "hello "}
    assert submitted.execution_id == run.execution_id


async def test_resume_run_restores_interrupted_state_when_submit_fails(run_storage, monkeypatch):
    from agentseek_api.core.orm import Run, Thread
    await _seed_run(run_storage, status="interrupted")
    monkeypatch.setattr(run_prep_module, "get_executor", lambda: RaisingExecutor())
    with pytest.raises(RuntimeError, match="submit failed"):
        await run_prep_module.resume_run(
            thread_id="t1", run_id="r1", resume="world", user=User(identity="u1", is_authenticated=True))
    async with run_storage() as session:
        row = await session.get(Run, "r1")
        assert row.status == "interrupted" and row.last_error == "submit failed"
        assert row.output_json == {"interrupts": [{"value": "Provide value:"}]}
        assert (await session.get(Thread, "t1")).status == "interrupted"
    assert run_prep_module.thread_protocol_broker._active_runs["t1"] == 0


@pytest.mark.asyncio
async def test_resume_run_rejects_when_thread_already_has_active_run(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_thread = type("FakeThread", (), {"thread_id": "t1", "user_id": "u1", "status": "busy", "state_updated_at": None, "metadata_json": {}})()
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
    fake_assistant = type("FakeAssistant", (), {"assistant_id": "a1", "graph_id": "subgraph_hitl_agent", "context_json": None})()
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
    fake_thread = type("FakeThread", (), {"thread_id": "t1", "user_id": "u1", "status": "idle", "state_updated_at": None, "metadata_json": {}})()
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
    fake_assistant = type("FakeAssistant", (), {"assistant_id": "a1", "graph_id": "subgraph_hitl_agent", "context_json": None})()
    session_factory = FakeSessionFactory([FakeSession([fake_thread, db_run, fake_assistant, None], execute_rowcounts=[0])])

    monkeypatch.setattr("agentseek_api.services.run_preparation.db_manager.get_session_factory", lambda: session_factory)

    with pytest.raises(RuntimeError, match="Run is not interrupted"):
        await run_prep_module.resume_run(
            thread_id="t1",
            run_id="r1",
            resume="world",
            user=User(identity="u1", is_authenticated=True),
        )

def _make_assistant(*, config_json=None, context_json=None) -> object:
    """Build a fake assistant ORM row for _prepare_run tests."""
    return type(
        "FakeAssistant",
        (),
        {
            "assistant_id": "a1",
            "graph_id": "default",
            "context_json": context_json,
            "config_json": config_json,
        },
    )()


def _make_db_run() -> object:
    return type(
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


def _make_thread() -> object:
    return type(
        "FakeThread",
        (),
        {"thread_id": "t1", "user_id": "u1", "status": "idle", "state_updated_at": None, "metadata_json": {}},
    )()


async def _prepare_with_kwargs(
    monkeypatch: pytest.MonkeyPatch,
    *,
    assistant: object,
    kwargs: dict[str, Any] | None = None,
) -> DeferredExecutor:
    from agentseek_api.core.orm import Assistant
    async with run_prep_module.db_manager.get_session_factory()() as session:
        row = await session.get(Assistant, "a1")
        row.config_json = assistant.config_json or {}
        row.context_json = assistant.context_json or {}
        await session.commit()
    executor = DeferredExecutor()
    monkeypatch.setattr(run_prep_module, "get_executor", lambda: executor)
    await run_prep_module.prepare_and_submit_run(
        thread_id="t1",
        assistant_id="a1",
        payload={"x": 1},
        user=User(identity="u1", is_authenticated=True),
        kwargs=kwargs,
    )
    return executor


def test_merge_config_defaults() -> None:
    """Unit coverage for the aegra-style assistant config merge helper."""
    merge = run_prep_module._merge_config_defaults

    # No assistant config -> client config passes through untouched (copy).
    assert merge({}, {"configurable": {"a": 1}}) == {"configurable": {"a": 1}}

    # Assistant-only config surfaces as defaults.
    assert merge({"configurable": {"tender_text": "hello"}}, {}) == {"configurable": {"tender_text": "hello"}}

    # Client wins on top-level keys.
    assert merge({"recursion_limit": 10}, {"recursion_limit": 25}) == {"recursion_limit": 25}

    # configurable merged one level deeper: assistant defaults preserved, client wins per-key.
    merged = merge(
        {"configurable": {"tender_text": "hello", "model": "assistant-model"}},
        {"configurable": {"model": "client-model"}},
    )
    assert merged == {"configurable": {"tender_text": "hello", "model": "client-model"}}

    # Non-dict configurable values fall back to plain top-level merge.
    assert merge({"configurable": "bad"}, {"configurable": {"a": 1}}) == {"configurable": {"a": 1}}


@pytest.mark.asyncio
async def test_prepare_run_merges_assistant_config_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Assistant-level config.configurable becomes the run config default (aegra parity)."""
    assistant = _make_assistant(config_json={"configurable": {"tender_text": "hello"}})
    executor = await _prepare_with_kwargs(monkeypatch, assistant=assistant)

    assert len(executor.submitted) == 1
    assert executor.submitted[0].kwargs["config"] == {"configurable": {"tender_text": "hello"}}


@pytest.mark.asyncio
async def test_prepare_run_client_config_overrides_assistant_configurable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Client configurable keys win while assistant defaults are preserved."""
    assistant = _make_assistant(config_json={"configurable": {"tender_text": "hello", "model": "assistant-model"}})
    executor = await _prepare_with_kwargs(
        monkeypatch,
        assistant=assistant,
        kwargs={"config": {"configurable": {"model": "client-model"}}},
    )

    assert len(executor.submitted) == 1
    assert executor.submitted[0].kwargs["config"] == {
        "configurable": {"tender_text": "hello", "model": "client-model"}
    }


@pytest.mark.asyncio
async def test_prepare_run_merges_assistant_config_and_context(monkeypatch: pytest.MonkeyPatch) -> None:
    """Assistant config and context both flow into run kwargs."""
    assistant = _make_assistant(
        config_json={"configurable": {"tender_text": "hello"}},
        context_json={"tenant": "acme"},
    )
    executor = await _prepare_with_kwargs(monkeypatch, assistant=assistant)

    assert len(executor.submitted) == 1
    submitted = executor.submitted[0].kwargs
    assert submitted["config"] == {"configurable": {"tender_text": "hello"}}
    assert submitted["context"] == {"tenant": "acme"}


@pytest.mark.asyncio
async def test_prepare_run_without_assistant_config_keeps_client_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """No assistant config -> client config passes through unchanged."""
    assistant = _make_assistant(config_json=None)
    executor = await _prepare_with_kwargs(
        monkeypatch,
        assistant=assistant,
        kwargs={"config": {"configurable": {"model": "client-model"}}},
    )

    assert len(executor.submitted) == 1
    assert executor.submitted[0].kwargs["config"] == {"configurable": {"model": "client-model"}}
