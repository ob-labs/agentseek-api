import asyncio
from datetime import UTC, datetime, timedelta

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from agentseek_api.core.database import db_manager
from agentseek_api.core.orm import Base, Run, Thread
from agentseek_api.services import run_jobs, run_preparation, run_state, thread_protocol
from agentseek_api.settings import settings


@pytest_asyncio.fixture
async def recovery_db(tmp_path, monkeypatch):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/recovery.db")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    monkeypatch.setattr(db_manager, "get_engine", lambda: engine)
    monkeypatch.setattr(db_manager, "get_session_factory", lambda: factory)
    monkeypatch.setattr(settings, "EXECUTOR_BACKEND", "inline")
    broker = thread_protocol.ThreadProtocolEventBroker()
    run_broker = run_state.RunEventBroker()
    for module in (run_jobs, run_preparation, thread_protocol):
        monkeypatch.setattr(module, "thread_protocol_broker", broker)
    for module in (run_jobs, run_preparation, run_state):
        monkeypatch.setattr(module, "run_broker", run_broker)
    async with factory() as session:
        session.add(Thread(thread_id="t", user_id="u", status="busy"))
        session.add(Run(run_id="r", thread_id="t", assistant_id="a", user_id="u", status="pending"))
        await session.commit()
    job = run_jobs.RunExecutionJob(run_id="r", thread_id="t", user_id="u", graph_id="default", payload={})
    yield factory, job
    await engine.dispose()


async def test_initial_lifecycle_failure_releases_accounting(recovery_db, monkeypatch):
    factory, job = recovery_db

    async def unavailable(*_args, **_kwargs):
        raise RuntimeError("start append unavailable")

    monkeypatch.setattr(run_preparation, "_publish_lifecycle", unavailable)
    with pytest.raises(RuntimeError, match="start append unavailable"):
        await run_preparation._submit_prepared_run(
            run_id=job.run_id, thread_id=job.thread_id, user_id=job.user_id,
            payload={}, graph_id="default", failure_run_status="error", failure_thread_status="idle",
        )
    assert thread_protocol.thread_protocol_broker._active_runs.get("t", 0) == 0
    async with factory() as session:
        assert (await session.get(Run, "r")).status == "error"


async def test_expired_running_claim_is_recovered_but_active_duplicate_is_not(recovery_db):
    from agentseek_api.services.run_dispatch import claim_execution

    factory, job = recovery_db
    assert await claim_execution(job, owner_id="old", recovered=False) == "claimed"
    assert await claim_execution(job, owner_id="old", recovered=False) == "active"
    assert await claim_execution(job, owner_id="new", recovered=False) == "active"
    async with factory() as session:
        row = await session.get(Run, "r")
        row.execution_lease_until = datetime.now(UTC) - timedelta(seconds=1)
        await session.commit()
    assert await claim_execution(job, owner_id="new", recovered=False) == "claimed"
    async with factory() as session:
        row = await session.get(Run, "r")
        assert row.status == "running" and row.execution_owner == "new"


async def test_old_payload_cannot_claim_resumed_generation(recovery_db):
    from agentseek_api.services.run_dispatch import claim_execution

    factory, job = recovery_db
    assert await claim_execution(job, owner_id="old", recovered=False) == "claimed"
    async with factory() as session:
        row = await session.get(Run, "r")
        row.status = "pending"
        row.execution_id = "resumed-generation"
        row.execution_owner = None
        await session.commit()
    assert await claim_execution(job, owner_id="new", recovered=True) == "stale"


async def test_successful_handoff_keeps_accounting_until_task_finishes(recovery_db, monkeypatch):
    from agentseek_api.services.executor import InlineExecutor

    factory, job = recovery_db
    release = asyncio.Event()

    async def execute(**_kwargs):
        await release.wait()
        return run_jobs.RunExecutionResult(output={"ok": True}, interrupted=False, interrupts=[])

    monkeypatch.setattr(run_jobs, "execute_run", execute)
    monkeypatch.setattr(run_preparation, "get_executor", lambda: InlineExecutor())
    try:
        await run_preparation._submit_prepared_run(
            run_id=job.run_id, thread_id=job.thread_id, user_id=job.user_id,
            payload={}, graph_id="default", failure_run_status="error", failure_thread_status="idle",
        )
        assert thread_protocol.thread_protocol_broker._active_runs["t"] == 1
    finally:
        release.set()
    async with asyncio.timeout(3):
        while thread_protocol.thread_protocol_broker._active_runs["t"]:
            await asyncio.sleep(0.01)
    async with factory() as session:
        assert (await session.get(Run, "r")).status == "success"


async def test_terminal_transaction_rolls_back_both_logs_and_state(recovery_db, monkeypatch):
    from agentseek_api.core.orm import RunStreamEvent, ThreadStreamEvent
    from agentseek_api.services import terminal_delivery
    from agentseek_api.services.run_dispatch import claim_execution
    from sqlalchemy import select

    factory, job = recovery_db
    await claim_execution(job, owner_id="owner", recovered=False)
    original = terminal_delivery.add_thread_stream_event_to_session
    async def fail(*args, **kwargs):
        await original(*args, **kwargs)
        raise RuntimeError("terminal insert failed")
    monkeypatch.setattr(terminal_delivery, "add_thread_stream_event_to_session", fail)
    with pytest.raises(RuntimeError, match="terminal insert failed"):
        await terminal_delivery.finish_run(job, terminal_delivery.TerminalResult(status="success", output={"answer": 1}))
    async with factory() as session:
        assert (await session.get(Run, "r")).status == "running"
        assert list(await session.scalars(select(RunStreamEvent))) == []
        assert list(await session.scalars(select(ThreadStreamEvent))) == []
    assert run_state.run_broker.snapshot_records("r") == []
    monkeypatch.setattr(terminal_delivery, "add_thread_stream_event_to_session", original)
    await terminal_delivery.finish_run(job, terminal_delivery.TerminalResult(status="success", output={"answer": 1}))
    await terminal_delivery.finish_run(job, terminal_delivery.TerminalResult(status="success", output={"answer": 2}))
    async with factory() as session:
        assert (await session.get(Run, "r")).output_json == {"answer": 1}
        assert len(list(await session.scalars(select(RunStreamEvent)))) == 1
        assert len(list(await session.scalars(select(ThreadStreamEvent)))) == 1


async def test_stale_owner_cannot_finalize_or_overwrite_cancellation(recovery_db):
    from agentseek_api.services import terminal_delivery
    from agentseek_api.services.run_dispatch import claim_execution

    factory, job = recovery_db
    await claim_execution(job, owner_id="old", recovered=False)
    async with factory() as session:
        row = await session.get(Run, "r")
        row.execution_owner = "new"
        await session.commit()
    await terminal_delivery.finish_run(job, terminal_delivery.TerminalResult(status="success"))
    async with factory() as session:
        row = await session.get(Run, "r")
        assert row.status == "running"
        row.status = "error"
        row.last_error = "Run cancelled"
        await session.commit()
    job.owner_id = "new"
    await terminal_delivery.finish_run(job, terminal_delivery.TerminalResult(status="success"))
    async with factory() as session:
        row = await session.get(Run, "r")
        assert row.status == "error" and row.last_error == "Run cancelled"


async def test_redis_terminal_retry_recovers_stored_result_without_graph(recovery_db, monkeypatch):
    from agentseek_api.services import terminal_delivery
    from agentseek_api.services.run_dispatch import claim_execution

    factory, job = recovery_db
    monkeypatch.setattr(settings, "EXECUTOR_BACKEND", "redis")
    await claim_execution(job, owner_id="owner", recovered=False)
    delivered = {}
    broken = True
    async def append(*, scope, stream_id, payload, operation_id):
        if broken and scope == "thread":
            raise RuntimeError("second log unavailable")
        return delivered.setdefault(operation_id, (1, payload))
    monkeypatch.setattr(terminal_delivery, "append_redis_envelope", append)
    with pytest.raises(RuntimeError, match="second log unavailable"):
        await terminal_delivery.finish_run(job, terminal_delivery.TerminalResult(status="success", output={"answer": []}))
    async with factory() as session:
        row = await session.get(Run, "r")
        assert row.status == "terminal_pending"
        assert row.terminal_result["output"] == {"answer": []}
    assert len(delivered) == 1
    broken = False
    assert await terminal_delivery.reconcile_terminal_deliveries() == 1
    assert len(delivered) == 2
    assert await terminal_delivery.reconcile_terminal_deliveries() == 0
    async with factory() as session:
        row = await session.get(Run, "r")
        assert row.status == "success" and row.output_json == {"answer": []}


async def test_duplicate_execution_does_not_run_graph_twice(recovery_db, monkeypatch):
    factory, job = recovery_db
    release = asyncio.Event()
    entered = asyncio.Event()
    calls = []
    async def execute(**kwargs):
        calls.append(kwargs)
        entered.set()
        await release.wait()
        return run_jobs.RunExecutionResult(output={}, interrupted=False, interrupts=[])
    monkeypatch.setattr(run_jobs, "execute_run", execute)
    first = asyncio.create_task(run_jobs.execute_run_job(job))
    try:
        await asyncio.wait_for(entered.wait(), 2)
        duplicate = run_jobs.RunExecutionJob.from_payload(job.to_payload())
        await asyncio.wait_for(run_jobs.execute_run_job(duplicate), 1)
        assert len(calls) == 1
    finally:
        release.set()
        await first


async def test_queue_accept_then_timeout_retains_unknown_dispatch(recovery_db, monkeypatch):
    factory, job = recovery_db
    monkeypatch.setattr(settings, "EXECUTOR_BACKEND", "redis")
    accepted = []
    class AmbiguousExecutor:
        async def submit(self, queued):
            accepted.append(queued)
            raise TimeoutError("ack lost")
    async def lifecycle(*_args, **_kwargs):
        pass
    monkeypatch.setattr(run_preparation, "get_executor", lambda: AmbiguousExecutor())
    monkeypatch.setattr(run_preparation, "_publish_lifecycle", lifecycle)
    with pytest.raises(TimeoutError, match="ack lost"):
        await run_preparation._submit_prepared_run(
            run_id="r", thread_id="t", user_id="u", payload={}, graph_id="default",
            failure_run_status="error", failure_thread_status="idle",
        )
    async with factory() as session:
        row = await session.get(Run, "r")
        assert row.status == "pending" and row.dispatch_state == "submitted_unknown"
        assert row.dispatch_payload["execution_id"] == accepted[0].execution_id


async def test_start_and_compensation_failure_remain_recoverable(recovery_db, monkeypatch):
    from agentseek_api.services import run_dispatch
    factory, job = recovery_db
    async def fail(*_args, **_kwargs):
        raise RuntimeError("storage unavailable")
    monkeypatch.setattr(run_preparation, "_publish_lifecycle", fail)
    monkeypatch.setattr(run_preparation, "_persist_submission_failure", fail)
    with pytest.raises(RuntimeError, match="storage unavailable"):
        await run_preparation._submit_prepared_run(
            run_id="r", thread_id="t", user_id="u", payload={}, graph_id="default",
            failure_run_status="error", failure_thread_status="idle",
        )
    assert thread_protocol.thread_protocol_broker._active_runs.get("t", 0) == 0
    recovered = []
    class Executor:
        async def submit(self, job):
            recovered.append(job)
    monkeypatch.setattr("agentseek_api.services.executor.get_executor", lambda: Executor())
    assert await run_dispatch.reconcile_dispatches() == 1
    assert recovered[0].execution_id is not None
    async with factory() as session:
        assert (await session.get(Run, "r")).status == "pending"


@pytest.mark.parametrize("phase", [1, 3])
async def test_terminal_commit_failure_is_restart_recoverable(recovery_db, monkeypatch, phase):
    from sqlalchemy import event
    from agentseek_api.services import terminal_delivery
    from agentseek_api.services.run_dispatch import claim_execution
    factory, job = recovery_db
    await claim_execution(job, owner_id="owner", recovered=False)
    monkeypatch.setattr(settings, "EXECUTOR_BACKEND", "redis")
    committed = {}
    async def append(**envelope):
        return committed.setdefault(envelope["operation_id"], (1, envelope["payload"]))
    monkeypatch.setattr(terminal_delivery, "append_redis_envelope", append)
    def fail_commit(session):
        target = "terminal_pending" if phase == 1 else "success"
        session.flush()
        state = session.connection().exec_driver_sql("SELECT status FROM runs WHERE run_id = 'r'").scalar()
        if state == target:
            raise RuntimeError("commit unavailable")
    event.listen(factory.class_.sync_session_class, "before_commit", fail_commit)
    try:
        with pytest.raises(RuntimeError, match="commit unavailable"):
            await terminal_delivery.finish_run(job, terminal_delivery.TerminalResult(status="success", output={"result": 42}))
    finally:
        event.remove(factory.class_.sync_session_class, "before_commit", fail_commit)
    async with factory() as session:
        assert (await session.get(Run, "r")).status == ("running" if phase == 1 else "terminal_pending")
    if phase == 1:
        assert committed == {}
        await terminal_delivery.finish_run(job, terminal_delivery.TerminalResult(status="success", output={"result": 42}))
    else:
        assert len(committed) == 2
        assert await terminal_delivery.reconcile_terminal_deliveries() == 1
    assert len(committed) == 2
    async with factory() as session:
        row = await session.get(Run, "r")
        assert row.status == "success" and row.output_json == {"result": 42}


async def test_broker_failure_does_not_undo_terminal_commit(recovery_db, monkeypatch):
    from sqlalchemy import select
    from agentseek_api.core.orm import RunStreamEvent, ThreadStreamEvent
    from agentseek_api.services.terminal_delivery import TerminalResult, finish_run
    from agentseek_api.services.run_dispatch import claim_execution
    factory, job = recovery_db
    await claim_execution(job, owner_id="owner", recovered=False)
    def fail(*args, **kwargs):
        raise RuntimeError("broker unavailable")
    monkeypatch.setattr(run_jobs.run_broker, "publish", fail)
    monkeypatch.setattr(run_jobs.thread_protocol_broker, "publish", fail)
    await finish_run(job, TerminalResult(status="success"))
    await finish_run(job, TerminalResult(status="success"))
    async with factory() as session:
        assert (await session.get(Run, "r")).status == "success"
        assert len(list(await session.scalars(select(RunStreamEvent)))) == 1
        assert len(list(await session.scalars(select(ThreadStreamEvent)))) == 1


async def test_populated_legacy_runs_upgrade_is_idempotent(tmp_path, monkeypatch):
    from agentseek_api.core.database import DatabaseManager
    from sqlalchemy import text
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/legacy.db")
    async with engine.begin() as connection:
        await connection.execute(text("""CREATE TABLE runs (
            run_id VARCHAR(36) PRIMARY KEY, thread_id VARCHAR(36) NOT NULL,
            assistant_id VARCHAR(36) NOT NULL, user_id VARCHAR(255) NOT NULL,
            status VARCHAR(32) NOT NULL, input JSON NOT NULL, output JSON,
            metadata JSON NOT NULL, kwargs JSON NOT NULL, multitask_strategy VARCHAR(32) NOT NULL,
            last_error TEXT, created_at DATETIME NOT NULL, updated_at DATETIME NOT NULL)"""))
        await connection.execute(text("""INSERT INTO runs VALUES
            ('legacy', 'thread', 'assistant', 'user', 'running', '{}', NULL, '{}', '{}', 'enqueue',
             NULL, '2026-01-01 00:00:00', '2026-01-01 00:00:00')"""))
        await connection.run_sync(Base.metadata.create_all)
        await connection.run_sync(DatabaseManager._apply_additive_migrations)
        await connection.run_sync(DatabaseManager._apply_additive_migrations)
    async with async_sessionmaker(engine)() as session:
        row = await session.get(Run, "legacy")
        assert row.status == "running" and row.input_json == {}
        assert row.dispatch_state == "pending" and row.execution_id is None
        assert row.terminal_result is None and row.execution_owner is None
    from agentseek_api.services.run_dispatch import reconcile_legacy_runs
    factory = async_sessionmaker(engine, expire_on_commit=False)
    monkeypatch.setattr(db_manager, "get_session_factory", lambda: factory)
    monkeypatch.setattr(settings, "EXECUTOR_BACKEND", "inline")
    async with factory() as session:
        session.add(Thread(thread_id="thread", user_id="user", status="busy"))
        await session.commit()
    await reconcile_legacy_runs()
    async with factory() as session:
        row = await session.get(Run, "legacy")
        assert row.status == "error"
        assert "upgrade" in row.last_error.lower()
        assert (await session.get(Thread, "thread")).status == "error"
    assert await reconcile_legacy_runs() == 0
    await engine.dispose()
async def test_reclaimed_execution_cannot_publish_more_protocol_frames(recovery_db, monkeypatch):
    from agentseek_api.core.orm import RunStreamEvent, ThreadStreamEvent
    from sqlalchemy import select
    factory, job = recovery_db

    async def execute(**_kwargs):
        async with factory() as session:
            row = await session.get(Run, "r")
            row.execution_owner = "replacement-owner"
            await session.commit()
        await thread_protocol.apublish_values_event("t", values={"stale": True}, run_id="r")
        return run_jobs.RunExecutionResult(output={}, interrupted=False, interrupts=[])
    monkeypatch.setattr(run_jobs, "execute_run", execute)
    await run_jobs.execute_run_job(job)
    async with factory() as session:
        assert not list(await session.scalars(select(RunStreamEvent).where(RunStreamEvent.event == "values")))
        assert not list(await session.scalars(select(ThreadStreamEvent).where(ThreadStreamEvent.method == "values")))


@pytest.mark.parametrize("failure", ["reclaimed", "database"])
async def test_lease_loss_cancels_execution_and_stops_heartbeat(recovery_db, monkeypatch, failure):
    from agentseek_api.services import run_dispatch
    factory, job = recovery_db
    await run_dispatch.claim_execution(job, owner_id="owner", recovered=False)
    monkeypatch.setattr(run_dispatch, "LEASE_SECONDS", 0.015)
    renewed = asyncio.Event()
    original = run_dispatch.retry_transaction

    async def observe(operation):
        if failure == "database":
            raise RuntimeError("database unavailable")
        result = await original(operation)
        renewed.set()
        return result
    monkeypatch.setattr(run_dispatch, "retry_transaction", observe)

    async def execute():
        async with run_dispatch.execution_lease(job):
            if failure == "reclaimed":
                await renewed.wait()
                async with factory() as session:
                    row = await session.get(Run, "r")
                    assert row.execution_lease_until is not None
                    row.execution_owner = "new"
                    await session.commit()
            await asyncio.Event().wait()
    task = asyncio.create_task(execute())
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 2)
    assert not [item for item in asyncio.all_tasks() if item.get_name() == "run-lease:r"]
@pytest.mark.parametrize("race", ["delete", "cancel"])
async def test_redis_start_is_fenced_after_claim(recovery_db, monkeypatch, race):
    from agentseek_api.services import run_dispatch, stream_persistence
    factory, job = recovery_db
    monkeypatch.setattr(settings, "EXECUTOR_BACKEND", "redis")
    original_claim = run_dispatch.claim_execution
    published, executed = [], []

    async def raced_claim(*args, **kwargs):
        result = await original_claim(*args, **kwargs)
        async with factory() as session:
            row = await session.get(Run, "r")
            if race == "delete":
                await session.delete(row)
            else:
                row.status, row.last_error = "error", "Run cancelled"
            await session.commit()
        return result

    async def append(*args, **kwargs):
        published.append((args, kwargs))
        return 1, {"event": "start"}

    async def execute(**kwargs):
        executed.append(kwargs)
        return run_jobs.RunExecutionResult(output={}, interrupted=False, interrupts=[])
    monkeypatch.setattr(run_dispatch, "claim_execution", raced_claim)
    monkeypatch.setattr(run_jobs, "append_redis_run_stream_event", append)
    monkeypatch.setattr(stream_persistence, "append_redis_envelope", append)
    monkeypatch.setattr(run_jobs, "execute_run", execute)
    await run_jobs.execute_run_job(job)
    assert published == []
    assert executed == []
async def test_legacy_reconciliation_does_not_terminate_new_generation(recovery_db, monkeypatch):
    from agentseek_api.services import terminal_delivery
    from agentseek_api.services.run_dispatch import reconcile_legacy_runs
    factory, _job = recovery_db
    original = terminal_delivery.finish_run
    async def concurrently_claimed(job, result, **kwargs):
        async with factory() as session:
            row = await session.get(Run, "r")
            row.execution_id, row.execution_owner = "new-generation", "new-owner"
            await session.commit()
        return await original(job, result, **kwargs)
    monkeypatch.setattr(terminal_delivery, "finish_run", concurrently_claimed)
    await reconcile_legacy_runs()
    async with factory() as session:
        assert (await session.get(Run, "r")).status == "pending"
