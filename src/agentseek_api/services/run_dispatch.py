"""Durable dispatch intent and generation-fenced execution claims."""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from contextvars import ContextVar
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from sqlalchemy import or_, select, update

from agentseek_api.core.orm import Run, Thread
from agentseek_api.services.transaction_retry import retry_transaction
from agentseek_api.settings import settings

LEASE_SECONDS = 30
TERMINAL = {"success", "error", "interrupted"}
logger = logging.getLogger(__name__)
current_execution: ContextVar[tuple[str, str, str] | None] = ContextVar("current_execution", default=None)


async def fence_execution_writes(session) -> None:
    """Fence graph publications in the same transaction that saves their data."""
    execution = current_execution.get()
    if execution is None or session.info.get("stream_execution_fence") == execution:
        return
    run_id, generation, owner = execution
    locked = await session.execute(update(Run).where(
        Run.run_id == run_id, Run.execution_id == generation,
        Run.execution_owner == owner, Run.status == "running",
    ).values(execution_owner=owner))
    if locked.rowcount != 1:
        raise RuntimeError("Execution no longer owns stream publication")
    session.info["stream_execution_fence"] = execution


def register_dispatch(run, job, *, new_generation: bool = False) -> None:
    if new_generation or not run.execution_id:
        run.execution_id = str(uuid4())
        run.execution_owner = None
        run.execution_lease_until = None
        run.terminal_result = None
        run.dispatch_state = "pending"
    job.execution_id = run.execution_id
    run.dispatch_payload = {**job.to_payload(), "backend": settings.EXECUTOR_BACKEND.strip().lower()}


async def ensure_dispatch(job) -> None:
    async def register(session):
        run = await session.scalar(select(Run).where(Run.run_id == job.run_id).with_for_update())
        if run is None:
            raise ValueError("Run not found")
        register_dispatch(run, job)
    await retry_transaction(register)


async def set_dispatch_state(job, state: str) -> None:
    async def change(session):
        await session.execute(update(Run).where(
            Run.run_id == job.run_id, Run.execution_id == job.execution_id,
            Run.status == "pending", Run.dispatch_state.in_(["pending", "submitted_unknown"]),
        ).values(dispatch_state=state))
    await retry_transaction(change)


async def claim_execution(job, *, owner_id: str, recovered: bool) -> str:
    """Only a worker holding the global Redis lease may pass recovered=True."""
    async def claim(session):
        run = await session.scalar(select(Run).where(Run.run_id == job.run_id).with_for_update())
        if run is None:
            return "deleted"
        # Pre-upgrade queue payloads belong only to the legacy generation.
        generation = job.execution_id or job.run_id
        if run.execution_id is not None and run.execution_id != generation:
            return "stale"
        if run.status in TERMINAL:
            return "terminal"
        if run.status == "terminal_pending" or run.terminal_result is not None:
            return "terminal_pending"
        now = datetime.now(UTC)
        expires = run.execution_lease_until
        if expires is not None and expires.tzinfo is None:
            expires = expires.replace(tzinfo=UTC)
        if run.execution_owner is not None:
            if run.execution_owner == owner_id:
                return "active"
            if not recovered and expires is not None and expires > now:
                return "active"
        result = await session.execute(update(Run).where(
            Run.run_id == run.run_id, Run.execution_id == run.execution_id,
            Run.execution_owner == run.execution_owner, Run.status == run.status,
        ).values(
            execution_id=generation, execution_owner=owner_id,
            execution_lease_until=now + timedelta(seconds=LEASE_SECONDS),
            dispatch_state="accepted", status="running", last_error=None,
        ))
        if result.rowcount != 1:
            return "active"
        await session.execute(update(Thread).where(Thread.thread_id == job.thread_id).values(status="busy", state_updated_at=now))
        job.execution_id = generation
        job.owner_id = owner_id
        return "claimed"
    return await retry_transaction(claim)


@asynccontextmanager
async def execution_lease(job):
    parent = asyncio.current_task()
    token = current_execution.set((job.run_id, job.execution_id, job.owner_id))

    async def heartbeat():
        while True:
            await asyncio.sleep(LEASE_SECONDS / 3)
            async def renew(session):
                result = await session.execute(update(Run).where(
                    Run.run_id == job.run_id, Run.execution_id == job.execution_id,
                    Run.execution_owner == job.owner_id, Run.status == "running",
                ).values(execution_lease_until=datetime.now(UTC) + timedelta(seconds=LEASE_SECONDS)))
                return result.rowcount == 1
            try:
                alive = await retry_transaction(renew)
            except Exception:
                parent.cancel()
                raise
            if not alive:
                parent.cancel()
                return

    task = asyncio.create_task(heartbeat(), name=f"run-lease:{job.run_id}")
    try:
        yield
    finally:
        current_execution.reset(token)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def reconcile_dispatches(*, limit: int = 100, queue=None) -> int:
    """Repair unsent/ambiguous intent and abandoned inline executions.

    Redis processing tokens are recovered by its single leased worker. Inline
    executions use their SQL lease; a scheduling race is resolved by the claim.
    Neither an intent nor a local broker counter proves execution is alive.
    """
    from agentseek_api.core.database import db_manager
    from agentseek_api.services.executor import RedisExecutor, get_executor
    from agentseek_api.services.run_jobs import RunExecutionJob
    from agentseek_api.services.thread_protocol import thread_protocol_broker

    backend = settings.EXECUTOR_BACKEND.strip().lower()
    async with db_manager.get_session_factory()() as session:
        query = select(Run.dispatch_payload).where(
            Run.dispatch_payload.is_not(None),
            or_(Run.status == "pending", (Run.status == "running") & or_(
                Run.execution_lease_until.is_(None), Run.execution_lease_until < datetime.now(UTC),
            )),
        ).order_by(Run.updated_at).limit(limit)
        payloads = list(await session.scalars(query))
    submitted = 0
    executor = RedisExecutor(queue=queue) if queue is not None else get_executor()
    for payload in payloads:
        if not payload or payload.get("backend", backend) != backend:
            continue
        job = RunExecutionJob.from_payload(payload)
        try:
            if backend == "redis":
                if await executor.queue.contains_run(run_id=job.run_id, execution_id=job.execution_id):
                    continue
            else:
                thread_protocol_broker.run_started(job.thread_id)
                job.owns_accounting = True
            try:
                await executor.submit(job)
            except BaseException:
                if job.owns_accounting:
                    thread_protocol_broker.run_finished(job.thread_id)
                    job.owns_accounting = False
                raise
            await set_dispatch_state(job, "accepted")
            submitted += 1
        except Exception:
            logger.exception("Dispatch remains recoverable", extra={"run_id": job.run_id})
    return submitted


@asynccontextmanager
async def run_recovery_service(*, queue=None):
    from agentseek_api.services.terminal_delivery import reconcile_terminal_deliveries
    from agentseek_api.services.redis_delivery import reconcile_protocol_deliveries

    # Pre-upgrade inline tasks cannot survive process restart. They did not
    # persist a resume command/generation, so do not guess and rerun side effects.
    legacy_pending = settings.EXECUTOR_BACKEND.strip().lower() == "inline"
    started_at = datetime.now(UTC)

    async def reconcile():
        nonlocal legacy_pending
        while True:
            try:
                if legacy_pending:
                    while await reconcile_legacy_runs(created_before=started_at):
                        pass
                    legacy_pending = False
                if settings.EXECUTOR_BACKEND.strip().lower() == "redis":
                    await reconcile_protocol_deliveries()
                await reconcile_terminal_deliveries()
                await reconcile_dispatches(queue=queue)
            except Exception:
                logger.exception("Run recovery scan failed; retrying")
            await asyncio.sleep(2)

    task = asyncio.create_task(reconcile(), name="run-recovery")
    try:
        yield
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def reconcile_legacy_runs(*, limit: int = 100, created_before: datetime | None = None) -> int:
    from agentseek_api.core.database import db_manager
    from agentseek_api.services.run_jobs import RunExecutionJob
    from agentseek_api.services.terminal_delivery import TerminalResult, finish_run
    if settings.EXECUTOR_BACKEND.strip().lower() != "inline":
        return 0  # Redis preserves/requeues legacy payloads under its worker lease.
    async with db_manager.get_session_factory()() as session:
        query = select(Run).where(
            Run.execution_id.is_(None), Run.status.in_(["pending", "running"]),
        )
        if created_before is not None:
            query = query.where(Run.created_at < created_before)
        rows = list(await session.scalars(query.limit(limit)))
    for run in rows:
        await finish_run(RunExecutionJob(run_id=run.run_id, thread_id=run.thread_id,
            user_id=run.user_id, graph_id="default", payload=run.input_json),
            TerminalResult(status="error", output=run.output_json,
                error="RunInterruptedByUpgrade: Legacy inline execution cannot be recovered safely after upgrade; resubmit the run"))
    return len(rows)
