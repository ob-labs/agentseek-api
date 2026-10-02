"""Transactional terminal state and a durable two-envelope Redis outbox.

Redis deliberately has three phases: pending SQL result, stream delivery, final
SQL state. An end frame may precede the final GET status during phase 3 retry.
The stored result is immutable within an execution generation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import asyncio
from datetime import UTC, datetime
import logging
from typing import Any

from sqlalchemy import select, update

from agentseek_api.core.database import db_manager
from agentseek_api.core.orm import Run, Thread, StreamCleanup
from agentseek_api.services.stream_persistence import (
    add_run_stream_event_to_session,
    add_thread_stream_event_to_session,
)
from agentseek_api.services.transaction_retry import retry_transaction
from agentseek_api.settings import settings

logger = logging.getLogger(__name__)
TERMINAL = {"success", "error", "interrupted"}


class _CancellationConflict(RuntimeError):
    """A concurrent claim invalidated cancellation's read of an active run."""


@dataclass(frozen=True)
class TerminalResult:
    status: str
    output: Any = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    thread_status: str | None = None


async def append_redis_envelope(**kwargs):
    # Lazy import avoids the protocol/persistence module import cycle.
    from agentseek_api.services.stream_persistence import append_redis_envelope as append
    return await append(**kwargs, retain=True)


def _envelopes(job, result: TerminalResult):
    from agentseek_api.services.thread_protocol import protocol_timestamp_ms
    lifecycle = {"success": "completed", "error": "failed", "interrupted": "interrupted"}[result.status]
    data = {"event": lifecycle, "graph_name": job.graph_id}
    if result.error is not None:
        data["error"] = result.error
    end = {"event": "end", "status": result.status}
    if result.error is not None:
        end["error"] = result.error
    if result.status == "interrupted" and isinstance(result.output, dict):
        end["interrupts"] = result.output.get("interrupts", [])
    return [
        {"scope": "run", "stream_id": job.run_id,
         "operation_id": f"terminal:{job.run_id}:{job.execution_id}:run",
         "payload": end},
        {"scope": "thread", "stream_id": job.thread_id,
         "operation_id": f"terminal:{job.run_id}:{job.execution_id}:thread",
         "payload": {"method": "lifecycle", "params": {
             "namespace": [], "timestamp": protocol_timestamp_ms(), "data": data}}},
    ]


async def _apply_result(session, run, result):
    run.status = result["status"]
    run.output_json = result["output"]
    run.last_error = result["error"]
    run.metadata_json = {**(run.metadata_json or {}), **result["metadata"]}
    run.execution_owner = None
    run.execution_lease_until = None
    run.dispatch_state = "finished"
    await session.execute(update(Thread).where(Thread.thread_id == run.thread_id).values(
        status=result.get("thread_status") or {"success": "idle", "error": "error", "interrupted": "interrupted"}[run.status],
        state_updated_at=datetime.now(UTC),
    ))


def _publish(envelopes, records):
    from agentseek_api.services import run_jobs
    for envelope, (seq, payload) in zip(envelopes, records, strict=True):
        try:
            if envelope["scope"] == "run":
                run_jobs.run_broker.publish(envelope["stream_id"], payload["event"], seq=seq,
                    **{k: v for k, v in payload.items() if k != "event"})
            else:
                run_jobs.thread_protocol_broker.publish(envelope["stream_id"], payload, seq=seq, persist=False)
        except Exception:
            # Durable replay, not re-appending, is the fallback for a failed notification.
            logger.exception("Terminal broker notification failed after durable commit")


async def finish_run(job, result: TerminalResult, *, cancel: bool = False) -> bool:
    if result.status not in TERMINAL:
        raise ValueError(f"Invalid terminal status: {result.status}")
    redis = settings.EXECUTOR_BACKEND.strip().lower() == "redis"
    async def stage(session):
        run = await session.scalar(select(Run).where(Run.run_id == job.run_id).with_for_update())
        if run is None or run.thread_id != job.thread_id or run.status in TERMINAL:
            return None
        if cancel:
            # Persisting a terminal intent is the completion linearization
            # point. Cancellation cannot rewrite an already delivered outcome.
            if run.status == "terminal_pending":
                return None
            job.execution_id, job.owner_id = run.execution_id, run.execution_owner
        elif run.execution_id != job.execution_id:
            return None
        if run.status == "terminal_pending":
            return (run.terminal_result["envelopes"], []) if redis else None
        if run.execution_owner != job.owner_id:
            return None
        # Acquire a write lock on SQLite too, and fence a concurrently reclaimed
        # owner. Retrying the entire transaction rereads all predicates.
        locked = await session.execute(update(Run).where(
            Run.run_id == job.run_id, Run.execution_id == job.execution_id,
            Run.execution_owner == job.owner_id, Run.status == run.status,
        ).values(status="terminal_pending" if redis else run.status))
        if locked.rowcount != 1:
            if cancel:
                raise _CancellationConflict("Cancellation raced with an execution claim")
            return None
        envelopes = _envelopes(job, result)
        stored = {**asdict(result), "envelopes": envelopes}
        if cancel:
            stored["output"] = run.output_json
        if redis:
            run.status = "terminal_pending"
            run.terminal_result = stored
            run.execution_lease_until = None
            session.add(StreamCleanup(operation_id=f"terminal:{job.run_id}:{job.execution_id}",
                run_id=job.run_id, execution_id=job.execution_id,
                envelopes=[{k: v for k, v in envelope.items() if k != "payload"} for envelope in envelopes]))
            return envelopes, []
        await _apply_result(session, run, stored)
        end = await add_run_stream_event_to_session(session, job.run_id, payload=envelopes[0]["payload"])
        lifecycle = await add_thread_stream_event_to_session(session, job.thread_id, payload=envelopes[1]["payload"])
        return envelopes, [end, lifecycle]

    for attempt in range(8):
        try:
            staged = await retry_transaction(stage)
            break
        except _CancellationConflict:
            # SQLite's SELECT FOR UPDATE does not lock the row. A worker
            # claim must not silently discard cancellation; reread the run
            # in a fresh transaction, still respecting a completed result.
            if attempt == 7:
                raise
            await asyncio.sleep(min(0.005 * 2 ** attempt, 0.2))
    if staged is None:
        return False
    if redis:
        await _deliver_pending(job.run_id, job.execution_id)
    else:
        _publish(*staged)
    return True


async def _deliver_pending(run_id: str, generation: str) -> bool:
    from agentseek_api.services.redis_delivery import reconcile_protocol_deliveries
    await reconcile_protocol_deliveries(run_id=run_id)
    # Hold the SQL row lock through the bounded external writes. This serializes
    # cancellation/deletion and multiple reconcilers with this terminal unit.
    async def deliver(session):
        run = await session.scalar(select(Run).where(Run.run_id == run_id).with_for_update())
        if run is None or run.execution_id != generation or run.status != "terminal_pending":
            return None
        locked = await session.execute(update(Run).where(
            Run.run_id == run_id, Run.execution_id == generation, Run.status == "terminal_pending",
        ).values(status="terminal_pending"))
        if locked.rowcount != 1:
            return None
        stored = run.terminal_result
        async with asyncio.timeout(5):
            records = [await append_redis_envelope(**envelope) for envelope in stored["envelopes"]]
        await _apply_result(session, run, stored)
        return stored["envelopes"], records
    delivered = await retry_transaction(deliver)
    if delivered is None:
        return False
    _publish(*delivered)
    await _cleanup_terminal_markers(run_id, generation)
    return True


async def _cleanup_terminal_markers(run_id: str, generation: str | None) -> None:
    from agentseek_api.services.stream_persistence import expire_redis_envelope, _get_redis_client, _operation_key
    try:
        async def cleanup(session):
            # Same lock order as terminal delivery. The cleanup row is not a
            # child FK: deletion/resume must not destroy cleanup ownership.
            run = await session.scalar(select(Run).where(Run.run_id == run_id).with_for_update())
            intent = await session.scalar(select(StreamCleanup).where(
                StreamCleanup.operation_id == f"terminal:{run_id}:{generation}").with_for_update())
            if intent is None:
                return
            await session.execute(update(StreamCleanup).where(
                StreamCleanup.operation_id == intent.operation_id).values(operation_id=intent.operation_id))
            same_generation = run is not None and run.execution_id == generation
            if same_generation and run.status not in TERMINAL:
                return  # Pending delivery still needs non-expiring markers.
            async with asyncio.timeout(5):
                for envelope in intent.envelopes:
                    if run is None:
                        await _get_redis_client().delete(_operation_key(**envelope))
                    else:
                        await expire_redis_envelope(**envelope)
            await session.delete(intent)
            if same_generation:
                run.terminal_result = None
        await retry_transaction(cleanup)
    except Exception:
        logger.exception("Terminal marker cleanup remains pending", extra={"run_id": run_id})


async def cleanup_terminal_markers_for_runs(run_ids: list[str] | None = None, *, limit: int = 100) -> None:
    async with db_manager.get_session_factory()() as session:
        query = select(StreamCleanup.run_id, StreamCleanup.execution_id).order_by(StreamCleanup.created_at)
        if run_ids is not None:
            query = query.where(StreamCleanup.run_id.in_(run_ids))
        else:
            query = query.limit(limit)
        pending = list((await session.execute(query)).all())
    for run_id, generation in pending:
        await _cleanup_terminal_markers(run_id, generation)

async def reconcile_terminal_deliveries(*, limit: int = 100) -> int:
    async with db_manager.get_session_factory()() as session:
        pending = list((await session.execute(select(Run.run_id, Run.execution_id, Run.status).where(
            Run.terminal_result.is_not(None),
        ).order_by(Run.updated_at).limit(limit))).all())
    completed = 0
    for run_id, generation, status in pending:
        try:
            if status == "terminal_pending":
                completed += await _deliver_pending(run_id, generation)
            else:
                await _cleanup_terminal_markers(run_id, generation)
        except Exception:
            logger.exception("Terminal delivery remains pending", extra={"run_id": run_id})
    await cleanup_terminal_markers_for_runs(limit=limit)
    return completed
