from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from agentseek_api.core.orm import Run
from agentseek_api.settings import settings
from agentseek_api.services.run_executor import RunExecutionResult, UNSET, execute_run
from agentseek_api.services.run_state import run_broker
from agentseek_api.services.stream_persistence import (
    add_thread_stream_event_to_session,
    add_run_stream_event_to_session,
    append_redis_run_stream_event,
    append_run_stream_event_atomic,
    append_thread_stream_event_atomic,
    buffered_stream_persistence,
    buffer_durable_event,
    next_run_stream_seq,  # noqa: F401 - module attribute; tests assert the redis path never calls it
    next_thread_stream_seq,  # noqa: F401 - module attribute; tests assert the redis path never calls it
    persist_thread_stream_events,
)
from agentseek_api.services.thread_checkpoint_store import checkpoint_to_payload, get_latest_checkpoint
from agentseek_api.services.thread_protocol import (
    apublish_lifecycle_event,
    protocol_timestamp_ms,
    publish_lifecycle_event,  # noqa: F401 - run_preparation rebinds this module attribute
    thread_protocol_broker,
)

RUN_EXECUTION_JOB_KIND = "run.execute"
TERMINAL_RUN_STATUSES = {"success", "error", "interrupted"}
RUN_CHECKPOINT_ID_METADATA_KEY = "__agentseek_checkpoint_id"
logger = logging.getLogger(__name__)


@dataclass(slots=True)
class RunExecutionJob:
    run_id: str
    thread_id: str
    user_id: str
    payload: Any
    graph_id: str
    kwargs: dict[str, Any] = field(default_factory=dict)
    resume: Any | None = None
    is_resume: bool = False
    kind: str = RUN_EXECUTION_JOB_KIND
    execution_id: str | None = None
    owner_id: str | None = None
    recover_running: bool = False
    owns_accounting: bool = False

    def to_payload(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "run_id": self.run_id,
            "thread_id": self.thread_id,
            "user_id": self.user_id,
            "payload": self.payload,
            "kwargs": self.kwargs,
            "graph_id": self.graph_id,
            "resume": self.resume if self.is_resume else None,
            "is_resume": self.is_resume,
            "execution_id": self.execution_id,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> RunExecutionJob:
        kind = payload.get("kind", RUN_EXECUTION_JOB_KIND)
        if kind != RUN_EXECUTION_JOB_KIND:
            raise ValueError(f"Unsupported run job kind: {kind}")
        return cls(
            run_id=str(payload["run_id"]),
            thread_id=str(payload["thread_id"]),
            user_id=str(payload["user_id"]),
            payload=payload["payload"],
            kwargs=dict(payload.get("kwargs", {})),
            graph_id=str(payload["graph_id"]),
            resume=payload.get("resume"),
            is_resume=bool(payload.get("is_resume", False)),
            kind=kind,
            execution_id=payload.get("execution_id"),
        )


def _is_cancelled_run(run: Run) -> bool:
    return run.status == "error" and run.last_error == "Run cancelled"


async def _publish_lifecycle(
    thread_id: str,
    *,
    event: str,
    graph_name: str | None = None,
    error: str | None = None,
    session: AsyncSession | None = None,
) -> tuple[int, dict[str, Any]] | None:
    """Publish a thread lifecycle event.

    Durable-before-expose ordering: the event row (and its seq) is appended
    atomically before the in-memory broker makes it visible, so a client can
    never receive a seq that was not durably committed. When ``session`` is
    given (terminal lifecycle), the row is staged inside that transaction
    instead - the caller must commit and only then expose the event to the
    broker, keeping it atomic with the run status.
    """
    if settings.EXECUTOR_BACKEND.strip().lower() == "redis":
        await apublish_lifecycle_event(
            thread_id,
            event=event,
            graph_name=graph_name,
            error=error,
        )
        return None
    data: dict[str, Any] = {"event": event}
    if graph_name is not None:
        data["graph_name"] = graph_name
    if error is not None:
        data["error"] = error
    payload: dict[str, Any] = {
        "method": "lifecycle",
        "params": {
            "namespace": [],
            "timestamp": protocol_timestamp_ms(),
            "data": data,
        },
    }
    if session is None:
        seq, _ = await append_thread_stream_event_atomic(thread_id, payload)
        thread_protocol_broker.publish(thread_id, payload, persist=False, seq=seq)
        return seq, payload
    seq, _ = await add_thread_stream_event_to_session(session, thread_id, payload=payload)
    return seq, payload


async def _publish_run_event(
    run_id: str,
    event: str,
    **payload: Any,
) -> tuple[int, dict[str, Any]] | None:
    """Append a run lifecycle record durably, then expose it to the broker.

    The metadata-DB path allocates the seq and inserts the row in one atomic
    transaction and only then publishes to the in-memory broker, so the broker
    never exposes a seq that is not durable.
    """
    if settings.EXECUTOR_BACKEND.strip().lower() != "redis":
        if await buffer_durable_event(
            "run", run_id, {"event": event, **payload},
            lambda seq, saved: run_broker.publish(run_id, saved["event"], seq=seq, **{k: v for k, v in saved.items() if k != "event"}),
        ):
            return None
    if settings.EXECUTOR_BACKEND.strip().lower() == "redis":
        event_payload = {"event": event, **payload}
        from agentseek_api.services.run_dispatch import current_execution, fence_execution_writes
        execution = current_execution.get()
        if execution is None:
            seq, _ = await append_redis_run_stream_event(run_id, event_payload)
        else:
            from uuid import uuid4
            from agentseek_api.services.stream_persistence import append_redis_envelope
            from agentseek_api.services.transaction_retry import retry_transaction
            import asyncio
            operation_id = f"execution:{execution[1]}:{uuid4()}"
            async def append(session):
                # Hold the execution row's write lock across Redis delivery.
                # A deletion/cancellation either wins before this fence or
                # waits for this append and can then remove/terminate it.
                await fence_execution_writes(session)
                async with asyncio.timeout(5):
                    return await append_redis_envelope(scope="run", stream_id=run_id,
                        operation_id=operation_id, payload=event_payload)
            seq, _ = await retry_transaction(append)
        return run_broker.publish(run_id, event, seq=seq, **payload)
    seq, _ = await append_run_stream_event_atomic(run_id, {"event": event, **payload})
    return run_broker.publish(run_id, event, seq=seq, **payload)


async def _publish_terminal_run_event(session: AsyncSession, run_id: str, *, status: str) -> tuple[int, dict[str, Any]] | None:
    """Record the terminal ``end`` event.

    Redis: the atomic Lua append already makes the record durable before the
    broker publishes it, so this delegates to ``_publish_run_event`` unchanged.
    Inline: the row is staged inside ``session`` (allocating its seq from the
    locked counter row) without touching the broker; the caller commits and
    then exposes the event so the terminal status and its stream record are
    durable as one unit.
    """
    if settings.EXECUTOR_BACKEND.strip().lower() == "redis":
        return await _publish_run_event(run_id, "end", status=status)
    return await add_run_stream_event_to_session(
        session,
        run_id,
        payload={"event": "end", "status": status},
    )


async def _persist_thread_snapshot(thread_id: str) -> None:
    if settings.EXECUTOR_BACKEND.strip().lower() == "redis":
        return
    await persist_thread_stream_events(thread_id, thread_protocol_broker.snapshot_records(thread_id))


def _apply_execution_result(db_run: Run, result: RunExecutionResult) -> None:
    db_run.output_json = result.output
    db_run.last_error = None
    db_run.status = "interrupted" if result.interrupted else "success"


async def execute_run_job(job: RunExecutionJob) -> None:
    from uuid import uuid4
    from agentseek_api.services.run_dispatch import claim_execution, execution_lease
    from agentseek_api.services.terminal_delivery import TerminalResult, finish_run, _deliver_pending

    try:
        claim = await claim_execution(job, owner_id=job.owner_id or str(uuid4()), recovered=job.recover_running)
        if claim == "terminal_pending":
            await _deliver_pending(job.run_id, job.execution_id or job.run_id)
            return
        if claim != "claimed":
            return
        async with execution_lease(job):
            try:
                # Start append belongs to the same compensation boundary as the graph.
                await _publish_run_event(job.run_id, "start")
                execute_kwargs = {
                    "thread_id": job.thread_id, "run_id": job.run_id,
                    "payload": job.payload, "user_id": job.user_id,
                    "graph_id": job.graph_id,
                    "resume": job.resume if job.is_resume else UNSET,
                }
                if job.kwargs:
                    execute_kwargs["kwargs"] = job.kwargs
                async with buffered_stream_persistence(run_id=job.run_id, thread_id=job.thread_id):
                    result = await execute_run(**execute_kwargs)
                metadata = {}
                try:
                    latest_checkpoint = await get_latest_checkpoint(job.thread_id)
                    if latest_checkpoint is not None:
                        metadata[RUN_CHECKPOINT_ID_METADATA_KEY] = checkpoint_to_payload(latest_checkpoint)["checkpoint"]["checkpoint_id"]
                except Exception:
                    logger.debug("Checkpoint metadata unavailable", exc_info=True)
                terminal = TerminalResult(
                    status="interrupted" if result.interrupted else "success",
                    output=result.output, metadata=metadata,
                )
            except Exception as exc:
                terminal = TerminalResult(status="error", error=f"{type(exc).__name__}: {exc}")
            # Failure here deliberately leaves a recoverable lease/pending result.
            # Do not turn a completed graph into another execution on delivery retry.
        await finish_run(job, terminal)
    finally:
        if job.owns_accounting:
            job.owns_accounting = False
            thread_protocol_broker.run_finished(job.thread_id)
