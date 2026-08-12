from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from agentseek_api.core.database import db_manager
from agentseek_api.core.orm import Run, Thread
from agentseek_api.settings import settings
from agentseek_api.services.run_executor import RunExecutionResult, UNSET, execute_run
from agentseek_api.services.run_state import run_broker
from agentseek_api.services.stream_persistence import (
    add_thread_stream_event_to_session,
    add_run_stream_event_to_session,
    append_redis_run_stream_event,
    append_run_stream_event_atomic,
    append_thread_stream_event_atomic,
    next_run_stream_seq,  # noqa: F401 - module attribute; tests assert the redis path never calls it
    next_thread_stream_seq,  # noqa: F401 - module attribute; tests assert the redis path never calls it
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
    if settings.EXECUTOR_BACKEND.strip().lower() == "redis":
        event_payload = {"event": event, **payload}
        try:
            seq, _ = await append_redis_run_stream_event(run_id, event_payload)
        except Exception:
            logger.warning(
                "Failed to atomically append Redis run stream event",
                extra={"run_id": run_id, "event": event},
                exc_info=True,
            )
            seq = None
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


def _apply_execution_result(db_run: Run, result: RunExecutionResult) -> None:
    db_run.output_json = result.output
    db_run.last_error = None
    db_run.status = "interrupted" if result.interrupted else "success"


async def execute_run_job(job: RunExecutionJob) -> None:
    session_factory = db_manager.get_session_factory()
    try:
        async with session_factory() as execution_session:
            db_run = await execution_session.scalar(select(Run).where(Run.run_id == job.run_id))
            if db_run is None:
                await _publish_lifecycle(
                    job.thread_id,
                    event="failed",
                    graph_name=job.graph_id,
                    error="Run was deleted before execution started",
                )
                return
            if _is_cancelled_run(db_run):
                await _publish_lifecycle(
                    job.thread_id,
                    event="failed",
                    graph_name=job.graph_id,
                    error=db_run.last_error,
                )
                return
            if db_run.status in TERMINAL_RUN_STATUSES:
                return

            db_run.status = "running"
            db_run.last_error = None
            thread = await execution_session.scalar(select(Thread).where(Thread.thread_id == job.thread_id))
            if thread is not None:
                thread.status = "busy"
                thread.state_updated_at = db_run.updated_at
            await execution_session.commit()
            await _publish_run_event(job.run_id, "start")

            try:
                execute_kwargs = {
                    "thread_id": job.thread_id,
                    "run_id": job.run_id,
                    "payload": job.payload,
                    "user_id": job.user_id,
                    "graph_id": job.graph_id,
                    "resume": job.resume if job.is_resume else UNSET,
                }
                if job.kwargs:
                    execute_kwargs["kwargs"] = job.kwargs
                result = await execute_run(**execute_kwargs)
                await execution_session.refresh(db_run)
                if not _is_cancelled_run(db_run):
                    # A missing checkpoint lookup should not turn a successful run into a failed one.
                    try:
                        latest_checkpoint = await get_latest_checkpoint(job.thread_id)
                    except Exception:  # noqa: BLE001
                        latest_checkpoint = None
                    if latest_checkpoint is not None:
                        checkpoint_id = checkpoint_to_payload(latest_checkpoint)["checkpoint"]["checkpoint_id"]
                        db_run.metadata_json = {
                            **(db_run.metadata_json or {}),
                            RUN_CHECKPOINT_ID_METADATA_KEY: checkpoint_id,
                        }
                    _apply_execution_result(db_run, result)
            except Exception as exc:  # noqa: BLE001
                await execution_session.refresh(db_run)
                if not _is_cancelled_run(db_run):
                    db_run.status = "error"
                    db_run.last_error = f"{type(exc).__name__}: {exc}"

            thread = await execution_session.scalar(select(Thread).where(Thread.thread_id == job.thread_id))
            if thread is not None:
                thread.status = "interrupted" if db_run.status == "interrupted" else ("error" if db_run.status == "error" else "idle")
                thread.state_updated_at = db_run.updated_at
            is_redis_executor = settings.EXECUTOR_BACKEND.strip().lower() == "redis"
            terminal = await _publish_terminal_run_event(execution_session, job.run_id, status=db_run.status)
            lifecycle_state = "completed"
            if db_run.status == "interrupted":
                lifecycle_state = "interrupted"
            elif db_run.status == "error":
                lifecycle_state = "failed"
            lifecycle = await _publish_lifecycle(
                job.thread_id,
                event=lifecycle_state,
                graph_name=job.graph_id,
                error=db_run.last_error,
                session=execution_session,
            )
            await execution_session.commit()
            if not is_redis_executor and terminal is not None and lifecycle is not None:
                # The terminal records are durable with the run state now; only
                # then expose them to the in-memory brokers, so a client never
                # sees a seq that was not durably committed.
                run_broker.publish(job.run_id, "end", seq=terminal[0], status=db_run.status)
                thread_protocol_broker.publish(
                    job.thread_id,
                    lifecycle[1],
                    persist=False,
                    seq=lifecycle[0],
                )
    finally:
        thread_protocol_broker.run_finished(job.thread_id)
