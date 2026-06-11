from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from agentseek_api.core.database import db_manager
from agentseek_api.core.orm import Run, Thread
from agentseek_api.services.run_executor import RunExecutionResult, UNSET, execute_run
from agentseek_api.services.run_state import run_broker
from agentseek_api.services.stream_persistence import (
    add_thread_stream_event_to_session,
    add_run_stream_event_to_session,
    next_run_stream_seq,
    next_thread_stream_seq,
    persist_run_stream_event,
    persist_thread_stream_event,
)
from agentseek_api.services.thread_checkpoint_store import checkpoint_to_payload, get_latest_checkpoint
from agentseek_api.services.thread_protocol import publish_lifecycle_event, thread_protocol_broker

RUN_EXECUTION_JOB_KIND = "run.execute"
TERMINAL_RUN_STATUSES = {"success", "error", "interrupted"}
RUN_CHECKPOINT_ID_METADATA_KEY = "__agentseek_checkpoint_id"


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
) -> None:
    kwargs: dict[str, Any] = {"event": event}
    if graph_name is not None:
        kwargs["graph_name"] = graph_name
    if error is not None:
        kwargs["error"] = error
    seq = await next_thread_stream_seq(thread_id)
    published = publish_lifecycle_event(thread_id, persist=False, seq=seq, **kwargs)
    if session is None:
        await persist_thread_stream_event(thread_id, published)
        return
    await add_thread_stream_event_to_session(
        session,
        thread_id,
        seq=int(published["seq"]),
        payload=published,
    )


async def _publish_run_event(
    run_id: str,
    event: str,
    *,
    persist: bool = True,
    **payload: Any,
) -> tuple[int, dict[str, Any]] | None:
    seq = await next_run_stream_seq(run_id)
    published = run_broker.publish(run_id, event, seq=seq, **payload)
    if published is None:
        return None
    seq, event_payload = published
    if persist:
        await persist_run_stream_event(run_id, seq=seq, payload=event_payload)
    return seq, event_payload


async def _persist_thread_snapshot(thread_id: str) -> None:
    for event in thread_protocol_broker.snapshot_records(thread_id):
        await persist_thread_stream_event(thread_id, event)


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
                await _persist_thread_snapshot(job.thread_id)
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
            terminal_run_event = await _publish_run_event(job.run_id, "end", status=db_run.status, persist=False)
            if terminal_run_event is not None:
                seq, event_payload = terminal_run_event
                await add_run_stream_event_to_session(execution_session, job.run_id, seq=seq, payload=event_payload)
            lifecycle_state = "completed"
            if db_run.status == "interrupted":
                lifecycle_state = "interrupted"
            elif db_run.status == "error":
                lifecycle_state = "failed"
            await _publish_lifecycle(
                job.thread_id,
                event=lifecycle_state,
                graph_name=job.graph_id,
                error=db_run.last_error,
                session=execution_session,
            )
            await execution_session.commit()
    finally:
        thread_protocol_broker.run_finished(job.thread_id)
