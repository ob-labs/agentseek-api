from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select, update

from agentseek_api.core.database import db_manager
from agentseek_api.core.orm import Assistant, CronTick, Run, Thread
from agentseek_api.models.auth import User
from agentseek_api.services.executor import get_executor
from agentseek_api.services import run_jobs as run_jobs_module
from agentseek_api.services.run_jobs import RunExecutionJob

execute_run = run_jobs_module.execute_run
run_broker = run_jobs_module.run_broker
_publish_run_event = run_jobs_module._publish_run_event
_persist_thread_snapshot = run_jobs_module._persist_thread_snapshot
add_run_stream_event_to_session = run_jobs_module.add_run_stream_event_to_session
publish_lifecycle_event = run_jobs_module.publish_lifecycle_event
thread_protocol_broker = run_jobs_module.thread_protocol_broker
RunExecutionResult = run_jobs_module.RunExecutionResult
UNSET = run_jobs_module.UNSET
TERMINAL_RUN_STATUSES = {"success", "error", "interrupted"}
ACTIVE_THREAD_RUN_CONFLICT = "Another run is already active for this thread"


class ActiveThreadRunConflictError(RuntimeError):
    pass


async def _publish_lifecycle(
    thread_id: str,
    *,
    event: str,
    graph_name: str | None = None,
    error: str | None = None,
) -> None:
    run_jobs_module.publish_lifecycle_event = publish_lifecycle_event
    await run_jobs_module._publish_lifecycle(thread_id, event=event, graph_name=graph_name, error=error)


async def _persist_submission_failure(
    *,
    thread_id: str,
    run_id: str,
    error: str,
    run_status: str,
    thread_status: str,
) -> None:
    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        db_run = await session.scalar(select(Run).where(Run.run_id == run_id))
        if db_run is not None:
            db_run.status = run_status
            db_run.last_error = error

        thread = await session.scalar(select(Thread).where(Thread.thread_id == thread_id))
        if thread is not None:
            thread.status = thread_status
            thread.state_updated_at = datetime.now(UTC)

        await session.commit()


def _active_run_exists_query(*, thread_id: str, user_id: str, exclude_run_id: str | None = None):
    query = select(Run.run_id).where(
        Run.thread_id == thread_id,
        Run.user_id == user_id,
        Run.status.not_in(TERMINAL_RUN_STATUSES),
    )
    if exclude_run_id is not None:
        query = query.where(Run.run_id != exclude_run_id)
    return query.exists()


async def _claim_thread_for_run(*, session, thread_id: str, user_id: str, claimed_at: datetime) -> bool:
    result = await session.execute(
        update(Thread)
        .where(
            Thread.thread_id == thread_id,
            Thread.user_id == user_id,
            ~_active_run_exists_query(thread_id=thread_id, user_id=user_id),
        )
        .values(status="busy", state_updated_at=claimed_at)
    )
    return result.rowcount == 1


async def _claim_thread_for_resume(
    *,
    session,
    thread_id: str,
    run_id: str,
    user_id: str,
    claimed_at: datetime,
) -> bool:
    resumable_run_exists = select(Run.run_id).where(
        Run.run_id == run_id,
        Run.thread_id == thread_id,
        Run.user_id == user_id,
        Run.status == "interrupted",
    ).exists()
    result = await session.execute(
        update(Thread)
        .where(
            Thread.thread_id == thread_id,
            Thread.user_id == user_id,
            resumable_run_exists,
            ~_active_run_exists_query(thread_id=thread_id, user_id=user_id),
        )
        .values(status="busy", state_updated_at=claimed_at)
    )
    return result.rowcount == 1


async def _execute_and_persist(
    *,
    run_id: str,
    thread_id: str,
    user_id: str,
    payload: Any,
    graph_id: str,
    kwargs: dict[str, Any] | None = None,
    resume: Any | None = None,
    is_resume: bool = False,
) -> None:
    run_jobs_module.execute_run = execute_run
    run_jobs_module.run_broker = run_broker
    run_jobs_module._publish_run_event = _publish_run_event
    run_jobs_module._persist_thread_snapshot = _persist_thread_snapshot
    run_jobs_module.add_run_stream_event_to_session = add_run_stream_event_to_session
    run_jobs_module.publish_lifecycle_event = publish_lifecycle_event
    run_jobs_module.thread_protocol_broker = thread_protocol_broker
    await run_jobs_module.execute_run_job(
        RunExecutionJob(
            run_id=run_id,
            thread_id=thread_id,
            user_id=user_id,
            payload=payload,
            graph_id=graph_id,
            kwargs=kwargs or {},
            resume=resume,
            is_resume=is_resume,
        )
    )


async def _load_run(run_id: str) -> Run | None:
    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        return await session.scalar(select(Run).where(Run.run_id == run_id))


async def _load_run_with_graph(run_id: str) -> tuple[Run, str]:
    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        run = await session.scalar(select(Run).where(Run.run_id == run_id))
        if run is None:
            raise ValueError("Run not found")
        assistant = await session.scalar(select(Assistant).where(Assistant.assistant_id == run.assistant_id))
        if assistant is None:
            raise ValueError("Assistant not found")
        return run, assistant.graph_id


async def _prepare_run(
    *,
    thread_id: str,
    assistant_id: str,
    payload: Any,
    user: User,
    metadata: dict | None = None,
    kwargs: dict | None = None,
    multitask_strategy: str = "enqueue",
    tick_id: int | None = None,
) -> tuple[Run, str]:
    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        thread = await session.scalar(
            select(Thread).where(Thread.thread_id == thread_id, Thread.user_id == user.identity)
        )
        if thread is None:
            raise ValueError("Thread not found")
        assistant = await session.scalar(select(Assistant).where(Assistant.assistant_id == assistant_id))
        if assistant is None:
            raise ValueError("Assistant not found")
        graph_id = assistant.graph_id
        claimed_at = datetime.now(UTC)
        if not await _claim_thread_for_run(
            session=session,
            thread_id=thread_id,
            user_id=user.identity,
            claimed_at=claimed_at,
        ):
            raise ActiveThreadRunConflictError(ACTIVE_THREAD_RUN_CONFLICT)
        thread.status = "busy"
        thread.state_updated_at = claimed_at
        run = Run(
            thread_id=thread_id,
            assistant_id=assistant_id,
            user_id=user.identity,
            status="pending",
            input_json=payload,
            metadata_json=metadata or {},
            kwargs_json=kwargs or {},
            multitask_strategy=multitask_strategy,
        )
        session.add(run)
        await session.flush()
        if tick_id is not None:
            tick = await session.scalar(select(CronTick).where(CronTick.id == tick_id))
            if tick is None:
                raise RuntimeError(f"Cron tick {tick_id} not found")
            tick.run_id = run.run_id
            tick.thread_id = thread_id
            tick.status = "queued"
            tick.skip_reason = None
        await session.commit()
        await session.refresh(run)
    return run, graph_id


async def _submit_prepared_run(
    *,
    run_id: str,
    thread_id: str,
    user_id: str,
    payload: Any,
    graph_id: str,
    kwargs: dict[str, Any] | None = None,
    resume: Any = UNSET,
    is_resume: bool = False,
    failure_run_status: str,
    failure_thread_status: str,
) -> Run:
    thread_protocol_broker.run_started(thread_id)
    await _publish_lifecycle(thread_id, event="started", graph_name=graph_id)
    try:
        await get_executor().submit(
            RunExecutionJob(
                run_id=run_id,
                thread_id=thread_id,
                user_id=user_id,
                payload=payload,
                graph_id=graph_id,
                kwargs=kwargs or {},
                resume=resume,
                is_resume=is_resume,
            )
        )
    except Exception as exc:
        await _persist_submission_failure(
            thread_id=thread_id,
            run_id=run_id,
            error=str(exc),
            run_status=failure_run_status,
            thread_status=failure_thread_status,
        )
        await _publish_lifecycle(
            thread_id,
            event="failed",
            graph_name=graph_id,
            error=str(exc),
        )
        thread_protocol_broker.run_finished(thread_id)
        raise
    loaded = await _load_run(run_id)
    if loaded is None:
        raise ValueError("Run not found")
    return loaded


async def prepare_run(
    *,
    thread_id: str,
    assistant_id: str,
    payload: Any,
    user: User,
    metadata: dict | None = None,
    kwargs: dict | None = None,
    multitask_strategy: str = "enqueue",
    tick_id: int | None = None,
) -> tuple[Run, str]:
    return await _prepare_run(
        thread_id=thread_id,
        assistant_id=assistant_id,
        payload=payload,
        user=user,
        metadata=metadata,
        kwargs=kwargs,
        multitask_strategy=multitask_strategy,
        tick_id=tick_id,
    )


async def submit_existing_run(
    *,
    run_id: str,
    failure_run_status: str = "error",
    failure_thread_status: str = "error",
) -> Run:
    run, graph_id = await _load_run_with_graph(run_id)
    return await _submit_prepared_run(
        run_id=run.run_id,
        thread_id=run.thread_id,
        user_id=run.user_id,
        payload=run.input_json,
        graph_id=graph_id,
        kwargs=getattr(run, "kwargs_json", {}) or {},
        failure_run_status=failure_run_status,
        failure_thread_status=failure_thread_status,
    )


async def prepare_and_submit_run(
    *,
    thread_id: str,
    assistant_id: str,
    payload: Any,
    user: User,
    metadata: dict | None = None,
    kwargs: dict | None = None,
    multitask_strategy: str = "enqueue",
) -> Run:
    run, graph_id = await _prepare_run(
        thread_id=thread_id,
        assistant_id=assistant_id,
        payload=payload,
        user=user,
        metadata=metadata,
        kwargs=kwargs,
        multitask_strategy=multitask_strategy,
    )
    return await _submit_prepared_run(
        run_id=run.run_id,
        thread_id=run.thread_id,
        user_id=run.user_id,
        payload=run.input_json,
        graph_id=graph_id,
        kwargs=getattr(run, "kwargs_json", {}) or {},
        failure_run_status="error",
        failure_thread_status="error",
    )


async def resume_run(*, thread_id: str, run_id: str, resume: Any, user: User) -> Run:
    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        thread = await session.scalar(
            select(Thread).where(Thread.thread_id == thread_id, Thread.user_id == user.identity)
        )
        if thread is None:
            raise ValueError("Thread not found")
        run = await session.scalar(
            select(Run).where(Run.run_id == run_id, Run.thread_id == thread_id, Run.user_id == user.identity)
        )
        if run is None:
            raise ValueError("Run not found")
        assistant = await session.scalar(select(Assistant).where(Assistant.assistant_id == run.assistant_id))
        if assistant is None:
            raise ValueError("Assistant not found")
        claimed_at = datetime.now(UTC)
        if not await _claim_thread_for_resume(
            session=session,
            thread_id=thread_id,
            run_id=run_id,
            user_id=user.identity,
            claimed_at=claimed_at,
        ):
            if await session.scalar(
                select(Run.run_id).where(
                    Run.thread_id == thread_id,
                    Run.user_id == user.identity,
                    Run.status.not_in(TERMINAL_RUN_STATUSES),
                )
            ):
                raise ActiveThreadRunConflictError(ACTIVE_THREAD_RUN_CONFLICT)
            raise RuntimeError("Run is not interrupted")
        graph_id = assistant.graph_id
        payload = run.input_json
        run.status = "pending"
        run.last_error = None
        thread.status = "busy"
        thread.state_updated_at = claimed_at
        await session.commit()

    return await _submit_prepared_run(
        run_id=run_id,
        thread_id=thread_id,
        user_id=run.user_id,
        payload=payload,
        graph_id=graph_id,
        kwargs=getattr(run, "kwargs_json", {}) or {},
        resume=resume,
        is_resume=True,
        failure_run_status="interrupted",
        failure_thread_status="interrupted",
    )
