from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select

from agentseek_api.core.database import db_manager
from agentseek_api.core.orm import Assistant, Run, Thread
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


async def _execute_and_persist(
    *,
    run_id: str,
    thread_id: str,
    user_id: str,
    payload: dict[str, Any],
    graph_id: str,
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
            resume=resume,
            is_resume=is_resume,
        )
    )


async def _load_run(run_id: str) -> Run | None:
    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        return await session.scalar(select(Run).where(Run.run_id == run_id))


async def prepare_and_submit_run(
    *,
    thread_id: str,
    assistant_id: str,
    payload: dict,
    user: User,
    metadata: dict | None = None,
    kwargs: dict | None = None,
    multitask_strategy: str = "enqueue",
) -> Run:
    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        thread = await session.scalar(select(Thread).where(Thread.thread_id == thread_id, Thread.user_id == user.identity))
        if thread is None:
            raise ValueError("Thread not found")
        assistant = await session.scalar(select(Assistant).where(Assistant.assistant_id == assistant_id))
        if assistant is None:
            raise ValueError("Assistant not found")
        graph_id = assistant.graph_id
        thread.status = "busy"
        thread.state_updated_at = datetime.now(UTC)
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
        await session.commit()
        await session.refresh(run)

    thread_protocol_broker.run_started(thread_id)
    await _publish_lifecycle(thread_id, event="started", graph_name=graph_id)
    try:
        await get_executor().submit(
            RunExecutionJob(
                run_id=run.run_id,
                thread_id=run.thread_id,
                user_id=run.user_id,
                payload=run.input_json,
                graph_id=graph_id,
            )
        )
    except Exception as exc:
        await _persist_submission_failure(
            thread_id=thread_id,
            run_id=run.run_id,
            error=str(exc),
            run_status="error",
            thread_status="error",
        )
        await _publish_lifecycle(
            thread_id,
            event="failed",
            graph_name=graph_id,
            error=str(exc),
        )
        thread_protocol_broker.run_finished(thread_id)
        raise
    return await _load_run(run.run_id) or run


async def resume_run(*, thread_id: str, run_id: str, resume: Any, user: User) -> Run:
    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        run = await session.scalar(
            select(Run).where(Run.run_id == run_id, Run.thread_id == thread_id, Run.user_id == user.identity)
        )
        if run is None:
            raise ValueError("Run not found")
        if run.status != "interrupted":
            raise RuntimeError("Run is not interrupted")
        assistant = await session.scalar(select(Assistant).where(Assistant.assistant_id == run.assistant_id))
        if assistant is None:
            raise ValueError("Assistant not found")
        graph_id = assistant.graph_id
        payload = run.input_json
        run.status = "pending"
        run.last_error = None
        thread = await session.scalar(select(Thread).where(Thread.thread_id == thread_id, Thread.user_id == user.identity))
        if thread is not None:
            thread.status = "busy"
            thread.state_updated_at = datetime.now(UTC)
        await session.commit()

    thread_protocol_broker.run_started(thread_id)
    await _publish_lifecycle(thread_id, event="started", graph_name=graph_id)
    try:
        await get_executor().submit(
            RunExecutionJob(
                run_id=run_id,
                thread_id=thread_id,
                user_id=run.user_id,
                payload=payload,
                graph_id=graph_id,
                resume=resume,
                is_resume=True,
            )
        )
    except Exception as exc:
        await _persist_submission_failure(
            thread_id=thread_id,
            run_id=run_id,
            error=str(exc),
            run_status="interrupted",
            thread_status="interrupted",
        )
        await _publish_lifecycle(
            thread_id,
            event="failed",
            graph_name=graph_id,
            error=str(exc),
        )
        thread_protocol_broker.run_finished(thread_id)
        raise
    return await _load_run(run_id) or run
