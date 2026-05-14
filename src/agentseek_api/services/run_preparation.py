from typing import Any

from sqlalchemy import select

from agentseek_api.core.database import db_manager
from agentseek_api.core.orm import Assistant, Run, Thread
from agentseek_api.models.auth import User
from agentseek_api.services.executor import get_executor
from agentseek_api.services.run_executor import RunExecutionResult, UNSET, execute_run
from agentseek_api.services.run_state import run_broker


async def _execute_and_persist(
    *,
    run_id: str,
    thread_id: str,
    payload: dict[str, Any],
    graph_id: str,
    resume: Any | None = None,
    is_resume: bool = False,
) -> None:
    session_factory = db_manager.get_session_factory()
    async with session_factory() as execution_session:
        db_run = await execution_session.scalar(select(Run).where(Run.run_id == run_id))
        if db_run is None:
            return

        db_run.status = "running"
        db_run.last_error = None
        await execution_session.commit()
        run_broker.publish(run_id, "start")

        try:
            result = await execute_run(
                thread_id=thread_id,
                run_id=run_id,
                payload=payload,
                graph_id=graph_id,
                resume=resume if is_resume else UNSET,
            )
            _apply_execution_result(db_run, result)
        except Exception as exc:  # noqa: BLE001
            db_run.status = "error"
            db_run.last_error = str(exc)

        await execution_session.commit()
        run_broker.publish(run_id, "end", status=db_run.status)


def _apply_execution_result(db_run: Run, result: RunExecutionResult) -> None:
    db_run.output_json = result.output
    db_run.last_error = None
    db_run.status = "interrupted" if result.interrupted else "success"


async def _load_run(run_id: str) -> Run | None:
    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        return await session.scalar(select(Run).where(Run.run_id == run_id))


async def prepare_and_submit_run(*, thread_id: str, assistant_id: str, payload: dict, user: User) -> Run:
    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        thread = await session.scalar(select(Thread).where(Thread.thread_id == thread_id, Thread.user_id == user.identity))
        if thread is None:
            raise ValueError("Thread not found")
        assistant = await session.scalar(select(Assistant).where(Assistant.assistant_id == assistant_id))
        if assistant is None:
            raise ValueError("Assistant not found")
        graph_id = assistant.graph_id
        run = Run(
            thread_id=thread_id,
            assistant_id=assistant_id,
            user_id=user.identity,
            status="pending",
            input_json=payload,
        )
        session.add(run)
        await session.commit()
        await session.refresh(run)

    await get_executor().submit(
        lambda: _execute_and_persist(
            run_id=run.run_id,
            thread_id=run.thread_id,
            payload=run.input_json,
            graph_id=graph_id,
        )
    )
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
        await session.commit()

    await get_executor().submit(
        lambda: _execute_and_persist(
            run_id=run_id,
            thread_id=thread_id,
            payload=payload,
            graph_id=graph_id,
            resume=resume,
            is_resume=True,
        )
    )
    return await _load_run(run_id) or run
