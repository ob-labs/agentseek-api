from sqlalchemy import select

from agentseek_api.core.database import db_manager
from agentseek_api.core.orm import Assistant, Run, Thread
from agentseek_api.models.auth import User
from agentseek_api.services.executor import get_executor
from agentseek_api.services.run_executor import execute_run
from agentseek_api.services.run_state import run_broker


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

    async def _execute() -> None:
        session_factory_inner = db_manager.get_session_factory()
        async with session_factory_inner() as execution_session:
            db_run = await execution_session.scalar(select(Run).where(Run.run_id == run.run_id))
            if db_run is None:
                return
            db_run.status = "running"
            run_broker.publish(run.run_id, "start")
            await execution_session.commit()
            try:
                output = await execute_run(
                    thread_id=run.thread_id,
                    run_id=run.run_id,
                    payload=run.input_json,
                    graph_id=graph_id,
                )
                db_run.status = "success"
                db_run.output_json = output
            except Exception as exc:  # noqa: BLE001
                db_run.status = "error"
                db_run.last_error = str(exc)
            run_broker.publish(run.run_id, "end")
            await execution_session.commit()

    await get_executor().submit(_execute)
    return run
