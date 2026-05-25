from datetime import UTC, datetime, timedelta

from sqlalchemy import select

from fastapi import APIRouter, Depends, HTTPException

from agentseek_api.core.auth_deps import get_current_user
from agentseek_api.core.database import db_manager
from agentseek_api.core.orm import Assistant, CronJob, Thread
from agentseek_api.models.api import CronCreate, CronRead
from agentseek_api.models.auth import User

router = APIRouter(tags=["Crons"])


def _placeholder_next_run_at() -> datetime:
    return datetime.now(UTC) + timedelta(minutes=1)


def _to_read_model(row: CronJob) -> CronRead:
    return CronRead(
        cron_id=row.cron_id,
        assistant_id=row.assistant_id,
        thread_id=row.thread_id,
        enabled=row.enabled,
        schedule=row.schedule,
        next_run_at=row.next_run_at,
    )


async def _ensure_assistant_exists(*, assistant_id: str) -> None:
    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        existing = await session.scalar(select(Assistant.assistant_id).where(Assistant.assistant_id == assistant_id))
        if existing is None:
            raise HTTPException(status_code=404, detail="Assistant not found")


async def _create_cron(
    *,
    assistant_id: str,
    thread_id: str | None,
    payload: CronCreate,
    user: User,
) -> CronRead:
    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        row = CronJob(
            assistant_id=assistant_id,
            thread_id=thread_id,
            user_id=user.identity,
            schedule=payload.schedule,
            enabled=payload.enabled,
            input_json=payload.input,
            next_run_at=_placeholder_next_run_at(),
        )
        session.add(row)
        await session.commit()
        await session.refresh(row)
        return _to_read_model(row)


@router.post("/runs/crons", response_model=CronRead)
async def create_stateless_cron(payload: CronCreate, user: User = Depends(get_current_user)) -> CronRead:
    await _ensure_assistant_exists(assistant_id=payload.assistant_id)
    return await _create_cron(assistant_id=payload.assistant_id, thread_id=None, payload=payload, user=user)


@router.post("/threads/{thread_id}/runs/crons", response_model=CronRead)
async def create_thread_cron(thread_id: str, payload: CronCreate, user: User = Depends(get_current_user)) -> CronRead:
    await _ensure_assistant_exists(assistant_id=payload.assistant_id)
    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        thread = await session.scalar(select(Thread).where(Thread.thread_id == thread_id, Thread.user_id == user.identity))
        if thread is None:
            raise HTTPException(status_code=404, detail="Thread not found")
    return await _create_cron(assistant_id=payload.assistant_id, thread_id=thread_id, payload=payload, user=user)
