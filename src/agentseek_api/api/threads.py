from sqlalchemy import select

from fastapi import APIRouter, Depends, HTTPException

from agentseek_api.core.auth_deps import get_current_user
from agentseek_api.core.database import db_manager
from agentseek_api.core.orm import Thread
from agentseek_api.models.api import ThreadCreate, ThreadRead
from agentseek_api.models.auth import User

router = APIRouter(prefix="/threads", tags=["Threads"])


@router.post("", response_model=ThreadRead)
async def create_thread(payload: ThreadCreate, user: User = Depends(get_current_user)) -> ThreadRead:
    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        row = Thread(user_id=user.identity, metadata_json=payload.metadata)
        session.add(row)
        await session.commit()
        await session.refresh(row)
        return ThreadRead(thread_id=row.thread_id, user_id=row.user_id, metadata=row.metadata_json, created_at=row.created_at)


@router.get("", response_model=list[ThreadRead])
async def list_threads(user: User = Depends(get_current_user)) -> list[ThreadRead]:
    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        rows = (
            await session.scalars(select(Thread).where(Thread.user_id == user.identity).order_by(Thread.created_at.desc()))
        ).all()
        return [ThreadRead(thread_id=row.thread_id, user_id=row.user_id, metadata=row.metadata_json, created_at=row.created_at) for row in rows]


@router.get("/{thread_id}", response_model=ThreadRead)
async def get_thread(thread_id: str, user: User = Depends(get_current_user)) -> ThreadRead:
    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        row = await session.scalar(select(Thread).where(Thread.thread_id == thread_id, Thread.user_id == user.identity))
        if row is None:
            raise HTTPException(status_code=404, detail="Thread not found")
        return ThreadRead(thread_id=row.thread_id, user_id=row.user_id, metadata=row.metadata_json, created_at=row.created_at)
