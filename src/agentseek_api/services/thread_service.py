from typing import Any

from fastapi import HTTPException
from sqlalchemy import select

from agentseek_api.core.database import db_manager
from agentseek_api.core.orm import Thread
from agentseek_api.models.api import ThreadCreate, ThreadRead
from agentseek_api.models.auth import User


def _public_thread_config(config: dict[str, object] | None) -> dict[str, object]:
    if not isinstance(config, dict):
        return {}
    return dict(config)


def to_read_model(
    row: Thread,
    *,
    select: set[str] | None = None,
    values: dict[str, Any] | None = None,
    interrupts: dict[str, Any] | None = None,
) -> ThreadRead:
    def _include(field: str) -> bool:
        return select is None or field in select

    return ThreadRead(
        thread_id=row.thread_id,
        created_at=row.created_at,
        updated_at=row.updated_at,
        metadata=row.metadata_json,
        status=row.status,
        state_updated_at=row.state_updated_at if _include("state_updated_at") else None,
        config=_public_thread_config(row.config_json) if _include("config") else {},
        values=(values or {}) if _include("values") else {},
        interrupts=interrupts if _include("interrupts") else None,
    )


async def create_thread_for_user(*, payload: ThreadCreate, user: User) -> ThreadRead:
    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        if payload.thread_id is not None:
            existing = await session.scalar(
                select(Thread).where(Thread.thread_id == payload.thread_id, Thread.user_id == user.identity)
            )
            if existing is not None:
                if payload.if_exists == "raise":
                    raise HTTPException(status_code=409, detail="Thread already exists")
                return to_read_model(existing)

        kwargs: dict[str, object] = {
            "user_id": user.identity,
            "metadata_json": payload.metadata,
        }
        if payload.thread_id is not None:
            kwargs["thread_id"] = payload.thread_id

        row = Thread(**kwargs)
        session.add(row)
        await session.commit()
        await session.refresh(row)
        return to_read_model(row)
