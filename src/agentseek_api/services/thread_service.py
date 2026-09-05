import logging
from typing import Any

from fastapi import HTTPException
from sqlalchemy import delete, select

from agentseek_api.core.database import db_manager
from agentseek_api.core.orm import Run, Thread
from agentseek_api.models.api import ThreadCreate, ThreadRead
from agentseek_api.models.auth import User
from agentseek_api.services.stream_persistence import (
    delete_run_stream_events,
    delete_thread_stream_events,
)
from agentseek_api.services.thread_protocol import thread_protocol_broker

logger = logging.getLogger(__name__)


async def _best_effort_checkpointer_call(method_name: str, *args: object, **kwargs: object) -> None:
    """Call an optional checkpointer method, awaiting if needed, swallowing
    NotImplementedError for backends that do not support it."""
    method = getattr(db_manager.get_langgraph_checkpointer(), method_name, None)
    if method is None:
        return
    try:
        result = method(*args, **kwargs)
        if hasattr(result, "__await__"):
            await result
    except NotImplementedError:
        return


async def delete_threads_cascade(thread_ids: list[str]) -> list[str]:
    """Delete threads and all their runs, plus best-effort checkpointer/stream
    cleanup. Shared by the HTTP delete endpoint and the cron scheduler so the
    teardown sequence cannot drift between them.

    Returns the run_ids that were removed. The relational deletes run in a
    single transaction; per-thread best-effort cleanup follows.
    """
    if not thread_ids:
        return []
    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        run_ids = list(
            (await session.scalars(select(Run.run_id).where(Run.thread_id.in_(thread_ids)))).all()
        )
        await session.execute(delete(Run).where(Run.thread_id.in_(thread_ids)))
        await session.execute(delete(Thread).where(Thread.thread_id.in_(thread_ids)))
        await session.commit()

    for thread_id in thread_ids:
        await _best_effort_checkpointer_call("adelete_thread", thread_id)
    if run_ids:
        await _best_effort_checkpointer_call("adelete_for_runs", run_ids)
        await delete_run_stream_events(run_ids)
    for thread_id in thread_ids:
        thread_protocol_broker.delete_thread(thread_id)
        await delete_thread_stream_events(thread_id)
    return run_ids


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
                select(Thread).where(Thread.thread_id == payload.thread_id)
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
