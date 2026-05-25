from sqlalchemy import delete, func, select

from fastapi import HTTPException

from agentseek_api.core.database import db_manager
from agentseek_api.core.orm import CronJob
from agentseek_api.models.api import (
    CronCountRequest,
    CronCountResponse,
    CronCreate,
    CronPatch,
    CronRead,
    CronSearchRequest,
    CronSearchResponse,
)
from agentseek_api.models.auth import User
from agentseek_api.services.cron_rrule import compute_next_run_at, validate_schedule


def _to_read_model(row: CronJob) -> CronRead:
    return CronRead(
        cron_id=row.cron_id,
        assistant_id=row.assistant_id,
        thread_id=row.thread_id,
        enabled=row.enabled,
        schedule=row.schedule,
        next_run_at=row.next_run_at,
    )


def _search_stmt(*, user_id: str, payload: CronSearchRequest):
    stmt = select(CronJob).where(CronJob.user_id == user_id)
    if payload.assistant_id is not None:
        stmt = stmt.where(CronJob.assistant_id == payload.assistant_id)
    if payload.enabled is not None:
        stmt = stmt.where(CronJob.enabled == payload.enabled)
    if payload.thread_id is not None:
        stmt = stmt.where(CronJob.thread_id == payload.thread_id)
    return stmt


async def create_cron(*, assistant_id: str, thread_id: str | None, payload: CronCreate, user: User) -> CronRead:
    validate_schedule(payload.schedule, timezone_name="UTC")
    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        row = CronJob(
            assistant_id=assistant_id,
            thread_id=thread_id,
            user_id=user.identity,
            schedule=payload.schedule,
            enabled=payload.enabled,
            input_json=payload.input,
            next_run_at=compute_next_run_at(payload.schedule, timezone_name="UTC"),
        )
        session.add(row)
        await session.commit()
        await session.refresh(row)
        return _to_read_model(row)


async def get_cron(*, cron_id: str, user: User) -> CronRead:
    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        row = await session.scalar(select(CronJob).where(CronJob.cron_id == cron_id, CronJob.user_id == user.identity))
        if row is None:
            raise HTTPException(status_code=404, detail="Cron not found")
        return _to_read_model(row)


async def patch_cron(*, cron_id: str, payload: CronPatch, user: User) -> CronRead:
    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        row = await session.scalar(select(CronJob).where(CronJob.cron_id == cron_id, CronJob.user_id == user.identity))
        if row is None:
            raise HTTPException(status_code=404, detail="Cron not found")

        schedule_was_updated = False
        if payload.schedule is not None:
            validate_schedule(payload.schedule, timezone_name="UTC")
            row.schedule = payload.schedule
            row.next_run_at = compute_next_run_at(payload.schedule, timezone_name="UTC")
            schedule_was_updated = True
        if payload.enabled is not None:
            if payload.enabled and not row.enabled and not schedule_was_updated:
                row.next_run_at = compute_next_run_at(row.schedule, timezone_name="UTC")
            row.enabled = payload.enabled
        if "input" in payload.model_fields_set:
            if payload.input is None:
                raise ValueError("input cannot be null")
            row.input_json = payload.input

        await session.commit()
        await session.refresh(row)
        return _to_read_model(row)


async def delete_cron(*, cron_id: str, user: User) -> None:
    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        result = await session.execute(delete(CronJob).where(CronJob.cron_id == cron_id, CronJob.user_id == user.identity))
        if result.rowcount == 0:
            raise HTTPException(status_code=404, detail="Cron not found")
        await session.commit()


async def search_crons(*, payload: CronSearchRequest, user: User) -> CronSearchResponse:
    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        stmt = (
            _search_stmt(user_id=user.identity, payload=payload)
            .order_by(CronJob.created_at.desc(), CronJob.cron_id.desc())
            .limit(payload.limit)
            .offset(payload.offset)
        )
        rows = list((await session.scalars(stmt)).all())
    return CronSearchResponse(items=[_to_read_model(row) for row in rows])


async def count_crons(*, payload: CronCountRequest, user: User) -> CronCountResponse:
    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        stmt = _search_stmt(user_id=user.identity, payload=payload).with_only_columns(func.count(CronJob.cron_id))
        count = await session.scalar(stmt)
    return CronCountResponse(count=int(count or 0))
