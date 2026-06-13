from typing import Any
from urllib.parse import urlparse

from sqlalchemy import func, select

from fastapi import HTTPException

from agentseek_api.core.auth_deps import apply_metadata_filters
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


def _normalize_timezone(timezone_name: str | None) -> str:
    return timezone_name or "UTC"


def _validate_webhook(webhook: str | None) -> str | None:
    if webhook is None:
        return None
    parsed = urlparse(webhook)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("webhook must be an absolute http or https URL")
    return webhook


def _cron_kwargs(*, config: dict, context: dict) -> dict:
    return {"config": config, "context": context}


def _to_read_model(row: CronJob) -> CronRead:
    kwargs = row.kwargs_json or {}
    payload = {
        "input": row.input_json,
        "config": kwargs.get("config", {}),
        "context": kwargs.get("context", {}),
    }
    return CronRead(
        cron_id=row.cron_id,
        assistant_id=row.assistant_id,
        thread_id=row.thread_id,
        user_id=row.user_id,
        enabled=row.enabled,
        schedule=row.schedule,
        payload=payload,
        metadata=row.metadata_json or {},
        next_run_date=row.next_run_at,
        next_run_at=row.next_run_at,
        end_time=row.end_time,
        created_at=row.created_at,
        updated_at=row.updated_at,
        timezone=row.timezone,
        webhook=row.webhook,
        last_run_at=row.last_run_at,
        last_tick_status=row.last_tick_status,
        last_error=row.last_error,
    )


def _search_stmt(*, payload: CronSearchRequest):
    stmt = select(CronJob)
    if payload.assistant_id is not None:
        stmt = stmt.where(CronJob.assistant_id == payload.assistant_id)
    if payload.enabled is not None:
        stmt = stmt.where(CronJob.enabled == payload.enabled)
    if payload.thread_id is not None:
        stmt = stmt.where(CronJob.thread_id == payload.thread_id)
    return stmt


async def create_cron(*, assistant_id: str, thread_id: str | None, payload: CronCreate, user: User) -> CronRead:
    timezone_name = _normalize_timezone(payload.timezone)
    validate_schedule(payload.schedule, timezone_name=timezone_name)
    webhook = _validate_webhook(payload.webhook)
    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        row = CronJob(
            assistant_id=assistant_id,
            thread_id=thread_id,
            user_id=user.identity,
            schedule=payload.schedule,
            timezone=timezone_name,
            enabled=payload.enabled,
            input_json=payload.input,
            metadata_json=payload.metadata,
            kwargs_json=_cron_kwargs(config=payload.config, context=payload.context),
            webhook=webhook,
            next_run_at=compute_next_run_at(payload.schedule, timezone_name=timezone_name),
        )
        session.add(row)
        await session.commit()
        await session.refresh(row)
        return _to_read_model(row)


async def get_cron(*, cron_id: str, user: User, filters: dict[str, Any] | None = None) -> CronRead:
    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        stmt = select(CronJob).where(CronJob.cron_id == cron_id)
        stmt = apply_metadata_filters(stmt, CronJob, filters)
        row = await session.scalar(stmt)
        if row is None:
            raise HTTPException(status_code=404, detail="Cron not found")
        return _to_read_model(row)


async def patch_cron(*, cron_id: str, payload: CronPatch, user: User, filters: dict[str, Any] | None = None) -> CronRead:
    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        stmt = select(CronJob).where(CronJob.cron_id == cron_id)
        stmt = apply_metadata_filters(stmt, CronJob, filters)
        row = await session.scalar(stmt)
        if row is None:
            raise HTTPException(status_code=404, detail="Cron not found")

        schedule_was_updated = False
        current_timezone = row.timezone
        if "timezone" in payload.model_fields_set:
            row.timezone = _normalize_timezone(payload.timezone)
            current_timezone = row.timezone
            validate_schedule(row.schedule, timezone_name=current_timezone)
        if "webhook" in payload.model_fields_set:
            row.webhook = _validate_webhook(payload.webhook)
        if payload.schedule is not None:
            validate_schedule(payload.schedule, timezone_name=current_timezone)
            row.schedule = payload.schedule
            row.next_run_at = compute_next_run_at(payload.schedule, timezone_name=current_timezone)
            schedule_was_updated = True
        if payload.enabled is not None:
            if payload.enabled and not row.enabled and not schedule_was_updated:
                row.next_run_at = compute_next_run_at(row.schedule, timezone_name=current_timezone)
            row.enabled = payload.enabled
        if "input" in payload.model_fields_set:
            if payload.input is None:
                raise ValueError("input cannot be null")
            row.input_json = payload.input
        if "metadata" in payload.model_fields_set:
            if payload.metadata is None:
                raise ValueError("metadata cannot be null")
            row.metadata_json = payload.metadata
        if "config" in payload.model_fields_set:
            if payload.config is None:
                raise ValueError("config cannot be null")
            row.kwargs_json = _cron_kwargs(
                config=payload.config,
                context=row.kwargs_json.get("context", {}),
            )
        if "context" in payload.model_fields_set:
            if payload.context is None:
                raise ValueError("context cannot be null")
            row.kwargs_json = _cron_kwargs(
                config=row.kwargs_json.get("config", {}),
                context=payload.context,
            )
        if "timezone" in payload.model_fields_set and payload.schedule is None and row.enabled:
            row.next_run_at = compute_next_run_at(row.schedule, timezone_name=current_timezone)

        await session.commit()
        await session.refresh(row)
        return _to_read_model(row)


async def delete_cron(*, cron_id: str, user: User, filters: dict[str, Any] | None = None) -> None:
    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        stmt = select(CronJob).where(CronJob.cron_id == cron_id)
        stmt = apply_metadata_filters(stmt, CronJob, filters)
        row = await session.scalar(stmt)
        if row is None:
            raise HTTPException(status_code=404, detail="Cron not found")
        await session.delete(row)
        await session.commit()


async def search_crons(*, payload: CronSearchRequest, user: User, filters: dict[str, Any] | None = None) -> CronSearchResponse:
    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        stmt = _search_stmt(payload=payload)
        stmt = apply_metadata_filters(stmt, CronJob, filters)
        stmt = (
            stmt
            .order_by(CronJob.created_at.desc(), CronJob.cron_id.desc())
            .limit(payload.limit)
            .offset(payload.offset)
        )
        rows = list((await session.scalars(stmt)).all())
    return CronSearchResponse(items=[_to_read_model(row) for row in rows])


async def count_crons(*, payload: CronCountRequest, user: User, filters: dict[str, Any] | None = None) -> CronCountResponse:
    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        stmt = _search_stmt(payload=payload)
        stmt = apply_metadata_filters(stmt, CronJob, filters)
        stmt = stmt.with_only_columns(func.count(CronJob.cron_id))
        count = await session.scalar(stmt)
    return CronCountResponse(count=int(count or 0))
