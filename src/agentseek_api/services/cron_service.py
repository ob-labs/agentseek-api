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
    ThreadCronCreate,
)
from agentseek_api.models.auth import User
from agentseek_api.services.cron_rrule import compute_next_run_at, validate_schedule
from agentseek_api.services.stream_modes import normalize_stream_modes

DEFAULT_MULTITASK_STRATEGY = "enqueue"
DEFAULT_DURABILITY = "async"
DEFAULT_ON_RUN_COMPLETED = "delete"


def _declared_field(payload: Any, field: str, default: Any) -> Any:
    """Read a field only if it is DECLARED on the model, else the default.

    CronCreate and ThreadCronCreate use ``extra="allow"``, so a field that
    belongs to the other model (e.g. on_run_completed sent to ThreadCronCreate)
    lands in ``model_extra`` where a plain ``getattr`` would still find it. This
    guard ensures such cross-model fields are ignored rather than honored.
    """
    if field in type(payload).model_fields:
        return getattr(payload, field)
    return default


def _normalize_timezone(timezone_name: str | None) -> str:
    return timezone_name or "UTC"


def _validate_webhook(webhook: str | None) -> str | None:
    if webhook is None:
        return None
    parsed = urlparse(webhook)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("webhook must be an absolute http or https URL")
    return webhook


def _cron_kwargs(
    *,
    config: dict,
    context: dict,
    stream_mode: Any = None,
    interrupt_before: Any = None,
    interrupt_after: Any = None,
    durability: str = DEFAULT_DURABILITY,
    stream_subgraphs: bool = False,
    stream_resumable: bool = False,
    multitask_strategy: str = DEFAULT_MULTITASK_STRATEGY,
) -> dict:
    kwargs: dict[str, Any] = {"config": config, "context": context}
    kwargs["stream_modes"] = normalize_stream_modes(stream_mode)
    if interrupt_before is not None:
        kwargs["interrupt_before"] = interrupt_before
    if interrupt_after is not None:
        kwargs["interrupt_after"] = interrupt_after
    if durability != DEFAULT_DURABILITY:
        kwargs["durability"] = durability
    if stream_subgraphs:
        kwargs["stream_subgraphs"] = True
    if stream_resumable:
        kwargs["stream_resumable"] = True
    if multitask_strategy != DEFAULT_MULTITASK_STRATEGY:
        kwargs["multitask_strategy"] = multitask_strategy
    return kwargs


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


def _search_stmt(*, payload: CronSearchRequest | CronCountRequest):
    stmt = select(CronJob)
    if payload.assistant_id is not None:
        stmt = stmt.where(CronJob.assistant_id == payload.assistant_id)
    if payload.enabled is not None:
        stmt = stmt.where(CronJob.enabled == payload.enabled)
    if payload.thread_id is not None:
        stmt = stmt.where(CronJob.thread_id == payload.thread_id)
    if payload.metadata is not None:
        for key, value in payload.metadata.items():
            stmt = stmt.where(CronJob.metadata_json[key].as_string() == str(value))
    return stmt


async def create_cron(*, assistant_id: str, thread_id: str | None, payload: CronCreate | ThreadCronCreate, user: User) -> CronRead:
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
            kwargs_json=_cron_kwargs(
                config=payload.config,
                context=payload.context,
                stream_mode=payload.stream_mode,
                interrupt_before=payload.interrupt_before,
                interrupt_after=payload.interrupt_after,
                durability=payload.durability,
                stream_subgraphs=payload.stream_subgraphs,
                stream_resumable=payload.stream_resumable,
                multitask_strategy=_declared_field(payload, "multitask_strategy", DEFAULT_MULTITASK_STRATEGY),
            ),
            webhook=webhook,
            end_time=payload.end_time,
            on_run_completed=_declared_field(payload, "on_run_completed", DEFAULT_ON_RUN_COMPLETED),
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
        if "config" in payload.model_fields_set and payload.config is None:
            raise ValueError("config cannot be null")
        if "context" in payload.model_fields_set and payload.context is None:
            raise ValueError("context cannot be null")

        # Maps each patchable run-control field to the _cron_kwargs() keyword
        # and the kwargs_json storage key its current value is read from.
        # Adding a new run-control field is a single entry here.
        run_control_fields = {
            "config": ("config", "config"),
            "context": ("context", "context"),
            "stream_mode": ("stream_mode", "stream_modes"),
            "interrupt_before": ("interrupt_before", "interrupt_before"),
            "interrupt_after": ("interrupt_after", "interrupt_after"),
            "durability": ("durability", "durability"),
            "stream_subgraphs": ("stream_subgraphs", "stream_subgraphs"),
            "stream_resumable": ("stream_resumable", "stream_resumable"),
            "multitask_strategy": ("multitask_strategy", "multitask_strategy"),
        }
        if run_control_fields.keys() & payload.model_fields_set:
            existing = row.kwargs_json or {}
            # Start from the cron's current run-control values, then overlay
            # only the fields explicitly present in this patch.
            fold_kwargs: dict[str, Any] = {
                kwarg: existing.get(stored_key)
                for kwarg, stored_key in run_control_fields.values()
                if existing.get(stored_key) is not None
            }
            for field, (kwarg, _stored_key) in run_control_fields.items():
                if field in payload.model_fields_set:
                    value = getattr(payload, field)
                    if value is None:
                        # Explicit null clears the field back to its default.
                        fold_kwargs.pop(kwarg, None)
                    else:
                        fold_kwargs[kwarg] = value
            row.kwargs_json = _cron_kwargs(
                config=fold_kwargs.pop("config", {}),
                context=fold_kwargs.pop("context", {}),
                **fold_kwargs,
            )

        if "end_time" in payload.model_fields_set:
            row.end_time = payload.end_time
        if "on_run_completed" in payload.model_fields_set:
            if payload.on_run_completed is None:
                raise ValueError("on_run_completed cannot be null")
            row.on_run_completed = payload.on_run_completed

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

        sort_columns = {
            "next_run_date": CronJob.next_run_at,
            "end_time": CronJob.end_time,
        }
        if payload.sort_by is not None:
            sort_column = sort_columns.get(payload.sort_by, getattr(CronJob, payload.sort_by, CronJob.created_at))
            ascending = payload.sort_order == "asc"
            primary = sort_column.asc() if ascending else sort_column.desc()
            # Tiebreaker follows the primary direction so equal-key rows page
            # deterministically (a desc tiebreaker under an asc sort interleaves).
            tiebreaker = CronJob.cron_id.asc() if ascending else CronJob.cron_id.desc()
            stmt = stmt.order_by(primary, tiebreaker)
        else:
            stmt = stmt.order_by(CronJob.created_at.desc(), CronJob.cron_id.desc())

        stmt = stmt.limit(payload.limit).offset(payload.offset)
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
