from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

from sqlalchemy import select

from agentseek_api.core.database import db_manager
from agentseek_api.core.orm import CronJob, CronTick, Run, Thread
from agentseek_api.models.api import ThreadCreate
from agentseek_api.models.auth import User
from agentseek_api.services.cron_models import ClaimedCron, CronDispatchResult
from agentseek_api.services.cron_rrule import compute_next_run_at
from agentseek_api.services.cron_webhooks import (
    build_webhook_payload,
    deliver_webhook_with_retries,
    get_webhook_http_client,
)
from agentseek_api.services.run_preparation import (
    ActiveThreadRunConflictError,
    prepare_and_submit_run,
)
from agentseek_api.services.thread_service import create_thread_for_user
from agentseek_api.settings import settings

TERMINAL_RUN_STATUSES = {"success", "error", "interrupted"}
TERMINAL_TICK_STATUSES = {"success", "error", "interrupted", "skipped"}


def _cron_user(user_id: str) -> User:
    return User(identity=user_id, is_authenticated=True)


def _as_utc(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def _started_tick_stale_after_seconds() -> int:
    configured = settings.SCHEDULER_STARTED_TICK_STALE_AFTER_SECONDS
    if configured is not None:
        return max(1, configured)
    return max(1, settings.REDIS_SCHEDULER_LOCK_TTL_SECONDS)


async def _thread_is_busy(*, thread_id: str, user_id: str) -> bool:
    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        thread = await session.scalar(
            select(Thread).where(Thread.thread_id == thread_id, Thread.user_id == user_id)
        )
    return thread is not None and thread.status == "busy"


async def _mark_tick_outcome(
    *,
    tick_id: int,
    status: str,
    run_id: str | None = None,
    skip_reason: str | None = None,
) -> None:
    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        tick = await session.scalar(select(CronTick).where(CronTick.id == tick_id))
        if tick is None:
            raise RuntimeError(f"Cron tick {tick_id} not found")
        tick.status = status
        tick.run_id = run_id
        tick.skip_reason = skip_reason
        cron = await session.scalar(select(CronJob).where(CronJob.cron_id == tick.cron_id))
        if cron is not None and status in TERMINAL_TICK_STATUSES:
            cron.last_tick_status = status
            cron.last_error = skip_reason if status == "error" else None
            if status != "skipped":
                cron.last_run_at = datetime.now(UTC)
        await session.commit()


def _tick_stale_before(*, current_time: datetime) -> datetime:
    return current_time - timedelta(seconds=_started_tick_stale_after_seconds())


def _webhook_delivery_stale_before(*, current_time: datetime) -> datetime:
    return current_time - timedelta(seconds=_started_tick_stale_after_seconds())


async def _reconcile_terminal_ticks(
    *,
    http_client=None,
    sleep=asyncio.sleep,
) -> None:
    current_time = datetime.now(UTC)
    stale_before = _webhook_delivery_stale_before(current_time=current_time)
    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        ticks = list(
            (
                await session.scalars(
                    select(CronTick)
                    .where(
                        (CronTick.status == "queued")
                        | (
                            (CronTick.status.in_(TERMINAL_TICK_STATUSES))
                            & (
                                (CronTick.webhook_delivery_status.is_(None))
                                | (
                                    (CronTick.webhook_delivery_status == "delivering")
                                    & (CronTick.updated_at <= stale_before)
                                )
                            )
                        )
                    )
                    .order_by(CronTick.id.asc())
                    .with_for_update()
                )
            ).all()
        )

        deliveries: list[tuple[int, str, int, dict[str, object]]] = []
        for tick in ticks:
            cron = await session.scalar(select(CronJob).where(CronJob.cron_id == tick.cron_id))
            if cron is None:
                continue
            if tick.status == "queued" and tick.run_id is not None:
                run = await session.scalar(select(Run).where(Run.run_id == tick.run_id))
                if run is not None and run.status in TERMINAL_RUN_STATUSES:
                    tick.status = run.status
                    tick.skip_reason = run.last_error if run.status == "error" else None
            if tick.status in TERMINAL_TICK_STATUSES:
                cron.last_tick_status = tick.status
                cron.last_error = tick.skip_reason if tick.status == "error" else None
                if tick.status != "skipped":
                    cron.last_run_at = _as_utc(tick.updated_at)
            delivery_is_available = tick.webhook_delivery_status is None or (
                tick.webhook_delivery_status == "delivering" and _as_utc(tick.updated_at) <= stale_before
            )
            if cron.webhook and tick.status in TERMINAL_TICK_STATUSES and delivery_is_available:
                tick.webhook_delivery_status = "delivering"
                deliveries.append(
                    (
                        tick.id,
                        cron.webhook,
                        cron.max_webhook_attempts,
                        build_webhook_payload(cron=cron, tick=tick),
                    )
                )
        await session.commit()

    webhook_client = http_client or get_webhook_http_client()
    for tick_id, webhook_url, max_attempts, payload in deliveries:
        await deliver_webhook_with_retries(
            webhook_url=webhook_url,
            payload=payload,
            tick_id=tick_id,
            max_attempts=max_attempts,
            http_client=webhook_client,
            sleep=sleep,
        )


async def claim_due_crons(
    *,
    limit: int,
    scheduler_id: str,
    now: datetime | None = None,
) -> list[ClaimedCron]:
    current_time = now or datetime.now(UTC)
    stale_before = _tick_stale_before(current_time=current_time)
    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        reclaimed_ticks = list(
            (
                await session.scalars(
                    select(CronTick)
                    .where(
                        CronTick.status == "started",
                        CronTick.updated_at <= stale_before,
                    )
                    .order_by(CronTick.scheduled_for.asc(), CronTick.id.asc())
                    .limit(limit)
                    .with_for_update()
                )
            ).all()
        )

        claimed: list[ClaimedCron] = []
        seen_cron_ids: set[str] = set()
        for tick in reclaimed_ticks:
            row = await session.scalar(select(CronJob).where(CronJob.cron_id == tick.cron_id))
            if row is None:
                continue
            tick.scheduler_id = scheduler_id
            tick.updated_at = current_time
            claimed.append(ClaimedCron.from_row(row, tick_id=tick.id, scheduled_for=tick.scheduled_for))
            seen_cron_ids.add(row.cron_id)

        remaining = max(0, limit - len(claimed))
        if remaining == 0:
            await session.commit()
            return claimed

        rows = list(
            (
                await session.scalars(
                    select(CronJob)
                    .where(
                        CronJob.enabled.is_(True),
                        CronJob.next_run_at <= current_time,
                    )
                    .order_by(CronJob.next_run_at.asc(), CronJob.cron_id.asc())
                    .limit(remaining)
                    .with_for_update()
                )
            ).all()
        )

        for row in rows:
            if row.cron_id in seen_cron_ids:
                continue
            scheduled_for = row.next_run_at
            tick = CronTick(
                cron_id=row.cron_id,
                thread_id=row.thread_id,
                scheduler_id=scheduler_id,
                scheduled_for=scheduled_for,
                status="started",
            )
            session.add(tick)
            await session.flush()
            claimed.append(ClaimedCron.from_row(row, tick_id=tick.id, scheduled_for=scheduled_for))
            try:
                row.next_run_at = compute_next_run_at(row.schedule, timezone_name="UTC", now=current_time)
            except ValueError:
                row.enabled = False
        await session.commit()
    return claimed


async def dispatch_claimed_cron(claim: ClaimedCron) -> CronDispatchResult:
    user = _cron_user(claim.user_id)
    scheduled_for_iso = _as_utc(claim.scheduled_for).isoformat()
    try:
        if claim.thread_id is None:
            thread = await create_thread_for_user(
                payload=ThreadCreate(metadata={"stateless": True, "cron_id": claim.cron_id}),
                user=user,
            )
            run = await prepare_and_submit_run(
                thread_id=thread.thread_id,
                assistant_id=claim.assistant_id,
                payload=claim.input_json,
                user=user,
                metadata={
                    **claim.metadata_json,
                    "cron_id": claim.cron_id,
                    "scheduled_for": scheduled_for_iso,
                },
                kwargs=claim.kwargs_json,
            )
            await _mark_tick_outcome(tick_id=claim.tick_id, status="queued", run_id=run.run_id)
            return CronDispatchResult(
                cron_id=claim.cron_id,
                status="queued",
                thread_id=thread.thread_id,
                run_id=run.run_id,
            )

        if await _thread_is_busy(thread_id=claim.thread_id, user_id=claim.user_id):
            await _mark_tick_outcome(tick_id=claim.tick_id, status="skipped", skip_reason="thread_busy")
            return CronDispatchResult(
                cron_id=claim.cron_id,
                status="skipped",
                thread_id=claim.thread_id,
                skip_reason="thread_busy",
            )

        run = await prepare_and_submit_run(
            thread_id=claim.thread_id,
            assistant_id=claim.assistant_id,
            payload=claim.input_json,
            user=user,
            metadata={
                **claim.metadata_json,
                "cron_id": claim.cron_id,
                "scheduled_for": scheduled_for_iso,
            },
            kwargs=claim.kwargs_json,
        )
    except ActiveThreadRunConflictError:
        await _mark_tick_outcome(tick_id=claim.tick_id, status="skipped", skip_reason="thread_busy")
        return CronDispatchResult(
            cron_id=claim.cron_id,
            status="skipped",
            thread_id=claim.thread_id,
            skip_reason="thread_busy",
        )
    except Exception:
        await _mark_tick_outcome(tick_id=claim.tick_id, status="error", skip_reason="submission_failed")
        return CronDispatchResult(
            cron_id=claim.cron_id,
            status="error",
            thread_id=claim.thread_id,
            skip_reason="submission_failed",
        )

    await _mark_tick_outcome(tick_id=claim.tick_id, status="queued", run_id=run.run_id)
    return CronDispatchResult(
        cron_id=claim.cron_id,
        status="queued",
        thread_id=claim.thread_id,
        run_id=run.run_id,
    )


async def dispatch_due_crons(
    *,
    limit: int,
    scheduler_id: str,
    now: datetime | None = None,
    webhook_sleep=asyncio.sleep,
) -> list[CronDispatchResult]:
    await _reconcile_terminal_ticks(sleep=webhook_sleep)
    claimed = await claim_due_crons(limit=limit, scheduler_id=scheduler_id, now=now)
    results: list[CronDispatchResult] = []
    for item in claimed:
        results.append(await dispatch_claimed_cron(item))
    await _reconcile_terminal_ticks(sleep=webhook_sleep)
    return results
