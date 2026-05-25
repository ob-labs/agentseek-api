from __future__ import annotations

from datetime import UTC, datetime, timedelta

from sqlalchemy import select

from agentseek_api.core.database import db_manager
from agentseek_api.core.orm import CronJob, CronTick, Thread
from agentseek_api.models.api import ThreadCreate
from agentseek_api.models.auth import User
from agentseek_api.services.cron_models import ClaimedCron, CronDispatchResult
from agentseek_api.services.cron_rrule import compute_next_run_at
from agentseek_api.services.run_preparation import (
    ActiveThreadRunConflictError,
    prepare_and_submit_run,
)
from agentseek_api.services.thread_service import create_thread_for_user
from agentseek_api.settings import settings


def _cron_user(user_id: str) -> User:
    return User(identity=user_id, is_authenticated=True)


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
        await session.commit()


async def claim_due_crons(
    *,
    limit: int,
    scheduler_id: str,
    now: datetime | None = None,
) -> list[ClaimedCron]:
    current_time = now or datetime.now(UTC)
    stale_before = current_time - timedelta(seconds=_started_tick_stale_after_seconds())
    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        reclaimed_ticks = list(
            (
                await session.scalars(
                    select(CronTick)
                    .where(
                        CronTick.status == "started",
                        CronTick.created_at <= stale_before,
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
            claimed.append(ClaimedCron.from_row(row, tick_id=tick.id, scheduled_for=tick.scheduled_for))
            seen_cron_ids.add(row.cron_id)

        remaining = max(0, limit - len(claimed))
        if remaining == 0:
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
                metadata={"cron_id": claim.cron_id, "scheduled_for": claim.scheduled_for.isoformat()},
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
            metadata={"cron_id": claim.cron_id, "scheduled_for": claim.scheduled_for.isoformat()},
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
) -> list[CronDispatchResult]:
    claimed = await claim_due_crons(limit=limit, scheduler_id=scheduler_id, now=now)
    results: list[CronDispatchResult] = []
    for item in claimed:
        results.append(await dispatch_claimed_cron(item))
    return results
