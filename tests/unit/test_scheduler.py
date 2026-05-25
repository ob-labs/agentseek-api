from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import select

from agentseek_api.core.database import db_manager
from agentseek_api.core.orm import Assistant, CronJob, CronTick


class _FakeCheckpointer:
    def __init__(self, connection_args: dict[str, str]) -> None:
        self.connection_args = connection_args

    def setup(self) -> None:
        return None


def _as_utc(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


async def _age_tick(*, cron_id: str, stale_at: datetime) -> None:
    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        tick = await session.scalar(select(CronTick).where(CronTick.cron_id == cron_id))
        assert tick is not None
        tick.created_at = stale_at
        tick.updated_at = stale_at
        await session.commit()


@pytest.mark.asyncio
async def test_claim_due_crons_returns_each_due_cron_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    from agentseek_api.services.cron_scheduler import _mark_tick_outcome, claim_due_crons
    from agentseek_api.settings import settings

    monkeypatch.setattr(settings, "SEEKDB_URL", f"sqlite+aiosqlite:///{tmp_path}/scheduler-unit.db")
    monkeypatch.setattr("agentseek_api.core.database.OceanBaseCheckpointSaver", _FakeCheckpointer)
    await db_manager.close()
    await db_manager.initialize()
    try:
        session_factory = db_manager.get_session_factory()
        due_at = datetime.now(UTC) - timedelta(minutes=1)
        async with session_factory() as session:
            assistant = Assistant(name="scheduler-unit", graph_id="default")
            session.add(assistant)
            await session.flush()
            cron = CronJob(
                assistant_id=assistant.assistant_id,
                thread_id=None,
                user_id="u1",
                schedule="FREQ=MINUTELY;INTERVAL=1",
                enabled=True,
                input_json={"kind": "unit"},
                next_run_at=due_at,
            )
            session.add(cron)
            await session.commit()
            cron_id = cron.cron_id

        first = await claim_due_crons(limit=10, scheduler_id="scheduler-a", now=due_at)
        await _mark_tick_outcome(tick_id=first[0].tick_id, status="queued", run_id="run-1")
        second = await claim_due_crons(limit=10, scheduler_id="scheduler-b", now=due_at)

        assert [item.cron_id for item in first] == [cron_id]
        assert second == []

        async with session_factory() as session:
            persisted = await session.scalar(select(CronJob).where(CronJob.cron_id == cron_id))
            ticks = list((await session.scalars(select(CronTick).where(CronTick.cron_id == cron_id))).all())
            assert persisted is not None
            assert _as_utc(persisted.next_run_at) > due_at
            assert len(ticks) == 1
            assert ticks[0].status == "queued"
            assert ticks[0].scheduler_id == "scheduler-a"
            assert _as_utc(ticks[0].scheduled_for) == due_at
    finally:
        await db_manager.close()


@pytest.mark.asyncio
async def test_claim_due_crons_only_reclaims_abandoned_started_ticks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    from agentseek_api.services.cron_scheduler import claim_due_crons
    from agentseek_api.settings import settings

    monkeypatch.setattr(settings, "SEEKDB_URL", f"sqlite+aiosqlite:///{tmp_path}/scheduler-reclaim.db")
    monkeypatch.setattr("agentseek_api.core.database.OceanBaseCheckpointSaver", _FakeCheckpointer)
    monkeypatch.setattr(settings, "REDIS_SCHEDULER_LOCK_TTL_SECONDS", 30)
    await db_manager.close()
    await db_manager.initialize()
    try:
        session_factory = db_manager.get_session_factory()
        due_at = datetime.now(UTC) - timedelta(minutes=1)
        async with session_factory() as session:
            assistant = Assistant(name="scheduler-reclaim", graph_id="default")
            session.add(assistant)
            await session.flush()
            cron = CronJob(
                assistant_id=assistant.assistant_id,
                thread_id=None,
                user_id="u1",
                schedule="FREQ=MINUTELY;INTERVAL=1",
                enabled=True,
                input_json={"kind": "unit"},
                next_run_at=due_at,
            )
            session.add(cron)
            await session.commit()
            cron_id = cron.cron_id

        first = await claim_due_crons(limit=10, scheduler_id="scheduler-a", now=due_at)
        second = await claim_due_crons(limit=10, scheduler_id="scheduler-b", now=due_at + timedelta(seconds=10))
        stale_at = due_at - timedelta(seconds=31)
        await _age_tick(cron_id=cron_id, stale_at=stale_at)
        third = await claim_due_crons(limit=10, scheduler_id="scheduler-c", now=due_at + timedelta(seconds=11))

        assert [item.tick_id for item in first] == [first[0].tick_id]
        assert second == []
        assert [item.tick_id for item in third] == [first[0].tick_id]

        async with session_factory() as session:
            persisted = await session.scalar(select(CronJob).where(CronJob.cron_id == cron_id))
            ticks = list((await session.scalars(select(CronTick).where(CronTick.cron_id == cron_id))).all())
            assert persisted is not None
            assert _as_utc(persisted.next_run_at) > due_at
            assert len(ticks) == 1
            assert ticks[0].status == "started"
            assert ticks[0].scheduler_id == "scheduler-a"
    finally:
        await db_manager.close()
