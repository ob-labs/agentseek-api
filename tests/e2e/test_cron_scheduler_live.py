"""Real-DB (seekdb / OceanBase / MySQL) coverage for cron behaviors that the
HTTP surface cannot reach: the additive startup migration's ``ALTER TABLE ADD
COLUMN`` DDL on a legacy table, and the scheduler's cross-table cascade delete.

These exercise the two most backend-DDL/SQL-sensitive paths in the change, which
the unit/integration tests only prove on SQLite. The scheduler functions and the
migration helper operate through the in-process ``db_manager`` (wired to the real
backend by the ``e2e_db`` fixture), so they can be invoked directly — the HTTP
server runs as a separate subprocess and the scheduler is a separate process.

The tests are ``async def`` (not ``asyncio.run`` inside a sync test) so they run
on pytest-asyncio's event loop — the same loop ``e2e_db`` bound the engine pool
to. Mixing loops would raise "attached to a different loop".

Deliberately out of scope here: the scheduler's ``with_for_update`` row-locking
under concurrent schedulers. That is a no-op on SQLite and only meaningful on a
real backend, but reproducing a true double-claim race needs two concurrent
scheduler loops against the same DB — too flaky for CI. The ``end_time``
predicate added in this change is a plain ``WHERE`` filter inside the existing
``FOR UPDATE`` query and does not alter the locking semantics, so the locking
path is validated by the existing checkpoint/runtime suites, not re-proven here.
"""

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import Column, Integer, String, inspect, select, text

from agentseek_api.core.database import DatabaseManager, db_manager
from agentseek_api.core.orm import Base, CronJob, CronTick, Run, Thread


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_additive_migration_adds_column_on_real_db(e2e_db: None) -> None:
    """The migration's ALTER TABLE ADD COLUMN must execute on the real engine,
    not just SQLite. Create a legacy table missing a column, seed a row, run the
    real migration, and assert the column was added + backfilled on the live
    backend."""
    table_name = "cron_migration_e2e_probe"

    # Register a throwaway mapped table with a column we withhold from the
    # initial CREATE, so _apply_additive_migrations must ADD it on the real DB.
    probe = type(
        "CronMigrationProbe",
        (Base,),
        {
            "__tablename__": table_name,
            "id": Column(Integer, primary_key=True, autoincrement=True),
            "added_col": Column(
                "added_col", String(16), nullable=False, server_default=text("'present'")
            ),
        },
    )
    engine = db_manager.get_engine()
    try:
        async with engine.begin() as conn:
            await conn.run_sync(lambda c: probe.__table__.drop(c, checkfirst=True))
            # Legacy table: id only, no added_col, with an existing row.
            await conn.execute(text(f"CREATE TABLE {table_name} (id INTEGER PRIMARY KEY)"))
            await conn.execute(text(f"INSERT INTO {table_name} (id) VALUES (1)"))

        # The real migration runs the real dialect's ADD COLUMN DDL.
        async with engine.begin() as conn:
            await conn.run_sync(DatabaseManager._apply_additive_migrations)

        async with engine.connect() as conn:
            columns = await conn.run_sync(
                lambda c: {col["name"] for col in inspect(c).get_columns(table_name)}
            )
            assert "added_col" in columns
            backfilled = (
                await conn.execute(text(f"SELECT added_col FROM {table_name} WHERE id = 1"))
            ).scalar()
            # server_default backfilled the legacy row on the real backend.
            assert backfilled == "present"
    finally:
        async with engine.begin() as conn:
            await conn.run_sync(lambda c: probe.__table__.drop(c, checkfirst=True))
        Base.metadata.remove(probe.__table__)


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_reconcile_cascade_deletes_stateless_thread_on_real_db(e2e_db: None) -> None:
    """on_run_completed='delete' deletes the ephemeral stateless thread AND its
    runs via delete_threads_cascade (multi-row IN delete + cross-table). Prove
    that cascade SQL executes on the real backend, not just SQLite."""
    from agentseek_api.services import cron_scheduler as cron_scheduler_module

    session_factory = db_manager.get_session_factory()
    user = "cron-sched-e2e"

    async def _no_sleep(_: float) -> None:
        return None

    # Seed a stateless cron (thread_id NULL) + an ephemeral thread + a successful
    # run + a queued tick pointing at it — the exact post-dispatch state the
    # reconcile loop acts on when on_run_completed='delete'.
    async with session_factory() as session:
        cron = CronJob(
            assistant_id="assistant-e2e",
            thread_id=None,
            user_id=user,
            schedule="FREQ=MINUTELY;INTERVAL=1",
            enabled=True,
            input_json={"kind": "sched-e2e"},
            metadata_json={},
            kwargs_json={"config": {}, "context": {}, "stream_modes": ["values"]},
            on_run_completed="delete",
            next_run_at=datetime.now(UTC) + timedelta(minutes=5),
        )
        session.add(cron)
        await session.flush()
        cron_id = cron.cron_id

        thread = Thread(
            user_id=user,
            metadata_json={"stateless": True, "cron_id": cron_id},
            status="idle",
        )
        session.add(thread)
        await session.flush()
        thread_id = thread.thread_id

        run = Run(
            thread_id=thread_id,
            assistant_id="assistant-e2e",
            user_id=user,
            status="success",
            input_json={"kind": "sched-e2e"},
        )
        session.add(run)
        await session.flush()
        run_id = run.run_id

        tick = CronTick(
            cron_id=cron_id,
            thread_id=thread_id,
            run_id=run_id,
            scheduler_id="scheduler-e2e",
            scheduled_for=datetime.now(UTC) - timedelta(minutes=1),
            status="queued",
        )
        session.add(tick)
        await session.commit()

    try:
        # Real reconcile: promotes the queued tick to success and runs the cascade.
        await cron_scheduler_module._reconcile_terminal_ticks(sleep=_no_sleep)

        async with session_factory() as session:
            assert await session.scalar(select(Thread).where(Thread.thread_id == thread_id)) is None
            assert await session.scalar(select(Run).where(Run.run_id == run_id)) is None
            persisted_tick = await session.scalar(select(CronTick).where(CronTick.cron_id == cron_id))
            assert persisted_tick is not None
            assert persisted_tick.status == "success"
    finally:
        async with session_factory() as session:
            await session.execute(text("DELETE FROM cron_ticks WHERE cron_id = :c"), {"c": cron_id})
            await session.execute(text("DELETE FROM cron_jobs WHERE cron_id = :c"), {"c": cron_id})
            await session.commit()
