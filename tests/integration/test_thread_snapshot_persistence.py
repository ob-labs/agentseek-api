from contextlib import contextmanager

import pytest
import pytest_asyncio
from sqlalchemy import event, insert, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from agentseek_api.core.database import db_manager
from agentseek_api.core.orm import ThreadStreamEvent
from agentseek_api.services import run_jobs
from agentseek_api.services.thread_protocol import ThreadProtocolEventBroker
from agentseek_api.settings import settings


@pytest_asyncio.fixture
async def snapshot_db(tmp_path, monkeypatch):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/snapshot.db")
    async with engine.begin() as connection:
        await connection.run_sync(ThreadStreamEvent.__table__.create)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    broker = ThreadProtocolEventBroker()
    monkeypatch.setattr(settings, "EXECUTOR_BACKEND", "inline")
    monkeypatch.setattr(db_manager, "get_engine", lambda: engine)
    monkeypatch.setattr(db_manager, "get_session_factory", lambda: factory)
    monkeypatch.setattr(run_jobs, "thread_protocol_broker", broker)
    try:
        yield engine, factory, broker
    finally:
        await engine.dispose()


@contextmanager
def capture_database_calls(engine):
    calls = {"statements": [], "commits": 0}

    def before_execute(
        _connection, _cursor, statement, _parameters, _context, _executemany
    ):
        calls["statements"].append(statement)

    def on_commit(_connection):
        calls["commits"] += 1

    event.listen(engine.sync_engine, "before_cursor_execute", before_execute)
    event.listen(engine.sync_engine, "commit", on_commit)
    try:
        yield calls
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", before_execute)
        event.remove(engine.sync_engine, "commit", on_commit)


async def read_events(factory, thread_id="thread-1"):
    async with factory() as session:
        rows = await session.scalars(
            select(ThreadStreamEvent)
            .where(ThreadStreamEvent.thread_id == thread_id)
            .order_by(ThreadStreamEvent.seq)
        )
        return [row.payload_json for row in rows]


@pytest.mark.parametrize("event_count,query_budget", [(270, 4), (1200, 8)])
@pytest.mark.parametrize("stored_fraction", [0, 0.5, 1])
async def test_snapshot_recovers_missing_history_with_bounded_database_calls(
    snapshot_db,
    event_count,
    query_budget,
    stored_fraction,
):
    engine, factory, broker = snapshot_db
    events = [
        broker.publish(
            "thread-1", {"method": "values", "params": {"data": i}}, persist=False
        )
        for i in range(event_count)
    ]
    stored_count = int(event_count * stored_fraction)
    # Existing rows must remain unchanged, even if the in-memory payload differs.
    saved_events = [{**item, "saved": True} for item in events[:stored_count]]
    async with factory() as session:
        if saved_events:
            await session.execute(
                insert(ThreadStreamEvent),
                [
                    {
                        "thread_id": "thread-1",
                        "seq": item["seq"],
                        "method": "values",
                        "payload_json": item,
                    }
                    for item in saved_events
                ],
            )
        # The same sequence on another thread must not suppress this thread's event.
        await session.execute(
            insert(ThreadStreamEvent),
            [
                {
                    "thread_id": "thread-2",
                    "seq": event_count,
                    "method": "values",
                    "payload_json": {"other": True},
                }
            ],
        )
        await session.commit()

    with capture_database_calls(engine) as calls:
        await run_jobs._persist_thread_snapshot("thread-1")

    assert await read_events(factory) == saved_events + events[stored_count:]
    assert await read_events(factory, "thread-2") == [{"other": True}]
    assert len(calls["statements"]) <= query_budget, calls
    assert calls["commits"] <= 3, calls


async def test_snapshot_without_events_does_not_access_database(snapshot_db):
    engine, _, _ = snapshot_db
    with capture_database_calls(engine) as calls:
        await run_jobs._persist_thread_snapshot("thread-1")
    assert calls == {"statements": [], "commits": 0}


async def test_snapshot_recovers_remaining_events_after_concurrent_duplicate(
    snapshot_db, monkeypatch
):
    engine, factory, broker = snapshot_db
    events = [
        broker.publish(
            "thread-1", {"method": "values", "params": {"data": i}}, persist=False
        )
        for i in range(3)
    ]
    raced_event = {**events[0], "concurrent": True}
    raced = False

    class RacingSession(AsyncSession):
        async def execute(self, statement, *args, **kwargs):
            nonlocal raced
            if statement.is_insert and not raced:
                raced = True
                # A background publisher commits after the snapshot's lookup.
                async with factory() as other:
                    await other.execute(
                        insert(ThreadStreamEvent),
                        [
                            {
                                "thread_id": "thread-1",
                                "seq": 1,
                                "method": "values",
                                "payload_json": raced_event,
                            }
                        ],
                    )
                    await other.commit()
            return await super().execute(statement, *args, **kwargs)

    racing_factory = async_sessionmaker(
        engine, class_=RacingSession, expire_on_commit=False
    )
    monkeypatch.setattr(db_manager, "get_session_factory", lambda: racing_factory)
    await run_jobs._persist_thread_snapshot("thread-1")

    assert raced
    assert await read_events(factory) == [raced_event, *events[1:]]


async def test_snapshot_retries_events_after_batch_commit_failure(
    snapshot_db, monkeypatch
):
    engine, factory, broker = snapshot_db
    events = [
        broker.publish(
            "thread-1", {"method": "values", "params": {"data": i}}, persist=False
        )
        for i in range(3)
    ]
    failed = False

    class FailOnceSession(AsyncSession):
        async def commit(self):
            nonlocal failed
            if not failed:
                failed = True
                raise RuntimeError("temporary commit failure")
            await super().commit()

    failing_factory = async_sessionmaker(
        engine, class_=FailOnceSession, expire_on_commit=False
    )
    monkeypatch.setattr(db_manager, "get_session_factory", lambda: failing_factory)
    await run_jobs._persist_thread_snapshot("thread-1")

    assert failed
    assert await read_events(factory) == events


@pytest.mark.parametrize("backend", ["redis", "uninitialized"])
async def test_snapshot_skips_unavailable_sql_storage(
    snapshot_db, monkeypatch, backend
):
    engine, factory, broker = snapshot_db
    broker.publish("thread-1", {"method": "values"}, persist=False)
    if backend == "redis":
        monkeypatch.setattr(settings, "EXECUTOR_BACKEND", "redis")
    else:

        def unavailable_engine():
            raise RuntimeError("not initialized")

        monkeypatch.setattr(db_manager, "get_engine", unavailable_engine)

    with capture_database_calls(engine) as calls:
        await run_jobs._persist_thread_snapshot("thread-1")
    assert calls == {"statements": [], "commits": 0}
    assert await read_events(factory) == []


async def test_snapshot_ignores_invalid_sequences_and_preserves_first_duplicate(
    snapshot_db,
):
    _, factory, broker = snapshot_db
    broker.publish("thread-1", {"method": "values"}, seq=0, persist=False)
    first = broker.publish(
        "thread-1",
        {"method": "values", "params": {"data": "first"}},
        seq=1,
        persist=False,
    )
    broker.publish(
        "thread-1",
        {"method": "values", "params": {"data": "second"}},
        seq=1,
        persist=False,
    )

    await run_jobs._persist_thread_snapshot("thread-1")

    assert await read_events(factory) == [first]
