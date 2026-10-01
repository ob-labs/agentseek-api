import asyncio

import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from agentseek_api.core.database import db_manager
from agentseek_api.core.orm import Base, RunStreamEvent
from agentseek_api.services import run_jobs, stream_persistence, thread_protocol
from agentseek_api.services.run_state import RunEventBroker
from agentseek_api.settings import settings


@pytest_asyncio.fixture
async def durability_db(tmp_path, monkeypatch):
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{tmp_path}/durability.db",
        pool_size=1, max_overflow=0, pool_timeout=0.1,
    )
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    monkeypatch.setattr(db_manager, "get_engine", lambda: engine)
    monkeypatch.setattr(db_manager, "get_session_factory", lambda: factory)
    monkeypatch.setattr(settings, "EXECUTOR_BACKEND", "inline")
    monkeypatch.setattr(run_jobs, "run_broker", RunEventBroker())
    monkeypatch.setattr(thread_protocol, "thread_protocol_broker", thread_protocol.ThreadProtocolEventBroker())
    yield factory
    await engine.dispose()


async def test_first_append_uses_one_pool_connection(durability_db):
    seq, _ = await stream_persistence.append_run_stream_event_atomic("pool", {"event": "start"})
    assert seq == 1
    async with durability_db() as session:
        assert list(await session.scalars(select(RunStreamEvent.seq))) == [1]


async def test_missing_counter_concurrent_recovery_reloads_winner(durability_db):
    async with durability_db() as session:
        session.add(RunStreamEvent(run_id="recovery", seq=10, event="start", payload_json={"event": "start"}))
        await session.commit()
    results = await asyncio.gather(*(
        stream_persistence.append_run_stream_event_atomic("recovery", {"event": "message_chunk", "content": str(i)})
        for i in range(2)
    ))
    assert sorted(seq for seq, _ in results) == [11, 12]


async def test_buffer_exposes_no_event_until_its_batch_commits(durability_db):
    # Seed first so this regression isolates exposure ordering from pool seeding.
    from agentseek_api.core.orm import StreamSequence
    async with durability_db() as session:
        session.add(StreamSequence(scope="run", scope_id="buffer", seq=0))
        await session.commit()
    async with stream_persistence.buffered_stream_persistence(run_id="buffer", thread_id="thread"):
        await run_jobs._publish_run_event("buffer", "message_chunk", content="first")
        assert run_jobs.run_broker.snapshot_records("buffer") == []
    records = run_jobs.run_broker.snapshot_records("buffer")
    assert records == [(1, {"event": "message_chunk", "content": "first"})]
    async with durability_db() as session:
        rows = list(await session.scalars(select(RunStreamEvent)))
        assert [(row.seq, row.payload_json) for row in rows] == records


def test_broker_preserves_durable_cursor_when_notification_arrives_late():
    broker = RunEventBroker()
    broker.publish("run", "message_chunk", seq=2, content="second")
    broker.publish("run", "message_chunk", seq=1, content="first")
    broker.publish("run", "message_chunk", seq=2, content="second")
    assert broker.snapshot_records("run") == [
        (1, {"event": "message_chunk", "content": "first"}),
        (2, {"event": "message_chunk", "content": "second"}),
    ]


async def test_append_retries_sqlite_busy_in_a_fresh_transaction(durability_db, monkeypatch):
    from sqlalchemy.exc import OperationalError

    original = stream_persistence._stage_db_event
    sessions = []

    async def busy_once(session, *args, **kwargs):
        sessions.append(session)
        if len(sessions) == 1:
            raise OperationalError("INSERT", {}, Exception("database is locked"))
        return await original(session, *args, **kwargs)

    monkeypatch.setattr(stream_persistence, "_stage_db_event", busy_once)
    seq, _ = await stream_persistence.append_run_stream_event_atomic("retry", {"event": "start"})
    assert seq == 1
    assert len(sessions) == 2 and sessions[0] is not sessions[1]
