import asyncio
from collections import Counter

import pytest
import pytest_asyncio
from sqlalchemy import event, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from agentseek_api.core.database import db_manager
from agentseek_api.core.orm import Base, Run, RunStreamEvent, Thread, ThreadStreamEvent
from agentseek_api.services import run_jobs, stream_persistence, thread_protocol
from agentseek_api.services.run_state import RunEventBroker
from agentseek_api.settings import settings


@pytest_asyncio.fixture
async def stream_db(tmp_path, monkeypatch):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/stream.db")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    # A small replay window proves that SQL buffering does not rely on retained history.
    broker = thread_protocol.ThreadProtocolEventBroker(max_events_per_thread=16)
    monkeypatch.setattr(settings, "EXECUTOR_BACKEND", "inline")
    monkeypatch.setattr(db_manager, "get_engine", lambda: engine)
    monkeypatch.setattr(db_manager, "get_session_factory", lambda: factory)
    monkeypatch.setattr(run_jobs, "run_broker", RunEventBroker())
    monkeypatch.setattr(run_jobs, "thread_protocol_broker", broker)
    monkeypatch.setattr(thread_protocol, "thread_protocol_broker", broker)
    async with factory() as session:
        session.add(Thread(thread_id="thread-1", user_id="user-1"))
        session.add(
            Run(
                run_id="run-1",
                thread_id="thread-1",
                assistant_id="default",
                user_id="user-1",
                status="pending",
            )
        )
        await session.commit()
    try:
        yield engine, factory, broker
    finally:
        await engine.dispose()


async def read_payloads(factory, model):
    async with factory() as session:
        rows = await session.scalars(select(model).order_by(model.seq))
        return [row.payload_json for row in rows]


async def test_inline_job_batches_live_writes_without_losing_evicted_history(
    stream_db, monkeypatch
):
    engine, factory, broker = stream_db
    calls = Counter()

    def count_sql(_conn, _cursor, statement, _params, _context, _many):
        calls[statement.split()[0]] += 1

    def count_commit(_conn):
        calls["COMMIT"] += 1

    async def emit_events(**_kwargs):
        for i in range(270):
            await run_jobs._publish_run_event("run-1", "message_chunk", content=str(i))
            await thread_protocol.apublish_values_event("thread-1", values={"index": i})
        return run_jobs.RunExecutionResult(
            output={"count": 270}, interrupted=False, interrupts=[]
        )

    monkeypatch.setattr(run_jobs, "execute_run", emit_events)
    event.listen(engine.sync_engine, "before_cursor_execute", count_sql)
    event.listen(engine.sync_engine, "commit", count_commit)
    try:
        await run_jobs.execute_run_job(
            run_jobs.RunExecutionJob(
                run_id="run-1",
                thread_id="thread-1",
                user_id="user-1",
                payload={},
                graph_id="default",
            )
        )
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", count_sql)
        event.remove(engine.sync_engine, "commit", count_commit)

    run_events = await read_payloads(factory, RunStreamEvent)
    thread_events = await read_payloads(factory, ThreadStreamEvent)
    assert [
        item["content"] for item in run_events if item["event"] == "message_chunk"
    ] == [str(i) for i in range(270)]
    assert [
        item["params"]["data"]["index"]
        for item in thread_events
        if item["method"] == "values"
    ] == list(range(270))
    assert run_events[-1] == {"event": "end", "status": "success"}
    assert thread_events[-1]["params"]["data"]["event"] == "completed"
    assert len(broker.snapshot_records("thread-1")) == 16
    async with factory() as session:
        run = await session.get(Run, "run-1")
        assert run.status == "success" and run.output_json == {"count": 270}
    assert calls["INSERT"] < 30, calls
    assert calls["SELECT"] < 40, calls
    assert calls["COMMIT"] < 15, calls


async def test_buffer_keeps_payload_at_publication_time_and_flushes_before_exit(
    stream_db,
):
    _, factory, _ = stream_db
    payload = {"event": "message_chunk", "data": {"text": "original"}}
    async with stream_persistence.buffered_stream_persistence(
        run_id="run-1", thread_id="thread-1"
    ):
        await stream_persistence.persist_run_stream_event(
            "run-1", seq=1, payload=payload
        )
        payload["data"]["text"] = "mutated"
    assert await read_payloads(factory, RunStreamEvent) == [
        {"event": "message_chunk", "data": {"text": "original"}}
    ]


async def test_replay_preserves_saved_payload_after_producer_mutates_live_value(
    stream_db, monkeypatch
):
    import json

    from agentseek_api.api import streaming
    from agentseek_api.models.auth import User
    from agentseek_api.models.protocol import ProtocolEventStreamRequest

    _, factory, broker = stream_db
    monkeypatch.setattr(streaming, "thread_protocol_broker", broker)
    values = {"messages": [{"text": "original"}]}
    async with stream_persistence.buffered_stream_persistence(
        run_id="run-1", thread_id="thread-1"
    ):
        await thread_protocol.apublish_values_event("thread-1", values=values)
        values["messages"][0]["text"] = "mutated"
    saved = await read_payloads(factory, ThreadStreamEvent)
    assert saved[0]["params"]["data"]["messages"][0]["text"] == "original"
    response = await streaming.stream_thread_protocol_events(
        "thread-1",
        ProtocolEventStreamRequest(channels=["values"]),
        user=User(identity="user-1"),
        last_event_id=None,
    )
    chunks = [chunk async for chunk in response.body_iterator]
    assert len(chunks) == 1
    replayed = json.loads(chunks[0].split("data: ", 1)[1])
    assert replayed == saved[0]


async def test_buffer_flushes_before_graph_finishes(stream_db):
    _, factory, _ = stream_db
    async with stream_persistence.buffered_stream_persistence(
        run_id="run-1", thread_id="thread-1"
    ):
        await stream_persistence.persist_run_stream_event(
            "run-1", seq=1, payload={"event": "message_chunk"}
        )
        async with asyncio.timeout(2):
            while not await read_payloads(factory, RunStreamEvent):
                await asyncio.sleep(0.01)
    assert await read_payloads(factory, RunStreamEvent) == [{"event": "message_chunk"}]


@pytest.mark.parametrize("failure", [RuntimeError, asyncio.CancelledError])
async def test_buffer_flushes_when_execution_raises(stream_db, failure):
    _, factory, _ = stream_db
    with pytest.raises(failure):
        async with stream_persistence.buffered_stream_persistence(
            run_id="run-1", thread_id="thread-1"
        ):
            await stream_persistence.persist_run_stream_event(
                "run-1", seq=1, payload={"event": "message_chunk"}
            )
            raise failure()
    assert await read_payloads(factory, RunStreamEvent) == [{"event": "message_chunk"}]


@pytest.mark.parametrize("failure", ["duplicate", "commit", "lookup"])
async def test_live_batch_recovers_after_transaction_failure(
    stream_db, monkeypatch, failure
):
    from sqlalchemy import insert
    from sqlalchemy.ext.asyncio import AsyncSession

    engine, factory, _ = stream_db
    failed = False

    class FailingSession(AsyncSession):
        async def execute(self, statement, *args, **kwargs):
            nonlocal failed
            if failure == "duplicate" and statement.is_insert and not failed:
                failed = True
                async with factory() as other:
                    await other.execute(
                        insert(RunStreamEvent),
                        [
                            {
                                "run_id": "run-1",
                                "seq": 1,
                                "event": "message_chunk",
                                "payload_json": {
                                    "event": "message_chunk",
                                    "content": "concurrent",
                                },
                            }
                        ],
                    )
                    await other.commit()
            return await super().execute(statement, *args, **kwargs)

        async def commit(self):
            nonlocal failed
            if failure == "commit" and not failed:
                failed = True
                raise RuntimeError("temporary commit failure")
            await super().commit()

        async def scalars(self, statement, *args, **kwargs):
            nonlocal failed
            if failure == "lookup" and not failed:
                failed = True
                raise RuntimeError("temporary lookup failure")
            return await super().scalars(statement, *args, **kwargs)

    failing_factory = async_sessionmaker(
        engine, class_=FailingSession, expire_on_commit=False
    )
    monkeypatch.setattr(db_manager, "get_session_factory", lambda: failing_factory)
    async with stream_persistence.buffered_stream_persistence(
        run_id="run-1", thread_id="thread-1"
    ):
        for seq in (1, 2):
            await stream_persistence.persist_run_stream_event(
                "run-1",
                seq=seq,
                payload={"event": "message_chunk", "content": str(seq)},
            )
        await stream_persistence.persist_thread_stream_event(
            "thread-1", {"seq": 1, "method": "values", "params": {"data": "saved"}}
        )
    assert failed
    assert await read_payloads(factory, RunStreamEvent) == [
        {
            "event": "message_chunk",
            "content": "concurrent" if failure == "duplicate" else "1",
        },
        {"event": "message_chunk", "content": "2"},
    ]
    assert await read_payloads(factory, ThreadStreamEvent) == [
        {"seq": 1, "method": "values", "params": {"data": "saved"}}
    ]


async def test_independent_runs_do_not_flush_each_others_buffers(
    stream_db, monkeypatch
):
    from functools import partial
    from agentseek_api.services.stream_event_buffer import StreamEventBuffer

    _, factory, _ = stream_db
    monkeypatch.setattr(
        stream_persistence,
        "StreamEventBuffer",
        partial(StreamEventBuffer, flush_interval=60),
    )
    ready = [asyncio.Event(), asyncio.Event()]
    release = [asyncio.Event(), asyncio.Event()]

    async def execute(index):
        async with stream_persistence.buffered_stream_persistence(
            run_id=f"run-{index + 1}", thread_id=f"thread-{index + 1}"
        ):
            await stream_persistence.persist_run_stream_event(
                f"run-{index + 1}",
                seq=1,
                payload={"event": "message_chunk", "index": index},
            )
            ready[index].set()
            await release[index].wait()

    tasks = [asyncio.create_task(execute(index)) for index in range(2)]
    try:
        await asyncio.wait_for(asyncio.gather(*(item.wait() for item in ready)), 1)
        release[0].set()
        await tasks[0]
        assert await read_payloads(factory, RunStreamEvent) == [
            {"event": "message_chunk", "index": 0}
        ]
    finally:
        for item in release:
            item.set()
        await asyncio.gather(*tasks)
    assert len(await read_payloads(factory, RunStreamEvent)) == 2


async def test_live_event_delivery_does_not_wait_for_sql(stream_db, monkeypatch):
    from functools import partial
    from agentseek_api.services.stream_event_buffer import StreamEventBuffer

    engine, factory, _ = stream_db
    monkeypatch.setattr(
        stream_persistence,
        "StreamEventBuffer",
        partial(StreamEventBuffer, flush_interval=60),
    )

    sql_calls = []

    def unexpected_sql(*_args):
        sql_calls.append(True)

    async with stream_persistence.buffered_stream_persistence(
        run_id="run-1", thread_id="thread-1"
    ):
        event.listen(engine.sync_engine, "before_cursor_execute", unexpected_sql)
        try:
            await run_jobs._publish_run_event("run-1", "message_chunk", content="hello")
            stream = run_jobs.run_broker.stream_records("run-1")
            try:
                assert await anext(stream) == (
                    1,
                    {"event": "message_chunk", "content": "hello"},
                )
            finally:
                await stream.aclose()
            assert sql_calls == []
        finally:
            event.remove(engine.sync_engine, "before_cursor_execute", unexpected_sql)
    assert await read_payloads(factory, RunStreamEvent) == [
        {"event": "message_chunk", "content": "hello"}
    ]


async def test_late_child_and_unrelated_stream_use_immediate_persistence(
    stream_db, monkeypatch
):
    from functools import partial
    from agentseek_api.services.stream_event_buffer import StreamEventBuffer

    _, factory, _ = stream_db
    monkeypatch.setattr(
        stream_persistence,
        "StreamEventBuffer",
        partial(StreamEventBuffer, flush_interval=60),
    )
    release = asyncio.Event()

    async def late_publish():
        await release.wait()
        await stream_persistence.persist_run_stream_event(
            "run-1", seq=2, payload={"event": "late"}
        )

    async with stream_persistence.buffered_stream_persistence(
        run_id="run-1", thread_id="thread-1"
    ):
        await stream_persistence.persist_run_stream_event(
            "run-1", seq=1, payload={"event": "buffered"}
        )
        await stream_persistence.persist_run_stream_event(
            "other-run", seq=1, payload={"event": "unrelated"}
        )
        assert await read_payloads(factory, RunStreamEvent) == [{"event": "unrelated"}]
        child = asyncio.create_task(late_publish())
    release.set()
    await child
    payloads = await read_payloads(factory, RunStreamEvent)
    assert sorted(item["event"] for item in payloads) == [
        "buffered",
        "late",
        "unrelated",
    ]


@pytest.mark.parametrize("route", ["protocol", "thread", "run"])
async def test_replay_orders_buffered_events_before_later_persisted_events(
    stream_db, monkeypatch, route
):
    from datetime import UTC, datetime
    from functools import partial

    from agentseek_api.api import runs, streaming, threads
    from agentseek_api.models.api import RunRead
    from agentseek_api.models.auth import User
    from agentseek_api.models.protocol import ProtocolEventStreamRequest
    from agentseek_api.services.stream_event_buffer import StreamEventBuffer

    _, factory, broker = stream_db
    for module in (runs, streaming, threads):
        monkeypatch.setattr(module, "thread_protocol_broker", broker)
    monkeypatch.setattr(
        stream_persistence,
        "StreamEventBuffer",
        partial(StreamEventBuffer, flush_interval=60),
    )
    published = asyncio.Event()
    release = asyncio.Event()

    async def first_run():
        async with stream_persistence.buffered_stream_persistence(
            run_id="run-1", thread_id="thread-1"
        ):
            await thread_protocol.apublish_values_event(
                "thread-1", values={"index": 1}, run_id="run-1"
            )
            published.set()
            await release.wait()

    task = asyncio.create_task(first_run())
    try:
        await asyncio.wait_for(published.wait(), 1)
        # A different task has no buffer, so seq 2 reaches SQL before seq 1.
        await thread_protocol.apublish_values_event(
            "thread-1", values={"index": 2}, run_id="run-1"
        )
        assert [
            item["seq"] for item in await read_payloads(factory, ThreadStreamEvent)
        ] == [2]
        user = User(identity="user-1")
        if route == "protocol":
            response = await streaming.stream_thread_protocol_events(
                "thread-1",
                ProtocolEventStreamRequest(channels=["values"]),
                user=user,
                last_event_id=None,
            )
        elif route == "thread":
            response = await threads.join_thread_stream(
                "thread-1", user=user, last_event_id=None
            )
        else:
            now = datetime.now(UTC)
            response = runs._build_create_run_stream_response(
                thread_id="thread-1",
                created=RunRead(
                    run_id="run-1",
                    thread_id="thread-1",
                    assistant_id="default",
                    status="running",
                    output=None,
                    created_at=now,
                    updated_at=now,
                ),
                user=user,
                stream_modes=["values"],
                after_seq=0,
                location="/runs/run-1",
                content_location="/runs/run-1/stream",
                include_metadata=False,
            )
        stream = response.body_iterator
        try:
            async with asyncio.timeout(2):
                if route == "thread":
                    assert await anext(stream) == ": stream-open\n\n"
                assert (await anext(stream)).startswith("id: 1\n")
                assert (await anext(stream)).startswith("id: 2\n")
        finally:
            await stream.aclose()
    finally:
        release.set()
        await task
    assert [
        item["seq"] for item in await read_payloads(factory, ThreadStreamEvent)
    ] == [1, 2]
