import asyncio

import pytest

from agentseek_api.services.stream_event_buffer import StreamEvent, StreamEventBuffer


def record(seq, text="text"):
    return StreamEvent("run", "run-1", seq, {"text": text})


async def test_count_limit_applies_backpressure_to_concurrent_producers():
    entered = asyncio.Event()
    release = asyncio.Event()
    batches = []

    async def write(records):
        entered.set()
        await release.wait()
        batches.append(list(records))

    async with StreamEventBuffer(
        write, run_id="run-1", thread_id="thread-1", max_events=2, flush_interval=60
    ) as buffer:
        assert await buffer.append(record(1))
        producer = asyncio.create_task(buffer.append(record(2)))
        await asyncio.wait_for(entered.wait(), 1)
        waiting_producer = asyncio.create_task(buffer.append(record(3)))
        await asyncio.sleep(0)
        assert not producer.done() and not waiting_producer.done()
        release.set()
        assert await producer and await waiting_producer
    assert [[item.seq for item in batch] for batch in batches] == [[1, 2], [3]]


async def test_byte_limit_flushes_before_overflow_and_writes_oversized_event_alone():
    batches = []

    async def write(records):
        batches.append(list(records))

    async with StreamEventBuffer(
        write, run_id="run-1", thread_id="thread-1", max_bytes=40, flush_interval=60
    ) as buffer:
        await buffer.append(record(1, "a" * 10))
        await buffer.append(record(2, "b" * 10))
        await buffer.append(record(3, "c" * 100))
        await buffer.append(record(4, "d"))
    assert [[item.seq for item in batch] for batch in batches] == [[1], [2], [3], [4]]


async def test_cancellation_during_periodic_write_retries_retained_batch_on_close():
    entered = asyncio.Event()
    interrupted = asyncio.Event()
    batches = []
    attempts = 0

    async def write(records):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                interrupted.set()
        batches.append(list(records))

    async with StreamEventBuffer(
        write, run_id="run-1", thread_id="thread-1", flush_interval=0.001
    ) as buffer:
        await buffer.append(record(1))
        await asyncio.wait_for(entered.wait(), 1)
        buffer._worker.cancel()
        await asyncio.wait_for(interrupted.wait(), 1)
    assert interrupted.is_set()
    assert [[item.seq for item in batch] for batch in batches] == [[1]]


async def test_repeated_cancellation_waits_for_drain_and_propagates():
    draining = asyncio.Event()
    release = asyncio.Event()
    saved = []

    async def write(records):
        draining.set()
        await release.wait()
        saved.extend(item.seq for item in records)

    async def execute():
        async with StreamEventBuffer(
            write, run_id="run-1", thread_id="thread-1", flush_interval=60
        ) as buffer:
            await buffer.append(record(1))

    task = asyncio.create_task(execute())
    await asyncio.wait_for(draining.wait(), 1)
    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert saved == [1]
    assert not any(
        item.get_name() in {"stream-event-flush", "stream-event-drain"}
        for item in asyncio.all_tasks()
    )


async def test_closed_buffer_rejects_late_child_task_events():
    saved = []

    async def write(records):
        saved.extend(item.seq for item in records)

    async with StreamEventBuffer(write, run_id="run-1", thread_id="thread-1") as buffer:
        await buffer.append(record(1))
    assert not await buffer.append(record(2))
    assert saved == [1]


async def test_normal_close_does_not_abort_an_in_flight_database_write():
    entered = asyncio.Event()
    release = asyncio.Event()
    interrupted = []
    saved = []
    attempts = 0

    async def write(records):
        nonlocal attempts
        attempts += 1
        entered.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            interrupted.append(True)
            raise
        saved.extend(item.seq for item in records)

    buffer = StreamEventBuffer(
        write, run_id="run-1", thread_id="thread-1", flush_interval=0.001
    )
    await buffer.__aenter__()
    await buffer.append(record(1))
    await asyncio.wait_for(entered.wait(), 1)
    closing = asyncio.create_task(buffer.__aexit__(None, None, None))
    try:
        # Let the close task, drain task, and any cancelled writer all run.
        for _ in range(5):
            await asyncio.sleep(0)
        assert not closing.done()
    finally:
        release.set()
        await closing
    assert interrupted == []
    assert attempts == 1
    assert saved == [1]
