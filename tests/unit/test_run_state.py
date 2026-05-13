import asyncio

import pytest

from agentseek_api.services.run_state import RunEventBroker


@pytest.mark.asyncio
async def test_stream_yields_events_and_stops_on_end() -> None:
    broker = RunEventBroker()
    run_id = "r1"
    broker.publish(run_id, "start")
    broker.publish(run_id, "end")

    events = []
    async for event in broker.stream(run_id):
        events.append(event)
    assert events == ["start", "end"]


@pytest.mark.asyncio
async def test_stream_waits_for_future_event() -> None:
    broker = RunEventBroker()
    run_id = "r2"

    async def produce() -> None:
        await asyncio.sleep(0.01)
        broker.publish(run_id, "start")
        broker.publish(run_id, "end")

    producer = asyncio.create_task(produce())
    events = []
    async for event in broker.stream(run_id):
        events.append(event)
    await producer
    assert events == ["start", "end"]


def test_snapshot_returns_copy() -> None:
    broker = RunEventBroker()
    run_id = "r3"
    broker.publish(run_id, "start")
    snap = broker.snapshot(run_id)
    snap.append("mutate")
    assert broker.snapshot(run_id) == ["start"]


@pytest.mark.asyncio
async def test_completed_stream_can_be_replayed_for_late_subscriber() -> None:
    broker = RunEventBroker()
    run_id = "r4"
    broker.publish(run_id, "start")
    broker.publish(run_id, "end")

    first_events = []
    async for event in broker.stream(run_id):
        first_events.append(event)

    second_events = []

    async def consume_late_subscriber() -> None:
        async for event in broker.stream(run_id):
            second_events.append(event)

    await asyncio.wait_for(consume_late_subscriber(), timeout=0.1)

    assert first_events == ["start", "end"]
    assert second_events == ["start", "end"]


def test_completed_run_cache_is_bounded() -> None:
    broker = RunEventBroker(max_completed_runs=1)
    broker.publish("r5", "start")
    broker.publish("r5", "end")
    broker.publish("r6", "start")
    broker.publish("r6", "end")

    assert broker.snapshot("r5") == []
    assert broker.snapshot("r6") == ["start", "end"]
