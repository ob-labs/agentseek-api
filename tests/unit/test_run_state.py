import asyncio

import pytest

from agentseek_api.services.run_state import RunEventBroker


@pytest.mark.asyncio
async def test_stream_yields_events_and_stops_on_end() -> None:
    broker = RunEventBroker()
    run_id = "r1"
    broker.publish(run_id, "start")
    broker.publish(run_id, "end", status="success")

    events = []
    async for event in broker.stream(run_id):
        events.append(event)
    assert events == [{"event": "start"}, {"event": "end", "status": "success"}]


@pytest.mark.asyncio
async def test_stream_waits_for_future_event() -> None:
    broker = RunEventBroker()
    run_id = "r2"

    async def produce() -> None:
        await asyncio.sleep(0.01)
        broker.publish(run_id, "start")
        broker.publish(run_id, "end", status="success")

    producer = asyncio.create_task(produce())
    events = []
    async for event in broker.stream(run_id):
        events.append(event)
    await producer
    assert events == [{"event": "start"}, {"event": "end", "status": "success"}]


def test_snapshot_returns_copy() -> None:
    broker = RunEventBroker()
    run_id = "r3"
    broker.publish(run_id, "start")
    snap = broker.snapshot(run_id)
    snap.append({"event": "mutate"})
    assert broker.snapshot(run_id) == [{"event": "start"}]


@pytest.mark.asyncio
async def test_completed_stream_can_be_replayed_for_late_subscriber() -> None:
    broker = RunEventBroker()
    run_id = "r4"
    broker.publish(run_id, "start")
    broker.publish(run_id, "end", status="success")

    first_events = []
    async for event in broker.stream(run_id):
        first_events.append(event)

    second_events = []

    async def consume_late_subscriber() -> None:
        async for event in broker.stream(run_id):
            second_events.append(event)

    await asyncio.wait_for(consume_late_subscriber(), timeout=0.1)

    assert first_events == [{"event": "start"}, {"event": "end", "status": "success"}]
    assert second_events == [{"event": "start"}, {"event": "end", "status": "success"}]


def test_completed_run_cache_is_bounded() -> None:
    broker = RunEventBroker(max_completed_runs=1)
    broker.publish("r5", "start")
    broker.publish("r5", "end", status="success")
    broker.publish("r6", "start")
    broker.publish("r6", "end", status="success")

    assert broker.snapshot("r5") == []
    assert broker.snapshot("r6") == [{"event": "start"}, {"event": "end", "status": "success"}]


@pytest.mark.asyncio
async def test_start_reopens_a_completed_run_for_resume() -> None:
    broker = RunEventBroker()
    broker.publish("r7", "start")
    broker.publish("r7", "end", status="interrupted")
    broker.publish("r7", "start")
    broker.publish("r7", "end", status="success")

    events = []
    async for event in broker.stream("r7"):
        events.append(event)

    assert events == [
        {"event": "start"},
        {"event": "end", "status": "interrupted"},
        {"event": "start"},
        {"event": "end", "status": "success"},
    ]
