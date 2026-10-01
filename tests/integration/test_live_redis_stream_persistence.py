import json
import os
from uuid import uuid4

import pytest
from redis.asyncio import from_url

from agentseek_api.services import stream_persistence as stream_module


_TEST_REDIS_URL = os.getenv("AGENTSEEK_TEST_REDIS_URL")
pytestmark = pytest.mark.skipif(
    not _TEST_REDIS_URL,
    reason="AGENTSEEK_TEST_REDIS_URL is required for live Redis script tests",
)


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param(
            {
                "method": "messages/partial",
                "params": {
                    "data": {
                        "tool_calls": [],
                        "invalid_tool_calls": [],
                    }
                },
            },
            id="message-with-empty-arrays",
        ),
        pytest.param({}, id="empty-object"),
        pytest.param(
            {
                "type": "payload-event",
                "event_id": "payload-event-id",
                "seq": 0,
                "method": "values",
            },
            id="reserved-envelope-fields",
        ),
    ],
)
@pytest.mark.asyncio
async def test_thread_stream_lua_splices_header_without_reencoding_payload(
    monkeypatch: pytest.MonkeyPatch,
    payload: dict[str, object],
) -> None:
    assert _TEST_REDIS_URL is not None
    redis = from_url(_TEST_REDIS_URL, decode_responses=True)
    thread_id = 'thread-"' + "\\" + uuid4().hex[:8]
    sequence_key = f"agentseek:threads:stream-seq:{thread_id}"
    stream_key = f"agentseek:threads:stream:{thread_id}"
    monkeypatch.setattr(stream_module, "_redis_client", redis)

    try:
        seq, event = await stream_module.append_redis_thread_stream_event(thread_id, payload)
        rows = await redis.xrange(stream_key, min="-", max="+")
    finally:
        await redis.delete(sequence_key, stream_key)
        await redis.aclose()

    assert seq == 1
    assert event["type"] == "event"
    assert event["event_id"] == f"{thread_id}:1"
    assert event["seq"] == 1
    payload_body = {key: value for key, value in payload.items() if key not in {"type", "event_id", "seq"}}
    expected_event = {"type": "event", "event_id": f"{thread_id}:1", "seq": 1, **payload_body}
    assert event == expected_event
    assert len(rows) == 1
    stored_payload = rows[0][1]["payload"]
    stored_event = json.loads(stored_payload)
    assert stored_event == event
    assert stored_payload == json.dumps(expected_event, ensure_ascii=False, separators=(",", ":"))


@pytest.mark.parametrize("fault", ["before_xadd", "after_xadd", "lost_ack"])
async def test_idempotent_envelope_repairs_runtime_failure(monkeypatch, fault):
    redis = from_url(_TEST_REDIS_URL, decode_responses=True)
    stream_id = uuid4().hex
    operation = uuid4().hex
    monkeypatch.setattr(stream_module, "_redis_client", redis)
    original = stream_module._APPEND_REDIS_ENVELOPE_SCRIPT
    injected = original
    if fault == "before_xadd":
        injected = original.replace("-- BEFORE_XADD", "error('injected before XADD')")
    elif fault == "after_xadd":
        injected = original.replace("-- AFTER_XADD", "error('injected after XADD')")
    monkeypatch.setattr(stream_module, "_APPEND_REDIS_ENVELOPE_SCRIPT", injected)
    kwargs = dict(scope="thread", stream_id=stream_id, operation_id=operation,
                  payload={"method": "values", "params": {"data": []}})
    try:
        if fault != "lost_ack":
            with pytest.raises(Exception, match="injected"):
                await stream_module.append_redis_envelope(**kwargs)
        else:
            await stream_module.append_redis_envelope(**kwargs)  # Pretend response was lost.
        monkeypatch.setattr(stream_module, "_APPEND_REDIS_ENVELOPE_SCRIPT", original)
        # An unrelated writer may advance the stream after a failed reservation.
        await stream_module.append_redis_envelope(scope="thread", stream_id=stream_id,
            operation_id="other-" + operation, payload={"method": "values"})
        first = await stream_module.append_redis_envelope(**kwargs)
        assert await stream_module.append_redis_envelope(**kwargs) == first
        rows = await redis.xrange(stream_module._thread_stream_key(stream_id))
        assert len(rows) == 2
        assert first[1]["params"]["data"] == []
    finally:
        keys = await redis.keys(f"*{stream_id}*")
        if keys:
            await redis.delete(*keys)
        await redis.aclose()


async def test_envelope_validates_wrong_type_before_counter_mutation(monkeypatch):
    redis = from_url(_TEST_REDIS_URL, decode_responses=True)
    stream_id = uuid4().hex
    monkeypatch.setattr(stream_module, "_redis_client", redis)
    key = stream_module._run_stream_key(stream_id)
    try:
        await redis.set(key, "wrong-type")
        with pytest.raises(Exception, match="WRONGTYPE"):
            await stream_module.append_redis_envelope(scope="run", stream_id=stream_id,
                operation_id="test", payload={"event": "end"})
        assert await redis.get(f"agentseek:runs:stream-seq:{stream_id}") is None
    finally:
        await redis.delete(key)
        await redis.aclose()


async def test_protocol_pair_reconciles_after_second_log_failure(run_storage, monkeypatch):
    from agentseek_api.services import redis_delivery
    redis = from_url(_TEST_REDIS_URL, decode_responses=True)
    identity = uuid4().hex
    monkeypatch.setattr(stream_module, "_redis_client", redis)
    original = stream_module.append_redis_envelope
    async def fail_thread(**kwargs):
        if kwargs["scope"] == "thread":
            raise RuntimeError("second log failed")
        return await original(**kwargs)
    monkeypatch.setattr(stream_module, "append_redis_envelope", fail_thread)
    args = dict(operation_id=identity, run_id=identity, thread_id=identity,
                payload={"method": "messages/partial", "params": {"data": {"tool_calls": []}}})
    try:
        with pytest.raises(RuntimeError, match="second log failed"):
            await stream_module.append_redis_protocol_event(**args)
        assert await redis.xlen(stream_module._run_stream_key(identity)) == 1
        assert await redis.xlen(stream_module._thread_stream_key(identity)) == 0
        monkeypatch.setattr(stream_module, "append_redis_envelope", original)
        assert await redis_delivery.reconcile_protocol_deliveries() == 1
        first = await stream_module.append_redis_protocol_event(**args)
        assert await stream_module.append_redis_protocol_event(**args) == first
        assert await redis.xlen(stream_module._run_stream_key(identity)) == 1
        assert await redis.xlen(stream_module._thread_stream_key(identity)) == 1
        assert first[1][1]["params"]["data"]["tool_calls"] == []
    finally:
        keys = await redis.keys(f"*{identity}*")
        if keys:
            await redis.delete(*keys)
        await redis.aclose()


async def test_pending_protocol_delivery_does_not_resurrect_deleted_run(run_storage, monkeypatch):
    from agentseek_api.core.orm import Run, StreamDelivery
    from agentseek_api.services import redis_delivery
    redis = from_url(_TEST_REDIS_URL, decode_responses=True)
    identity = uuid4().hex
    monkeypatch.setattr(stream_module, "_redis_client", redis)
    async with run_storage() as session:
        session.add(Run(run_id=identity, thread_id="t1", assistant_id="a1", user_id="u1", status="running", execution_id="generation"))
        await session.commit()
    original = stream_module.append_redis_envelope
    async def fail_thread(**kwargs):
        if kwargs["scope"] == "thread":
            raise RuntimeError("second log failed")
        return await original(**kwargs)
    monkeypatch.setattr(stream_module, "append_redis_envelope", fail_thread)
    try:
        with pytest.raises(RuntimeError):
            await stream_module.append_redis_protocol_event(operation_id=identity, run_id=identity,
                thread_id=identity, payload={"method": "values", "params": {"data": []}})
        async with run_storage() as session:
            await session.delete(await session.get(Run, identity))
            await session.commit()
        await redis.delete(stream_module._run_stream_key(identity))
        monkeypatch.setattr(stream_module, "append_redis_envelope", original)
        expire = stream_module.expire_redis_envelope
        async def fail_cleanup(**kwargs):
            raise RuntimeError("cleanup unavailable")
        monkeypatch.setattr(stream_module, "expire_redis_envelope", fail_cleanup)
        await redis_delivery.reconcile_protocol_deliveries()
        async with run_storage() as session:
            pending = await session.get(StreamDelivery, identity)
            assert pending is not None, "failed marker cleanup must retain its recovery record"
            assert pending.records == []
        monkeypatch.setattr(stream_module, "expire_redis_envelope", expire)
        await redis_delivery.reconcile_protocol_deliveries()
        assert await redis.xlen(stream_module._run_stream_key(identity)) == 0
        assert await redis.xlen(stream_module._thread_stream_key(identity)) == 0
        assert await redis.ttl(stream_module._operation_key("run", identity, f"{identity}:run")) > 0
        async with run_storage() as session:
            assert await session.get(StreamDelivery, identity) is None
    finally:
        keys = await redis.keys(f"*{identity}*")
        if keys:
            await redis.delete(*keys)
        await redis.aclose()


@pytest.mark.parametrize("cleanup", ["deleted", "acknowledged"])
async def test_delivery_reloads_after_competing_dispatcher_wins(run_storage, monkeypatch, cleanup):
    from sqlalchemy.ext.asyncio import AsyncSession
    from sqlalchemy.sql.dml import Update
    from agentseek_api.core.orm import StreamDelivery
    from agentseek_api.services import redis_delivery

    redis = from_url(_TEST_REDIS_URL, decode_responses=True)
    identity = uuid4().hex
    monkeypatch.setattr(stream_module, "_redis_client", redis)
    async with run_storage() as session:
        session.add(StreamDelivery(operation_id=identity, run_id=identity, thread_id=identity,
            run_bound=False, payload={"method": "values", "params": {"data": ["once"]}}))
        await session.commit()

    if cleanup == "acknowledged":
        async def fail_expiry(**kwargs):
            raise RuntimeError("cleanup temporarily unavailable")
        monkeypatch.setattr(stream_module, "expire_redis_envelope", fail_expiry)

    execute = AsyncSession.execute
    raced = False
    winner = None

    async def complete_competitor_before_lock(session, statement, *args, **kwargs):
        nonlocal raced, winner
        if not raced and isinstance(statement, Update) and statement.table.name == "stream_deliveries":
            raced = True
            winner = await redis_delivery._deliver(identity)
            if cleanup == "acknowledged":
                # SQL acknowledgment must prevent reappend even after expiry.
                markers = await redis.keys(f"*{identity}*:op:*")
                assert markers
                await redis.delete(*markers)
        return await execute(session, statement, *args, **kwargs)

    monkeypatch.setattr(AsyncSession, "execute", complete_competitor_before_lock)
    try:
        observed = await redis_delivery._deliver(identity)
        assert raced and winner is not None
        assert observed == (None if cleanup == "deleted" else winner)
        assert await redis.xlen(stream_module._run_stream_key(identity)) == 1
        assert await redis.xlen(stream_module._thread_stream_key(identity)) == 1
    finally:
        keys = await redis.keys(f"*{identity}*")
        if keys:
            await redis.delete(*keys)
        await redis.aclose()


@pytest.mark.parametrize("cleanup", ["deleted", "acknowledged"])
async def test_protocol_pair_recovers_stale_ack_without_duplicate_events(run_storage, monkeypatch, cleanup):
    from sqlalchemy.ext.asyncio import AsyncSession
    from sqlalchemy.orm.exc import StaleDataError
    from agentseek_api.core.orm import StreamDelivery
    from agentseek_api.services import redis_delivery

    redis = from_url(_TEST_REDIS_URL, decode_responses=True)
    identity = uuid4().hex
    monkeypatch.setattr(stream_module, "_redis_client", redis)
    commit = AsyncSession.commit
    raced = False
    winner = None

    async def competing_ack_before_commit(session):
        nonlocal raced, winner
        if not raced and any(isinstance(row, StreamDelivery) for row in session.dirty):
            raced = True
            # Reproduce the failed SQL acknowledgment after Redis accepted both
            # envelopes. Release SQLite's writer lock so a real competing
            # dispatcher can commit the winning acknowledgment/cleanup.
            await session.rollback()
            if cleanup == "acknowledged":
                async def fail_expiry(**kwargs):
                    raise RuntimeError("cleanup temporarily unavailable")
                monkeypatch.setattr(stream_module, "expire_redis_envelope", fail_expiry)
            winner = await redis_delivery._deliver(identity)
            if cleanup == "acknowledged":
                markers = await redis.keys(f"*{identity}*:op:*")
                assert markers
                await redis.delete(*markers)
            raise StaleDataError("UPDATE stream_deliveries expected 1 row; 0 matched")
        await commit(session)

    monkeypatch.setattr(AsyncSession, "commit", competing_ack_before_commit)
    try:
        observed = await stream_module.append_redis_protocol_event(
            operation_id=identity, run_id=identity, thread_id=identity,
            payload={"method": "values", "params": {"data": ["once"]}},
        )
        assert raced and winner is not None
        assert observed == winner
        assert await redis.xlen(stream_module._run_stream_key(identity)) == 1
        assert await redis.xlen(stream_module._thread_stream_key(identity)) == 1
    finally:
        keys = await redis.keys(f"*{identity}*")
        if keys:
            await redis.delete(*keys)
        await redis.aclose()


@pytest.mark.parametrize("race", ["delete", "resume"])
async def test_terminal_marker_cleanup_survives_run_replacement(run_storage, monkeypatch, race):
    from agentseek_api.core.orm import Run
    from agentseek_api.services import terminal_delivery
    from agentseek_api.services.run_jobs import RunExecutionJob
    redis = from_url(_TEST_REDIS_URL, decode_responses=True)
    identity = uuid4().hex
    monkeypatch.setattr(stream_module, "_redis_client", redis)
    monkeypatch.setattr(stream_module.settings, "EXECUTOR_BACKEND", "redis")
    job = RunExecutionJob(run_id=identity, thread_id=identity, user_id="u1", graph_id="g", payload={},
        execution_id="generation", owner_id="owner")
    async with run_storage() as session:
        session.add(Run(run_id=identity, thread_id=identity, assistant_id="a1", user_id="u1",
            status="running", execution_id="generation", execution_owner="owner"))
        await session.commit()
    original_append = terminal_delivery.append_redis_envelope
    original_cleanup = terminal_delivery._cleanup_terminal_markers
    async def append(**kwargs):
        if race == "delete" and kwargs["scope"] == "thread":
            raise RuntimeError("second terminal log failed")
        return await original_append(**kwargs)
    async def pause_cleanup(*args, **kwargs):
        pass
    monkeypatch.setattr(terminal_delivery, "append_redis_envelope", append)
    monkeypatch.setattr(terminal_delivery, "_cleanup_terminal_markers", pause_cleanup)
    try:
        if race == "delete":
            with pytest.raises(RuntimeError, match="second terminal"):
                await terminal_delivery.finish_run(job, terminal_delivery.TerminalResult(status="success"))
        else:
            await terminal_delivery.finish_run(job, terminal_delivery.TerminalResult(status="interrupted"))
        markers = await redis.keys(f"*{identity}*:op:*")
        assert markers and all([await redis.ttl(key) == -1 for key in markers])
        async with run_storage() as session:
            row = await session.get(Run, identity)
            if race == "delete":
                await session.delete(row)
            else:
                row.execution_id, row.terminal_result, row.status = "new-generation", None, "pending"
            await session.commit()
        monkeypatch.setattr(terminal_delivery, "_cleanup_terminal_markers", original_cleanup)
        await terminal_delivery.reconcile_terminal_deliveries()
        assert all([await redis.ttl(key) != -1 for key in markers])
        if race == "delete":
            assert not await redis.keys(f"*{identity}*:op:*")
    finally:
        keys = await redis.keys(f"*{identity}*")
        if keys:
            await redis.delete(*keys)
        await redis.aclose()
