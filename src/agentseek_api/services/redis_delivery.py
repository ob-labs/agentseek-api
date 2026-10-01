"""Recoverable Redis protocol pairs. SQL owns the envelope until both logs ack."""
from __future__ import annotations

import asyncio
import json
import logging
from sqlalchemy import delete, select, update

from agentseek_api.core.database import db_manager
from agentseek_api.core.orm import Run, StreamDelivery
from agentseek_api.services.transaction_retry import retry_transaction

logger = logging.getLogger(__name__)


async def append_protocol_pair(*, operation_id, run_id, thread_id, payload):
    # Compare the immutable JSON wire shape, not Python-only tuple/key types.
    payload = json.loads(json.dumps(payload, ensure_ascii=False))
    async def register(session):
        from agentseek_api.services.run_dispatch import fence_execution_writes
        await fence_execution_writes(session)
        # Same envelope may be retried concurrently by its producer/reconciler.
        dialect = session.bind.dialect.name
        run = await session.get(Run, run_id)
        values = dict(operation_id=operation_id, run_id=run_id, thread_id=thread_id, payload=payload,
                      run_bound=run is not None, execution_id=run.execution_id if run else None)
        if dialect == "sqlite":
            from sqlalchemy.dialects.sqlite import insert
            statement = insert(StreamDelivery).values(**values).on_conflict_do_nothing(index_elements=["operation_id"])
        elif dialect == "postgresql":
            from sqlalchemy.dialects.postgresql import insert
            statement = insert(StreamDelivery).values(**values).on_conflict_do_nothing(index_elements=["operation_id"])
        else:
            from sqlalchemy.dialects.mysql import insert
            statement = insert(StreamDelivery).values(**values).on_duplicate_key_update(operation_id=operation_id)
        await session.execute(statement)
        row = await session.get(StreamDelivery, operation_id)
        if row.run_id != run_id or row.thread_id != thread_id or row.payload != payload:
            raise ValueError("Operation ID reused for a different stream envelope")
    await retry_transaction(register)
    records = await _deliver(operation_id)
    if records is None:
        # Another dispatcher completed and removed the row between registration
        # and our lock. Its Redis markers still provide the original cursors.
        return await append_protocol_pair(operation_id=operation_id, run_id=run_id, thread_id=thread_id, payload=payload)
    return records


async def _deliver(operation_id):
    from agentseek_api.services.stream_persistence import append_redis_envelope, expire_redis_envelope

    async def deliver(session):
        # Acquire the write lock before loading ORM state. SQLite ignores
        # SELECT FOR UPDATE: a producer/reconciler may otherwise retain an old
        # row after another dispatcher acknowledges or removes it. That can
        # cause StaleDataError or reappend a delivered envelope after expiry.
        locked = await session.execute(update(StreamDelivery).where(
            StreamDelivery.operation_id == operation_id).values(operation_id=operation_id))
        if locked.rowcount == 0:
            return None
        row = await session.scalar(select(StreamDelivery).where(
            StreamDelivery.operation_id == operation_id).with_for_update())
        if row is None:
            return None
        envelopes = [dict(scope=scope, stream_id=identity, operation_id=f"{operation_id}:{scope}", payload=row.payload)
                     for scope, identity in (("run", row.run_id), ("thread", row.thread_id))]
        if row.run_bound:
            run = await session.scalar(select(Run).where(Run.run_id == row.run_id).with_for_update())
            if run is None or run.execution_id != row.execution_id or run.status in {"success", "error", "interrupted"}:
                # Discard delivery but retain the cleanup intent until Redis
                # acknowledges expiry; a transient failure must not leak markers.
                row.records = []
                return envelopes, []
        if row.records is None:
            async with asyncio.timeout(5):
                row.records = [await append_redis_envelope(**envelope, retain=True) for envelope in envelopes]
        return envelopes, row.records
    committed = await retry_transaction(deliver)
    if committed is None:
        return None
    envelopes, records = committed
    # A SQL acknowledgment precedes retention cleanup. If cleanup or its commit
    # fails, the saved cursors prevent another append even after marker expiry.
    try:
        async with asyncio.timeout(5):
            for envelope in envelopes:
                await expire_redis_envelope(**envelope)
        async def cleanup(session):
            await session.execute(delete(StreamDelivery).where(
                StreamDelivery.operation_id == operation_id, StreamDelivery.records.is_not(None)))
        await retry_transaction(cleanup)
    except Exception:
        logger.exception("Protocol marker cleanup remains pending")
    return [(int(seq), payload) for seq, payload in records]


async def reconcile_protocol_deliveries(*, run_id=None, limit=100):
    async with db_manager.get_session_factory()() as session:
        query = select(StreamDelivery.operation_id).order_by(StreamDelivery.created_at)
        if run_id is not None:
            query = query.where(StreamDelivery.run_id == run_id)
        else:
            query = query.limit(limit)
        pending = list(await session.scalars(query))
    completed = 0
    for operation_id in pending:
        try:
            completed += (await _deliver(operation_id)) is not None
        except Exception:
            if run_id is not None:
                raise  # Never append an end ahead of an incomplete protocol pair.
            logger.exception("Protocol pair remains pending", extra={"operation_id": operation_id})
    return completed
