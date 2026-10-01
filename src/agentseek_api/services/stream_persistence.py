from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from contextvars import ContextVar
from typing import Any

from redis.asyncio import Redis, from_url
from sqlalchemy import case, delete, func, insert, select
from sqlalchemy.ext.asyncio import AsyncSession

from agentseek_api.core.database import db_manager
from agentseek_api.core.orm import RunStreamEvent, StreamSequence, ThreadStreamEvent
from agentseek_api.settings import settings
from agentseek_api.services.thread_protocol import _namespace_matches, protocol_channel_for_method
from agentseek_api.services.stream_event_buffer import StreamEvent, StreamEventBuffer
from agentseek_api.services.transaction_retry import retry_transaction

_RUN_STREAM_SEQ_KEY_PREFIX = "agentseek:runs:stream-seq"
_THREAD_STREAM_SEQ_KEY_PREFIX = "agentseek:threads:stream-seq"
_RUN_STREAM_KEY_PREFIX = "agentseek:runs:stream"
_THREAD_STREAM_KEY_PREFIX = "agentseek:threads:stream"
_THREAD_STREAM_ENVELOPE_FIELDS = frozenset({"type", "event_id", "seq"})
# Serializes counter-row seeding per stream so a first-append burst opens one
# seed connection instead of one per publisher (the seed runs in its own short
# transaction; unbounded simultaneous seeds would exhaust the metadata pool).
_stream_seed_locks: dict[tuple[str, str], asyncio.Lock] = {}
_THREAD_SNAPSHOT_BATCH_SIZE = 500
_redis_client: Redis | None = None
logger = logging.getLogger(__name__)
_stream_buffer: ContextVar[StreamEventBuffer | None] = ContextVar("stream_persistence_buffer", default=None)

_APPEND_REDIS_STREAM_EVENT_SCRIPT = """
local seq = redis.call('INCR', KEYS[1])
local payload = ARGV[1]
if ARGV[4] ~= '' then
  -- Inject type/event_id/seq WITHOUT a cjson decode/encode round-trip.
  -- Redis' bundled lua-cjson cannot distinguish an empty array from an empty
  -- object, so cjson.encode(cjson.decode('{"tool_calls":[]}')) returns
  -- '{"tool_calls":{}}'. That silently corrupts every streamed message
  -- (tool_calls / invalid_tool_calls become {}), and langgraph-sdk's
  -- convertToChunk() then throws on `{}.map`, so the client cannot concat
  -- message chunks by id and each token replaces the previous one instead of
  -- accumulating. Splice the header in as a string to keep the original
  -- payload (and its empty arrays) byte-for-byte intact.
  local rest = string.sub(payload, 2)
  local event_id = cjson.encode(ARGV[4] .. ':' .. tostring(seq))
  local head = '{"type":"event","event_id":' .. event_id .. ',"seq":' .. tostring(seq)
  if rest == '}' then
    payload = head .. '}'
  else
    payload = head .. ',' .. rest
  end
end
redis.call('XADD', KEYS[2], 'MAXLEN', '~', ARGV[2], tostring(seq) .. '-0', 'payload', payload)
redis.call('EXPIRE', KEYS[2], ARGV[3])
return {seq, payload}
"""

# The operation marker is written BEFORE XADD. Lua execution is isolated but
# runtime errors do not roll back earlier commands. A retry inspects the reserved
# ID to distinguish a missing append from a committed append with a lost ack.
_APPEND_REDIS_ENVELOPE_SCRIPT = """
local expected = {'string', 'stream', 'hash'}
for i = 1, 3 do
  local actual = redis.call('TYPE', KEYS[i]).ok
  if actual ~= 'none' and actual ~= expected[i] then
    return redis.error_reply('WRONGTYPE stream envelope key')
  end
end
local maxlen = tonumber(ARGV[2])
local ttl = tonumber(ARGV[3])
if not maxlen or maxlen < 1 or not ttl or ttl < 1 or string.sub(ARGV[1], 1, 1) ~= '{' then
  return redis.error_reply('Invalid stream envelope arguments')
end
local counter = redis.call('GET', KEYS[1])
if counter and (not tonumber(counter) or tonumber(counter) < 0) then
  return redis.error_reply('Invalid stream sequence counter')
end
local seq = redis.call('HGET', KEYS[3], 'seq')
local payload = redis.call('HGET', KEYS[3], 'payload')
if seq then
  local rows = redis.call('XRANGE', KEYS[2], seq .. '-0', seq .. '-0')
  if redis.call('HGET', KEYS[3], 'done') == '1' or #rows > 0 then
    redis.call('HSET', KEYS[3], 'done', '1')
    return {tonumber(seq), payload}
  end
end
local last = redis.call('XREVRANGE', KEYS[2], '+', '-', 'COUNT', 1)
local lastseq = 0
if #last > 0 then lastseq = tonumber(string.match(last[1][1], '^(%d+)')) end
seq = math.max(tonumber(counter) or 0, lastseq) + 1
redis.call('SET', KEYS[1], tostring(seq))
payload = ARGV[1]
if ARGV[4] ~= '' then
  local rest = string.sub(payload, 2)
  local head = '{"type":"event","event_id":' .. cjson.encode(ARGV[4] .. ':' .. tostring(seq)) .. ',"seq":' .. tostring(seq)
  if rest == '}' then payload = head .. '}' else payload = head .. ',' .. rest end
end
redis.call('HSET', KEYS[3], 'seq', tostring(seq), 'payload', payload)
-- BEFORE_XADD
redis.call('XADD', KEYS[2], 'MAXLEN', '~', maxlen, tostring(seq) .. '-0', 'payload', payload)
-- AFTER_XADD
redis.call('HSET', KEYS[3], 'done', '1')
redis.call('EXPIRE', KEYS[2], ttl)
return {seq, payload}
"""


def _operation_key(scope: str, stream_id: str, operation_id: str) -> str:
    key = _run_stream_key(stream_id) if scope == "run" else _thread_stream_key(stream_id)
    return f"{key}:op:{operation_id}"


async def expire_redis_envelope(*, scope: str, stream_id: str, operation_id: str, **_kwargs) -> None:
    await _get_redis_client().expire(_operation_key(scope, stream_id, operation_id), max(1, settings.REDIS_STREAM_TTL_SECONDS))


async def append_redis_envelope(*, scope: str, stream_id: str, operation_id: str,
                                payload: dict[str, Any], retain: bool = False) -> tuple[int, dict[str, Any]]:
    if scope not in {"run", "thread"} or not operation_id:
        raise ValueError("A stream scope and stable operation_id are required")
    if scope == "thread":
        payload = {k: v for k, v in payload.items() if k not in _THREAD_STREAM_ENVELOPE_FIELDS}
    prefix = _RUN_STREAM_SEQ_KEY_PREFIX if scope == "run" else _THREAD_STREAM_SEQ_KEY_PREFIX
    stream_key = _run_stream_key(stream_id) if scope == "run" else _thread_stream_key(stream_id)
    result = await _get_redis_client().eval(
        _APPEND_REDIS_ENVELOPE_SCRIPT, 3, f"{prefix}:{stream_id}", stream_key,
        _operation_key(scope, stream_id, operation_id),
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        str(max(1, settings.REDIS_STREAM_MAXLEN)), str(max(1, settings.REDIS_STREAM_TTL_SECONDS)),
        stream_id if scope == "thread" else "",
    )
    if not retain:
        await expire_redis_envelope(scope=scope, stream_id=stream_id, operation_id=operation_id)
    return int(result[0]), json.loads(result[1])


async def append_redis_protocol_event(*, operation_id: str, run_id: str, thread_id: str, payload: dict[str, Any]):
    from agentseek_api.services.redis_delivery import append_protocol_pair
    return await append_protocol_pair(operation_id=operation_id, run_id=run_id, thread_id=thread_id, payload=payload)


def _metadata_db_ready() -> bool:
    try:
        db_manager.get_engine()
    except RuntimeError:
        return False
    return True


def _uses_redis_executor() -> bool:
    return settings.EXECUTOR_BACKEND.strip().lower() == "redis"


def _get_redis_client() -> Redis:
    global _redis_client
    if _redis_client is None:
        _redis_client = from_url(settings.REDIS_URL, decode_responses=True)
    return _redis_client


def _run_stream_key(run_id: str) -> str:
    return f"{_RUN_STREAM_KEY_PREFIX}:{run_id}"


def _thread_stream_key(thread_id: str) -> str:
    return f"{_THREAD_STREAM_KEY_PREFIX}:{thread_id}"


async def _append_redis_stream_event_atomic(
    *,
    sequence_key: str,
    stream_key: str,
    payload: dict[str, Any],
    event_prefix: str = "",
) -> tuple[int, dict[str, Any]]:
    if event_prefix:
        payload = {key: value for key, value in payload.items() if key not in _THREAD_STREAM_ENVELOPE_FIELDS}
    result = await _get_redis_client().eval(
        _APPEND_REDIS_STREAM_EVENT_SCRIPT,
        2,
        sequence_key,
        stream_key,
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        str(max(1, settings.REDIS_STREAM_MAXLEN)),
        str(max(1, settings.REDIS_STREAM_TTL_SECONDS)),
        event_prefix,
    )
    seq = int(result[0])
    encoded_payload = result[1]
    if isinstance(encoded_payload, bytes):
        encoded_payload = encoded_payload.decode()
    event = json.loads(encoded_payload)
    if not isinstance(event, dict):
        raise TypeError("Redis stream event payload must be a JSON object")
    return seq, event


async def append_redis_run_stream_event(run_id: str, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    return await _append_redis_stream_event_atomic(
        sequence_key=f"{_RUN_STREAM_SEQ_KEY_PREFIX}:{run_id}",
        stream_key=_run_stream_key(run_id),
        payload=payload,
    )


async def append_redis_thread_stream_event(thread_id: str, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    return await _append_redis_stream_event_atomic(
        sequence_key=f"{_THREAD_STREAM_SEQ_KEY_PREFIX}:{thread_id}",
        stream_key=_thread_stream_key(thread_id),
        payload=payload,
        event_prefix=thread_id,
    )


async def _load_redis_stream_events(key: str, *, after_seq: int) -> list[tuple[int, dict[str, Any]]]:
    rows = await _get_redis_client().xrange(key, min=f"({after_seq}-0", max="+")
    events: list[tuple[int, dict[str, Any]]] = []
    for entry_id, fields in rows:
        try:
            seq = int(entry_id.split("-", 1)[0])
            payload = json.loads(fields["payload"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            events.append((seq, payload))
    return events


def _scope_event_model(scope: str) -> type[RunStreamEvent] | type[ThreadStreamEvent]:
    if scope == "run":
        return RunStreamEvent
    if scope == "thread":
        return ThreadStreamEvent
    raise ValueError(f"Unsupported stream scope: {scope}")


def _thread_envelope(thread_id: str, seq: int, payload: dict[str, Any]) -> dict[str, Any]:
    """Mirror the wire envelope ``ThreadProtocolEventBroker._record_event`` builds.

    Keeping the persisted row byte-compatible with the in-memory broker event
    means ``_record_event(seq=...)`` reproduces exactly what was already
    committed, so the broker never re-derives a different identity.
    """
    return {
        "type": "event",
        "event_id": f"{thread_id}:{seq}",
        "seq": seq,
        **payload,
    }


async def _ensure_stream_sequence(session: AsyncSession, scope: str, scope_id: str) -> StreamSequence:
    """Create/reconcile and lock the counter using the caller's one connection."""
    model = _scope_event_model(scope)
    id_column = model.run_id if scope == "run" else model.thread_id
    maximum = select(func.coalesce(func.max(model.seq), 0)).where(id_column == scope_id).scalar_subquery()
    dialect = session.get_bind().dialect.name
    values = {"scope": scope, "scope_id": scope_id, "seq": maximum}
    if dialect == "sqlite":
        from sqlalchemy.dialects.sqlite import insert as dialect_insert
        statement = dialect_insert(StreamSequence).values(**values)
        statement = statement.on_conflict_do_update(
            index_elements=["scope", "scope_id"],
            set_={"seq": case((StreamSequence.seq < statement.excluded.seq, statement.excluded.seq), else_=StreamSequence.seq)},
        )
    elif dialect == "postgresql":
        from sqlalchemy.dialects.postgresql import insert as dialect_insert
        statement = dialect_insert(StreamSequence).values(**values)
        statement = statement.on_conflict_do_update(
            index_elements=["scope", "scope_id"],
            set_={"seq": func.greatest(StreamSequence.seq, statement.excluded.seq)},
        )
    elif dialect in {"mysql", "mariadb"}:
        from sqlalchemy.dialects.mysql import insert as dialect_insert
        statement = dialect_insert(StreamSequence).values(**values)
        statement = statement.on_duplicate_key_update(seq=func.greatest(StreamSequence.seq, statement.inserted.seq))
    else:
        raise ValueError(f"Unsupported stream counter dialect: {dialect}")
    await session.execute(statement)
    # populate_existing is essential if the session saw this counter before
    # another transaction committed; allocation must read the winning row.
    return await session.scalar(
        select(StreamSequence).where(
            StreamSequence.scope == scope, StreamSequence.scope_id == scope_id
        ).with_for_update().execution_options(populate_existing=True)
    )


async def _stage_db_event(
    session: AsyncSession,
    scope: str,
    scope_id: str,
    payload: dict[str, Any],
    *,
    seq: int | None,
) -> tuple[int, dict[str, Any]]:
    """Allocate (or commit) the stream seq and stage the event row in ``session``.

    Does not commit: the standalone path commits explicitly, while the
    in-session path (terminal events) commits together with the run/thread
    status so ``seq`` and state are durable as one unit.
    """
    from agentseek_api.services.run_dispatch import fence_execution_writes
    await fence_execution_writes(session)
    counter = await _ensure_stream_sequence(session, scope, scope_id)
    new_seq = counter.seq + 1 if seq is None else seq
    counter.seq = max(counter.seq, new_seq)
    if scope == "run":
        session.add(
            RunStreamEvent(
                run_id=scope_id,
                seq=new_seq,
                event=str(payload.get("method") or payload.get("event", "message")),
                payload_json=dict(payload),
            )
        )
    else:
        session.add(
            ThreadStreamEvent(
                thread_id=scope_id,
                seq=new_seq,
                method=str(payload.get("method", "event")),
                payload_json=dict(_thread_envelope(scope_id, new_seq, payload)),
            )
        )
    return new_seq, dict(payload)


# Upper bound on uniqueness retries. The metadata-DB append relies on the
# per-stream counter row's row lock to serialize publishers; SQLite ignores
# ``SELECT ... FOR UPDATE``, so concurrent publishers can read the same counter
# value and collide on the ``UNIQUE(scope_id, seq)`` constraint. Each retry
# rolls back and re-reads the counter, so every successful commit advances the
# stream by one - concurrent appends converge to unique, gapless seqs. The
# bound is a safety valve; normal (non-concurrent) appends never retry.
_MAX_ATOMIC_APPEND_RETRIES = 32


async def _db_append(
    scope: str,
    scope_id: str,
    payload: dict[str, Any],
    *,
    seq: int | None = None,
) -> tuple[int, dict[str, Any]]:
    async def append(session: AsyncSession):
        return await _stage_db_event(session, scope, scope_id, payload, seq=seq)
    return await retry_transaction(append)


async def append_run_stream_event_atomic(
    run_id: str,
    payload: dict[str, Any],
    *,
    seq: int | None = None,
) -> tuple[int, dict[str, Any]]:
    """Durably append a run-scoped stream event and return its seq.

    Redis executor: single Lua ``INCR``+``XADD`` (atomic by construction).
    Inline executor: single metadata-DB transaction. The caller must only
    expose the event to clients after this returns.
    """
    if _uses_redis_executor():
        if seq is not None:  # pragma: no cover - redis appends always allocate
            raise ValueError("Redis stream append allocates its own seq")
        return await append_redis_run_stream_event(run_id, payload)
    if not _metadata_db_ready():
        # No metadata DB at all (offline tests / pre-initialization): there is
        # nothing durable to protect, so fall back to broker-local sequence
        # allocation (seq=None) exactly like the legacy path. Production runs
        # always have the DB initialized, so this is a startup/offline posture,
        # not a durable-path fallback.
        return (None, dict(payload))
    return await _db_append("run", run_id, payload, seq=seq)


async def append_thread_stream_event_atomic(
    thread_id: str,
    payload: dict[str, Any],
    *,
    seq: int | None = None,
) -> tuple[int, dict[str, Any]]:
    """Durably append a thread-protocol event and return its seq (see run twin)."""
    if _uses_redis_executor():
        if seq is not None:  # pragma: no cover - redis appends always allocate
            raise ValueError("Redis stream append allocates its own seq")
        return await append_redis_thread_stream_event(thread_id, payload)
    if not _metadata_db_ready():
        # No metadata DB at all (offline tests / pre-initialization): nothing
        # durable to protect, fall back to broker-local sequence allocation.
        return (None, dict(payload))
    return await _db_append("thread", thread_id, payload, seq=seq)


async def next_run_stream_seq(run_id: str) -> int | None:
    # Legacy helper, retained only for tests and backward compatibility.
    # Production callers must use append_run_stream_event_atomic so the
    # allocation and the durable write are one atomic unit.
    if not _uses_redis_executor():
        if not _metadata_db_ready():
            return None
        try:
            session_factory = db_manager.get_session_factory()
        except RuntimeError:
            return None
        async with session_factory() as session:
            row = await session.scalar(
                select(func.max(RunStreamEvent.seq)).where(RunStreamEvent.run_id == run_id)
            )
        return (row or 0) + 1
    return int(await _get_redis_client().incr(f"{_RUN_STREAM_SEQ_KEY_PREFIX}:{run_id}"))


async def next_thread_stream_seq(thread_id: str) -> int | None:
    # Legacy helper, retained only for tests and backward compatibility.
    # Production callers must use append_thread_stream_event_atomic.
    if not _uses_redis_executor():
        if not _metadata_db_ready():
            return None
        try:
            session_factory = db_manager.get_session_factory()
        except RuntimeError:
            return None
        async with session_factory() as session:
            row = await session.scalar(
                select(func.max(ThreadStreamEvent.seq)).where(ThreadStreamEvent.thread_id == thread_id)
            )
        return (row or 0) + 1
    return int(await _get_redis_client().incr(f"{_THREAD_STREAM_SEQ_KEY_PREFIX}:{thread_id}"))


def parse_last_event_id(raw_value: str | None) -> int | None:
    if not isinstance(raw_value, str):
        return None
    if raw_value is None or raw_value == "":
        return None
    try:
        value = int(raw_value)
    except (ValueError, TypeError):
        return None
    if value < 0:
        return None
    return value


async def persist_run_stream_event(run_id: str, *, seq: int, payload: dict[str, Any]) -> None:
    if _uses_redis_executor():
        logger.warning(
            "Skipped non-atomic Redis stream append from legacy run persistence helper",
            extra={"run_id": run_id, "seq": seq},
        )
        return
    if not _metadata_db_ready():
        return
    if await _buffer_stream_event(StreamEvent("run", run_id, seq, payload)):
        return
    await _persist_run_stream_event(run_id, seq=seq, payload=payload)


async def _persist_run_stream_event(run_id: str, *, seq: int, payload: dict[str, Any]) -> None:
    try:
        session_factory = db_manager.get_session_factory()
        async with session_factory() as session:
            existing = await session.scalar(
                select(RunStreamEvent.id).where(RunStreamEvent.run_id == run_id, RunStreamEvent.seq == seq)
            )
            if existing is None:
                session.add(
                    RunStreamEvent(
                        run_id=run_id,
                        seq=seq,
                        event=str(payload.get("method") or payload.get("event", "message")),
                        payload_json=dict(payload),
                    )
                )
                await session.commit()
    except Exception:
        return


async def add_run_stream_event_to_session(
    session: AsyncSession,
    run_id: str,
    *,
    seq: int | None = None,
    payload: dict[str, Any],
) -> tuple[int, dict[str, Any]]:
    """Stage a run stream event inside the caller's transaction.

    Allocates the seq from the locked counter row when ``seq`` is not given
    (terminal events committed atomically with the run status) or commits a
    pre-assigned seq when it is (idempotent: an existing row is skipped). The
    caller is responsible for committing, and must publish to the in-memory
    broker only after that commit.
    """
    if _uses_redis_executor():
        logger.warning(
            "Skipped non-atomic Redis stream append from legacy run session helper",
            extra={"run_id": run_id, "seq": seq},
        )
        return (seq or 0, payload)
    if not _metadata_db_ready():
        # No metadata DB (offline tests / pre-initialization): there is nothing
        # durable to stage, so defer to broker-local sequence allocation.
        return (seq or 0, payload)
    if seq is not None:
        existing = await session.scalar(
            select(RunStreamEvent.id).where(RunStreamEvent.run_id == run_id, RunStreamEvent.seq == seq)
        )
        if existing is not None:
            return seq, payload
    return await _stage_db_event(session, "run", run_id, payload, seq=seq)


async def add_thread_stream_event_to_session(
    session: AsyncSession,
    thread_id: str,
    *,
    seq: int | None = None,
    payload: dict[str, Any],
) -> tuple[int, dict[str, Any]]:
    """Stage a thread-protocol event inside the caller's transaction (see run twin)."""
    if _uses_redis_executor():
        logger.warning(
            "Skipped non-atomic Redis stream append from legacy thread session helper",
            extra={"thread_id": thread_id, "seq": seq},
        )
        return (seq or 0, payload)
    if not _metadata_db_ready():
        return (seq or 0, payload)
    if seq is not None:
        existing = await session.scalar(
            select(ThreadStreamEvent.id).where(ThreadStreamEvent.thread_id == thread_id, ThreadStreamEvent.seq == seq)
        )
        if existing is not None:
            return seq, payload
    return await _stage_db_event(session, "thread", thread_id, payload, seq=seq)


async def load_run_stream_events(run_id: str, *, after_seq: int = 0) -> list[tuple[int, dict[str, Any]]]:
    if _uses_redis_executor():
        return await _load_redis_stream_events(_run_stream_key(run_id), after_seq=after_seq)
    if not _metadata_db_ready():
        return []
    try:
        session_factory = db_manager.get_session_factory()
    except RuntimeError:
        return []
    async with session_factory() as session:
        rows = (
            await session.scalars(
                select(RunStreamEvent)
                .where(RunStreamEvent.run_id == run_id, RunStreamEvent.seq > after_seq)
                .order_by(RunStreamEvent.seq.asc())
            )
        ).all()
    return [(row.seq, dict(row.payload_json)) for row in rows]


async def delete_run_stream_events(run_ids: list[str]) -> None:
    if not run_ids:
        return
    for run_id in run_ids:
        _stream_seed_locks.pop(("run", run_id), None)
    if _uses_redis_executor():
        keys = [key for run_id in run_ids for key in (_run_stream_key(run_id), f"{_RUN_STREAM_SEQ_KEY_PREFIX}:{run_id}")]
        try:
            from agentseek_api.services.terminal_delivery import cleanup_terminal_markers_for_runs
            await cleanup_terminal_markers_for_runs(run_ids)
        except Exception:
            logger.warning("Redis terminal marker cleanup remains queued", exc_info=True)
        try:
            await _get_redis_client().delete(*keys)
        except Exception:
            logger.warning("Failed to delete Redis run stream keys", extra={"run_ids": run_ids}, exc_info=True)
            return
        return
    if not _metadata_db_ready():
        return
    try:
        session_factory = db_manager.get_session_factory()
        async with session_factory() as session:
            await session.execute(delete(RunStreamEvent).where(RunStreamEvent.run_id.in_(run_ids)))
            await session.execute(
                delete(StreamSequence).where(
                    StreamSequence.scope == "run", StreamSequence.scope_id.in_(run_ids)
                )
            )
            await session.commit()
    except Exception:
        return


async def persist_thread_stream_event(thread_id: str, event: dict[str, Any] | None) -> None:
    if event is None:
        return
    seq = int(event.get("seq", 0))
    if seq <= 0:
        return
    if _uses_redis_executor():
        logger.warning(
            "Skipped non-atomic Redis stream append from legacy thread persistence helper",
            extra={"thread_id": thread_id, "seq": seq},
        )
        return
    if not _metadata_db_ready():
        return
    if await _buffer_stream_event(StreamEvent("thread", thread_id, seq, event)):
        return
    await _persist_thread_stream_event(thread_id, event)


async def _persist_thread_stream_event(thread_id: str, event: dict[str, Any]) -> None:
    seq = int(event["seq"])
    try:
        session_factory = db_manager.get_session_factory()
        async with session_factory() as session:
            existing = await session.scalar(
                select(ThreadStreamEvent.id).where(ThreadStreamEvent.thread_id == thread_id, ThreadStreamEvent.seq == seq)
            )
            if existing is None:
                session.add(
                    ThreadStreamEvent(
                        thread_id=thread_id,
                        seq=seq,
                        method=str(event.get("method", "event")),
                        payload_json=dict(event),
                    )
                )
                await session.commit()
    except Exception:
        return


@asynccontextmanager
async def buffered_stream_persistence(*, run_id: str, thread_id: str):
    if _uses_redis_executor() or not _metadata_db_ready():
        yield
        return
    async with StreamEventBuffer(_persist_stream_event_batch, run_id=run_id, thread_id=thread_id) as buffer:
        token = _stream_buffer.set(buffer)
        try:
            yield
        finally:
            _stream_buffer.reset(token)


async def _buffer_stream_event(record: StreamEvent) -> bool:
    buffer = _stream_buffer.get()
    if buffer is None:
        return False
    stream_id = buffer.run_id if record.kind == "run" else buffer.thread_id
    if record.stream_id != stream_id:
        return False
    try:
        return await buffer.append(record)
    except Exception:
        logger.warning("Failed to buffer stream event; persisting individually", exc_info=True)
        return False


async def buffer_durable_event(kind: str, stream_id: str, payload: dict[str, Any], publish) -> bool:
    """Queue an unallocated event; only the committed batch may publish it."""
    buffer = _stream_buffer.get()
    if buffer is None or stream_id != (buffer.run_id if kind == "run" else buffer.thread_id):
        return False
    return await buffer.append(StreamEvent(kind, stream_id, 0, payload, publish))


async def _commit_allocated_stream_batch(records: list[StreamEvent]) -> None:
    async def stage(session: AsyncSession):
        from agentseek_api.services.run_dispatch import fence_execution_writes
        await fence_execution_writes(session)
        results = []
        groups: dict[tuple[str, str], list[StreamEvent]] = {}
        for record in records:
            groups.setdefault((record.kind, record.stream_id), []).append(record)
        for (kind, stream_id), group in sorted(groups.items()):
            counter = await _ensure_stream_sequence(session, kind, stream_id)
            model = _scope_event_model(kind)
            id_field = "run_id" if kind == "run" else "thread_id"
            name_field = "event" if kind == "run" else "method"
            rows = []
            for record in group:
                counter.seq += 1
                seq = counter.seq
                payload = dict(record.payload) if kind == "run" else _thread_envelope(stream_id, seq, record.payload)
                rows.append({id_field: stream_id, "seq": seq, name_field: str(payload.get("method") or payload.get("event", "message")), "payload_json": payload})
                results.append((record, seq, payload))
            await session.execute(insert(model), rows)
        return results

    results = await retry_transaction(stage)
    for record, seq, payload in results:
        if record.publish is not None:
            try:
                record.publish(seq, payload)
            except Exception:
                # The commit already succeeded. Re-appending would duplicate
                # durable history; a reconnect can replay the committed record.
                logger.exception("Broker notification failed after stream batch commit")


async def _persist_stream_event_batch(records: list[StreamEvent]) -> None:
    allocated = [record for record in records if record.publish is not None]
    if allocated:
        await _commit_allocated_stream_batch(allocated)
    records = [record for record in records if record.publish is None]
    if not records:
        return
    groups: dict[tuple[str, str], dict[int, StreamEvent]] = {}
    for record in records:
        groups.setdefault((record.kind, record.stream_id), {}).setdefault(record.seq, record)
    try:
        session_factory = db_manager.get_session_factory()
        async with session_factory() as session:
            inserted = False
            for (kind, stream_id), events in groups.items():
                model = RunStreamEvent if kind == "run" else ThreadStreamEvent
                id_field, name_field = ("run_id", "event") if kind == "run" else ("thread_id", "method")
                existing = set(await session.scalars(select(model.seq).where(
                    getattr(model, id_field) == stream_id, model.seq.in_(list(events)),
                )))
                missing = [
                    {
                        id_field: stream_id,
                        "seq": seq,
                        name_field: str(record.payload.get(name_field, "message" if kind == "run" else "event")),
                        "payload_json": record.payload,
                    }
                    for seq, record in events.items() if seq not in existing
                ]
                if missing:
                    await session.execute(insert(model), missing)
                    inserted = True
            if inserted:
                await session.commit()
    except Exception:
        logger.warning("Failed to persist live stream batch; retrying individually", exc_info=True)
        # The failed session has closed before these idempotent retries. Bypass
        # buffering here so an error cannot requeue its own batch indefinitely.
        for record in records:
            if record.kind == "run":
                await _persist_run_stream_event(record.stream_id, seq=record.seq, payload=record.payload)
            else:
                await _persist_thread_stream_event(record.stream_id, record.payload)


async def persist_thread_stream_events(thread_id: str, events: list[dict[str, Any]]) -> None:
    """Recover a buffered SQL snapshot without rechecking each event separately."""
    if not events or _uses_redis_executor() or not _metadata_db_ready():
        return
    events_by_seq: dict[int, dict[str, Any]] = {}
    for event in events:
        seq = int(event.get("seq", 0))
        if seq > 0:
            events_by_seq.setdefault(seq, event)
    snapshot = list(events_by_seq.values())
    for offset in range(0, len(snapshot), _THREAD_SNAPSHOT_BATCH_SIZE):
        batch = snapshot[offset : offset + _THREAD_SNAPSHOT_BATCH_SIZE]
        try:
            session_factory = db_manager.get_session_factory()
            async with session_factory() as session:
                existing = set(await session.scalars(
                    select(ThreadStreamEvent.seq).where(
                        ThreadStreamEvent.thread_id == thread_id,
                        ThreadStreamEvent.seq.in_([int(event["seq"]) for event in batch]),
                    )
                ))
                missing = [
                    {
                        "thread_id": thread_id,
                        "seq": int(event["seq"]),
                        "method": str(event.get("method", "event")),
                        "payload_json": dict(event),
                    }
                    for event in batch
                    if int(event["seq"]) not in existing
                ]
                if missing:
                    await session.execute(insert(ThreadStreamEvent), missing)
                    await session.commit()
        except Exception:
            # A background publisher can insert after our lookup. Roll back the
            # batch before retrying individually so one conflict cannot lose
            # the other events. This also retains best-effort failure handling.
            logger.warning("Failed to persist thread event batch; retrying individually", exc_info=True)
            for event in batch:
                await persist_thread_stream_event(thread_id, event)


async def load_thread_stream_events(
    thread_id: str,
    *,
    channels: list[str],
    namespaces: list[list[str]] | None,
    depth: int | None,
    after_seq: int = 0,
) -> list[dict[str, Any]]:
    if _uses_redis_executor():
        records = await _load_redis_stream_events(_thread_stream_key(thread_id), after_seq=after_seq)
        payloads = [event for _, event in records]
    else:
        if not _metadata_db_ready():
            return []
        try:
            session_factory = db_manager.get_session_factory()
        except RuntimeError:
            return []
        async with session_factory() as session:
            rows = (
                await session.scalars(
                    select(ThreadStreamEvent)
                    .where(ThreadStreamEvent.thread_id == thread_id, ThreadStreamEvent.seq > after_seq)
                    .order_by(ThreadStreamEvent.seq.asc())
                )
            ).all()
        payloads = [dict(row.payload_json) for row in rows]
    events: list[dict[str, Any]] = []
    for event in payloads:
        channel = protocol_channel_for_method(str(event.get("method", "")))
        namespace = event.get("params", {}).get("namespace", [])
        if not isinstance(namespace, list):
            namespace = []
        if channel not in channels:
            continue
        if not _namespace_matches(namespace, namespaces=namespaces, depth=depth):
            continue
        events.append(event)
    return events


async def delete_thread_stream_events(thread_id: str) -> None:
    _stream_seed_locks.pop(("thread", thread_id), None)
    if _uses_redis_executor():
        try:
            await _get_redis_client().delete(
                _thread_stream_key(thread_id),
                f"{_THREAD_STREAM_SEQ_KEY_PREFIX}:{thread_id}",
            )
        except Exception:
            logger.warning("Failed to delete Redis thread stream keys", extra={"thread_id": thread_id}, exc_info=True)
            return
        return
    if not _metadata_db_ready():
        return
    try:
        session_factory = db_manager.get_session_factory()
        async with session_factory() as session:
            await session.execute(delete(ThreadStreamEvent).where(ThreadStreamEvent.thread_id == thread_id))
            await session.execute(
                delete(StreamSequence).where(
                    StreamSequence.scope == "thread", StreamSequence.scope_id == thread_id
                )
            )
            await session.commit()
    except Exception:
        return
