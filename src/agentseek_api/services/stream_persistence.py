from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from redis.asyncio import Redis, from_url
from sqlalchemy import delete, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from agentseek_api.core.database import db_manager
from agentseek_api.core.orm import RunStreamEvent, StreamSequence, ThreadStreamEvent
from agentseek_api.settings import settings
from agentseek_api.services.thread_protocol import _namespace_matches, protocol_channel_for_method

_RUN_STREAM_SEQ_KEY_PREFIX = "agentseek:runs:stream-seq"
_THREAD_STREAM_SEQ_KEY_PREFIX = "agentseek:threads:stream-seq"
_RUN_STREAM_KEY_PREFIX = "agentseek:runs:stream"
_THREAD_STREAM_KEY_PREFIX = "agentseek:threads:stream"
_THREAD_STREAM_ENVELOPE_FIELDS = frozenset({"type", "event_id", "seq"})
# Serializes counter-row seeding per stream so a first-append burst opens one
# seed connection instead of one per publisher (the seed runs in its own short
# transaction; unbounded simultaneous seeds would exhaust the metadata pool).
_stream_seed_locks: dict[tuple[str, str], asyncio.Lock] = {}
_redis_client: Redis | None = None
logger = logging.getLogger(__name__)

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


async def _seed_stream_sequence(scope: str, scope_id: str) -> None:
    """Create the per-stream counter row in its own short transaction.

    Runs outside the caller's transaction so the create-race never executes
    inside a longer append/terminal transaction: MySQL raises
    ``SAVEPOINT ... does not exist`` when a concurrent creator collides inside
    ``begin_nested``. Seeding separately keeps the append transaction purely
    lock-then-allocate. A concurrent creator is resolved by the unique
    constraint; the row is seeded from ``MAX(seq)`` so it can be re-created from
    durable state even if it was deleted out from under an active stream.
    """
    session_factory = db_manager.get_session_factory()
    model = _scope_event_model(scope)
    id_column = model.run_id if scope == "run" else model.thread_id
    async with session_factory() as session:
        max_seq = await session.scalar(select(func.max(model.seq)).where(id_column == scope_id))
        try:
            session.add(StreamSequence(scope=scope, scope_id=scope_id, seq=max_seq or 0))
            await session.commit()
        except IntegrityError:
            # A concurrent publisher created the row first (or a concurrent
            # seed committed between our SELECT and INSERT): nothing to do.
            await session.rollback()


async def _ensure_stream_sequence(session: AsyncSession, scope: str, scope_id: str) -> StreamSequence:
    """Return the per-stream counter row, creating it if missing.

    The row is locked (``SELECT ... FOR UPDATE``) so the caller's transaction
    holds the only allocation right for this stream; concurrent publishers
    serialize on this single row and can never observe the same ``MAX(seq)+1``.
    Two database pitfalls are deliberately avoided:

    - ``FOR UPDATE`` is never issued against a missing row: on MySQL/InnoDB a
      point ``FOR UPDATE`` over a non-existent unique key takes a gap lock, and
      the separate-transaction seed then deadlocks against it (``Lock wait
      timeout``). The row is seeded first (its own short transaction) and only
      then locked.
    - After seeding in a separate transaction, the row is confirmed with a
      ``FOR UPDATE`` current read: under MySQL REPEATABLE READ a plain ``SELECT``
      would keep returning the pre-seed snapshot within the caller's
      transaction.

    A missing row (first append, or deleted out from under an active stream by
    the two-phase delete path in ``threads.py``) is re-seeded from ``MAX(seq)``.
    The in-process per-stream lock serializes the seed so a first-append burst
    opens one seed transaction instead of exhausting the metadata pool.
    """
    stmt = select(StreamSequence).where(
        StreamSequence.scope == scope, StreamSequence.scope_id == scope_id
    )
    lock_key = (scope, scope_id)
    lock = _stream_seed_locks.setdefault(lock_key, asyncio.Lock())
    async with lock:
        for _ in range(_MAX_ATOMIC_APPEND_RETRIES):
            row = await session.scalar(stmt)
            if row is not None:
                # Row exists: the unique index turns this into a record lock
                # only - no gap lock, safe to hold across the allocation.
                locked = await session.scalar(stmt.with_for_update())
                if locked is not None:
                    return locked
            await _seed_stream_sequence(scope, scope_id)
            # Current read: sees the seed even under snapshot isolation.
            locked = await session.scalar(stmt.with_for_update())
            if locked is not None:
                return locked
    raise RuntimeError(f"Failed to establish {scope} stream sequence counter for {scope_id}")


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
    """Append a stream event to the metadata DB atomically.

    One transaction per attempt: lock the per-stream counter row, allocate the
    seq, insert the event row. A uniqueness collision (concurrent publisher, or
    a pre-assigned ``seq`` that collides with an out-of-band row) is healed by
    rolling back and re-allocating from durable state, keeping the stream
    monotonic instead of dropping the frame. Any other failure raises - the
    caller must not expose an event whose durable append did not succeed.
    """
    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        for _ in range(_MAX_ATOMIC_APPEND_RETRIES):
            try:
                result = await _stage_db_event(session, scope, scope_id, payload, seq=seq)
                await session.commit()
                return result
            except IntegrityError:
                await session.rollback()
                # Re-allocate from durable state next attempt instead of
                # reusing the colliding seq.
                seq = None
        raise RuntimeError(
            f"Failed to atomically append {scope} stream event after "
            f"{_MAX_ATOMIC_APPEND_RETRIES} attempts (scope_id={scope_id})"
        )


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
