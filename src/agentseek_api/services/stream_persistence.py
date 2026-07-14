from __future__ import annotations

import json
import logging
from typing import Any

from redis.asyncio import Redis, from_url
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from agentseek_api.core.database import db_manager
from agentseek_api.core.orm import RunStreamEvent, ThreadStreamEvent
from agentseek_api.settings import settings
from agentseek_api.services.thread_protocol import _namespace_matches, protocol_channel_for_method

_RUN_STREAM_SEQ_KEY_PREFIX = "agentseek:runs:stream-seq"
_THREAD_STREAM_SEQ_KEY_PREFIX = "agentseek:threads:stream-seq"
_RUN_STREAM_KEY_PREFIX = "agentseek:runs:stream"
_THREAD_STREAM_KEY_PREFIX = "agentseek:threads:stream"
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


async def next_run_stream_seq(run_id: str) -> int | None:
    if not _uses_redis_executor():
        return None
    return int(await _get_redis_client().incr(f"{_RUN_STREAM_SEQ_KEY_PREFIX}:{run_id}"))


async def next_thread_stream_seq(thread_id: str) -> int | None:
    if not _uses_redis_executor():
        return None
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
                        event=str(payload.get("event", "message")),
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
    seq: int,
    payload: dict[str, Any],
) -> None:
    if _uses_redis_executor():
        logger.warning(
            "Skipped non-atomic Redis stream append from legacy run session helper",
            extra={"run_id": run_id, "seq": seq},
        )
        return
    existing = await session.scalar(
        select(RunStreamEvent.id).where(RunStreamEvent.run_id == run_id, RunStreamEvent.seq == seq)
    )
    if existing is not None:
        return
    session.add(
        RunStreamEvent(
            run_id=run_id,
            seq=seq,
            event=str(payload.get("event", "message")),
            payload_json=dict(payload),
        )
    )


async def add_thread_stream_event_to_session(
    session: AsyncSession,
    thread_id: str,
    *,
    seq: int,
    payload: dict[str, Any],
) -> None:
    if _uses_redis_executor():
        logger.warning(
            "Skipped non-atomic Redis stream append from legacy thread session helper",
            extra={"thread_id": thread_id, "seq": seq},
        )
        return
    existing = await session.scalar(
        select(ThreadStreamEvent.id).where(ThreadStreamEvent.thread_id == thread_id, ThreadStreamEvent.seq == seq)
    )
    if existing is not None:
        return
    session.add(
        ThreadStreamEvent(
            thread_id=thread_id,
            seq=seq,
            method=str(payload.get("method", "event")),
            payload_json=dict(payload),
        )
    )


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
            await session.commit()
    except Exception:
        return
