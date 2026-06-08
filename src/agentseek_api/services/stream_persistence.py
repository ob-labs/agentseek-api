from __future__ import annotations

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
_redis_client: Redis | None = None


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
    if not _metadata_db_ready():
        return
    seq = int(event.get("seq", 0))
    if seq <= 0:
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
    events: list[dict[str, Any]] = []
    for row in rows:
        event = dict(row.payload_json)
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
    if not _metadata_db_ready():
        return
    try:
        session_factory = db_manager.get_session_factory()
        async with session_factory() as session:
            await session.execute(delete(ThreadStreamEvent).where(ThreadStreamEvent.thread_id == thread_id))
            await session.commit()
    except Exception:
        return
