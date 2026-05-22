from __future__ import annotations

from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from agentseek_api.core.database import db_manager
from agentseek_api.core.orm import RunStreamEvent, ThreadStreamEvent
from agentseek_api.services.thread_protocol import _namespace_matches, protocol_channel_for_method


def _metadata_db_ready() -> bool:
    try:
        db_manager.get_engine()
    except RuntimeError:
        return False
    return True


def parse_last_event_id(raw_value: str | None) -> int | None:
    if not isinstance(raw_value, str):
        return None
    if raw_value is None or raw_value == "":
        return None
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError("Last-Event-ID must be an integer event sequence.") from exc
    if value < 0:
        raise ValueError("Last-Event-ID must be an integer event sequence.")
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
