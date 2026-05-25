import asyncio
from copy import deepcopy
from datetime import UTC, datetime
import json
from collections.abc import AsyncIterator
from typing import Any

from sqlalchemy import delete, select

from fastapi import APIRouter, Depends, Header, HTTPException, Response
from fastapi.responses import JSONResponse, StreamingResponse

from agentseek_api.core.auth_deps import get_current_user
from agentseek_api.core.database import db_manager
from agentseek_api.core.orm import Run, Thread
from agentseek_api.models.api import ThreadCreate, ThreadPatch, ThreadPruneRequest, ThreadRead, ThreadSearchRequest
from agentseek_api.models.auth import User
from agentseek_api.models.protocol import ProtocolCommandRequest, ProtocolEventStreamRequest
from agentseek_api.services.run_preparation import (
    ActiveThreadRunConflictError,
    prepare_and_submit_run,
    resume_run,
)
from agentseek_api.services.stream_persistence import (
    delete_run_stream_events,
    delete_thread_stream_events,
    load_thread_stream_events,
    parse_last_event_id,
)
from agentseek_api.services.thread_checkpoint_store import (
    checkpoint_to_payload,
    copy_checkpoints,
    list_checkpoints,
    put_checkpoint,
    prune_checkpoints,
)
from agentseek_api.services.thread_protocol import thread_protocol_broker
from agentseek_api.settings import settings

router = APIRouter(prefix="/threads", tags=["Threads"])

TERMINAL_RUN_STATUSES = ("success", "error", "interrupted")
REDIS_STREAM_POLL_INTERVAL_SECONDS = 0.05
REDIS_STREAM_TERMINAL_IDLE_POLLS = 2
THREAD_STREAM_CHANNELS = ["input", "lifecycle", "messages", "tools", "values"]


async def _best_effort_checkpointer_call(method_name: str, *args: object, **kwargs: object) -> None:
    method = getattr(db_manager.get_langgraph_checkpointer(), method_name, None)
    if method is None:
        return
    try:
        result = method(*args, **kwargs)
        if hasattr(result, "__await__"):
            await result
    except NotImplementedError:
        return


def _to_read_model(row: Thread) -> ThreadRead:
    return ThreadRead(
        thread_id=row.thread_id,
        user_id=row.user_id,
        metadata=row.metadata_json,
        created_at=row.created_at,
        updated_at=row.updated_at,
        state_updated_at=row.state_updated_at,
        config=_public_thread_config(row.config_json),
        status=row.status,
    )


def _public_thread_config(config: dict[str, object] | None) -> dict[str, object]:
    if not isinstance(config, dict):
        return {}
    return dict(config)


def _uses_redis_executor() -> bool:
    return settings.EXECUTOR_BACKEND.strip().lower() == "redis"


async def _thread_has_active_runs(*, thread_id: str, user_id: str) -> bool:
    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        active_run_id = await session.scalar(
            select(Run.run_id).where(
                Run.thread_id == thread_id,
                Run.user_id == user_id,
                Run.status.not_in(TERMINAL_RUN_STATUSES),
            )
        )
    return active_run_id is not None


async def _iter_persisted_thread_events(
    *,
    thread_id: str,
    payload: ProtocolEventStreamRequest,
    user_id: str,
    after_seq: int,
    wait_for_future_runs: bool = False,
) -> AsyncIterator[dict[str, Any]]:
    current_seq = after_seq
    terminal_idle_polls = 0
    while True:
        events = await load_thread_stream_events(
            thread_id,
            channels=payload.channels,
            namespaces=payload.namespaces,
            depth=payload.depth,
            after_seq=current_seq,
        )
        if events:
            terminal_idle_polls = 0
            for event in events:
                current_seq = max(current_seq, int(event.get("seq", 0)))
                yield event
            continue

        if not await _thread_has_active_runs(thread_id=thread_id, user_id=user_id):
            if wait_for_future_runs:
                await asyncio.sleep(REDIS_STREAM_POLL_INTERVAL_SECONDS)
                continue
            terminal_idle_polls += 1
            if terminal_idle_polls >= REDIS_STREAM_TERMINAL_IDLE_POLLS:
                return
        else:
            terminal_idle_polls = 0

        await asyncio.sleep(REDIS_STREAM_POLL_INTERVAL_SECONDS)


def _filtered_thread_rows(rows: list[Thread], payload: ThreadSearchRequest) -> list[Thread]:
    def matches(row: Thread) -> bool:
        if payload.ids is not None and row.thread_id not in payload.ids:
            return False
        if payload.status is not None and row.status != payload.status:
            return False
        if payload.metadata is not None:
            for key, value in payload.metadata.items():
                if row.metadata_json.get(key) != value:
                    return False
        return True

    return [row for row in rows if matches(row)]


def _checkpoint_lookup_payload(payload: dict[str, object] | None) -> str | None:
    if not isinstance(payload, dict):
        return None
    raw_checkpoint_id = payload.get("checkpoint_id")
    if isinstance(raw_checkpoint_id, str) and raw_checkpoint_id:
        return raw_checkpoint_id
    checkpoint = payload.get("checkpoint")
    if isinstance(checkpoint, dict):
        nested_checkpoint_id = checkpoint.get("checkpoint_id")
        if isinstance(nested_checkpoint_id, str) and nested_checkpoint_id:
            return nested_checkpoint_id
    config = payload.get("config")
    if isinstance(config, dict):
        configurable = config.get("configurable")
        if isinstance(configurable, dict):
            nested_config_id = configurable.get("checkpoint_id")
            if isinstance(nested_config_id, str) and nested_config_id:
                return nested_config_id
    return None


def _checkpoint_payload_with_thread_defaults(thread: Thread, payload: dict[str, object]) -> dict[str, object]:
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    payload["metadata"] = {
        "user_id": metadata.get("user_id", thread.user_id),
        "status": metadata.get("status", thread.status),
        **{key: value for key, value in metadata.items() if key not in {"user_id", "status"}},
    }
    return payload


def _normalized_datetime(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def _payload_created_at(payload: dict[str, object]) -> datetime:
    created_at = payload.get("created_at")
    if isinstance(created_at, datetime):
        return _normalized_datetime(created_at)
    return datetime.min.replace(tzinfo=UTC)


def _cancelled_run_windows(runs: list[Run]) -> list[tuple[datetime, datetime | None]]:
    sorted_runs = sorted(runs, key=lambda row: row.created_at)
    windows: list[tuple[datetime, datetime | None]] = []
    for index, run in enumerate(sorted_runs):
        if run.status != "error" or run.last_error != "Run cancelled":
            continue
        next_start = _normalized_datetime(sorted_runs[index + 1].created_at) if index + 1 < len(sorted_runs) else None
        windows.append((_normalized_datetime(run.created_at), next_start))
    return windows


def _is_checkpoint_visible(payload: dict[str, object], cancelled_windows: list[tuple[datetime, datetime | None]]) -> bool:
    created_at = payload.get("created_at")
    if not isinstance(created_at, datetime):
        return True
    created_at = _normalized_datetime(created_at)
    metadata = payload.get("metadata")
    source = metadata.get("source") if isinstance(metadata, dict) else None
    for start, end in cancelled_windows:
        if created_at < start:
            continue
        if end is not None and created_at >= end:
            continue
        if source != "update":
            return False
    return True


def _visible_checkpoint_payloads(thread: Thread, runs: list[Run], payloads: list[dict[str, object]]) -> list[dict[str, object]]:
    cancelled_windows = _cancelled_run_windows(runs)
    visible = [
        _checkpoint_payload_with_thread_defaults(thread, payload)
        for payload in payloads
        if _is_checkpoint_visible(payload, cancelled_windows)
    ]
    return sorted(visible, key=_payload_created_at, reverse=True)


def _empty_thread_state_payload(thread: Thread) -> dict[str, object]:
    return {
        "values": {},
        "next": [],
        "tasks": [],
        "checkpoint": {
            "thread_id": thread.thread_id,
            "checkpoint_ns": "",
            "checkpoint_id": thread.thread_id,
        },
        "metadata": {"user_id": thread.user_id, "status": thread.status},
        "created_at": thread.created_at,
        "parent_checkpoint": None,
        "interrupts": [],
    }


@router.post("", response_model=ThreadRead)
async def create_thread(payload: ThreadCreate, user: User = Depends(get_current_user)) -> ThreadRead:
    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        row = Thread(user_id=user.identity, metadata_json=payload.metadata, config_json=payload.config)
        session.add(row)
        await session.commit()
        await session.refresh(row)
        return _to_read_model(row)


@router.get("", response_model=list[ThreadRead])
async def list_threads(user: User = Depends(get_current_user)) -> list[ThreadRead]:
    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        rows = (
            await session.scalars(select(Thread).where(Thread.user_id == user.identity).order_by(Thread.created_at.desc()))
        ).all()
        return [_to_read_model(row) for row in rows]


@router.post("/search", response_model=list[ThreadRead])
async def search_threads(payload: ThreadSearchRequest, user: User = Depends(get_current_user)) -> list[ThreadRead]:
    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        rows = (
            await session.scalars(select(Thread).where(Thread.user_id == user.identity).order_by(Thread.created_at.desc()))
        ).all()

    filtered = _filtered_thread_rows(rows, payload)
    start = max(payload.offset, 0)
    end = start + max(payload.limit, 0)
    return [_to_read_model(row) for row in filtered[start:end]]


@router.post("/count", response_model=int)
async def count_threads(payload: ThreadSearchRequest, user: User = Depends(get_current_user)) -> int:
    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        rows = (
            await session.scalars(select(Thread).where(Thread.user_id == user.identity).order_by(Thread.created_at.desc()))
        ).all()
    return len(_filtered_thread_rows(rows, payload))


@router.post("/prune")
async def prune_threads(payload: ThreadPruneRequest, user: User = Depends(get_current_user)) -> dict[str, int]:
    session_factory = db_manager.get_session_factory()
    pruned_run_ids: list[str] = []
    async with session_factory() as session:
        rows = (
            await session.scalars(select(Thread).where(Thread.thread_id.in_(payload.thread_ids), Thread.user_id == user.identity))
        ).all()
        thread_ids = [row.thread_id for row in rows]
        if payload.strategy == "delete":
            pruned_run_ids = (
                await session.scalars(select(Run.run_id).where(Run.thread_id.in_(thread_ids), Run.user_id == user.identity))
            ).all()
            await session.execute(delete(Run).where(Run.thread_id.in_(thread_ids), Run.user_id == user.identity))
            await session.execute(delete(Thread).where(Thread.thread_id.in_(thread_ids), Thread.user_id == user.identity))
        elif payload.strategy == "keep_latest":
            runs = (
                await session.scalars(
                    select(Run)
                    .where(Run.thread_id.in_(thread_ids), Run.user_id == user.identity)
                    .order_by(Run.thread_id.asc(), Run.created_at.desc())
                )
            ).all()
            seen_thread_ids: set[str] = set()
            stale_run_ids: list[str] = []
            for run in runs:
                if run.thread_id in seen_thread_ids:
                    stale_run_ids.append(run.run_id)
                    continue
                seen_thread_ids.add(run.thread_id)
            if stale_run_ids:
                pruned_run_ids = stale_run_ids
                await session.execute(delete(Run).where(Run.run_id.in_(stale_run_ids), Run.user_id == user.identity))
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported prune strategy: {payload.strategy}")
        await session.commit()
    await prune_checkpoints(thread_ids, strategy=payload.strategy)
    if pruned_run_ids:
        await delete_run_stream_events(pruned_run_ids)
    if payload.strategy == "delete":
        for thread_id in thread_ids:
            thread_protocol_broker.delete_thread(thread_id)
            await delete_thread_stream_events(thread_id)
    return {"pruned_count": len(thread_ids)}


@router.get("/{thread_id}", response_model=ThreadRead)
async def get_thread(thread_id: str, user: User = Depends(get_current_user)) -> ThreadRead:
    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        row = await session.scalar(select(Thread).where(Thread.thread_id == thread_id, Thread.user_id == user.identity))
        if row is None:
            raise HTTPException(status_code=404, detail="Thread not found")
        return _to_read_model(row)


async def _get_thread_row(*, thread_id: str, user: User) -> Thread:
    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        row = await session.scalar(select(Thread).where(Thread.thread_id == thread_id, Thread.user_id == user.identity))
        if row is None:
            raise HTTPException(status_code=404, detail="Thread not found")
        return row


@router.patch("/{thread_id}", response_model=ThreadRead)
async def patch_thread(thread_id: str, payload: ThreadPatch, user: User = Depends(get_current_user)) -> ThreadRead:
    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        row = await session.scalar(select(Thread).where(Thread.thread_id == thread_id, Thread.user_id == user.identity))
        if row is None:
            raise HTTPException(status_code=404, detail="Thread not found")
        if payload.metadata is not None:
            row.metadata_json = {**row.metadata_json, **payload.metadata}
        await session.commit()
        await session.refresh(row)
        return _to_read_model(row)


@router.delete("/{thread_id}", status_code=204)
async def delete_thread(thread_id: str, user: User = Depends(get_current_user)) -> Response:
    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        row = await session.scalar(select(Thread).where(Thread.thread_id == thread_id, Thread.user_id == user.identity))
        if row is None:
            raise HTTPException(status_code=404, detail="Thread not found")
        run_ids = (
            await session.scalars(select(Run.run_id).where(Run.thread_id == thread_id, Run.user_id == user.identity))
        ).all()
        await session.execute(delete(Run).where(Run.thread_id == thread_id, Run.user_id == user.identity))
        await session.delete(row)
        await session.commit()
    await _best_effort_checkpointer_call("adelete_thread", thread_id)
    if run_ids:
        await _best_effort_checkpointer_call("adelete_for_runs", list(run_ids))
        await delete_run_stream_events(list(run_ids))
    thread_protocol_broker.delete_thread(thread_id)
    await delete_thread_stream_events(thread_id)
    return Response(status_code=204)


@router.post("/{thread_id}/copy", response_model=ThreadRead)
async def copy_thread(thread_id: str, user: User = Depends(get_current_user)) -> ThreadRead:
    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        source = await session.scalar(select(Thread).where(Thread.thread_id == thread_id, Thread.user_id == user.identity))
        if source is None:
            raise HTTPException(status_code=404, detail="Thread not found")
        source_runs = (
            await session.scalars(
                select(Run).where(Run.thread_id == thread_id, Run.user_id == user.identity).order_by(Run.created_at.asc())
            )
        ).all()
        copied = Thread(
            user_id=source.user_id,
            metadata_json=deepcopy(source.metadata_json),
            config_json=deepcopy(source.config_json),
            status=source.status,
            state_updated_at=source.state_updated_at,
        )
        session.add(copied)
        await session.flush()
        for source_run in source_runs:
            session.add(
                Run(
                    thread_id=copied.thread_id,
                    assistant_id=source_run.assistant_id,
                    user_id=source_run.user_id,
                    status=source_run.status,
                    input_json=deepcopy(source_run.input_json),
                    output_json=deepcopy(source_run.output_json),
                    metadata_json=deepcopy(source_run.metadata_json),
                    kwargs_json=deepcopy(source_run.kwargs_json),
                    multitask_strategy=source_run.multitask_strategy,
                    last_error=source_run.last_error,
                    created_at=source_run.created_at,
                    updated_at=source_run.updated_at,
                )
            )
        await session.commit()
        await session.refresh(copied)
    await copy_checkpoints(thread_id, copied.thread_id)
    return _to_read_model(copied)


@router.get("/{thread_id}/state")
async def get_thread_state(thread_id: str, user: User = Depends(get_current_user)) -> dict[str, object]:
    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        thread = await session.scalar(select(Thread).where(Thread.thread_id == thread_id, Thread.user_id == user.identity))
        if thread is None:
            raise HTTPException(status_code=404, detail="Thread not found")
        runs = (
            await session.scalars(
                select(Run).where(Run.thread_id == thread_id, Run.user_id == user.identity).order_by(Run.created_at.asc())
            )
        ).all()
    checkpoints = await list_checkpoints(thread_id)
    visible = _visible_checkpoint_payloads(
        thread,
        runs,
        [checkpoint_to_payload(item) for item in checkpoints],
    )
    if not visible:
        return _empty_thread_state_payload(thread)
    return visible[0]


@router.get("/{thread_id}/history")
async def get_thread_history(thread_id: str, user: User = Depends(get_current_user)) -> list[dict[str, object]]:
    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        thread = await session.scalar(select(Thread).where(Thread.thread_id == thread_id, Thread.user_id == user.identity))
        if thread is None:
            raise HTTPException(status_code=404, detail="Thread not found")
        runs = (
            await session.scalars(
                select(Run).where(Run.thread_id == thread_id, Run.user_id == user.identity).order_by(Run.created_at.asc())
            )
        ).all()
    return _visible_checkpoint_payloads(
        thread,
        runs,
        [checkpoint_to_payload(item) for item in await list_checkpoints(thread_id)],
    )


@router.post("/{thread_id}/history")
async def get_thread_history_post(thread_id: str, user: User = Depends(get_current_user)) -> list[dict[str, object]]:
    return await get_thread_history(thread_id, user)


async def _get_thread_state_at_checkpoint(*, thread_id: str, checkpoint_id: str, user: User) -> dict[str, object]:
    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        thread = await session.scalar(select(Thread).where(Thread.thread_id == thread_id, Thread.user_id == user.identity))
        if thread is None:
            raise HTTPException(status_code=404, detail="Thread not found")
        runs = (
            await session.scalars(
                select(Run).where(Run.thread_id == thread_id, Run.user_id == user.identity).order_by(Run.created_at.asc())
            )
        ).all()
    visible = _visible_checkpoint_payloads(
        thread,
        runs,
        [checkpoint_to_payload(item) for item in await list_checkpoints(thread_id)],
    )
    if checkpoint_id == thread.thread_id and not visible:
        return _empty_thread_state_payload(thread)
    for payload in visible:
        checkpoint = payload.get("checkpoint")
        if isinstance(checkpoint, dict) and checkpoint.get("checkpoint_id") == checkpoint_id:
            return payload
    raise HTTPException(status_code=404, detail="Checkpoint not found")


@router.get("/{thread_id}/state/{checkpoint_id}")
async def get_thread_state_at_checkpoint(
    thread_id: str,
    checkpoint_id: str,
    user: User = Depends(get_current_user),
) -> dict[str, object]:
    return await _get_thread_state_at_checkpoint(thread_id=thread_id, checkpoint_id=checkpoint_id, user=user)


@router.post("/{thread_id}/state")
async def update_thread_state(
    thread_id: str,
    payload: dict[str, object],
    user: User = Depends(get_current_user),
) -> dict[str, object]:
    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        thread = await session.scalar(select(Thread).where(Thread.thread_id == thread_id, Thread.user_id == user.identity))
        if thread is None:
            raise HTTPException(status_code=404, detail="Thread not found")
        values = payload.get("values", payload)
        if not isinstance(values, dict):
            values = {}
        checkpoint = await put_checkpoint(
            thread_id,
            values,
            metadata={"user_id": user.identity, "status": thread.status},
        )
        thread.state_updated_at = checkpoint_to_payload(checkpoint)["created_at"]
        await session.commit()
    return _checkpoint_payload_with_thread_defaults(thread, checkpoint_to_payload(checkpoint))


@router.post("/{thread_id}/state/checkpoint")
async def checkpoint_thread_state(
    thread_id: str,
    payload: dict[str, object],
    user: User = Depends(get_current_user),
) -> dict[str, object]:
    checkpoint_id = _checkpoint_lookup_payload(payload)
    if checkpoint_id is None:
        raise HTTPException(status_code=422, detail="checkpoint_id is required")
    return await _get_thread_state_at_checkpoint(thread_id=thread_id, checkpoint_id=checkpoint_id, user=user)


@router.get("/{thread_id}/stream")
async def stream_thread(
    thread_id: str,
    user: User = Depends(get_current_user),
    last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
) -> StreamingResponse:
    try:
        after_seq = parse_last_event_id(last_event_id) or 0
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        thread = await session.scalar(select(Thread).where(Thread.thread_id == thread_id, Thread.user_id == user.identity))
        if thread is None:
            raise HTTPException(status_code=404, detail="Thread not found")

    async def _event_iter() -> AsyncIterator[str]:
        payload = ProtocolEventStreamRequest(channels=THREAD_STREAM_CHANNELS)
        current_seq = after_seq
        yield ": stream-open\n\n"

        for event in await load_thread_stream_events(
            thread_id,
            channels=payload.channels,
            namespaces=payload.namespaces,
            depth=payload.depth,
            after_seq=after_seq,
        ):
            seq = int(event.get("seq", 0))
            current_seq = max(current_seq, seq)
            event_name = str(event.get("method", "event"))
            yield f"id: {seq}\nevent: {event_name}\ndata: {json.dumps(event)}\n\n"

        if _uses_redis_executor():
            async for event in _iter_persisted_thread_events(
                thread_id=thread_id,
                payload=payload,
                user_id=user.identity,
                after_seq=current_seq,
                wait_for_future_runs=True,
            ):
                seq = int(event.get("seq", 0))
                current_seq = max(current_seq, seq)
                event_name = str(event.get("method", "event"))
                yield f"id: {seq}\nevent: {event_name}\ndata: {json.dumps(event)}\n\n"
            return

        async for event in thread_protocol_broker.stream(
            thread_id,
            channels=payload.channels,
            namespaces=payload.namespaces,
            depth=payload.depth,
            since=current_seq,
            wait_for_future_runs=True,
        ):
            seq = int(event.get("seq", 0))
            current_seq = max(current_seq, seq)
            event_name = str(event.get("method", "event"))
            yield f"id: {seq}\nevent: {event_name}\ndata: {json.dumps(event)}\n\n"

    return StreamingResponse(_event_iter(), media_type="text/event-stream")


def _protocol_error(*, request_id: int | None, code: str, message: str, status_code: int) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "type": "error",
            "id": request_id,
            "error": code,
            "message": message,
        },
    )


def _coerce_protocol_input(raw_input: Any) -> dict[str, Any]:
    if isinstance(raw_input, dict):
        return raw_input
    if isinstance(raw_input, str):
        return {"message": raw_input}
    return {"value": raw_input}


@router.post("/{thread_id}/commands")
async def handle_protocol_command(
    thread_id: str,
    payload: ProtocolCommandRequest,
    user: User = Depends(get_current_user),
) -> JSONResponse:
    await _get_thread_row(thread_id=thread_id, user=user)

    if payload.method == "run.start":
        assistant_id = payload.params.get("assistant_id")
        if not isinstance(assistant_id, str) or not assistant_id:
            return _protocol_error(
                request_id=payload.id,
                code="invalid_argument",
                message="'assistant_id' is required for run.start",
                status_code=400,
            )

        try:
            run = await prepare_and_submit_run(
                thread_id=thread_id,
                assistant_id=assistant_id,
                payload=_coerce_protocol_input(payload.params.get("input")),
                user=user,
            )
        except ValueError as exc:
            return _protocol_error(request_id=payload.id, code="invalid_argument", message=str(exc), status_code=404)
        except ActiveThreadRunConflictError as exc:
            return _protocol_error(request_id=payload.id, code="thread_busy", message=str(exc), status_code=409)

        return JSONResponse(
            content={
                "type": "success",
                "id": payload.id,
                "result": {"run_id": run.run_id},
                "meta": {"applied_through_seq": thread_protocol_broker.latest_seq(thread_id)},
            }
        )

    if payload.method == "input.respond":
        interrupt_id = payload.params.get("interrupt_id")
        if not isinstance(interrupt_id, str) or not interrupt_id:
            return _protocol_error(
                request_id=payload.id,
                code="invalid_argument",
                message="'interrupt_id' is required for input.respond",
                status_code=400,
            )

        session_factory = db_manager.get_session_factory()
        async with session_factory() as session:
            run = await session.scalar(
                select(Run)
                .where(Run.thread_id == thread_id, Run.user_id == user.identity, Run.status == "interrupted")
                .order_by(Run.created_at.desc())
            )
            if run is None:
                return _protocol_error(
                    request_id=payload.id,
                    code="no_such_interrupt",
                    message="No interrupted run found for this thread",
                    status_code=404,
                )
            interrupts = run.output_json.get("interrupts") if isinstance(run.output_json, dict) else None
            if not isinstance(interrupts, list) or not any(item.get("id") == interrupt_id for item in interrupts if isinstance(item, dict)):
                return _protocol_error(
                    request_id=payload.id,
                    code="no_such_interrupt",
                    message=f"Interrupt '{interrupt_id}' was not found",
                    status_code=404,
                )
            run_id = run.run_id

        try:
            await resume_run(
                thread_id=thread_id,
                run_id=run_id,
                resume=payload.params.get("response"),
                user=user,
            )
        except (ValueError, RuntimeError) as exc:
            return _protocol_error(request_id=payload.id, code="unknown_error", message=str(exc), status_code=409)

        return JSONResponse(
            content={
                "type": "success",
                "id": payload.id,
                "result": {},
                "meta": {"applied_through_seq": thread_protocol_broker.latest_seq(thread_id)},
            }
        )

    return _protocol_error(
        request_id=payload.id,
        code="unknown_command",
        message=f"Unsupported command '{payload.method}'",
        status_code=400,
    )


@router.post("/{thread_id}/stream")
@router.post("/{thread_id}/stream/events")
async def stream_thread_protocol_events(
    thread_id: str,
    payload: ProtocolEventStreamRequest,
    user: User = Depends(get_current_user),
    last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
) -> StreamingResponse:
    try:
        header_since = parse_last_event_id(last_event_id) or 0
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    after_seq = max(header_since, payload.since or 0)
    await _get_thread_row(thread_id=thread_id, user=user)

    async def _event_iter() -> AsyncIterator[str]:
        current_seq = after_seq
        for event in await load_thread_stream_events(
            thread_id,
            channels=payload.channels,
            namespaces=payload.namespaces,
            depth=payload.depth,
            after_seq=after_seq,
        ):
            seq = int(event.get("seq", 0))
            current_seq = max(current_seq, seq)
            method = str(event.get("method", "event"))
            body = json.dumps(event)
            yield f"id: {seq}\nevent: {method}\ndata: {body}\n\n"

        if _uses_redis_executor():
            async for event in _iter_persisted_thread_events(
                thread_id=thread_id,
                payload=payload,
                user_id=user.identity,
                after_seq=current_seq,
            ):
                seq = int(event.get("seq", 0))
                method = str(event.get("method", "event"))
                body = json.dumps(event)
                yield f"id: {seq}\nevent: {method}\ndata: {body}\n\n"
            return

        async for event in thread_protocol_broker.stream(
            thread_id,
            channels=payload.channels,
            namespaces=payload.namespaces,
            depth=payload.depth,
            since=current_seq,
        ):
            seq = int(event.get("seq", 0))
            method = str(event.get("method", "event"))
            body = json.dumps(event)
            yield f"id: {seq}\nevent: {method}\ndata: {body}\n\n"

    return StreamingResponse(_event_iter(), media_type="text/event-stream")
