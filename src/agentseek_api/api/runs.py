import asyncio
import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Annotated, Any
from urllib.parse import urlencode

from sqlalchemy import delete, select

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Response
from fastapi.responses import JSONResponse, StreamingResponse

from agentseek_api.core.auth_deps import get_current_user
from agentseek_api.core.database import db_manager
from agentseek_api.core.orm import Run, Thread
from agentseek_api.models.api import RunCreateStateful, RunCreateStreamingStateful, RunRead, RunResume
from agentseek_api.models.auth import User
from agentseek_api.api.threads import get_thread_state
from agentseek_api.services.run_preparation import (
    ActiveThreadRunConflictError,
    prepare_and_submit_run,
    resume_run,
)
from agentseek_api.services.run_state import run_broker
from agentseek_api.services.stream_persistence import (
    delete_run_stream_events,
    load_thread_stream_events,
    load_run_stream_events,
    parse_last_event_id,
)
from agentseek_api.services.thread_protocol import thread_protocol_broker
from agentseek_api.settings import settings

router = APIRouter(prefix="/threads/{thread_id}/runs", tags=["Runs"])

TERMINAL_RUN_STATUSES = ("success", "error", "interrupted")
REDIS_STREAM_POLL_INTERVAL_SECONDS = 0.05
REDIS_STREAM_TERMINAL_IDLE_POLLS = 2
SUPPORTED_RUN_STREAM_MODES = {"values", "updates", "messages"}
RUN_STREAM_MODE_ALIASES = {"messages-tuple": "messages"}


async def _best_effort_delete_for_runs(run_ids: list[str]) -> None:
    try:
        await db_manager.get_langgraph_checkpointer().adelete_for_runs(run_ids)
    except NotImplementedError:
        return


def _to_read_model(run: Run) -> RunRead:
    interrupts = None
    if isinstance(run.output_json, dict):
        raw_interrupts = run.output_json.get("interrupts")
        if isinstance(raw_interrupts, list):
            interrupts = raw_interrupts
    return RunRead(
        run_id=run.run_id,
        thread_id=run.thread_id,
        assistant_id=run.assistant_id,
        status=run.status,
        output=run.output_json,
        interrupts=interrupts,
        last_error=run.last_error,
        created_at=run.created_at,
        updated_at=run.updated_at,
        metadata=run.metadata_json,
        kwargs=run.kwargs_json,
        multitask_strategy=run.multitask_strategy,
    )


def _uses_redis_executor() -> bool:
    return settings.EXECUTOR_BACKEND.strip().lower() == "redis"


def _normalize_stream_modes(stream_mode: str | list[str] | None) -> list[str]:
    raw_modes = [stream_mode] if isinstance(stream_mode, str) else list(stream_mode or ["values"])
    modes: list[str] = []
    for mode in raw_modes:
        normalized = RUN_STREAM_MODE_ALIASES.get(str(mode).strip(), str(mode).strip())
        if normalized and normalized not in modes:
            modes.append(normalized)
    if not modes:
        modes = ["values"]
    unsupported = [mode for mode in modes if mode not in SUPPORTED_RUN_STREAM_MODES]
    if unsupported:
        raise HTTPException(
            status_code=422,
            detail=(
                "Unsupported stream_mode value(s): "
                f"{', '.join(unsupported)}. Supported values: {', '.join(sorted(SUPPORTED_RUN_STREAM_MODES))}."
            ),
        )
    return modes


async def _wait_response_payload(run: RunRead, *, user: User) -> Any:
    if run.status == "interrupted" and run.interrupts:
        return {"__interrupt__": run.interrupts}
    if run.status == "error":
        return {"__error__": run.last_error} if run.last_error else {}
    state = await get_thread_state(run.thread_id, user)
    if isinstance(state, dict) and "values" in state:
        return state["values"]
    return {}


def _stream_response_headers(*, location: str, content_location: str) -> dict[str, str]:
    return {
        "Location": location,
        "Content-Location": content_location,
    }


def _wait_response_headers(*, thread_id: str, run_id: str) -> dict[str, str]:
    return _stream_response_headers(
        location=f"/threads/{thread_id}/runs/{run_id}/join",
        content_location=f"/threads/{thread_id}/runs/{run_id}",
    )


def _protocol_stream_location(*, thread_id: str, run_id: str, stream_modes: list[str]) -> str:
    query = urlencode([("stream_mode", mode) for mode in stream_modes], doseq=True)
    return f"/threads/{thread_id}/runs/{run_id}/stream?{query}"


def _interrupt_stream_event_name(stream_modes: list[str]) -> str | None:
    if "updates" in stream_modes:
        return "updates"
    if "values" in stream_modes:
        return "values"
    return None


def _protocol_stream_request(*, stream_modes: list[str]):
    from agentseek_api.models.protocol import ProtocolEventStreamRequest

    return ProtocolEventStreamRequest(channels=stream_modes)


async def _iter_persisted_protocol_run_events(
    *,
    thread_id: str,
    run_id: str,
    stream_modes: list[str],
    user_id: str,
    after_seq: int,
) -> AsyncIterator[dict[str, Any]]:
    payload = _protocol_stream_request(stream_modes=stream_modes)
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
        matching_events = [event for event in events if event.get("params", {}).get("run_id") == run_id]
        if matching_events:
            terminal_idle_polls = 0
            for event in matching_events:
                current_seq = max(current_seq, int(event.get("seq", 0)))
                yield event
            continue

        if await _is_run_terminal(run_id=run_id, thread_id=thread_id, user_id=user_id):
            terminal_idle_polls += 1
            if terminal_idle_polls >= REDIS_STREAM_TERMINAL_IDLE_POLLS:
                return
        else:
            terminal_idle_polls = 0

        await asyncio.sleep(REDIS_STREAM_POLL_INTERVAL_SECONDS)


def _protocol_event_sse(*, event_name: str, data: Any, seq: int | None = None) -> str:
    prefix = f"id: {seq}\n" if seq is not None else ""
    return f"{prefix}event: {event_name}\ndata: {json.dumps(data)}\n\n"


def _build_create_run_stream_response(
    *,
    thread_id: str,
    created: RunRead,
    user: User,
    stream_modes: list[str],
    after_seq: int,
    location: str,
    content_location: str,
    include_metadata: bool = True,
) -> StreamingResponse:
    protocol_channels = [mode for mode in stream_modes if mode in SUPPORTED_RUN_STREAM_MODES]

    async def _event_iter() -> AsyncIterator[str]:
        current_seq = after_seq
        if include_metadata:
            yield _protocol_event_sse(event_name="metadata", data={"run_id": created.run_id, "attempt": 1})

        for event in await load_thread_stream_events(
            thread_id,
            channels=protocol_channels,
            namespaces=None,
            depth=None,
            after_seq=after_seq,
        ):
            if event.get("params", {}).get("run_id") != created.run_id:
                continue
            current_seq = max(current_seq, int(event.get("seq", 0)))
            yield _protocol_event_sse(
                seq=current_seq,
                event_name=str(event.get("method", "message")),
                data=event.get("params", {}).get("data", {}),
            )

        if _uses_redis_executor():
            async for event in _iter_persisted_protocol_run_events(
                thread_id=thread_id,
                run_id=created.run_id,
                stream_modes=protocol_channels,
                user_id=user.identity,
                after_seq=current_seq,
            ):
                current_seq = max(current_seq, int(event.get("seq", 0)))
                yield _protocol_event_sse(
                    seq=current_seq,
                    event_name=str(event.get("method", "message")),
                    data=event.get("params", {}).get("data", {}),
                )
        else:
            async for event in thread_protocol_broker.stream(
                thread_id,
                channels=protocol_channels,
                namespaces=None,
                depth=None,
                since=current_seq,
            ):
                event_run_id = event.get("params", {}).get("run_id")
                if event_run_id != created.run_id:
                    continue
                current_seq = max(current_seq, int(event.get("seq", 0)))
                yield _protocol_event_sse(
                    seq=current_seq,
                    event_name=str(event.get("method", "message")),
                    data=event.get("params", {}).get("data", {}),
                )

        final_run = created if created.status in TERMINAL_RUN_STATUSES else await wait_run(thread_id, created.run_id, user)
        if final_run.status == "error":
            current_seq += 1
            yield _protocol_event_sse(
                seq=current_seq,
                event_name="error",
                data={"error": final_run.last_error, "run_id": created.run_id},
            )
            return
        interrupt_event = _interrupt_stream_event_name(stream_modes)
        if final_run.status == "interrupted" and final_run.interrupts and interrupt_event is not None:
            current_seq += 1
            yield _protocol_event_sse(
                seq=current_seq,
                event_name=interrupt_event,
                data={"__interrupt__": final_run.interrupts},
            )

    return StreamingResponse(
        _event_iter(),
        media_type="text/event-stream",
        headers=_stream_response_headers(location=location, content_location=content_location),
    )


async def _is_run_terminal(*, run_id: str, thread_id: str, user_id: str) -> bool:
    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        status = await session.scalar(
            select(Run.status).where(Run.run_id == run_id, Run.thread_id == thread_id, Run.user_id == user_id)
        )
    return status in TERMINAL_RUN_STATUSES


async def _iter_persisted_run_records(
    *,
    run_id: str,
    thread_id: str,
    user_id: str,
    after_seq: int,
) -> AsyncIterator[tuple[int, dict[str, object]]]:
    current_seq = after_seq
    terminal_idle_polls = 0
    while True:
        records = await load_run_stream_events(run_id, after_seq=current_seq)
        if records:
            terminal_idle_polls = 0
            for seq, event in records:
                current_seq = max(current_seq, seq)
                yield seq, event
            continue

        if await _is_run_terminal(run_id=run_id, thread_id=thread_id, user_id=user_id):
            terminal_idle_polls += 1
            if terminal_idle_polls >= REDIS_STREAM_TERMINAL_IDLE_POLLS:
                return
        else:
            terminal_idle_polls = 0

        await asyncio.sleep(REDIS_STREAM_POLL_INTERVAL_SECONDS)


@router.post("", response_model=RunRead)
async def create_run(thread_id: str, payload: RunCreateStateful, user: User = Depends(get_current_user)) -> RunRead:
    try:
        row = await prepare_and_submit_run(
            thread_id=thread_id,
            assistant_id=payload.assistant_id,
            payload=payload.input,
            user=user,
            metadata=payload.metadata,
            kwargs={"config": payload.config, "context": payload.context},
            multitask_strategy=getattr(payload, "multitask_strategy", "enqueue"),
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ActiveThreadRunConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return _to_read_model(row)


@router.get("", response_model=list[RunRead])
async def list_runs(thread_id: str, user: User = Depends(get_current_user)) -> list[RunRead]:
    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        rows = (
            await session.scalars(
                select(Run).where(Run.thread_id == thread_id, Run.user_id == user.identity).order_by(Run.created_at.desc())
            )
        ).all()
        return [_to_read_model(row) for row in rows]


@router.get("/{run_id}", response_model=RunRead)
async def get_run(thread_id: str, run_id: str, user: User = Depends(get_current_user)) -> RunRead:
    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        row = await session.scalar(
            select(Run).where(Run.run_id == run_id, Run.thread_id == thread_id, Run.user_id == user.identity)
        )
        if row is None:
            raise HTTPException(status_code=404, detail="Run not found")
        return _to_read_model(row)


@router.get("/{run_id}/wait", response_model=RunRead)
async def wait_run(thread_id: str, run_id: str, user: User = Depends(get_current_user)) -> RunRead:
    deadline = asyncio.get_event_loop().time() + 30
    while True:
        session_factory = db_manager.get_session_factory()
        async with session_factory() as session:
            row = await session.scalar(
                select(Run).where(Run.run_id == run_id, Run.thread_id == thread_id, Run.user_id == user.identity)
            )
            if row is None:
                raise HTTPException(status_code=404, detail="Run not found")
            if row.status in TERMINAL_RUN_STATUSES:
                return _to_read_model(row)
        if asyncio.get_event_loop().time() > deadline:
            raise HTTPException(status_code=408, detail="Run wait timeout")
        await asyncio.sleep(0.2)


@router.post(
    "/wait",
    response_class=JSONResponse,
    responses={
        200: {
            "content": {"application/json": {"schema": {}}},
            "headers": {
                "Location": {"schema": {"type": "string"}},
                "Content-Location": {"schema": {"type": "string"}},
            },
        }
    },
)
async def create_run_wait(
    thread_id: str,
    payload: RunCreateStreamingStateful,
    user: User = Depends(get_current_user),
) -> JSONResponse:
    _normalize_stream_modes(payload.stream_mode)
    created = await create_run(thread_id, payload, user)
    final_run = created if created.status in TERMINAL_RUN_STATUSES else await wait_run(thread_id, created.run_id, user)
    return JSONResponse(
        await _wait_response_payload(final_run, user=user),
        headers=_wait_response_headers(thread_id=thread_id, run_id=created.run_id),
    )


@router.post(
    "/stream",
    response_class=StreamingResponse,
    responses={
        200: {
            "content": {"text/event-stream": {"schema": {"type": "string"}}},
            "headers": {
                "Location": {"schema": {"type": "string"}},
                "Content-Location": {"schema": {"type": "string"}},
            },
        }
    },
)
async def create_run_stream(
    thread_id: str,
    payload: RunCreateStreamingStateful,
    user: User = Depends(get_current_user),
) -> StreamingResponse:
    stream_modes = _normalize_stream_modes(payload.stream_mode)
    after_seq = thread_protocol_broker.latest_seq(thread_id)
    created = await create_run(thread_id, payload, user)
    return _build_create_run_stream_response(
        thread_id=thread_id,
        created=created,
        user=user,
        stream_modes=stream_modes,
        after_seq=after_seq,
        location=_protocol_stream_location(thread_id=thread_id, run_id=created.run_id, stream_modes=stream_modes),
        content_location=f"/threads/{thread_id}/runs/{created.run_id}",
    )


@router.post("/{run_id}/resume", response_model=RunRead)
async def resume_existing_run(
    thread_id: str,
    run_id: str,
    payload: RunResume,
    user: User = Depends(get_current_user),
) -> RunRead:
    try:
        row = await resume_run(thread_id=thread_id, run_id=run_id, resume=payload.resume, user=user)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return _to_read_model(row)


@router.post("/{run_id}/cancel")
async def cancel_run(
    thread_id: str,
    run_id: str,
    user: User = Depends(get_current_user),
) -> dict[str, object]:
    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        row = await session.scalar(
            select(Run).where(Run.run_id == run_id, Run.thread_id == thread_id, Run.user_id == user.identity)
        )
        if row is None:
            raise HTTPException(status_code=404, detail="Run not found")
        if row.status not in TERMINAL_RUN_STATUSES:
            row.status = "error"
            row.last_error = "Run cancelled"
            thread = await session.scalar(
                select(Thread).where(Thread.thread_id == thread_id, Thread.user_id == user.identity)
            )
            if thread is not None:
                thread.status = "error"
                thread.state_updated_at = datetime.now(UTC)
            await session.commit()
    await _best_effort_delete_for_runs([run_id])
    return {}


@router.get("/{run_id}/join", response_model=RunRead)
async def join_run(thread_id: str, run_id: str, user: User = Depends(get_current_user)) -> RunRead:
    return await wait_run(thread_id, run_id, user)


@router.delete("/{run_id}", status_code=204)
async def delete_run(thread_id: str, run_id: str, user: User = Depends(get_current_user)) -> Response:
    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        row = await session.scalar(
            select(Run).where(Run.run_id == run_id, Run.thread_id == thread_id, Run.user_id == user.identity)
        )
        if row is None:
            raise HTTPException(status_code=404, detail="Run not found")
        await session.execute(delete(Run).where(Run.run_id == run_id, Run.thread_id == thread_id, Run.user_id == user.identity))
        await session.commit()
    await _best_effort_delete_for_runs([run_id])
    await delete_run_stream_events([run_id])
    return Response(status_code=204)


@router.get("/{run_id}/stream")
async def stream_run(
    thread_id: str,
    run_id: str,
    user: User = Depends(get_current_user),
    last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
    stream_mode: Annotated[list[str] | None, Query()] = None,
) -> StreamingResponse:
    try:
        after_seq = parse_last_event_id(last_event_id) or 0
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        row = await session.scalar(
            select(Run).where(Run.run_id == run_id, Run.thread_id == thread_id, Run.user_id == user.identity)
        )
        if row is None:
            raise HTTPException(status_code=404, detail="Run not found")
        if stream_mode is not None:
            stream_modes = _normalize_stream_modes(stream_mode)
            created = _to_read_model(row)
            return _build_create_run_stream_response(
                thread_id=thread_id,
                created=created,
                user=user,
                stream_modes=stream_modes,
                after_seq=after_seq,
                location=_protocol_stream_location(thread_id=thread_id, run_id=run_id, stream_modes=stream_modes),
                content_location=f"/threads/{thread_id}/runs/{run_id}",
                include_metadata=after_seq == 0,
            )

    async def _event_iter() -> AsyncIterator[str]:
        current_seq = after_seq
        use_redis_executor = _uses_redis_executor()
        records_by_seq: dict[int, dict[str, object]] = {
            seq: payload for seq, payload in await load_run_stream_events(run_id, after_seq=after_seq)
        }
        records_by_seq.update({seq: payload for seq, payload in run_broker.snapshot_records(run_id, after_seq=after_seq)})
        for seq in sorted(records_by_seq):
            event = records_by_seq[seq]
            current_seq = max(current_seq, seq)
            event_name = str(event.get("event", "message"))
            event_payload: dict[str, object] = {"run_id": run_id, **event}
            payload = json.dumps(event_payload)
            yield f"id: {seq}\nevent: {event_name}\ndata: {payload}\n\n"

        if use_redis_executor:
            async for seq, event in _iter_persisted_run_records(
                run_id=run_id,
                thread_id=thread_id,
                user_id=user.identity,
                after_seq=current_seq,
            ):
                event_name = str(event.get("event", "message"))
                event_payload = {"run_id": run_id, **event}
                yield f"id: {seq}\nevent: {event_name}\ndata: {json.dumps(event_payload)}\n\n"
            return

        if row.status in TERMINAL_RUN_STATUSES:
            return

        async for seq, event in run_broker.stream_records(run_id, after_seq=current_seq):
            event_name = str(event.get("event", "message"))
            event_payload = {"run_id": run_id, **event}
            yield f"id: {seq}\nevent: {event_name}\ndata: {json.dumps(event_payload)}\n\n"

    return StreamingResponse(_event_iter(), media_type="text/event-stream")
