import asyncio
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy import select

from agentseek_api.core.auth_deps import get_current_user
from agentseek_api.core.database import db_manager
from agentseek_api.core.orm import Run, Thread
from agentseek_api.models.auth import User
from agentseek_api.models.protocol import ProtocolCommandRequest, ProtocolEventStreamRequest
from agentseek_api.services.run_preparation import (
    ActiveThreadRunConflictError,
    prepare_and_submit_run,
    resume_run,
)
from agentseek_api.services.stream_persistence import (
    load_thread_stream_events,
    parse_last_event_id,
)
from agentseek_api.services.sse import iter_with_sse_keepalives, safe_json_dumps, sse_keepalive_comment
from agentseek_api.services.thread_protocol import thread_protocol_broker
from agentseek_api.settings import settings

router = APIRouter(prefix="/threads", tags=["Streaming"])

TERMINAL_RUN_STATUSES = ("success", "error", "interrupted")
REDIS_STREAM_POLL_INTERVAL_SECONDS = 0.05
REDIS_STREAM_TERMINAL_IDLE_POLLS = 2


async def _get_thread_row(*, thread_id: str, user: User) -> Thread:
    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        row = await session.scalar(select(Thread).where(Thread.thread_id == thread_id, Thread.user_id == user.identity))
        if row is None:
            raise HTTPException(status_code=404, detail="Thread not found")
        return row


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


@router.post("/{thread_id}/commands", summary="Protocol v2 Command​")
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
        except ActiveThreadRunConflictError as exc:
            return _protocol_error(request_id=payload.id, code="thread_busy", message=str(exc), status_code=409)
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



@router.post("/{thread_id}/stream/events", summary="Protocol v2 Event Stream (SSE)​")
async def stream_thread_protocol_events(
    thread_id: str,
    payload: ProtocolEventStreamRequest,
    user: User = Depends(get_current_user),
    last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
) -> StreamingResponse:
    header_since = parse_last_event_id(last_event_id) or 0
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
            body = safe_json_dumps(event)
            yield f"id: {seq}\nevent: {method}\ndata: {body}\n\n"

        if _uses_redis_executor():
            async for event in iter_with_sse_keepalives(
                _iter_persisted_thread_events(
                    thread_id=thread_id,
                    payload=payload,
                    user_id=user.identity,
                    after_seq=current_seq,
                )
            ):
                if event is None:
                    yield sse_keepalive_comment()
                    continue
                seq = int(event.get("seq", 0))
                method = str(event.get("method", "event"))
                body = safe_json_dumps(event)
                yield f"id: {seq}\nevent: {method}\ndata: {body}\n\n"
            return

        async for event in iter_with_sse_keepalives(
            thread_protocol_broker.stream(
                thread_id,
                channels=payload.channels,
                namespaces=payload.namespaces,
                depth=payload.depth,
                since=current_seq,
            )
        ):
            if event is None:
                yield sse_keepalive_comment()
                continue
            seq = int(event.get("seq", 0))
            method = str(event.get("method", "event"))
            body = safe_json_dumps(event)
            yield f"id: {seq}\nevent: {method}\ndata: {body}\n\n"

    return StreamingResponse(_event_iter(), media_type="text/event-stream")
