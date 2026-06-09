import asyncio
import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Annotated, Any
from urllib.parse import urlencode

from sqlalchemy import delete, select

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Response
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, StreamingResponse

from agentseek_api.core.auth_deps import get_current_user
from agentseek_api.core.database import db_manager
from agentseek_api.core.orm import Run, Thread
from agentseek_api.models.api import RunCreateStateful, RunCreateStreamingStateful, RunRead, RunResume
from agentseek_api.models.auth import User
from agentseek_api.api.threads import _find_thread_state_payload, _get_thread_state_at_checkpoint, get_thread_state
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
from agentseek_api.services.sse import iter_with_sse_keepalives, safe_json_dumps, sse_keepalive_comment
from agentseek_api.services.thread_protocol import thread_protocol_broker
from agentseek_api.settings import settings

router = APIRouter(prefix="/threads/{thread_id}/runs", tags=["Runs"])

TERMINAL_RUN_STATUSES = ("success", "error", "interrupted")
REDIS_STREAM_POLL_INTERVAL_SECONDS = 0.05
REDIS_STREAM_TERMINAL_IDLE_POLLS = 2
SUPPORTED_RUN_STREAM_MODES = {"values", "updates", "messages", "messages-tuple", "debug", "events", "tasks", "checkpoints", "custom"}
RUN_STREAM_MODE_ALIASES: dict[str, str] = {}
RUN_CHECKPOINT_ID_METADATA_KEY = "__agentseek_checkpoint_id"


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
        metadata=_public_run_metadata(run.metadata_json),
        kwargs=run.kwargs_json,
        multitask_strategy=run.multitask_strategy,
    )


def _public_run_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(metadata, dict):
        return {}
    return {key: value for key, value in metadata.items() if key != RUN_CHECKPOINT_ID_METADATA_KEY}


def _uses_redis_executor() -> bool:
    return settings.EXECUTOR_BACKEND.strip().lower() == "redis"


def _normalize_stream_modes(stream_mode: str | list[str] | None) -> list[str]:
    if stream_mode is None:
        return ["values"]

    raw_modes = [stream_mode] if isinstance(stream_mode, str) else list(stream_mode)
    modes: list[str] = []
    invalid_modes: list[str] = []

    if not raw_modes:
        invalid_modes.append("<empty>")

    for mode in raw_modes:
        raw_mode = str(mode).strip()
        if not raw_mode:
            invalid_modes.append("<empty>")
            continue
        normalized = RUN_STREAM_MODE_ALIASES.get(raw_mode, raw_mode)
        if normalized not in modes:
            modes.append(normalized)

    unsupported = invalid_modes + [mode for mode in modes if mode not in SUPPORTED_RUN_STREAM_MODES]
    if unsupported:
        raise HTTPException(
            status_code=422,
            detail=(
                "Unsupported stream_mode value(s): "
                f"{', '.join(unsupported)}. Supported values: {', '.join(sorted(SUPPORTED_RUN_STREAM_MODES))}."
            ),
        )
    return modes


def _parse_stream_mode_query_param(stream_mode: list[str] | None) -> list[str] | None:
    if stream_mode is None:
        return None
    if len(stream_mode) == 1:
        raw_value = stream_mode[0].strip()
        if raw_value.startswith("["):
            try:
                parsed = json.loads(raw_value)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, list):
                return [str(item) for item in parsed]
        return [raw_value]
    return [value.strip() for value in stream_mode]


def _validate_supported_run_controls(payload: Any, *, stateless: bool) -> None:
    unsupported_controls: list[str] = []

    if getattr(payload, "webhook", None) is not None:
        unsupported_controls.append("webhook")
    if getattr(payload, "feedback_keys", None):
        unsupported_controls.append("feedback_keys")
    if getattr(payload, "if_not_exists", "reject") != "reject":
        unsupported_controls.append("if_not_exists")
    if getattr(payload, "after_seconds", None) is not None:
        unsupported_controls.append("after_seconds")
    if stateless and getattr(payload, "on_completion", "keep") != "keep":
        unsupported_controls.append("on_completion")

    if unsupported_controls:
        raise HTTPException(
            status_code=422,
            detail=(
                "Unsupported run control field(s): "
                f"{', '.join(sorted(unsupported_controls))}. "
                "These controls are not implemented by agentseek-api."
            ),
        )


async def _wait_response_payload(run: RunRead, *, user: User) -> Any:
    if run.status == "interrupted" and run.interrupts:
        return {"__interrupt__": run.interrupts}
    if run.status == "error":
        return {"__error__": run.last_error} if run.last_error else {}
    checkpoint_id = await _load_run_checkpoint_id(run_id=run.run_id, thread_id=run.thread_id, user_id=user.identity)
    if checkpoint_id is not None:
        state = await _get_thread_state_at_checkpoint(thread_id=run.thread_id, checkpoint_id=checkpoint_id, user=user)
    else:
        state = await _find_thread_state_payload(thread_id=run.thread_id, user=user, checkpoint_ns=run.run_id)
        if state is None:
            state = await get_thread_state(run.thread_id, user)
    if isinstance(state, dict) and "values" in state:
        values = state["values"]
        if isinstance(values, dict):
            values.pop("__pregel_tasks", None)
        return values
    return {}


async def _load_run_checkpoint_id(*, run_id: str, thread_id: str, user_id: str) -> str | None:
    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        row = await session.scalar(
            select(Run).where(Run.run_id == run_id, Run.thread_id == thread_id, Run.user_id == user_id)
        )
        if row is None or not isinstance(row.metadata_json, dict):
            return None
        checkpoint_id = row.metadata_json.get(RUN_CHECKPOINT_ID_METADATA_KEY)
        if not isinstance(checkpoint_id, str) or not checkpoint_id:
            return None
        return checkpoint_id


async def _get_run_read(thread_id: str, run_id: str, user: User) -> RunRead:
    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        row = await session.scalar(
            select(Run).where(Run.run_id == run_id, Run.thread_id == thread_id, Run.user_id == user.identity)
        )
        if row is None:
            raise HTTPException(status_code=404, detail="Run not found")
        return _to_read_model(row)


def _wait_json_stream_response(
    *,
    run: RunRead,
    user: User,
    headers: dict[str, str],
    cancel_on_disconnect: bool = False,
) -> StreamingResponse:
    async def _body() -> AsyncIterator[bytes]:
        try:
            current_run = run
            while current_run.status not in TERMINAL_RUN_STATUSES:
                try:
                    current_run = await _wait_run_terminal(
                        current_run.thread_id,
                        current_run.run_id,
                        user,
                        timeout_seconds=5.0,
                    )
                except HTTPException as exc:
                    if exc.status_code != 408:
                        raise
                    yield b"\n"
            payload = await _wait_response_payload(current_run, user=user)
            yield json.dumps(jsonable_encoder(payload), separators=(",", ":")).encode()
        finally:
            if cancel_on_disconnect:
                try:
                    await _cancel_active_run(
                        thread_id=run.thread_id,
                        run_id=run.run_id,
                        user_id=user.identity,
                        require_existing=False,
                    )
                except Exception:
                    pass

    return StreamingResponse(
        _body(),
        media_type="application/json",
        headers=headers,
    )


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

    return ProtocolEventStreamRequest(channels=_protocol_channels_for_stream_modes(stream_modes))


def _protocol_channels_for_stream_modes(stream_modes: list[str]) -> list[str]:
    channels = list(stream_modes)
    if "input" not in channels:
        channels.append("input")
    return channels


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


_SSE_EVENT_NAME_MAP: dict[str, str] = {
    "messages-tuple": "messages",
}


def _protocol_event_sse(*, event_name: str, data: Any, seq: int | None = None) -> str:
    prefix = f"id: {seq}\n" if seq is not None else ""
    wire_name = _SSE_EVENT_NAME_MAP.get(event_name, event_name)
    return f"{prefix}event: {wire_name}\ndata: {safe_json_dumps(data)}\n\n"


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
    replay_existing: bool = True,
    cancel_on_disconnect: bool = False,
) -> StreamingResponse:
    protocol_channels = _protocol_channels_for_stream_modes(
        [mode for mode in stream_modes if mode in SUPPORTED_RUN_STREAM_MODES]
    )
    # When the client asked for ``messages``, the official LangGraph wire
    # contract is ``messages/metadata`` + ``messages/partial`` only. The
    # protocol-v2 block stream (message-start / content-block-* / message-finish)
    # is internal noise to that client and would fight the partial accumulator,
    # so suppress it here.
    suppress_block_messages = "messages" in stream_modes

    def _is_block_message_event(event: dict[str, Any]) -> bool:
        return str(event.get("method", "")) == "messages"

    async def _event_iter() -> AsyncIterator[str]:
        try:
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
                current_seq = max(current_seq, int(event.get("seq", 0)))
                if event.get("params", {}).get("run_id") != created.run_id:
                    continue
                if not replay_existing:
                    continue
                if suppress_block_messages and _is_block_message_event(event):
                    continue
                yield _protocol_event_sse(
                    seq=current_seq,
                    event_name=str(event.get("method", "message")),
                    data=event.get("params", {}).get("data", {}),
                )

            if _uses_redis_executor():
                async for event in iter_with_sse_keepalives(
                    _iter_persisted_protocol_run_events(
                        thread_id=thread_id,
                        run_id=created.run_id,
                        stream_modes=protocol_channels,
                        user_id=user.identity,
                        after_seq=current_seq,
                    )
                ):
                    if event is None:
                        yield sse_keepalive_comment()
                        continue
                    current_seq = max(current_seq, int(event.get("seq", 0)))
                    if suppress_block_messages and _is_block_message_event(event):
                        continue
                    yield _protocol_event_sse(
                        seq=current_seq,
                        event_name=str(event.get("method", "message")),
                        data=event.get("params", {}).get("data", {}),
                    )
            else:
                async for event in iter_with_sse_keepalives(
                    thread_protocol_broker.stream(
                        thread_id,
                        channels=protocol_channels,
                        namespaces=None,
                        depth=None,
                        since=current_seq,
                    )
                ):
                    if event is None:
                        yield sse_keepalive_comment()
                        continue
                    event_run_id = event.get("params", {}).get("run_id")
                    if event_run_id != created.run_id:
                        continue
                    current_seq = max(current_seq, int(event.get("seq", 0)))
                    if suppress_block_messages and _is_block_message_event(event):
                        continue
                    yield _protocol_event_sse(
                        seq=current_seq,
                        event_name=str(event.get("method", "message")),
                        data=event.get("params", {}).get("data", {}),
                    )

            final_run = (
                created
                if created.status in TERMINAL_RUN_STATUSES
                else await _wait_run_terminal(thread_id, created.run_id, user, timeout_seconds=None)
            )
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
        finally:
            # When the client disconnects mid-stream, Starlette closes this
            # generator and we transition the run to a terminal state if it's
            # still active. If the stream finished naturally the run is already
            # terminal, so _cancel_active_run is a no-op.
            if cancel_on_disconnect:
                try:
                    await _cancel_active_run(
                        thread_id=thread_id,
                        run_id=created.run_id,
                        user_id=user.identity,
                        require_existing=False,
                    )
                except Exception:
                    pass

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
    _validate_supported_run_controls(payload, stateless=False)
    run_kwargs: dict[str, Any] = {"config": payload.config, "context": payload.context}
    stream_modes = _normalize_stream_modes(payload.stream_mode)
    if stream_modes:
        run_kwargs["stream_modes"] = stream_modes
    interrupt_before = getattr(payload, "interrupt_before", None)
    if interrupt_before:
        run_kwargs["interrupt_before"] = interrupt_before
    interrupt_after = getattr(payload, "interrupt_after", None)
    if interrupt_after:
        run_kwargs["interrupt_after"] = interrupt_after
    command = getattr(payload, "command", None)
    if command is not None:
        run_kwargs["command"] = command
    durability = getattr(payload, "durability", "async")
    if durability != "async":
        run_kwargs["durability"] = durability
    if getattr(payload, "stream_subgraphs", False):
        run_kwargs["stream_subgraphs"] = True
    try:
        row = await prepare_and_submit_run(
            thread_id=thread_id,
            assistant_id=payload.assistant_id,
            payload=payload.input,
            user=user,
            metadata=payload.metadata,
            kwargs=run_kwargs,
            multitask_strategy=getattr(payload, "multitask_strategy", "enqueue"),
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ActiveThreadRunConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return _to_read_model(row)


_VALID_RUN_STATUSES = {"pending", "running", "error", "success", "timeout", "interrupted"}
_VALID_RUN_SELECT_FIELDS = {
    "run_id", "thread_id", "assistant_id", "created_at", "updated_at",
    "status", "metadata", "kwargs", "multitask_strategy",
}


@router.get("", response_model=list[RunRead])
async def list_runs(
    thread_id: str,
    user: User = Depends(get_current_user),
    limit: int = Query(default=10, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    status: str | None = Query(default=None),
    select_fields: Annotated[list[str] | None, Query(alias="select")] = None,
) -> Any:
    if status is not None and status not in _VALID_RUN_STATUSES:
        raise HTTPException(status_code=422, detail=f"Invalid status filter: {status}")
    query = select(Run).where(Run.thread_id == thread_id, Run.user_id == user.identity)
    if status is not None:
        query = query.where(Run.status == status)
    query = query.order_by(Run.created_at.desc()).limit(limit).offset(offset)
    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        rows = (await session.scalars(query)).all()
    if select_fields:
        fields = set(select_fields) & _VALID_RUN_SELECT_FIELDS
        data = [_to_read_model(row).model_dump(include=fields) for row in rows]
        return JSONResponse(content=jsonable_encoder(data))
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


async def _wait_run_terminal(
    thread_id: str,
    run_id: str,
    user: User,
    *,
    timeout_seconds: float | None = 30.0,
) -> RunRead:
    deadline = None if timeout_seconds is None else asyncio.get_event_loop().time() + timeout_seconds
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
        if deadline is not None and asyncio.get_event_loop().time() > deadline:
            raise HTTPException(status_code=408, detail="Run wait timeout")
        await asyncio.sleep(0.2)


@router.get("/{run_id}/wait", response_model=RunRead)
async def wait_run(thread_id: str, run_id: str, user: User = Depends(get_current_user)) -> RunRead:
    return await _wait_run_terminal(thread_id, run_id, user)


@router.post(
    "/wait",
    response_class=StreamingResponse,
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
) -> StreamingResponse:
    _normalize_stream_modes(payload.stream_mode)
    created = await create_run(thread_id, payload, user)
    return _wait_json_stream_response(
        run=created,
        user=user,
        headers=_wait_response_headers(thread_id=thread_id, run_id=created.run_id),
        cancel_on_disconnect=payload.on_disconnect == "cancel",
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
        cancel_on_disconnect=payload.on_disconnect == "cancel",
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


async def _cancel_active_run(
    *,
    thread_id: str,
    run_id: str,
    user_id: str,
    require_existing: bool = True,
) -> bool:
    """Mark an active run as cancelled and best-effort drop its checkpoints.

    Returns True if the run existed and was transitioned to a terminal state by
    this call. Returns False if the run did not exist (when ``require_existing``
    is False) or was already terminal.
    """
    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        row = await session.scalar(
            select(Run).where(Run.run_id == run_id, Run.thread_id == thread_id, Run.user_id == user_id)
        )
        if row is None:
            if require_existing:
                raise HTTPException(status_code=404, detail="Run not found")
            return False
        if row.status in TERMINAL_RUN_STATUSES:
            return False
        row.status = "error"
        row.last_error = "Run cancelled"
        thread = await session.scalar(
            select(Thread).where(Thread.thread_id == thread_id, Thread.user_id == user_id)
        )
        if thread is not None:
            thread.status = "error"
            thread.state_updated_at = datetime.now(UTC)
        await session.commit()
    await _best_effort_delete_for_runs([run_id])
    return True


@router.post("/{run_id}/cancel")
async def cancel_run(
    thread_id: str,
    run_id: str,
    user: User = Depends(get_current_user),
) -> dict[str, object]:
    await _cancel_active_run(thread_id=thread_id, run_id=run_id, user_id=user.identity)
    return {}


@router.get(
    "/{run_id}/join",
    response_class=StreamingResponse,
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
async def join_run(thread_id: str, run_id: str, user: User = Depends(get_current_user)) -> StreamingResponse:
    run = await _get_run_read(thread_id, run_id, user)
    return _wait_json_stream_response(
        run=run,
        user=user,
        headers=_wait_response_headers(thread_id=thread_id, run_id=run_id),
    )


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
    parsed_last_event_id = parse_last_event_id(last_event_id)
    after_seq = parsed_last_event_id or 0
    replay_existing = parsed_last_event_id is not None
    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        row = await session.scalar(
            select(Run).where(Run.run_id == run_id, Run.thread_id == thread_id, Run.user_id == user.identity)
        )
        if row is None:
            raise HTTPException(status_code=404, detail="Run not found")
        if stream_mode is not None:
            stream_modes = _normalize_stream_modes(_parse_stream_mode_query_param(stream_mode))
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
                replay_existing=replay_existing,
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
            payload = safe_json_dumps(event_payload)
            yield f"id: {seq}\nevent: {event_name}\ndata: {payload}\n\n"

        if use_redis_executor:
            async for item in iter_with_sse_keepalives(
                _iter_persisted_run_records(
                    run_id=run_id,
                    thread_id=thread_id,
                    user_id=user.identity,
                    after_seq=current_seq,
                )
            ):
                if item is None:
                    yield sse_keepalive_comment()
                    continue
                seq, event = item
                event_name = str(event.get("event", "message"))
                event_payload = {"run_id": run_id, **event}
                yield f"id: {seq}\nevent: {event_name}\ndata: {safe_json_dumps(event_payload)}\n\n"
            return

        if row.status in TERMINAL_RUN_STATUSES:
            return

        async for item in iter_with_sse_keepalives(run_broker.stream_records(run_id, after_seq=current_seq)):
            if item is None:
                yield sse_keepalive_comment()
                continue
            seq, event = item
            event_name = str(event.get("event", "message"))
            event_payload = {"run_id": run_id, **event}
            yield f"id: {seq}\nevent: {event_name}\ndata: {safe_json_dumps(event_payload)}\n\n"

    return StreamingResponse(
        _event_iter(),
        media_type="text/event-stream",
        headers=_stream_response_headers(
            location=f"/threads/{thread_id}/runs/{run_id}/stream",
            content_location=f"/threads/{thread_id}/runs/{run_id}",
        ),
    )
