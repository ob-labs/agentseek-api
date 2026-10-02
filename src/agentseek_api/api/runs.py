import asyncio
import json
from collections.abc import AsyncIterator
from typing import Annotated, Any, Literal
from urllib.parse import urlencode

from sqlalchemy import delete, select

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, Response, StreamingResponse

from agentseek_api.core.auth_deps import apply_metadata_filters, authorize, get_current_user
from agentseek_api.core.database import db_manager
from agentseek_api.core.orm import Run, Thread
from agentseek_api.models.api import ErrorDetailResponse, RunCreateStateful, RunCreateStreamingStateful, RunRead, RunResume
from agentseek_api.models.auth import User
from agentseek_api.api.threads import get_thread_state_internal
from agentseek_api.services.thread_checkpoint_store import snapshot_to_payload
from agentseek_api.services.run_preparation import (
    ActiveThreadRunConflictError,
    prepare_and_submit_run,
    resume_run,
)
from agentseek_api.services.stream_persistence import (
    delete_run_stream_events,
    load_run_stream_events,
    parse_last_event_id,
)
from agentseek_api.services.sse import iter_with_sse_keepalives, safe_json_dumps, sse_keepalive_comment
from agentseek_api.services.stream_modes import (
    SUPPORTED_RUN_STREAM_MODES,
    normalize_stream_modes as _normalize_stream_modes_shared,
)
from agentseek_api.services.thread_protocol import protocol_channel_for_method
from agentseek_api.settings import settings

router = APIRouter(prefix="/threads/{thread_id}/runs", tags=["Thread Runs"])

TERMINAL_RUN_STATUSES = ("success", "error", "interrupted")
REDIS_STREAM_POLL_INTERVAL_SECONDS = 0.05
REDIS_STREAM_TERMINAL_IDLE_POLLS = 2
RUN_CHECKPOINT_ID_METADATA_KEY = "__agentseek_checkpoint_id"


async def _has_thread_access(thread_id: str, user: User, *, action: str = "read") -> bool:
    filters = await authorize(user, "threads", action, {"thread_id": thread_id})
    if not filters:
        return True
    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        stmt = select(Thread.thread_id).where(Thread.thread_id == thread_id)
        stmt = apply_metadata_filters(stmt, Thread, filters)
        return await session.scalar(stmt) is not None


async def _verify_thread_access(thread_id: str, user: User, *, action: str = "read") -> None:
    if not await _has_thread_access(thread_id, user, action=action):
        raise HTTPException(status_code=404, detail="Thread not found")


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
        status="running" if run.status == "terminal_pending" else run.status,
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


def _format_run_error(last_error: str | None, run_id: str) -> dict[str, Any]:
    if last_error and ": " in last_error:
        error_type, _, message = last_error.partition(": ")
        return {"error": error_type, "message": message, "run_id": run_id}
    return {"error": last_error or "Unknown", "message": last_error or "", "run_id": run_id}


def _uses_redis_executor() -> bool:
    return settings.EXECUTOR_BACKEND.strip().lower() == "redis"


def _normalize_stream_modes(stream_mode: str | list[str] | None) -> list[str]:
    try:
        return _normalize_stream_modes_shared(stream_mode)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


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

    config = getattr(payload, "config", None) or {}
    context = getattr(payload, "context", None) or {}
    if isinstance(config, dict) and config.get("configurable") and context:
        raise HTTPException(
            status_code=400,
            detail=(
                "Cannot specify both configurable and context. Prefer setting context alone. "
                "Context was introduced in LangGraph 0.6.0 and is the long term planned "
                "replacement for configurable."
            ),
        )

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


async def _get_run_state(*, thread_id: str, run_id: str, user: User, checkpoint_id: str | None = None) -> dict[str, object] | None:
    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        thread = await session.scalar(select(Thread).where(Thread.thread_id == thread_id))
        if thread is None:
            return None

    graph_id = (thread.metadata_json or {}).get("graph_id")
    if not graph_id:
        return None

    from agentseek_api.api.threads import _build_compiled_graph
    graph = _build_compiled_graph(graph_id)

    config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}
    if checkpoint_id is not None:
        config["configurable"]["checkpoint_id"] = checkpoint_id
    else:
        config["configurable"]["checkpoint_ns"] = run_id

    snapshot = await graph.aget_state(config)
    if snapshot is None or snapshot.config is None:
        return None
    return snapshot_to_payload(snapshot, thread_id)


async def _wait_response_payload(run: RunRead, *, user: User) -> Any:
    if run.status == "interrupted" and run.interrupts:
        return {"__interrupt__": run.interrupts}
    if run.status == "error":
        return {"__error__": run.last_error} if run.last_error else {}
    checkpoint_id = await _load_run_checkpoint_id(run_id=run.run_id, thread_id=run.thread_id)
    state = await _get_run_state(
        thread_id=run.thread_id,
        run_id=run.run_id,
        user=user,
        checkpoint_id=checkpoint_id,
    )
    if state is None:
        state = await get_thread_state_internal(run.thread_id, user)
    if isinstance(state, dict) and "values" in state:
        values = state["values"]
        if isinstance(values, dict):
            values.pop("__pregel_tasks", None)
        return values
    return {}


async def _load_run_checkpoint_id(*, run_id: str, thread_id: str) -> str | None:
    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        row = await session.scalar(
            select(Run).where(Run.run_id == run_id, Run.thread_id == thread_id)
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
            select(Run).where(Run.run_id == run_id, Run.thread_id == thread_id)
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
            yield safe_json_dumps(jsonable_encoder(payload), separators=(",", ":")).encode("utf-8")
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
    protocol_channels = {
        *[mode for mode in stream_modes if mode in SUPPORTED_RUN_STREAM_MODES], "input"
    }

    async def _event_iter() -> AsyncIterator[str]:
        try:
            current_seq = after_seq
            saw_interrupt = False
            if include_metadata:
                yield _protocol_event_sse(event_name="metadata", data={"run_id": created.run_id, "attempt": 1})
            if not replay_existing:
                existing = await load_run_stream_events(created.run_id, after_seq=current_seq)
                # Joining without a cursor skips historical protocol frames, but
                # must still report an already committed terminal outcome.
                current_seq = max((seq for seq, event in existing if event.get("event") != "end"), default=current_seq)
            async for item in iter_with_sse_keepalives(_iter_persisted_run_records(
                run_id=created.run_id, thread_id=thread_id, after_seq=current_seq,
            )):
                if item is None:
                    yield sse_keepalive_comment()
                    continue
                seq, event = item
                if event.get("event") == "end":
                    if event.get("status") == "error":
                        yield _protocol_event_sse(seq=seq, event_name="error",
                            data=_format_run_error(event.get("error"), created.run_id))
                    else:
                        interrupt_event = _interrupt_stream_event_name(stream_modes)
                        if event.get("status") == "interrupted" and event.get("interrupts") and interrupt_event and not saw_interrupt:
                            yield _protocol_event_sse(seq=seq, event_name=interrupt_event,
                                data={"__interrupt__": event["interrupts"]})
                        else:
                            yield _protocol_event_sse(seq=seq, event_name="end", data={})
                    continue
                method = str(event.get("method", ""))
                if protocol_channel_for_method(method) not in protocol_channels or method == "messages":
                    continue
                event_data = event.get("params", {}).get("data", {})
                if isinstance(event_data, dict) and "__interrupt__" in event_data:
                    saw_interrupt = True
                yield _protocol_event_sse(seq=seq, event_name=method, data=event_data)
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


async def _is_run_terminal(*, run_id: str, thread_id: str) -> bool:
    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        status = await session.scalar(
            select(Run.status).where(Run.run_id == run_id, Run.thread_id == thread_id)
        )
    return status is None or status in TERMINAL_RUN_STATUSES


async def _iter_persisted_run_records(
    *,
    run_id: str,
    thread_id: str,
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

        if await _is_run_terminal(run_id=run_id, thread_id=thread_id):
            terminal_idle_polls += 1
            if terminal_idle_polls >= REDIS_STREAM_TERMINAL_IDLE_POLLS:
                return
        else:
            terminal_idle_polls = 0

        await asyncio.sleep(REDIS_STREAM_POLL_INTERVAL_SECONDS)


@router.post("", response_model=RunRead, response_model_exclude_none=True)
async def create_run(thread_id: str, payload: RunCreateStateful, user: User = Depends(get_current_user)) -> RunRead:
    await _verify_thread_access(thread_id, user, action="create_run")
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
        run_kwargs["command"] = command.model_dump(exclude_none=True)
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


_RunStatus = Literal["pending", "running", "error", "success", "timeout", "interrupted"]
_RunSelectField = Literal[
    "run_id", "thread_id", "assistant_id", "created_at", "updated_at",
    "status", "metadata", "kwargs", "multitask_strategy",
]
_VALID_RUN_SELECT_FIELDS = {
    "run_id", "thread_id", "assistant_id", "created_at", "updated_at",
    "status", "metadata", "kwargs", "multitask_strategy",
}


@router.get(
    "",
    response_model=list[RunRead],
    response_model_exclude_none=True,
    responses={404: {"model": ErrorDetailResponse}},
)
async def list_runs(
    thread_id: str,
    user: User = Depends(get_current_user),
    limit: int = Query(default=10, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    status: _RunStatus | None = Query(default=None),
    select_fields: Annotated[list[_RunSelectField] | None, Query(alias="select")] = None,
) -> Any:
    await _verify_thread_access(thread_id, user)
    query = select(Run).where(Run.thread_id == thread_id)
    if status is not None:
        query = query.where(Run.status.in_(["running", "terminal_pending"]) if status == "running" else Run.status == status)
    query = query.order_by(Run.created_at.desc()).limit(limit).offset(offset)
    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        rows = (await session.scalars(query)).all()
    if select_fields:
        fields = set(select_fields) & _VALID_RUN_SELECT_FIELDS
        data = [_to_read_model(row).model_dump(include=fields) for row in rows]
        return JSONResponse(content=jsonable_encoder(data))
    return [_to_read_model(row) for row in rows]


@router.get("/{run_id}", response_model=RunRead, response_model_exclude_none=True)
async def get_run(thread_id: str, run_id: str, user: User = Depends(get_current_user)) -> RunRead:
    await _verify_thread_access(thread_id, user)
    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        row = await session.scalar(
            select(Run).where(Run.run_id == run_id, Run.thread_id == thread_id)
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
                select(Run).where(Run.run_id == run_id, Run.thread_id == thread_id)
            )
            if row is None:
                raise HTTPException(status_code=404, detail="Run not found")
            if row.status in TERMINAL_RUN_STATUSES:
                return _to_read_model(row)
        if deadline is not None and asyncio.get_event_loop().time() > deadline:
            raise HTTPException(status_code=408, detail="Run wait timeout")
        await asyncio.sleep(0.2)


@router.get("/{run_id}/wait", response_model=RunRead, response_model_exclude_none=True)
async def wait_run(thread_id: str, run_id: str, user: User = Depends(get_current_user)) -> RunRead:
    await _verify_thread_access(thread_id, user)
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
    after_seq = 0
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
    await _verify_thread_access(thread_id, user, action="create_run")
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
            select(Run).where(Run.run_id == run_id, Run.thread_id == thread_id)
        )
        if row is None:
            if require_existing:
                raise HTTPException(status_code=404, detail="Run not found")
            return False
        if row.status in TERMINAL_RUN_STATUSES:
            return False
    from agentseek_api.services.run_jobs import RunExecutionJob
    from agentseek_api.services.terminal_delivery import TerminalResult, finish_run
    cancelled = await finish_run(RunExecutionJob(
        run_id=run_id, thread_id=thread_id, user_id=user_id, graph_id="default", payload=None,
    ), TerminalResult(status="error", error="Run cancelled"), cancel=True)
    if not cancelled:
        return False
    await _best_effort_delete_for_runs([run_id])
    return True


@router.post("/{run_id}/cancel")
async def cancel_run(
    thread_id: str,
    run_id: str,
    wait: bool = Query(False),
    action: Literal["interrupt", "rollback"] = Query("interrupt"),
    user: User = Depends(get_current_user),
) -> dict[str, object]:
    await _verify_thread_access(thread_id, user, action="update")
    cancelled = await _cancel_active_run(thread_id=thread_id, run_id=run_id, user_id=user.identity)
    if wait and cancelled:
        await _wait_run_terminal(thread_id, run_id, user)
    if action == "rollback" and cancelled:
        session_factory = db_manager.get_session_factory()
        async with session_factory() as session:
            await session.execute(delete(Run).where(Run.run_id == run_id, Run.thread_id == thread_id))
            await session.commit()
        await delete_run_stream_events([run_id])
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
async def join_run(
    thread_id: str,
    run_id: str,
    cancel_on_disconnect: bool = Query(False, description="If true, the run will be cancelled if the client disconnects."),
    user: User = Depends(get_current_user),
) -> StreamingResponse:
    await _verify_thread_access(thread_id, user)
    run = await _get_run_read(thread_id, run_id, user)
    return _wait_json_stream_response(
        run=run,
        user=user,
        headers=_wait_response_headers(thread_id=thread_id, run_id=run_id),
        cancel_on_disconnect=cancel_on_disconnect,
    )


@router.delete("/{run_id}", status_code=204)
async def delete_run(thread_id: str, run_id: str, user: User = Depends(get_current_user)) -> Response:
    await _verify_thread_access(thread_id, user, action="delete")
    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        row = await session.scalar(
            select(Run).where(Run.run_id == run_id, Run.thread_id == thread_id)
        )
        if row is None:
            raise HTTPException(status_code=404, detail="Run not found")
        await session.execute(delete(Run).where(Run.run_id == run_id, Run.thread_id == thread_id))
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
    cancel_on_disconnect: bool = Query(False, description="If true, the run will be cancelled if the client disconnects."),
) -> StreamingResponse:
    parsed_last_event_id = parse_last_event_id(last_event_id)
    after_seq = parsed_last_event_id or 0
    replay_existing = parsed_last_event_id is not None
    await _verify_thread_access(thread_id, user)
    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        row = await session.scalar(
            select(Run).where(Run.run_id == run_id, Run.thread_id == thread_id)
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
                cancel_on_disconnect=cancel_on_disconnect,
            )

    async def _event_iter() -> AsyncIterator[str]:
        # Brokers are notifications only. Read the durable cursor domain on
        # both backends, including when notifications are delayed or lost.
        async for item in iter_with_sse_keepalives(_iter_persisted_run_records(
            run_id=run_id, thread_id=thread_id, after_seq=after_seq,
        )):
            if item is None:
                yield sse_keepalive_comment()
                continue
            seq, event = item
            if "method" in event:
                yield _protocol_event_sse(seq=seq, event_name=str(event["method"]),
                    data=event.get("params", {}).get("data", {}))
            else:
                yield _protocol_event_sse(seq=seq, event_name=str(event.get("event", "message")),
                    data={"run_id": run_id, **event})

    return StreamingResponse(
        _event_iter(),
        media_type="text/event-stream",
        headers=_stream_response_headers(
            location=f"/threads/{thread_id}/runs/{run_id}/stream",
            content_location=f"/threads/{thread_id}/runs/{run_id}",
        ),
    )
