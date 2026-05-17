from copy import deepcopy
from datetime import UTC, datetime
import json
from collections.abc import AsyncIterator
from uuid import uuid4

from sqlalchemy import delete, select

from fastapi import APIRouter, Depends, HTTPException, Response
from fastapi.responses import StreamingResponse

from agentseek_api.core.auth_deps import get_current_user
from agentseek_api.core.database import db_manager
from agentseek_api.core.orm import Run, Thread
from agentseek_api.models.api import ThreadCreate, ThreadPatch, ThreadPruneRequest, ThreadRead, ThreadSearchRequest
from agentseek_api.models.auth import User
from agentseek_api.services.run_state import run_broker

router = APIRouter(prefix="/threads", tags=["Threads"])

_MANUAL_CHECKPOINTS_KEY = "__agentseek_manual_checkpoints__"
_INTERNAL_CONFIG_KEYS = {_MANUAL_CHECKPOINTS_KEY}


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
    return {key: value for key, value in config.items() if key not in _INTERNAL_CONFIG_KEYS}


def _manual_checkpoints(thread: Thread) -> list[dict[str, object]]:
    if not isinstance(thread.config_json, dict):
        return []
    checkpoints = thread.config_json.get(_MANUAL_CHECKPOINTS_KEY, [])
    if not isinstance(checkpoints, list):
        return []
    return [item for item in checkpoints if isinstance(item, dict)]


def _checkpoint_created_at(checkpoint: dict[str, object], *, fallback: datetime) -> datetime:
    raw_created_at = checkpoint.get("created_at")
    if isinstance(raw_created_at, str):
        try:
            return _normalized_datetime(datetime.fromisoformat(raw_created_at))
        except ValueError:
            return _normalized_datetime(fallback)
    return _normalized_datetime(fallback)


def _normalized_datetime(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def _manual_checkpoint_payload(thread: Thread, checkpoint: dict[str, object]) -> dict[str, object]:
    values = checkpoint.get("values", {})
    if not isinstance(values, dict):
        values = {}
    interrupts = checkpoint.get("interrupts", [])
    if not isinstance(interrupts, list):
        interrupts = []
    metadata = checkpoint.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    checkpoint_id = str(checkpoint.get("checkpoint_id", thread.thread_id))
    created_at = _checkpoint_created_at(checkpoint, fallback=thread.created_at)
    return {
        "values": values,
        "next": [],
        "tasks": [],
        "checkpoint": {
            "thread_id": thread.thread_id,
            "checkpoint_ns": checkpoint_id,
            "checkpoint_id": checkpoint_id,
        },
        "metadata": {
            "user_id": str(metadata.get("user_id", thread.user_id)),
            "status": str(metadata.get("status", thread.status)),
        },
        "created_at": created_at,
        "parent_checkpoint": None,
        "interrupts": interrupts,
    }


def _thread_state_payload(*, thread: Thread, run: Run | None) -> dict[str, object]:
    checkpoint_id = run.run_id if run is not None else thread.thread_id
    values = run.output_json if run is not None and run.output_json is not None else {}
    interrupts = []
    if run is not None and isinstance(run.output_json, dict):
        raw_interrupts = run.output_json.get("interrupts", [])
        if isinstance(raw_interrupts, list):
            interrupts = raw_interrupts
    return {
        "values": values,
        "next": [],
        "tasks": [],
        "checkpoint": {
            "thread_id": thread.thread_id,
            "checkpoint_ns": checkpoint_id,
            "checkpoint_id": checkpoint_id,
        },
        "metadata": {
            "user_id": thread.user_id,
            "status": run.status if run is not None else thread.status,
        },
        "created_at": run.created_at if run is not None else thread.created_at,
        "parent_checkpoint": None,
        "interrupts": interrupts,
    }


def _state_payload_created_at(payload: dict[str, object]) -> datetime:
    created_at = payload.get("created_at")
    if isinstance(created_at, datetime):
        return _normalized_datetime(created_at)
    if isinstance(created_at, str):
        try:
            return _normalized_datetime(datetime.fromisoformat(created_at))
        except ValueError:
            pass
    return datetime.min.replace(tzinfo=UTC)


def _latest_state_payload(*, thread: Thread, runs: list[Run]) -> dict[str, object]:
    latest_run_payload = _thread_state_payload(thread=thread, run=runs[0]) if runs else None
    manual_payloads = [_manual_checkpoint_payload(thread, checkpoint) for checkpoint in _manual_checkpoints(thread)]
    latest_manual_payload = max(manual_payloads, key=_state_payload_created_at) if manual_payloads else None
    if latest_run_payload is None and latest_manual_payload is None:
        return _thread_state_payload(thread=thread, run=None)
    if latest_run_payload is None:
        return latest_manual_payload
    if latest_manual_payload is None:
        return latest_run_payload
    if _state_payload_created_at(latest_manual_payload) >= _state_payload_created_at(latest_run_payload):
        return latest_manual_payload
    return latest_run_payload


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


@router.post("", response_model=ThreadRead)
async def create_thread(payload: ThreadCreate, user: User = Depends(get_current_user)) -> ThreadRead:
    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        row = Thread(user_id=user.identity, metadata_json=payload.metadata)
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
    async with session_factory() as session:
        rows = (
            await session.scalars(select(Thread).where(Thread.thread_id.in_(payload.thread_ids), Thread.user_id == user.identity))
        ).all()
        thread_ids = [row.thread_id for row in rows]
        if payload.strategy == "delete":
            await session.execute(delete(Run).where(Run.thread_id.in_(thread_ids), Run.user_id == user.identity))
            await session.execute(delete(Thread).where(Thread.thread_id.in_(thread_ids), Thread.user_id == user.identity))
        elif payload.strategy == "keep_latest":
            for row in rows:
                checkpoints = _manual_checkpoints(row)
                if len(checkpoints) <= 1:
                    continue
                latest_checkpoint = max(
                    checkpoints,
                    key=lambda checkpoint: _checkpoint_created_at(checkpoint, fallback=row.created_at),
                )
                next_config = deepcopy(row.config_json) if isinstance(row.config_json, dict) else {}
                next_config[_MANUAL_CHECKPOINTS_KEY] = [latest_checkpoint]
                row.config_json = next_config
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
                await session.execute(delete(Run).where(Run.run_id.in_(stale_run_ids), Run.user_id == user.identity))
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported prune strategy: {payload.strategy}")
        await session.commit()
    await _best_effort_checkpointer_call("aprune", thread_ids, strategy=payload.strategy)
    return {"pruned_count": len(thread_ids)}


@router.get("/{thread_id}", response_model=ThreadRead)
async def get_thread(thread_id: str, user: User = Depends(get_current_user)) -> ThreadRead:
    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        row = await session.scalar(select(Thread).where(Thread.thread_id == thread_id, Thread.user_id == user.identity))
        if row is None:
            raise HTTPException(status_code=404, detail="Thread not found")
        return _to_read_model(row)


@router.patch("/{thread_id}", response_model=ThreadRead)
async def patch_thread(thread_id: str, payload: ThreadPatch, user: User = Depends(get_current_user)) -> ThreadRead:
    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        row = await session.scalar(select(Thread).where(Thread.thread_id == thread_id, Thread.user_id == user.identity))
        if row is None:
            raise HTTPException(status_code=404, detail="Thread not found")
        if payload.metadata is not None:
            row.metadata_json = payload.metadata
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
    await _best_effort_checkpointer_call("acopy_thread", thread_id, copied.thread_id)
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
                select(Run).where(Run.thread_id == thread_id, Run.user_id == user.identity).order_by(Run.created_at.desc())
            )
        ).all()
    return _latest_state_payload(thread=thread, runs=runs)


@router.get("/{thread_id}/history")
async def get_thread_history(thread_id: str, user: User = Depends(get_current_user)) -> list[dict[str, object]]:
    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        thread = await session.scalar(select(Thread).where(Thread.thread_id == thread_id, Thread.user_id == user.identity))
        if thread is None:
            raise HTTPException(status_code=404, detail="Thread not found")
        runs = (
            await session.scalars(
                select(Run).where(Run.thread_id == thread_id, Run.user_id == user.identity).order_by(Run.created_at.desc())
            )
        ).all()
    run_payloads = [_thread_state_payload(thread=thread, run=run) for run in runs]
    manual_payloads = [_manual_checkpoint_payload(thread, checkpoint) for checkpoint in _manual_checkpoints(thread)]
    return sorted([*run_payloads, *manual_payloads], key=_state_payload_created_at, reverse=True)


@router.post("/{thread_id}/history")
async def get_thread_history_post(thread_id: str, user: User = Depends(get_current_user)) -> list[dict[str, object]]:
    return await get_thread_history(thread_id, user)


@router.get("/{thread_id}/state/{checkpoint_id}")
async def get_thread_state_at_checkpoint(
    thread_id: str,
    checkpoint_id: str,
    user: User = Depends(get_current_user),
) -> dict[str, object]:
    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        thread = await session.scalar(select(Thread).where(Thread.thread_id == thread_id, Thread.user_id == user.identity))
        if thread is None:
            raise HTTPException(status_code=404, detail="Thread not found")
        run = await session.scalar(
            select(Run).where(Run.run_id == checkpoint_id, Run.thread_id == thread_id, Run.user_id == user.identity)
        )
        if run is not None:
            return _thread_state_payload(thread=thread, run=run)
        for checkpoint in _manual_checkpoints(thread):
            if str(checkpoint.get("checkpoint_id")) == checkpoint_id:
                return _manual_checkpoint_payload(thread, checkpoint)
    raise HTTPException(status_code=404, detail="Checkpoint not found")


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
        checkpoint_id = str(uuid4())
        values = payload.get("values", payload)
        if not isinstance(values, dict):
            values = {}
        created_at = datetime.now(UTC)
        checkpoint = {
            "checkpoint_id": checkpoint_id,
            "values": values,
            "interrupts": [],
            "metadata": {
                "user_id": user.identity,
                "status": thread.status,
            },
            "created_at": created_at.isoformat(),
        }
        next_config = deepcopy(thread.config_json) if isinstance(thread.config_json, dict) else {}
        checkpoints = [* _manual_checkpoints(thread), checkpoint]
        next_config[_MANUAL_CHECKPOINTS_KEY] = checkpoints
        thread.config_json = next_config
        thread.state_updated_at = created_at
        await session.commit()
    return {
        "values": values,
        "checkpoint": {
            "thread_id": thread_id,
            "checkpoint_ns": checkpoint_id,
            "checkpoint_id": checkpoint_id,
        },
        "metadata": {"user_id": user.identity, "status": thread.status},
        "created_at": created_at,
        "interrupts": [],
    }


@router.post("/{thread_id}/state/checkpoint")
async def checkpoint_thread_state(
    thread_id: str,
    payload: dict[str, object],
    user: User = Depends(get_current_user),
) -> dict[str, object]:
    return await update_thread_state(thread_id, payload, user)


@router.get("/{thread_id}/stream")
async def stream_thread(thread_id: str, user: User = Depends(get_current_user)) -> StreamingResponse:
    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        thread = await session.scalar(select(Thread).where(Thread.thread_id == thread_id, Thread.user_id == user.identity))
        if thread is None:
            raise HTTPException(status_code=404, detail="Thread not found")
        run_ids = (
            await session.scalars(
                select(Run.run_id).where(Run.thread_id == thread_id, Run.user_id == user.identity).order_by(Run.created_at.asc())
            )
        ).all()

    async def _event_iter() -> AsyncIterator[str]:
        for run_id in run_ids:
            for event in run_broker.snapshot(run_id):
                event_name = str(event.get("event", "message"))
                event_payload: dict[str, object] = {"run_id": run_id, **event}
                yield f"event: {event_name}\ndata: {json.dumps(event_payload)}\n\n"

    return StreamingResponse(_event_iter(), media_type="text/event-stream")


@router.post("/{thread_id}/commands")
async def protocol_command(
    thread_id: str,
    payload: dict[str, object],
    user: User = Depends(get_current_user),
) -> dict[str, object]:
    method = payload.get("method")
    params = payload.get("params", {})
    if not isinstance(params, dict):
        params = {}
    if method == "run.start":
        from agentseek_api.api.runs import create_run
        from agentseek_api.models.api import RunCreate

        run = await create_run(
            thread_id,
            RunCreate(
                assistant_id=str(params.get("assistant_id", "")),
                input=params.get("input", {}),
            ),
            user,
        )
        return {"ok": True, "result": {"run_id": run.run_id}}
    return {"ok": False, "error": {"code": "not_supported", "message": f"Unsupported method: {method}"}}


@router.post("/{thread_id}/stream/events")
async def protocol_event_stream(
    thread_id: str,
    payload: dict[str, object],
    user: User = Depends(get_current_user),
) -> StreamingResponse:
    _ = payload
    return await stream_thread(thread_id, user)
