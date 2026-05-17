from copy import deepcopy
from datetime import UTC, datetime
import json
from collections.abc import AsyncIterator

from sqlalchemy import delete, select

from fastapi import APIRouter, Depends, HTTPException, Response
from fastapi.responses import StreamingResponse

from agentseek_api.core.auth_deps import get_current_user
from agentseek_api.core.database import db_manager
from agentseek_api.core.orm import Run, Thread
from agentseek_api.models.api import ThreadCreate, ThreadPatch, ThreadPruneRequest, ThreadRead, ThreadSearchRequest
from agentseek_api.models.auth import User
from agentseek_api.services.run_state import run_broker
from agentseek_api.services.thread_checkpoint_store import (
    checkpoint_to_payload,
    copy_checkpoints,
    get_checkpoint_by_id,
    list_checkpoints,
    put_checkpoint,
    prune_checkpoints,
)

router = APIRouter(prefix="/threads", tags=["Threads"])


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
    async with session_factory() as session:
        rows = (
            await session.scalars(select(Thread).where(Thread.thread_id.in_(payload.thread_ids), Thread.user_id == user.identity))
        ).all()
        thread_ids = [row.thread_id for row in rows]
        if payload.strategy == "delete":
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
                await session.execute(delete(Run).where(Run.run_id.in_(stale_run_ids), Run.user_id == user.identity))
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported prune strategy: {payload.strategy}")
        await session.commit()
    await prune_checkpoints(thread_ids, strategy=payload.strategy)
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
    checkpoint = await get_checkpoint_by_id(thread_id, checkpoint_id)
    if checkpoint is None:
        raise HTTPException(status_code=404, detail="Checkpoint not found")
    payload = checkpoint_to_payload(checkpoint)
    visible = _visible_checkpoint_payloads(thread, runs, [payload])
    if not visible:
        raise HTTPException(status_code=404, detail="Checkpoint not found")
    return visible[0]


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
