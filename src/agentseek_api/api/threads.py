import asyncio
import functools
import logging
from collections.abc import AsyncIterator
from copy import deepcopy
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Response
from fastapi.responses import StreamingResponse
from sqlalchemy import delete, func, select

from agentseek_api.core.auth_deps import apply_metadata_filters, authorize, get_current_user
from agentseek_api.core.database import db_manager
from agentseek_api.core.orm import Run, Thread
from agentseek_api.models.api import (
    ThreadCountRequest,
    ThreadCreate,
    ThreadPatch,
    ThreadPruneRequest,
    ThreadPruneResponse,
    ThreadRead,
    ThreadSearchRequest,
    ThreadStateSearch,
)
from agentseek_api.models.auth import User
from agentseek_api.models.protocol import ProtocolEventStreamRequest
from agentseek_api.services.langgraph_service import get_langgraph_service
from agentseek_api.services.sse import (
    iter_with_sse_keepalives,
    safe_json_dumps,
    sse_keepalive_comment,
)
from agentseek_api.services.stream_persistence import (
    delete_run_stream_events,
    delete_thread_stream_events,
    load_thread_stream_events,
    parse_last_event_id,
)
from agentseek_api.services.thread_checkpoint_store import (
    _filter_internal_channels,
    _make_serializable,
    checkpoint_to_payload,
    copy_checkpoints,
    prune_checkpoints,
    put_checkpoint,
    snapshot_to_payload,
)
from agentseek_api.services.thread_protocol import thread_protocol_broker
from agentseek_api.services.thread_service import create_thread_for_user, to_read_model
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


logger = logging.getLogger(__name__)

@functools.lru_cache(maxsize=64)
def _build_compiled_graph_cached(graph_id: str) -> Any:
    entry = get_langgraph_service().get_entry(graph_id)
    return entry.build_graph(checkpointer=None)


def _build_compiled_graph(graph_id: str) -> Any:
    graph = _build_compiled_graph_cached(graph_id)
    checkpointer = db_manager.get_langgraph_checkpointer()
    graph.checkpointer = checkpointer
    return graph


async def _enrich_thread_state(thread_id: str, graph: Any) -> tuple[dict[str, Any], dict[str, Any] | None]:
    try:
        config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}
        snapshot = await graph.aget_state(config)
        if not snapshot or not snapshot.values:
            return {}, None
        raw_values = _make_serializable(snapshot.values)
        values = _filter_internal_channels(raw_values) if isinstance(raw_values, dict) else {}
        interrupts: dict[str, Any] | None = None
        if snapshot.tasks:
            interrupt_list = [
                {"value": getattr(i, "value", None)}
                for task in snapshot.tasks
                for i in (task.interrupts or ())
            ]
            if interrupt_list:
                interrupts = {str(i): item for i, item in enumerate(interrupt_list)}
        return values, interrupts
    except Exception:
        logger.warning("Failed to load graph state for thread %s", thread_id, exc_info=True)
        return {}, None



def _uses_redis_executor() -> bool:
    return settings.EXECUTOR_BACKEND.strip().lower() == "redis"


async def _thread_has_active_runs(*, thread_id: str) -> bool:
    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        active_run_id = await session.scalar(
            select(Run.run_id).where(
                Run.thread_id == thread_id,
                Run.status.not_in(TERMINAL_RUN_STATUSES),
            )
        )
    return active_run_id is not None


async def _iter_persisted_thread_events(
    *,
    thread_id: str,
    payload: ProtocolEventStreamRequest,
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

        if not await _thread_has_active_runs(thread_id=thread_id):
            if wait_for_future_runs:
                await asyncio.sleep(REDIS_STREAM_POLL_INTERVAL_SECONDS)
                continue
            terminal_idle_polls += 1
            if terminal_idle_polls >= REDIS_STREAM_TERMINAL_IDLE_POLLS:
                return
        else:
            terminal_idle_polls = 0

        await asyncio.sleep(REDIS_STREAM_POLL_INTERVAL_SECONDS)



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


def _checkpoint_namespace(payload: dict[str, object]) -> str:
    checkpoint = payload.get("checkpoint")
    if not isinstance(checkpoint, dict):
        return ""
    checkpoint_ns = checkpoint.get("checkpoint_ns", "")
    return str(checkpoint_ns) if checkpoint_ns is not None else ""




@router.post("", response_model=ThreadRead, response_model_exclude_none=True)
async def create_thread(payload: ThreadCreate, user: User = Depends(get_current_user)) -> ThreadRead:
    if payload.ttl is not None:
        raise HTTPException(status_code=422, detail="'ttl' is not supported yet")
    if payload.supersteps is not None:
        raise HTTPException(status_code=422, detail="'supersteps' is not supported yet")

    value: dict = {"metadata": dict(payload.metadata or {})}
    if payload.thread_id is not None:
        value["thread_id"] = str(payload.thread_id)
    if payload.if_exists is not None:
        value["if_exists"] = payload.if_exists
    await authorize(user, "threads", "create", value)
    payload.metadata = value.get("metadata", payload.metadata)
    return await create_thread_for_user(payload=payload, user=user)


@router.post("/search", response_model=list[ThreadRead], response_model_exclude_none=True)
async def search_threads(payload: ThreadSearchRequest, user: User = Depends(get_current_user)) -> list[ThreadRead]:
    if payload.values is not None:
        raise HTTPException(status_code=422, detail="'values' filter is not supported yet")
    if payload.extract is not None:
        raise HTTPException(status_code=422, detail="'extract' is not supported yet")

    sort_by = payload.sort_by or "created_at"
    sort_order = payload.sort_order or "desc"
    sort_column = getattr(Thread, sort_by, Thread.created_at)
    order_clause = sort_column.desc() if sort_order == "desc" else sort_column.asc()

    value: dict = {"metadata": dict(payload.metadata or {}), "limit": payload.limit, "offset": payload.offset}
    if payload.status is not None:
        value["status"] = payload.status
    filters = await authorize(user, "threads", "search", value)

    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        stmt = select(Thread)
        stmt = apply_metadata_filters(stmt, Thread, filters)
        if payload.ids is not None:
            stmt = stmt.where(Thread.thread_id.in_(payload.ids))
        if payload.status is not None:
            stmt = stmt.where(Thread.status == payload.status)
        if payload.metadata is not None:
            for key, value in payload.metadata.items():
                stmt = stmt.where(Thread.metadata_json[key].as_string() == str(value))
        stmt = stmt.order_by(order_clause).offset(payload.offset).limit(payload.limit)
        rows = (await session.scalars(stmt)).all()

    selected_fields = set(payload.select) if payload.select else None

    need_state = selected_fields is None or "values" in selected_fields or "interrupts" in selected_fields
    if need_state:
        _STATE_BATCH_SIZE = 5

        async def _get_state(row: Thread) -> tuple[dict[str, Any], dict[str, Any] | None]:
            gid = (row.metadata_json or {}).get("graph_id")
            if not gid:
                return {}, None
            return await _enrich_thread_state(row.thread_id, _build_compiled_graph(gid))

        states: list[tuple[dict[str, Any], dict[str, Any] | None]] = []
        for i in range(0, len(rows), _STATE_BATCH_SIZE):
            batch = rows[i:i + _STATE_BATCH_SIZE]
            batch_states = await asyncio.gather(*[_get_state(row) for row in batch])
            states.extend(batch_states)
        return [
            to_read_model(row, select=selected_fields, values=values, interrupts=interrupts)
            for row, (values, interrupts) in zip(rows, states)
        ]
    return [to_read_model(row, select=selected_fields) for row in rows]


@router.post("/count", response_model=int)
async def count_threads(payload: ThreadCountRequest, user: User = Depends(get_current_user)) -> int:
    if payload.values is not None:
        raise HTTPException(status_code=422, detail="'values' filter is not supported yet")

    value: dict = {"metadata": dict(payload.metadata or {})}
    if payload.status is not None:
        value["status"] = payload.status
    filters = await authorize(user, "threads", "search", value)

    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        stmt = select(func.count()).select_from(Thread)
        stmt = apply_metadata_filters(stmt, Thread, filters)
        if payload.status is not None:
            stmt = stmt.where(Thread.status == payload.status)
        if payload.metadata is not None:
            for key, value in payload.metadata.items():
                stmt = stmt.where(Thread.metadata_json[key].as_string() == str(value))
        result = await session.scalar(stmt)
        return result or 0


@router.post("/prune", response_model=ThreadPruneResponse)
async def prune_threads(payload: ThreadPruneRequest, user: User = Depends(get_current_user)) -> ThreadPruneResponse:
    filters = await authorize(user, "threads", "delete", {"thread_ids": payload.thread_ids})

    session_factory = db_manager.get_session_factory()
    pruned_run_ids: list[str] = []
    async with session_factory() as session:
        thread_stmt = select(Thread).where(Thread.thread_id.in_(payload.thread_ids))
        thread_stmt = apply_metadata_filters(thread_stmt, Thread, filters)
        rows = (await session.scalars(thread_stmt)).all()
        thread_ids = [row.thread_id for row in rows]
        if payload.strategy == "delete":
            pruned_run_ids = (
                await session.scalars(select(Run.run_id).where(Run.thread_id.in_(thread_ids)))
            ).all()
            await session.execute(delete(Run).where(Run.thread_id.in_(thread_ids)))
            await session.execute(delete(Thread).where(Thread.thread_id.in_(thread_ids)))
        elif payload.strategy == "keep_latest":
            runs = (
                await session.scalars(
                    select(Run)
                    .where(Run.thread_id.in_(thread_ids))
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
                await session.execute(delete(Run).where(Run.run_id.in_(stale_run_ids)))
        await session.commit()
    await prune_checkpoints(thread_ids, strategy=payload.strategy)
    if pruned_run_ids:
        await delete_run_stream_events(pruned_run_ids)
    if payload.strategy == "delete":
        for thread_id in thread_ids:
            thread_protocol_broker.delete_thread(thread_id)
            await delete_thread_stream_events(thread_id)
    return ThreadPruneResponse(pruned_count=len(thread_ids))


@router.get("/{thread_id}", response_model=ThreadRead, response_model_exclude_none=True)
async def get_thread(
    thread_id: str,
    include: list[str] | None = Query(None, description="Additional fields to include (e.g. 'ttl')"),
    user: User = Depends(get_current_user),
) -> ThreadRead:
    filters = await authorize(user, "threads", "read", {"thread_id": thread_id})

    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        stmt = select(Thread).where(Thread.thread_id == thread_id)
        stmt = apply_metadata_filters(stmt, Thread, filters)
        row = await session.scalar(stmt)
        if row is None:
            raise HTTPException(status_code=404, detail="Thread not found")
        graph_id = (row.metadata_json or {}).get("graph_id")
        if graph_id:
            graph = _build_compiled_graph(graph_id)
            values, interrupts = await _enrich_thread_state(thread_id, graph)
        else:
            values, interrupts = {}, None
        return to_read_model(row, values=values, interrupts=interrupts)

@router.patch("/{thread_id}", response_model=ThreadRead, response_model_exclude_none=True)
async def patch_thread(thread_id: str, payload: ThreadPatch, user: User = Depends(get_current_user)) -> ThreadRead:
    filters = await authorize(user, "threads", "update", {"thread_id": thread_id, "metadata": dict(payload.metadata or {})})

    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        stmt = select(Thread).where(Thread.thread_id == thread_id)
        stmt = apply_metadata_filters(stmt, Thread, filters)
        row = await session.scalar(stmt)
        if row is None:
            raise HTTPException(status_code=404, detail="Thread not found")
        if payload.metadata is not None:
            row.metadata_json = {**row.metadata_json, **payload.metadata}
        await session.commit()
        await session.refresh(row)
        return to_read_model(row)


@router.delete("/{thread_id}", status_code=204)
async def delete_thread(thread_id: str, user: User = Depends(get_current_user)) -> Response:
    filters = await authorize(user, "threads", "delete", {"thread_id": thread_id})

    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        stmt = select(Thread).where(Thread.thread_id == thread_id)
        stmt = apply_metadata_filters(stmt, Thread, filters)
        row = await session.scalar(stmt)
        if row is None:
            raise HTTPException(status_code=404, detail="Thread not found")
        run_ids = (
            await session.scalars(select(Run.run_id).where(Run.thread_id == thread_id))
        ).all()
        await session.execute(delete(Run).where(Run.thread_id == thread_id))
        await session.delete(row)
        await session.commit()
    await _best_effort_checkpointer_call("adelete_thread", thread_id)
    if run_ids:
        await _best_effort_checkpointer_call("adelete_for_runs", list(run_ids))
        await delete_run_stream_events(list(run_ids))
    thread_protocol_broker.delete_thread(thread_id)
    await delete_thread_stream_events(thread_id)
    return Response(status_code=204)


@router.post("/{thread_id}/copy", response_model=ThreadRead, response_model_exclude_none=True)
async def copy_thread(thread_id: str, user: User = Depends(get_current_user)) -> ThreadRead:
    filters = await authorize(user, "threads", "create", {"thread_id": thread_id})

    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        stmt = select(Thread).where(Thread.thread_id == thread_id)
        stmt = apply_metadata_filters(stmt, Thread, filters)
        source = await session.scalar(stmt)
        if source is None:
            raise HTTPException(status_code=404, detail="Thread not found")
        source_runs = (
            await session.scalars(
                select(Run).where(Run.thread_id == thread_id).order_by(Run.created_at.asc())
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
    return to_read_model(copied)


async def get_thread_state_internal(
    thread_id: str,
    user: User,
    checkpoint_ns: str | None = None,
    filters: dict[str, Any] | None = None,
) -> dict[str, object] | None:
    if filters is None:
        filters = await authorize(user, "threads", "read", {"thread_id": thread_id})

    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        stmt = select(Thread).where(Thread.thread_id == thread_id)
        stmt = apply_metadata_filters(stmt, Thread, filters)
        thread = await session.scalar(stmt)
        if thread is None:
            raise HTTPException(status_code=404, detail="Thread not found")

    graph_id = (thread.metadata_json or {}).get("graph_id")
    if not graph_id:
        return None

    graph = _build_compiled_graph(graph_id)

    config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}
    if checkpoint_ns is not None:
        config["configurable"]["checkpoint_ns"] = checkpoint_ns

    snapshot = await graph.aget_state(config)
    if snapshot is None or snapshot.config is None:
        return None

    return snapshot_to_payload(snapshot, thread_id)


@router.get("/{thread_id}/state")
async def get_thread_state(
    thread_id: str,
    subgraphs: bool = Query(False, description="Whether to include subgraphs in the response"),
    checkpoint_ns: str | None = Query(None, description="Checkpoint namespace to scope lookup"),
    user: User = Depends(get_current_user),
) -> dict[str, object]:
    if subgraphs is True:
        raise HTTPException(status_code=422, detail="'subgraphs' is not supported yet")

    filters = await authorize(user, "threads", "read", {"thread_id": thread_id})

    ns = checkpoint_ns if isinstance(checkpoint_ns, str) else None
    state = await get_thread_state_internal(thread_id, user, checkpoint_ns=ns, filters=filters)
    if state is not None:
        return state

    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        stmt = select(Thread).where(Thread.thread_id == thread_id)
        stmt = apply_metadata_filters(stmt, Thread, filters)
        thread = await session.scalar(stmt)
        if thread is None:
            raise HTTPException(status_code=404, detail="Thread not found")
    return _empty_thread_state_payload(thread)


@router.get("/{thread_id}/history")
async def get_thread_history(
    thread_id: str,
    user: User = Depends(get_current_user),
    limit: int = Query(default=10, ge=1),
    before: str | None = Query(default=None),
) -> list[dict[str, object]]:
    filters = await authorize(user, "threads", "read", {"thread_id": thread_id})

    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        stmt = select(Thread).where(Thread.thread_id == thread_id)
        stmt = apply_metadata_filters(stmt, Thread, filters)
        thread = await session.scalar(stmt)
        if thread is None:
            raise HTTPException(status_code=404, detail="Thread not found")

        graph_id = (thread.metadata_json or {}).get("graph_id")
        if not graph_id:
            logger.info("history: no graph_id for thread %s", thread_id)
            return []

        runs = (
            await session.scalars(
                select(Run).where(Run.thread_id == thread_id).order_by(Run.created_at.asc())
            )
        ).all()

    graph = _build_compiled_graph(graph_id)

    config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}
    history_kwargs: dict[str, Any] = {"limit": limit}
    if before is not None:
        history_kwargs["before"] = {"configurable": {"checkpoint_id": before}}
    payloads: list[dict[str, object]] = []
    async for snapshot in graph.aget_state_history(config, **history_kwargs):
        payloads.append(snapshot_to_payload(snapshot, thread_id))

    return _visible_checkpoint_payloads(thread, runs, payloads)


@router.post("/{thread_id}/history")
async def get_thread_history_post(
    thread_id: str,
    payload: ThreadStateSearch,
    user: User = Depends(get_current_user),
) -> list[dict[str, object]]:
    before_id: str | None = None
    if payload.before is not None and payload.before.checkpoint_id:
        before_id = payload.before.checkpoint_id
    return await get_thread_history(thread_id, user, limit=payload.limit, before=before_id)


async def _get_thread_state_at_checkpoint(*, thread_id: str, checkpoint_id: str, user: User) -> dict[str, object]:
    filters = await authorize(user, "threads", "read", {"thread_id": thread_id})

    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        stmt = select(Thread).where(Thread.thread_id == thread_id)
        stmt = apply_metadata_filters(stmt, Thread, filters)
        thread = await session.scalar(stmt)
        if thread is None:
            raise HTTPException(status_code=404, detail="Thread not found")

    graph_id = (thread.metadata_json or {}).get("graph_id")
    if not graph_id:
        if checkpoint_id == thread.thread_id:
            return _empty_thread_state_payload(thread)
        raise HTTPException(status_code=404, detail="Checkpoint not found")

    graph = _build_compiled_graph(graph_id)

    config: dict[str, Any] = {"configurable": {"thread_id": thread_id, "checkpoint_id": checkpoint_id}}
    snapshot = await graph.aget_state(config)
    if snapshot is None or snapshot.config is None:
        if checkpoint_id == thread.thread_id:
            return _empty_thread_state_payload(thread)
        raise HTTPException(status_code=404, detail="Checkpoint not found")

    return snapshot_to_payload(snapshot, thread_id)


@router.get("/{thread_id}/state/{checkpoint_id}")
async def get_thread_state_at_checkpoint(
    thread_id: str,
    checkpoint_id: str,
    subgraphs: bool = Query(False, description="Whether to include subgraphs in the response"),
    checkpoint_ns: str | None = Query(None, description="Checkpoint namespace to scope lookup"),
    user: User = Depends(get_current_user),
) -> dict[str, object]:
    if subgraphs is True:
        raise HTTPException(status_code=422, detail="'subgraphs' is not supported yet")
    return await _get_thread_state_at_checkpoint(thread_id=thread_id, checkpoint_id=checkpoint_id, user=user)


@router.post("/{thread_id}/state")
async def update_thread_state(
    thread_id: str,
    payload: dict[str, object],
    user: User = Depends(get_current_user),
) -> dict[str, object]:
    filters = await authorize(user, "threads", "update", {"thread_id": thread_id})

    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        stmt = select(Thread).where(Thread.thread_id == thread_id)
        stmt = apply_metadata_filters(stmt, Thread, filters)
        thread = await session.scalar(stmt)
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


@router.post("/{thread_id}/state/checkpoint",summary="Get Thread State At Checkpoint Post")
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
async def join_thread_stream(
    thread_id: str,
    user: User = Depends(get_current_user),
    last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
) -> StreamingResponse:
    after_seq = parse_last_event_id(last_event_id) or 0
    filters = await authorize(user, "threads", "read", {"thread_id": thread_id})

    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        stmt = select(Thread).where(Thread.thread_id == thread_id)
        stmt = apply_metadata_filters(stmt, Thread, filters)
        thread = await session.scalar(stmt)
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
            yield f"id: {seq}\nevent: {event_name}\ndata: {safe_json_dumps(event)}\n\n"

        if _uses_redis_executor():
            async for event in iter_with_sse_keepalives(
                _iter_persisted_thread_events(
                    thread_id=thread_id,
                    payload=payload,
                    after_seq=current_seq,
                    wait_for_future_runs=True,
                )
            ):
                if event is None:
                    yield sse_keepalive_comment()
                    continue
                seq = int(event.get("seq", 0))
                current_seq = max(current_seq, seq)
                event_name = str(event.get("method", "event"))
                yield f"id: {seq}\nevent: {event_name}\ndata: {safe_json_dumps(event)}\n\n"
            return

        async for event in iter_with_sse_keepalives(
            thread_protocol_broker.stream(
                thread_id=thread_id,
                channels=payload.channels,
                namespaces=payload.namespaces,
                depth=payload.depth,
                since=current_seq,
                wait_for_future_runs=True,
            )
        ):
            if event is None:
                yield sse_keepalive_comment()
                continue
            seq = int(event.get("seq", 0))
            current_seq = max(current_seq, seq)
            event_name = str(event.get("method", "event"))
            yield f"id: {seq}\nevent: {event_name}\ndata: {safe_json_dumps(event)}\n\n"

    return StreamingResponse(_event_iter(), media_type="text/event-stream")


