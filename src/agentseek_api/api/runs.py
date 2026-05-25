import asyncio
import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime

from sqlalchemy import delete, select

from fastapi import APIRouter, Depends, Header, HTTPException, Response
from fastapi.responses import StreamingResponse

from agentseek_api.core.auth_deps import get_current_user
from agentseek_api.core.database import db_manager
from agentseek_api.core.orm import Run, Thread
from agentseek_api.models.api import RunCreate, RunRead, RunResume
from agentseek_api.models.auth import User
from agentseek_api.services.run_preparation import (
    ActiveThreadRunConflictError,
    prepare_and_submit_run,
    resume_run,
)
from agentseek_api.services.run_state import run_broker
from agentseek_api.services.stream_persistence import (
    delete_run_stream_events,
    load_run_stream_events,
    parse_last_event_id,
)
from agentseek_api.settings import settings

router = APIRouter(prefix="/threads/{thread_id}/runs", tags=["Runs"])

TERMINAL_RUN_STATUSES = ("success", "error", "interrupted")
REDIS_STREAM_POLL_INTERVAL_SECONDS = 0.05
REDIS_STREAM_TERMINAL_IDLE_POLLS = 2


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
async def create_run(thread_id: str, payload: RunCreate, user: User = Depends(get_current_user)) -> RunRead:
    try:
        row = await prepare_and_submit_run(
            thread_id=thread_id,
            assistant_id=payload.assistant_id,
            payload=payload.input,
            user=user,
            metadata=payload.metadata,
            kwargs={"config": payload.config, "context": payload.context},
            multitask_strategy=payload.multitask_strategy,
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


@router.post("/wait", response_model=RunRead)
async def create_run_wait(thread_id: str, payload: RunCreate, user: User = Depends(get_current_user)) -> RunRead:
    created = await create_run(thread_id, payload, user)
    if created.status in TERMINAL_RUN_STATUSES:
        return created
    return await wait_run(thread_id, created.run_id, user)


@router.post("/stream")
async def create_run_stream(thread_id: str, payload: RunCreate, user: User = Depends(get_current_user)) -> StreamingResponse:
    created = await create_run(thread_id, payload, user)
    return await stream_run(thread_id, created.run_id, user)


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

    async def _event_iter() -> AsyncIterator[str]:
        current_seq = after_seq
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

        if row.status in TERMINAL_RUN_STATUSES:
            return

        if _uses_redis_executor():
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

        async for seq, event in run_broker.stream_records(run_id, after_seq=current_seq):
            event_name = str(event.get("event", "message"))
            event_payload = {"run_id": run_id, **event}
            yield f"id: {seq}\nevent: {event_name}\ndata: {json.dumps(event_payload)}\n\n"

    return StreamingResponse(_event_iter(), media_type="text/event-stream")
