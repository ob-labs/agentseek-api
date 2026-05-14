import asyncio
import json
from collections.abc import AsyncIterator

from sqlalchemy import select

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from agentseek_api.core.auth_deps import get_current_user
from agentseek_api.core.database import db_manager
from agentseek_api.core.orm import Run
from agentseek_api.models.api import RunCreate, RunRead, RunResume
from agentseek_api.models.auth import User
from agentseek_api.services.run_preparation import prepare_and_submit_run, resume_run
from agentseek_api.services.run_state import run_broker

router = APIRouter(prefix="/threads/{thread_id}/runs", tags=["Runs"])


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
    )


@router.post("", response_model=RunRead)
async def create_run(thread_id: str, payload: RunCreate, user: User = Depends(get_current_user)) -> RunRead:
    try:
        row = await prepare_and_submit_run(
            thread_id=thread_id,
            assistant_id=payload.assistant_id,
            payload=payload.input,
            user=user,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
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
            if row.status in {"success", "error", "interrupted"}:
                return _to_read_model(row)
        if asyncio.get_event_loop().time() > deadline:
            raise HTTPException(status_code=408, detail="Run wait timeout")
        await asyncio.sleep(0.2)


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


@router.get("/{run_id}/stream")
async def stream_run(thread_id: str, run_id: str, user: User = Depends(get_current_user)) -> StreamingResponse:
    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        row = await session.scalar(
            select(Run).where(Run.run_id == run_id, Run.thread_id == thread_id, Run.user_id == user.identity)
        )
        if row is None:
            raise HTTPException(status_code=404, detail="Run not found")

    async def _event_iter() -> AsyncIterator[str]:
        async for event in run_broker.stream(run_id):
            event_name = str(event.get("event", "message"))
            event_payload: dict[str, object] = {"run_id": run_id, **event}
            payload = json.dumps(event_payload)
            yield f"event: {event_name}\ndata: {payload}\n\n"

    return StreamingResponse(_event_iter(), media_type="text/event-stream")
