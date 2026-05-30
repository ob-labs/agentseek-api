from datetime import UTC, datetime

from sqlalchemy import select

from fastapi import APIRouter, Depends, Response
from fastapi.responses import StreamingResponse

from agentseek_api.core.auth_deps import get_current_user
from agentseek_api.core.database import db_manager
from agentseek_api.core.orm import Run, Thread
from agentseek_api.models.api import (
    RunCreateStateless,
    RunCreateStreamingStateless,
    RunRead,
    RunsCancelRequest,
    ThreadCreate,
)
from agentseek_api.models.auth import User
from agentseek_api.services.thread_service import create_thread_for_user
from agentseek_api.api.runs import (
    _build_create_run_stream_response,
    _normalize_stream_modes,
    _protocol_stream_location,
    _stream_response_headers,
    _validate_supported_run_controls,
    _wait_json_stream_response,
    create_run,
)

router = APIRouter(prefix="/runs", tags=["Stateless Runs"])


async def _best_effort_delete_for_runs(run_ids: list[str]) -> None:
    try:
        await db_manager.get_langgraph_checkpointer().adelete_for_runs(run_ids)
    except NotImplementedError:
        return


@router.post("", response_model=RunRead)
async def create_stateless_run(payload: RunCreateStateless, user: User = Depends(get_current_user)) -> RunRead:
    _validate_supported_run_controls(payload, stateless=True)
    thread = await create_thread_for_user(payload=ThreadCreate(metadata={"stateless": True}), user=user)
    return await create_run(thread.thread_id, payload, user)


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
async def create_stateless_run_wait(payload: RunCreateStreamingStateless, user: User = Depends(get_current_user)) -> StreamingResponse:
    _normalize_stream_modes(payload.stream_mode)
    created = await create_stateless_run(payload, user)
    return _wait_json_stream_response(
        run=created,
        user=user,
        headers=_stream_response_headers(
            location=f"/threads/{created.thread_id}/runs/{created.run_id}/join",
            content_location=f"/threads/{created.thread_id}/runs/{created.run_id}",
        ),
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
async def create_stateless_run_stream(payload: RunCreateStreamingStateless, user: User = Depends(get_current_user)):
    stream_modes = _normalize_stream_modes(payload.stream_mode)
    _validate_supported_run_controls(payload, stateless=True)
    thread = await create_thread_for_user(payload=ThreadCreate(metadata={"stateless": True}), user=user)
    created = await create_run(thread.thread_id, payload, user)
    return _build_create_run_stream_response(
        thread_id=thread.thread_id,
        created=created,
        user=user,
        stream_modes=stream_modes,
        after_seq=0,
        location=_protocol_stream_location(thread_id=thread.thread_id, run_id=created.run_id, stream_modes=stream_modes),
        content_location=f"/threads/{thread.thread_id}/runs/{created.run_id}",
    )


@router.post("/batch", response_model=list[RunRead])
async def create_run_batch(payload: list[RunCreateStateless], user: User = Depends(get_current_user)) -> list[RunRead]:
    return [await create_stateless_run(item, user) for item in payload]


@router.post("/cancel", status_code=204)
async def cancel_runs(payload: RunsCancelRequest, user: User = Depends(get_current_user)) -> Response:
    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        query = select(Run).where(Run.user_id == user.identity)
        if payload.thread_id is not None:
            query = query.where(Run.thread_id == payload.thread_id)
        if payload.run_ids:
            query = query.where(Run.run_id.in_(payload.run_ids))
        if payload.status is not None and payload.status != "all":
            query = query.where(Run.status == payload.status)
        rows = (await session.scalars(query)).all()
        cancelled_thread_ids: set[str] = set()
        for row in rows:
            if row.status not in {"success", "error", "interrupted"}:
                row.status = "error"
                row.last_error = "Run cancelled"
                cancelled_thread_ids.add(row.thread_id)
        if cancelled_thread_ids:
            threads = (
                await session.scalars(
                    select(Thread).where(Thread.thread_id.in_(cancelled_thread_ids), Thread.user_id == user.identity)
                )
            ).all()
            for thread in threads:
                thread.status = "error"
                thread.state_updated_at = datetime.now(UTC)
        await session.commit()
    if rows:
        await _best_effort_delete_for_runs([row.run_id for row in rows])
    return Response(status_code=204)
