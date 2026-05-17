from sqlalchemy import select

from fastapi import APIRouter, Depends, Response

from agentseek_api.core.auth_deps import get_current_user
from agentseek_api.core.database import db_manager
from agentseek_api.core.orm import Run
from agentseek_api.models.api import RunCreate, RunRead, RunsCancelRequest, ThreadCreate
from agentseek_api.models.auth import User
from agentseek_api.services.thread_service import create_thread_for_user
from agentseek_api.api.runs import create_run, create_run_stream, wait_run

router = APIRouter(prefix="/runs", tags=["Stateless Runs"])


async def _best_effort_delete_for_runs(run_ids: list[str]) -> None:
    try:
        await db_manager.get_langgraph_checkpointer().adelete_for_runs(run_ids)
    except NotImplementedError:
        return


@router.post("", response_model=RunRead)
async def create_stateless_run(payload: RunCreate, user: User = Depends(get_current_user)) -> RunRead:
    thread = await create_thread_for_user(payload=ThreadCreate(metadata={"stateless": True}), user=user)
    return await create_run(thread.thread_id, payload, user)


@router.post("/wait", response_model=RunRead)
async def create_stateless_run_wait(payload: RunCreate, user: User = Depends(get_current_user)) -> RunRead:
    created = await create_stateless_run(payload, user)
    if created.status in {"success", "error", "interrupted"}:
        return created
    return await wait_run(created.thread_id, created.run_id, user)


@router.post("/stream")
async def create_stateless_run_stream(payload: RunCreate, user: User = Depends(get_current_user)):
    thread = await create_thread_for_user(payload=ThreadCreate(metadata={"stateless": True}), user=user)
    return await create_run_stream(thread.thread_id, payload, user)


@router.post("/batch", response_model=list[RunRead])
async def create_run_batch(payload: list[RunCreate], user: User = Depends(get_current_user)) -> list[RunRead]:
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
        for row in rows:
            if row.status not in {"success", "error", "interrupted"}:
                row.status = "error"
                row.last_error = "Run cancelled"
        await session.commit()
    if rows:
        await _best_effort_delete_for_runs([row.run_id for row in rows])
    return Response(status_code=204)
