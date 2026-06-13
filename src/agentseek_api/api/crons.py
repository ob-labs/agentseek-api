from sqlalchemy import select

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response

from agentseek_api.core.auth_deps import authorize, apply_metadata_filters, get_current_user
from agentseek_api.core.database import db_manager
from agentseek_api.core.orm import Assistant, Thread
from agentseek_api.models.api import (
    CronCountRequest,
    CronCountResponse,
    CronCreate,
    CronPatch,
    CronRead,
    CronSearchRequest,
    CronSearchResponse,
)
from agentseek_api.models.auth import User
from agentseek_api.services import cron_service
from agentseek_api.services.default_assistants import resolve_assistant_id

router = APIRouter(tags=["Crons"])


async def _ensure_assistant_exists(*, assistant_id: str) -> str:
    resolved_id = resolve_assistant_id(assistant_id)
    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        existing = await session.scalar(select(Assistant.assistant_id).where(Assistant.assistant_id == resolved_id))
        if existing is None:
            raise HTTPException(status_code=404, detail="Assistant not found")
    return resolved_id


async def _create_cron(
    *,
    assistant_id: str,
    thread_id: str | None,
    payload: CronCreate,
    user: User,
) -> CronRead:
    try:
        return await cron_service.create_cron(assistant_id=assistant_id, thread_id=thread_id, payload=payload, user=user)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/runs/crons", response_model=CronRead, response_model_exclude_none=True)
async def create_stateless_cron(payload: CronCreate, user: User = Depends(get_current_user)) -> CronRead:
    value: dict = {
        "metadata": dict(payload.metadata) if payload.metadata else {},
        "assistant_id": payload.assistant_id,
    }
    await authorize(user, "crons", "create", value)
    payload.metadata = value.get("metadata", payload.metadata)
    resolved_id = await _ensure_assistant_exists(assistant_id=payload.assistant_id)
    return await _create_cron(assistant_id=resolved_id, thread_id=None, payload=payload, user=user)


@router.post("/threads/{thread_id}/runs/crons", response_model=CronRead, response_model_exclude_none=True)
async def create_thread_cron(thread_id: str, payload: CronCreate, user: User = Depends(get_current_user)) -> CronRead:
    value: dict = {
        "metadata": dict(payload.metadata) if payload.metadata else {},
        "assistant_id": payload.assistant_id,
        "thread_id": thread_id,
    }
    await authorize(user, "crons", "create", value)
    payload.metadata = value.get("metadata", payload.metadata)
    resolved_id = await _ensure_assistant_exists(assistant_id=payload.assistant_id)
    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        thread = await session.scalar(select(Thread).where(Thread.thread_id == thread_id))
        if thread is None:
            raise HTTPException(status_code=404, detail="Thread not found")
    return await _create_cron(assistant_id=resolved_id, thread_id=thread_id, payload=payload, user=user)


@router.post("/runs/crons/search", response_model=CronSearchResponse)
async def search_crons(payload: CronSearchRequest, user: User = Depends(get_current_user)) -> CronSearchResponse:
    filters = await authorize(user, "crons", "search", {})
    return await cron_service.search_crons(payload=payload, user=user, filters=filters)


@router.post("/runs/crons/count", response_model=CronCountResponse)
async def count_crons(payload: CronCountRequest, user: User = Depends(get_current_user)) -> CronCountResponse:
    filters = await authorize(user, "crons", "search", {})
    return await cron_service.count_crons(payload=payload, user=user, filters=filters)


@router.get("/runs/crons/{cron_id}", response_model=CronRead, response_model_exclude_none=True)
async def get_cron(cron_id: str, user: User = Depends(get_current_user)) -> CronRead:
    filters = await authorize(user, "crons", "read", {"cron_id": cron_id})
    return await cron_service.get_cron(cron_id=cron_id, user=user, filters=filters)


@router.patch("/runs/crons/{cron_id}", response_model=CronRead, response_model_exclude_none=True)
async def patch_cron(cron_id: str, payload: CronPatch, user: User = Depends(get_current_user)) -> CronRead:
    filters = await authorize(user, "crons", "update", {"cron_id": cron_id})
    try:
        return await cron_service.patch_cron(cron_id=cron_id, payload=payload, user=user, filters=filters)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.delete("/runs/crons/{cron_id}", status_code=204)
async def delete_cron(cron_id: str, user: User = Depends(get_current_user)) -> Response:
    filters = await authorize(user, "crons", "delete", {"cron_id": cron_id})
    await cron_service.delete_cron(cron_id=cron_id, user=user, filters=filters)
    return Response(status_code=204)
