from datetime import UTC, datetime
from typing import Annotated

from sqlalchemy import select

from fastapi import APIRouter, Depends, HTTPException, Query, Response

from agentseek_api.core.auth_deps import get_current_user
from agentseek_api.core.database import db_manager
from agentseek_api.core.orm import StoreItem
from agentseek_api.models.api import (
    StoreDeleteRequest,
    StoreItemRead,
    StoreListNamespacesRequest,
    StorePutRequest,
    StoreSearchRequest,
    StoreSearchResponse,
)
from agentseek_api.models.auth import User

router = APIRouter(prefix="/store", tags=["Store"])


def _namespace_path(namespace: list[str]) -> str:
    if not namespace or not all(isinstance(part, str) and part for part in namespace):
        raise HTTPException(status_code=422, detail="namespace must contain at least one non-empty string")
    return "\x1f".join(namespace)


def _to_read_model(row: StoreItem) -> StoreItemRead:
    return StoreItemRead(
        namespace=list(row.namespace_json),
        key=row.key,
        value=dict(row.value_json),
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def _matches_filter(value: dict[str, object], filter_payload: dict[str, object] | None) -> bool:
    if not filter_payload:
        return True
    return all(value.get(key) == expected for key, expected in filter_payload.items())


def _matches_namespace(namespace: list[str], payload: StoreListNamespacesRequest) -> bool:
    if payload.prefix is not None and namespace[: len(payload.prefix)] != payload.prefix:
        return False
    if payload.suffix is not None and namespace[-len(payload.suffix) :] != payload.suffix:
        return False
    if payload.max_depth is not None and len(namespace) > payload.max_depth:
        return False
    return True


@router.put("/items", response_model=StoreItemRead)
async def put_item(payload: StorePutRequest, user: User = Depends(get_current_user)) -> StoreItemRead:
    namespace_path = _namespace_path(payload.namespace)
    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        row = await session.scalar(
            select(StoreItem).where(
                StoreItem.user_id == user.identity,
                StoreItem.namespace_path == namespace_path,
                StoreItem.key == payload.key,
            )
        )
        if row is None:
            row = StoreItem(
                user_id=user.identity,
                namespace_path=namespace_path,
                namespace_json=list(payload.namespace),
                key=payload.key,
                value_json=dict(payload.value),
            )
            session.add(row)
        else:
            row.value_json = dict(payload.value)
            row.updated_at = datetime.now(UTC)
        await session.commit()
        await session.refresh(row)
        return _to_read_model(row)


@router.get("/items", response_model=StoreItemRead)
async def get_item(
    namespace: Annotated[list[str], Query()],
    key: str,
    user: User = Depends(get_current_user),
) -> StoreItemRead:
    namespace_path = _namespace_path(namespace)
    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        row = await session.scalar(
            select(StoreItem).where(
                StoreItem.user_id == user.identity,
                StoreItem.namespace_path == namespace_path,
                StoreItem.key == key,
            )
        )
        if row is None:
            raise HTTPException(status_code=404, detail="Store item not found")
        return _to_read_model(row)


@router.delete("/items", status_code=204)
async def delete_item(payload: StoreDeleteRequest, user: User = Depends(get_current_user)) -> Response:
    namespace_path = _namespace_path(payload.namespace)
    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        row = await session.scalar(
            select(StoreItem).where(
                StoreItem.user_id == user.identity,
                StoreItem.namespace_path == namespace_path,
                StoreItem.key == payload.key,
            )
        )
        if row is not None:
            await session.delete(row)
            await session.commit()
    return Response(status_code=204)


@router.post("/items/search", response_model=StoreSearchResponse)
async def search_items(payload: StoreSearchRequest, user: User = Depends(get_current_user)) -> StoreSearchResponse:
    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        rows = (
            await session.scalars(select(StoreItem).where(StoreItem.user_id == user.identity).order_by(StoreItem.created_at.asc()))
        ).all()
    items = []
    for row in rows:
        namespace = list(row.namespace_json)
        if payload.namespace_prefix is not None and namespace[: len(payload.namespace_prefix)] != payload.namespace_prefix:
            continue
        if not _matches_filter(dict(row.value_json), payload.filter):
            continue
        items.append(_to_read_model(row))
    start = max(payload.offset, 0)
    end = start + max(payload.limit, 0)
    return StoreSearchResponse(items=items[start:end])


@router.post("/namespaces", response_model=list[list[str]])
async def list_namespaces(
    payload: StoreListNamespacesRequest,
    user: User = Depends(get_current_user),
) -> list[list[str]]:
    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        rows = (await session.scalars(select(StoreItem).where(StoreItem.user_id == user.identity))).all()
    namespaces = sorted({tuple(row.namespace_json) for row in rows})
    filtered = [list(namespace) for namespace in namespaces if _matches_namespace(list(namespace), payload)]
    start = max(payload.offset, 0)
    end = start + max(payload.limit, 0)
    return filtered[start:end]
