from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from langgraph.store.base import InvalidNamespaceError, Item, NOT_PROVIDED, SearchItem

from agentseek_api.core.auth_deps import get_current_user
from agentseek_api.core.database import db_manager
from agentseek_api.core.runtime_store import UserScopedStore
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


def _to_store_item_read(item: Item | SearchItem) -> StoreItemRead:
    return StoreItemRead(
        namespace=list(item.namespace),
        key=item.key,
        value=dict(item.value),
        created_at=item.created_at,
        updated_at=item.updated_at,
    )


def _user_store(user: User) -> UserScopedStore:
    return UserScopedStore(db_manager.get_store(), user_id=user.identity)


def _handle_store_error(exc: InvalidNamespaceError) -> HTTPException:
    return HTTPException(status_code=422, detail=str(exc))


@router.put("/items", status_code=204)
async def put_item(payload: StorePutRequest, user: User = Depends(get_current_user)) -> Response:
    ttl = payload.ttl if "ttl" in payload.model_fields_set else NOT_PROVIDED
    index = payload.index if "index" in payload.model_fields_set else None
    try:
        await _user_store(user).aput(
            tuple(payload.namespace),
            payload.key,
            dict(payload.value),
            index=index,
            ttl=ttl,
        )
    except InvalidNamespaceError as exc:
        raise _handle_store_error(exc) from exc
    return Response(status_code=204)


@router.get("/items", response_model=StoreItemRead)
async def get_item(
    key: str = Query(),
    namespace: list[str] = Query(default_factory=list),
    refresh_ttl: bool | None = Query(default=None),
    user: User = Depends(get_current_user),
) -> StoreItemRead:
    try:
        item = await _user_store(user).aget(tuple(namespace), key, refresh_ttl=refresh_ttl)
    except InvalidNamespaceError as exc:
        raise _handle_store_error(exc) from exc
    if item is None:
        raise HTTPException(status_code=404, detail="Store item not found")
    return _to_store_item_read(item)


@router.delete("/items", status_code=204)
async def delete_item(payload: StoreDeleteRequest, user: User = Depends(get_current_user)) -> Response:
    try:
        await _user_store(user).adelete(tuple(payload.namespace), payload.key)
    except InvalidNamespaceError as exc:
        raise _handle_store_error(exc) from exc
    return Response(status_code=204)


@router.post("/items/search", response_model=StoreSearchResponse)
async def search_items(payload: StoreSearchRequest, user: User = Depends(get_current_user)) -> StoreSearchResponse:
    try:
        items = await _user_store(user).asearch(
            tuple(payload.namespace_prefix or ()),
            query=payload.query,
            filter=payload.filter,
            limit=max(payload.limit, 0),
            offset=max(payload.offset, 0),
            refresh_ttl=payload.refresh_ttl,
        )
    except InvalidNamespaceError as exc:
        raise _handle_store_error(exc) from exc
    return StoreSearchResponse(items=[_to_store_item_read(item) for item in items])


@router.post("/namespaces", response_model=list[list[str]])
async def list_namespaces(payload: StoreListNamespacesRequest, user: User = Depends(get_current_user)) -> list[list[str]]:
    try:
        namespaces = await _user_store(user).alist_namespaces(
            prefix=tuple(payload.prefix) if payload.prefix is not None else None,
            suffix=tuple(payload.suffix) if payload.suffix is not None else None,
            max_depth=payload.max_depth,
            limit=max(payload.limit, 0),
            offset=max(payload.offset, 0),
        )
    except InvalidNamespaceError as exc:
        raise _handle_store_error(exc) from exc
    return [list(namespace) for namespace in namespaces]
