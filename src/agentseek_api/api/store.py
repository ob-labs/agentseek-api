from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import importlib
import importlib.util
import json
import math
from pathlib import Path
import sys
from typing import Annotated

from sqlalchemy import delete, select

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
from agentseek_api.settings import settings

router = APIRouter(prefix="/store", tags=["Store"])
EmbeddingFunction = Callable[[list[str]], list[list[float]]]


@dataclass(frozen=True)
class StoreTtlConfig:
    refresh_on_read: bool = True
    default_ttl: float | None = None
    sweep_interval_minutes: int | None = None


@dataclass(frozen=True)
class StoreIndexConfig:
    embed: str | None = None
    dims: int | None = None
    fields: list[str] | None = None
    embed_fn: EmbeddingFunction | None = None


@dataclass(frozen=True)
class StoreConfig:
    ttl: StoreTtlConfig = StoreTtlConfig()
    index: StoreIndexConfig = StoreIndexConfig()


_last_sweep_at: datetime | None = None


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _active_config_path() -> Path | None:
    if settings.AGENTSEEK_GRAPHS:
        path = Path(settings.AGENTSEEK_GRAPHS).expanduser().resolve()
        if path.exists():
            return path
    for candidate in ("agentseek.json", "langgraph.json"):
        path = Path(candidate).resolve()
        if path.exists():
            return path
    return None


def _load_embedding_function(reference: str, *, config_path: Path) -> EmbeddingFunction | None:
    if ":" not in reference:
        return None
    module_ref, symbol = reference.rsplit(":", maxsplit=1)
    if not module_ref or not symbol:
        return None
    if module_ref.endswith(".py") or module_ref.startswith(".") or "/" in module_ref or "\\" in module_ref:
        module_path = Path(module_ref).expanduser()
        if not module_path.is_absolute():
            module_path = config_path.parent / module_path
        module_path = module_path.resolve()
        spec = importlib.util.spec_from_file_location(f"agentseek_store_embeddings_{abs(hash(module_path))}", module_path)
        if spec is None or spec.loader is None:
            raise HTTPException(status_code=500, detail=f"Could not load store.index.embed module '{module_path}'.")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    else:
        try:
            module = importlib.import_module(module_ref)
        except ModuleNotFoundError:
            return None
    embed_fn = getattr(module, symbol, None)
    if not callable(embed_fn):
        return None
    return embed_fn


def _apply_config_dependencies(payload: dict[str, object], *, config_path: Path) -> None:
    dependencies = payload.get("dependencies")
    if not isinstance(dependencies, list):
        return
    for dependency in dependencies:
        if not isinstance(dependency, str):
            continue
        if dependency == ".":
            root = config_path.parent.resolve()
        else:
            candidate = Path(dependency).expanduser()
            root = candidate.resolve() if candidate.is_absolute() else (config_path.parent / candidate).resolve()
        if root.exists():
            root_text = str(root)
            if root_text not in sys.path:
                sys.path.insert(0, root_text)


def _load_store_config() -> StoreConfig:
    config_path = _active_config_path()
    if config_path is None:
        return StoreConfig()
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return StoreConfig()
    raw_store = payload.get("store")
    if not isinstance(raw_store, dict):
        return StoreConfig()
    _apply_config_dependencies(payload, config_path=config_path)

    ttl = StoreTtlConfig()
    raw_ttl = raw_store.get("ttl")
    if isinstance(raw_ttl, dict):
        refresh_on_read = raw_ttl.get("refresh_on_read", True)
        default_ttl = raw_ttl.get("default_ttl")
        sweep_interval = raw_ttl.get("sweep_interval_minutes")
        ttl = StoreTtlConfig(
            refresh_on_read=bool(refresh_on_read),
            default_ttl=float(default_ttl) if isinstance(default_ttl, (int, float)) else None,
            sweep_interval_minutes=int(sweep_interval) if isinstance(sweep_interval, int) else None,
        )

    index = StoreIndexConfig()
    raw_index = raw_store.get("index")
    if isinstance(raw_index, dict):
        embed = raw_index.get("embed")
        dims = raw_index.get("dims")
        raw_fields = raw_index.get("fields")
        fields = [item for item in raw_fields if isinstance(item, str)] if isinstance(raw_fields, list) else None
        embed_text = embed if isinstance(embed, str) else None
        index = StoreIndexConfig(
            embed=embed_text,
            dims=int(dims) if isinstance(dims, int) else None,
            fields=fields,
            embed_fn=_load_embedding_function(embed_text, config_path=config_path) if embed_text else None,
        )
    return StoreConfig(ttl=ttl, index=index)


def _namespace_path(namespace: list[str]) -> str:
    if not namespace or not all(isinstance(part, str) and part for part in namespace):
        raise HTTPException(status_code=422, detail="namespace must contain at least one non-empty string")
    return "\x1f".join(namespace)


def _expires_at_for_ttl(ttl_minutes: float | None, now: datetime) -> datetime | None:
    if ttl_minutes is None:
        return None
    return now + timedelta(minutes=ttl_minutes)


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _is_expired(row: StoreItem, now: datetime) -> bool:
    return row.expires_at is not None and _as_utc(row.expires_at) <= _as_utc(now)


def _should_sweep(last_sweep_at: datetime | None, now: datetime, interval_minutes: int) -> bool:
    return last_sweep_at is None or _as_utc(now) >= _as_utc(last_sweep_at) + timedelta(minutes=interval_minutes)


async def _sweep_expired_items(config: StoreConfig, now: datetime) -> None:
    global _last_sweep_at
    interval = config.ttl.sweep_interval_minutes
    if interval is None:
        return
    if not _should_sweep(_last_sweep_at, now, interval):
        return
    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        await session.execute(delete(StoreItem).where(StoreItem.expires_at.is_not(None), StoreItem.expires_at <= now))
        await session.commit()
    _last_sweep_at = now


def _extract_field_text(value: dict[str, object], fields: list[str] | None) -> str:
    if fields:
        parts = [value.get(field) for field in fields]
    else:
        parts = value.values()
    return "\n".join(str(part) for part in parts if isinstance(part, (str, int, float, bool)))


def _embed_one(config: StoreConfig, text: str) -> list[float] | None:
    if config.index.embed_fn is None or not text:
        return None
    vectors = config.index.embed_fn([text])
    if not vectors:
        return None
    vector = [float(item) for item in vectors[0]]
    if config.index.dims is not None and len(vector) != config.index.dims:
        raise HTTPException(status_code=500, detail="store.index.embed returned a vector with the wrong dimensions.")
    return vector


def _cosine_similarity(left: list[float] | None, right: list[float] | None) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=False))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


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
    config = _load_store_config()
    now = _utc_now()
    await _sweep_expired_items(config, now)
    namespace_path = _namespace_path(payload.namespace)
    ttl_minutes = payload.ttl if payload.ttl is not None else config.ttl.default_ttl
    embedding = _embed_one(config, _extract_field_text(payload.value, config.index.fields))
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
                embedding_json=embedding,
                expires_at=_expires_at_for_ttl(ttl_minutes, now),
            )
            session.add(row)
        else:
            row.value_json = dict(payload.value)
            row.embedding_json = embedding
            row.expires_at = _expires_at_for_ttl(ttl_minutes, now)
            row.updated_at = now
        await session.commit()
        await session.refresh(row)
        return _to_read_model(row)


@router.get("/items", response_model=StoreItemRead)
async def get_item(
    namespace: Annotated[list[str], Query()],
    key: str,
    refresh_ttl: bool | None = None,
    user: User = Depends(get_current_user),
) -> StoreItemRead:
    config = _load_store_config()
    now = _utc_now()
    await _sweep_expired_items(config, now)
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
        if row is None or _is_expired(row, now):
            if row is not None:
                await session.delete(row)
                await session.commit()
            raise HTTPException(status_code=404, detail="Store item not found")
        should_refresh = config.ttl.refresh_on_read if refresh_ttl is None else refresh_ttl
        if should_refresh and config.ttl.default_ttl is not None:
            row.expires_at = _expires_at_for_ttl(config.ttl.default_ttl, now)
            await session.commit()
            await session.refresh(row)
        return _to_read_model(row)


@router.delete("/items", status_code=204)
async def delete_item(payload: StoreDeleteRequest, user: User = Depends(get_current_user)) -> Response:
    await _sweep_expired_items(_load_store_config(), _utc_now())
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
    config = _load_store_config()
    now = _utc_now()
    await _sweep_expired_items(config, now)
    query_embedding = _embed_one(config, payload.query) if payload.query else None
    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        rows = (
            await session.scalars(select(StoreItem).where(StoreItem.user_id == user.identity).order_by(StoreItem.created_at.asc()))
        ).all()
        ranked: list[tuple[float, StoreItemRead]] = []
        should_refresh = config.ttl.refresh_on_read if payload.refresh_ttl is None else payload.refresh_ttl
        for row in rows:
            if _is_expired(row, now):
                await session.delete(row)
                continue
            namespace = list(row.namespace_json)
            if payload.namespace_prefix is not None and namespace[: len(payload.namespace_prefix)] != payload.namespace_prefix:
                continue
            if not _matches_filter(dict(row.value_json), payload.filter):
                continue
            score = _cosine_similarity(query_embedding, row.embedding_json)
            ranked.append((score, _to_read_model(row)))
            if should_refresh and config.ttl.default_ttl is not None:
                row.expires_at = _expires_at_for_ttl(config.ttl.default_ttl, now)
        await session.commit()
    if query_embedding is not None:
        ranked.sort(key=lambda item: item[0], reverse=True)
    items = [item for _, item in ranked]
    start = max(payload.offset, 0)
    end = start + max(payload.limit, 0)
    return StoreSearchResponse(items=items[start:end])


@router.post("/namespaces", response_model=list[list[str]])
async def list_namespaces(
    payload: StoreListNamespacesRequest,
    user: User = Depends(get_current_user),
) -> list[list[str]]:
    now = _utc_now()
    await _sweep_expired_items(_load_store_config(), now)
    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        rows = (await session.scalars(select(StoreItem).where(StoreItem.user_id == user.identity))).all()
    namespaces = sorted({tuple(row.namespace_json) for row in rows if not _is_expired(row, now)})
    filtered = [list(namespace) for namespace in namespaces if _matches_namespace(list(namespace), payload)]
    start = max(payload.offset, 0)
    end = start + max(payload.limit, 0)
    return filtered[start:end]
