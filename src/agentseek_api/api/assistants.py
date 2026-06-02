from typing import Any

from sqlalchemy import select

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from langchain_core.runnables.utils import create_model

from agentseek_api.core.auth_deps import get_current_user
from agentseek_api.core.database import db_manager
from agentseek_api.core.orm import Assistant
from agentseek_api.models.api import (
    AssistantCreate,
    AssistantPatch,
    AssistantRead,
    AssistantSearchRequest,
    AssistantVersionInfo,
    ErrorDetailResponse,
)
from agentseek_api.services.langgraph_service import get_langgraph_service

router = APIRouter(dependencies=[Depends(get_current_user)])
ASSISTANT_VERSION_PROMOTION_UNSUPPORTED = "Assistant version promotion is not supported"
DELETE_THREADS_UNSUPPORTED = "delete_threads=true is not supported"
SUBGRAPHS_UNSUPPORTED = "The graph does not support subgraphs"


def _detail_response(*, description: str, detail: str) -> dict[str, object]:
    return {
        "description": description,
        "model": ErrorDetailResponse,
        "content": {
            "application/json": {
                "example": {"detail": detail},
            }
        },
    }


def _to_read_model(row: Assistant) -> AssistantRead:
    return AssistantRead(
        assistant_id=row.assistant_id,
        name=row.name,
        graph_id=row.graph_id,
        created_at=row.created_at,
        updated_at=row.updated_at,
        metadata=row.metadata_json,
        config=row.config_json,
        context=row.context_json,
        version=row.version,
        description=row.description,
    )


def _filtered_assistant_rows(rows: list[Assistant], payload: AssistantSearchRequest) -> list[Assistant]:
    def matches(row: Assistant) -> bool:
        if payload.graph_id is not None and row.graph_id != payload.graph_id:
            return False
        if payload.name is not None and row.name != payload.name:
            return False
        if payload.metadata is not None:
            for key, value in payload.metadata.items():
                if row.metadata_json.get(key) != value:
                    return False
        return True

    return [row for row in rows if matches(row)]


@router.post("", response_model=AssistantRead)
async def create_assistant(payload: AssistantCreate) -> AssistantRead:
    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        row = Assistant(
            name=payload.name,
            graph_id=payload.graph_id,
            metadata_json=payload.metadata,
            config_json=payload.config,
            context_json=payload.context,
            description=payload.description,
        )
        session.add(row)
        await session.commit()
        await session.refresh(row)
        return _to_read_model(row)


@router.get("", response_model=list[AssistantRead])
async def list_assistants() -> list[AssistantRead]:
    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        rows = (await session.scalars(select(Assistant).order_by(Assistant.created_at.desc()))).all()
        return [_to_read_model(row) for row in rows]


@router.post("/search", response_model=list[AssistantRead])
async def search_assistants(payload: AssistantSearchRequest) -> list[AssistantRead]:
    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        rows = (await session.scalars(select(Assistant).order_by(Assistant.created_at.desc()))).all()

    filtered = _filtered_assistant_rows(rows, payload)
    start = max(payload.offset, 0)
    end = start + max(payload.limit, 0)
    return [_to_read_model(row) for row in filtered[start:end]]


@router.post("/count", response_model=int)
async def count_assistants(payload: AssistantSearchRequest) -> int:
    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        rows = (await session.scalars(select(Assistant).order_by(Assistant.created_at.desc()))).all()
    return len(_filtered_assistant_rows(rows, payload))


@router.get("/{assistant_id}", response_model=AssistantRead)
async def get_assistant(assistant_id: str) -> AssistantRead:
    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        row = await session.scalar(select(Assistant).where(Assistant.assistant_id == assistant_id))
        if row is None:
            raise HTTPException(status_code=404, detail="Assistant not found")
        return _to_read_model(row)


@router.patch("/{assistant_id}", response_model=AssistantRead)
async def patch_assistant(assistant_id: str, payload: AssistantPatch) -> AssistantRead:
    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        row = await session.scalar(select(Assistant).where(Assistant.assistant_id == assistant_id))
        if row is None:
            raise HTTPException(status_code=404, detail="Assistant not found")
        if payload.graph_id is not None:
            row.graph_id = payload.graph_id
        if payload.config is not None:
            row.config_json = payload.config
        if payload.context is not None:
            row.context_json = payload.context
        if payload.metadata is not None:
            row.metadata_json = {**row.metadata_json, **payload.metadata}
        if payload.name is not None:
            row.name = payload.name
        if payload.description is not None:
            row.description = payload.description
        await session.commit()
        await session.refresh(row)
        return _to_read_model(row)


@router.delete(
    "/{assistant_id}",
    status_code=204,
    responses={
        400: _detail_response(description="Unsupported delete option", detail=DELETE_THREADS_UNSUPPORTED),
        404: _detail_response(description="Assistant not found", detail="Assistant not found"),
    },
)
async def delete_assistant(assistant_id: str, delete_threads: bool = False) -> Response:
    if delete_threads:
        raise HTTPException(status_code=400, detail=DELETE_THREADS_UNSUPPORTED)
    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        row = await session.scalar(select(Assistant).where(Assistant.assistant_id == assistant_id))
        if row is None:
            raise HTTPException(status_code=404, detail="Assistant not found")
        await session.delete(row)
        await session.commit()
    return Response(status_code=204)


@router.get("/{assistant_id}/graph")
async def get_assistant_graph(
    assistant_id: str,
    xray: bool | int | None = Query(
        None,
        description="Expand subgraph nodes. Pass true or a positive integer depth.",
    ),
) -> dict[str, object]:
    assistant = await get_assistant(assistant_id)
    entry = get_langgraph_service().get_entry(assistant.graph_id)
    graph = entry.build_graph()
    if isinstance(xray, int) and not isinstance(xray, bool) and xray <= 0:
        raise HTTPException(status_code=422, detail="Invalid xray value")
    xray_value: bool | int = xray if xray is not None else False
    try:
        drawable_graph = await graph.aget_graph(xray=xray_value)
    except NotImplementedError as exc:
        raise HTTPException(status_code=422, detail="The graph does not support visualization") from exc
    json_graph = drawable_graph.to_json()
    for node in json_graph.get("nodes", []):
        data = node.get("data") if isinstance(node, dict) else None
        if isinstance(data, dict):
            data.pop("id", None)
    return json_graph


@router.get("/{assistant_id}/schemas")
async def get_assistant_schemas(assistant_id: str) -> dict[str, object]:
    assistant = await get_assistant(assistant_id)
    entry = get_langgraph_service().get_entry(assistant.graph_id)
    graph = entry.build_graph()
    return {"graph_id": assistant.graph_id, **_extract_graph_schemas(graph)}


def _safe_schema(getter, *args, **kwargs) -> dict[str, object] | None:
    try:
        return getter(*args, **kwargs)
    except Exception:  # noqa: BLE001 - graph helpers raise broad errors
        return None


def _state_jsonschema(graph) -> dict[str, object] | None:
    channel_list = getattr(graph, "stream_channels_list", None)
    channels = getattr(graph, "channels", None)
    if not channel_list or channels is None:
        return None
    fields: dict[str, tuple[object, object]] = {}
    for key in channel_list:
        channel = channels.get(key) if isinstance(channels, dict) else getattr(channels, key, None)
        update_type = getattr(channel, "UpdateType", Any) if channel is not None else Any
        fields[key] = (update_type, None)
    try:
        name = graph.get_name("State") if hasattr(graph, "get_name") else "State"
        return create_model(name, **fields).model_json_schema()
    except Exception:  # noqa: BLE001
        return None


def _extract_graph_schemas(graph) -> dict[str, object | None]:
    return {
        "input_schema": _safe_schema(graph.get_input_jsonschema),
        "output_schema": _safe_schema(graph.get_output_jsonschema),
        "state_schema": _state_jsonschema(graph),
        "config_schema": _safe_schema(graph.get_config_jsonschema) if hasattr(graph, "get_config_jsonschema") else None,
        "context_schema": _safe_schema(graph.get_context_jsonschema) if hasattr(graph, "get_context_jsonschema") else None,
    }


async def _collect_subgraphs(assistant_id: str, *, namespace: str | None, recurse: bool) -> dict[str, dict[str, object | None]]:
    assistant = await get_assistant(assistant_id)
    entry = get_langgraph_service().get_entry(assistant.graph_id)
    graph = entry.build_graph()
    aget_subgraphs = getattr(graph, "aget_subgraphs", None)
    if not callable(aget_subgraphs):
        raise HTTPException(status_code=422, detail=SUBGRAPHS_UNSUPPORTED)
    try:
        return {
            ns: _extract_graph_schemas(subgraph)
            async for ns, subgraph in aget_subgraphs(namespace=namespace, recurse=recurse)
        }
    except NotImplementedError as exc:
        raise HTTPException(status_code=422, detail=SUBGRAPHS_UNSUPPORTED) from exc


@router.get(
    "/{assistant_id}/subgraphs",
    response_model=dict[str, dict[str, object | None]],
    responses={
        404: _detail_response(description="Assistant not found", detail="Assistant not found"),
        422: _detail_response(description="Graph does not support subgraphs", detail=SUBGRAPHS_UNSUPPORTED),
    },
)
async def get_assistant_subgraphs(
    assistant_id: str,
    recurse: bool = Query(False, description="Recursively include nested subgraphs."),
    namespace: str | None = Query(None, description="Filter to a specific subgraph namespace."),
) -> dict[str, dict[str, object | None]]:
    return await _collect_subgraphs(assistant_id, namespace=namespace, recurse=recurse)


@router.get(
    "/{assistant_id}/subgraphs/{namespace}",
    response_model=dict[str, dict[str, object | None]],
    responses={
        404: _detail_response(description="Assistant not found", detail="Assistant not found"),
        422: _detail_response(description="Graph does not support subgraphs", detail=SUBGRAPHS_UNSUPPORTED),
    },
)
async def get_assistant_subgraphs_by_namespace(
    assistant_id: str,
    namespace: str,
    recurse: bool = Query(False, description="Recursively include nested subgraphs."),
) -> dict[str, dict[str, object | None]]:
    return await _collect_subgraphs(assistant_id, namespace=namespace, recurse=recurse)


@router.post(
    "/{assistant_id}/versions",
    response_model=AssistantVersionInfo,
    responses={
        404: _detail_response(description="Assistant not found", detail="Assistant not found"),
    },
)
async def get_assistant_versions(assistant_id: str) -> AssistantVersionInfo:
    assistant = await get_assistant(assistant_id)
    return AssistantVersionInfo(
        assistant_id=assistant.assistant_id,
        current_version=assistant.version,
        latest_version=assistant.version,
        available_versions=[assistant.version],
        supports_version_history=False,
    )


@router.post(
    "/{assistant_id}/latest",
    status_code=409,
    response_model=None,
    responses={
        404: _detail_response(description="Assistant not found", detail="Assistant not found"),
        409: _detail_response(
            description="Unsupported helper endpoint",
            detail=ASSISTANT_VERSION_PROMOTION_UNSUPPORTED,
        ),
    },
)
async def set_latest_assistant_version(assistant_id: str) -> None:
    _ = await get_assistant(assistant_id)
    raise HTTPException(status_code=409, detail=ASSISTANT_VERSION_PROMOTION_UNSUPPORTED)
