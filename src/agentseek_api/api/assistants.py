from sqlalchemy import select

from fastapi import APIRouter, HTTPException, Response

from agentseek_api.core.database import db_manager
from agentseek_api.core.orm import Assistant
from agentseek_api.models.api import AssistantCreate, AssistantPatch, AssistantRead, AssistantSearchRequest
from agentseek_api.services.langgraph_service import get_langgraph_service

router = APIRouter(prefix="/assistants", tags=["Assistants"])


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


@router.delete("/{assistant_id}", status_code=204)
async def delete_assistant(assistant_id: str, delete_threads: bool = False) -> Response:
    if delete_threads:
        raise HTTPException(status_code=400, detail="delete_threads=true is not supported")
    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        row = await session.scalar(select(Assistant).where(Assistant.assistant_id == assistant_id))
        if row is None:
            raise HTTPException(status_code=404, detail="Assistant not found")
        await session.delete(row)
        await session.commit()
    return Response(status_code=204)


@router.get("/{assistant_id}/graph")
async def get_assistant_graph(assistant_id: str) -> dict[str, object]:
    assistant = await get_assistant(assistant_id)
    return {
        "assistant_id": assistant.assistant_id,
        "graph_id": assistant.graph_id,
        "registered_graph_ids": get_langgraph_service().registered_graph_ids(),
    }


@router.get("/{assistant_id}/schemas")
async def get_assistant_schemas(assistant_id: str) -> dict[str, object]:
    assistant = await get_assistant(assistant_id)
    return {
        "assistant_id": assistant.assistant_id,
        "graph_id": assistant.graph_id,
        "input_schema": {"type": "object"},
        "output_schema": {"type": "object"},
    }


@router.get("/{assistant_id}/subgraphs")
async def get_assistant_subgraphs(assistant_id: str) -> list[dict[str, object]]:
    _ = await get_assistant(assistant_id)
    return []


@router.get("/{assistant_id}/subgraphs/{namespace}")
async def get_assistant_subgraphs_by_namespace(assistant_id: str, namespace: str) -> list[dict[str, object]]:
    _ = (await get_assistant(assistant_id), namespace)
    return []


@router.post("/{assistant_id}/versions")
async def get_assistant_versions(assistant_id: str) -> dict[str, object]:
    assistant = await get_assistant(assistant_id)
    return {"assistant_id": assistant.assistant_id, "version": assistant.version}


@router.post("/{assistant_id}/latest", response_model=AssistantRead)
async def set_latest_assistant_version(assistant_id: str) -> AssistantRead:
    return await get_assistant(assistant_id)
