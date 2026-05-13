from sqlalchemy import select

from fastapi import APIRouter, HTTPException

from agentseek_api.core.database import db_manager
from agentseek_api.core.orm import Assistant
from agentseek_api.models.api import AssistantCreate, AssistantRead

router = APIRouter(prefix="/assistants", tags=["Assistants"])


@router.post("", response_model=AssistantRead)
async def create_assistant(payload: AssistantCreate) -> AssistantRead:
    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        row = Assistant(name=payload.name, graph_id=payload.graph_id)
        session.add(row)
        await session.commit()
        await session.refresh(row)
        return AssistantRead(
            assistant_id=row.assistant_id,
            name=row.name,
            graph_id=row.graph_id,
            created_at=row.created_at,
        )


@router.get("", response_model=list[AssistantRead])
async def list_assistants() -> list[AssistantRead]:
    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        rows = (await session.scalars(select(Assistant).order_by(Assistant.created_at.desc()))).all()
        return [
            AssistantRead(assistant_id=row.assistant_id, name=row.name, graph_id=row.graph_id, created_at=row.created_at)
            for row in rows
        ]


@router.get("/{assistant_id}", response_model=AssistantRead)
async def get_assistant(assistant_id: str) -> AssistantRead:
    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        row = await session.scalar(select(Assistant).where(Assistant.assistant_id == assistant_id))
        if row is None:
            raise HTTPException(status_code=404, detail="Assistant not found")
        return AssistantRead(assistant_id=row.assistant_id, name=row.name, graph_id=row.graph_id, created_at=row.created_at)
