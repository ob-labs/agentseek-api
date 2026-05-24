from __future__ import annotations

from typing import Any

from fastapi import HTTPException
from sqlalchemy import select

from agentseek_api import __version__
from agentseek_api.core.database import db_manager
from agentseek_api.core.orm import Assistant
from agentseek_api.models.api import AssistantRead
from agentseek_api.services.langgraph_service import GraphEntry


def is_a2a_compatible_entry(entry: GraphEntry) -> bool:
    input_schema = entry.input_schema
    properties = input_schema.get("properties")
    required = input_schema.get("required")
    if not isinstance(properties, dict) or not isinstance(required, list):
        return False

    messages = properties.get("messages")
    if not isinstance(messages, dict):
        return False

    return messages.get("type") == "array" and "messages" in required


def build_agent_card(base_url: str, assistant: AssistantRead, entry: GraphEntry) -> dict[str, Any]:
    description = assistant.description or entry.description
    url = f"{base_url}/a2a/{assistant.assistant_id}"
    skill_description = description or f"Runs the {assistant.graph_id} graph."

    return {
        "name": assistant.name,
        "description": description,
        "supportedInterfaces": [
            {
                "url": url,
                "protocolBinding": "JSONRPC",
                "protocolVersion": "1.0",
            }
        ],
        "version": __version__,
        "capabilities": {"streaming": False, "pushNotifications": False},
        "defaultInputModes": ["text/plain"],
        "defaultOutputModes": ["text/plain"],
        "skills": [
            {
                "id": entry.tool_name,
                "name": assistant.name,
                "description": skill_description,
                "tags": [assistant.graph_id, entry.tool_name],
            }
        ],
    }


async def load_assistant(assistant_id: str) -> AssistantRead:
    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        row = await session.scalar(select(Assistant).where(Assistant.assistant_id == assistant_id))
        if row is None:
            raise HTTPException(status_code=404, detail="Assistant not found")
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
