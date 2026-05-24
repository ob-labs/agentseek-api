from __future__ import annotations

from typing import Any

from fastapi import HTTPException
from sqlalchemy import select

from agentseek_api import __version__
from agentseek_api.core.auth_middleware import get_config_auth_openapi
from agentseek_api.core.database import db_manager
from agentseek_api.core.orm import Assistant
from agentseek_api.models.api import AssistantRead
from agentseek_api.services.langgraph_service import GraphEntry
from agentseek_api.settings import settings


def is_a2a_compatible_entry(entry: GraphEntry) -> bool:
    input_schema = entry.input_schema
    if input_schema.get("type") != "object":
        return False

    properties = input_schema.get("properties")
    required = input_schema.get("required")
    if not isinstance(properties, dict) or not isinstance(required, list):
        return False

    messages = properties.get("messages")
    if not isinstance(messages, dict):
        return False

    return messages.get("type") == "array" and "messages" in required


def _agent_card_auth_metadata() -> dict[str, Any]:
    auth_openapi = get_config_auth_openapi()
    if isinstance(auth_openapi, dict):
        security_schemes = auth_openapi.get("securitySchemes")
        security = auth_openapi.get("security")
        if isinstance(security_schemes, dict) and isinstance(security, list):
            translated_schemes: dict[str, Any] = {}
            for scheme_name, scheme in security_schemes.items():
                if not isinstance(scheme_name, str) or not isinstance(scheme, dict):
                    continue
                translated = _translate_openapi_security_scheme(scheme)
                if translated is not None:
                    translated_schemes[scheme_name] = translated

            translated_security = _translate_openapi_security_requirements(
                security,
                retained_scheme_names=set(translated_schemes.keys()),
            )
            if translated_schemes:
                metadata: dict[str, Any] = {"securitySchemes": translated_schemes}
                if translated_security:
                    metadata["security"] = translated_security
                return metadata

    auth_type = settings.AUTH_TYPE.strip().lower()
    if auth_type == "api_key":
        return {
            "securitySchemes": {
                "apiKeyAuth": {
                    "apiKeySecurityScheme": {
                        "location": "header",
                        "name": "x-api-key",
                    }
                }
            },
            "security": [{"apiKeyAuth": []}],
        }
    if auth_type == "jwt":
        return {
            "securitySchemes": {
                "bearerAuth": {
                    "httpAuthSecurityScheme": {
                        "scheme": "bearer",
                        "bearerFormat": "JWT",
                    }
                }
            },
            "security": [{"bearerAuth": []}],
        }
    return {}


def _translate_openapi_security_scheme(scheme: dict[str, Any]) -> dict[str, Any] | None:
    description = scheme.get("description")
    description_value = description if isinstance(description, str) and description else None

    if scheme.get("type") == "apiKey":
        location = scheme.get("in")
        name = scheme.get("name")
        if isinstance(location, str) and isinstance(name, str):
            translated: dict[str, Any] = {
                "location": location,
                "name": name,
            }
            if description_value is not None:
                translated["description"] = description_value
            return {"apiKeySecurityScheme": translated}

    if scheme.get("type") == "http" and scheme.get("scheme") == "bearer":
        translated: dict[str, Any] = {"scheme": "bearer"}
        bearer_format = scheme.get("bearerFormat")
        if isinstance(bearer_format, str) and bearer_format:
            translated["bearerFormat"] = bearer_format
        if description_value is not None:
            translated["description"] = description_value
        return {"httpAuthSecurityScheme": translated}

    return None


def _translate_openapi_security_requirements(
    security: list[Any],
    *,
    retained_scheme_names: set[str],
) -> list[dict[str, list[Any]]]:
    translated: list[dict[str, list[Any]]] = []
    for item in security:
        if not isinstance(item, dict):
            continue
        filtered = {
            name: scopes
            for name, scopes in item.items()
            if isinstance(name, str) and name in retained_scheme_names and isinstance(scopes, list)
        }
        if filtered:
            translated.append(filtered)
    return translated


def build_agent_card(base_url: str, assistant: AssistantRead, entry: GraphEntry) -> dict[str, Any]:
    description = assistant.description or ""
    url = f"{base_url}/a2a/{assistant.assistant_id}"
    skill_description = assistant.description or entry.description or f"Runs the {assistant.graph_id} graph."

    card: dict[str, Any] = {
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
    card.update(_agent_card_auth_metadata())
    return card


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
