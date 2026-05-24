from __future__ import annotations

import json
from dataclasses import dataclass, field
from threading import Lock
from typing import Any
from uuid import uuid4

from fastapi import HTTPException
from langchain_core.messages import HumanMessage
from langgraph.constants import CONF, CONFIG_KEY_CHECKPOINTER
from sqlalchemy import select

from agentseek_api import __version__
from agentseek_api.core.auth_middleware import get_config_auth_openapi
from agentseek_api.core.database import db_manager
from agentseek_api.core.orm import Assistant
from agentseek_api.core.runtime_store import UserScopedStore
from agentseek_api.models.api import AssistantRead
from agentseek_api.models.auth import User
from agentseek_api.services.langgraph_service import GraphEntry
from agentseek_api.settings import settings


@dataclass
class A2ATaskRecord:
    task_id: str
    assistant_id: str
    context_id: str
    state: str = "submitted"
    status_message: str = ""
    artifacts: list[dict[str, Any]] = field(default_factory=list)


class A2ATaskRegistry:
    def __init__(self) -> None:
        self._tasks: dict[str, A2ATaskRecord] = {}
        self._lock = Lock()

    def save(self, record: A2ATaskRecord) -> None:
        with self._lock:
            self._tasks[record.task_id] = record

    def get(self, task_id: str) -> A2ATaskRecord:
        with self._lock:
            try:
                return self._tasks[task_id]
            except KeyError as exc:
                raise ValueError(f"Unknown task: {task_id}") from exc


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
                    metadata["securityRequirements"] = translated_security
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
            "securityRequirements": [{"apiKeyAuth": []}],
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
            "securityRequirements": [{"bearerAuth": []}],
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
        if not all(isinstance(name, str) and name in retained_scheme_names and isinstance(scopes, list) for name, scopes in item.items()):
            continue
        translated.append(item)
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


def build_graph_config(*, user: User, context_id: str) -> tuple[UserScopedStore, dict[str, Any]]:
    runtime_store = UserScopedStore(db_manager.get_store(), user_id=user.identity)
    checkpointer = db_manager.get_langgraph_checkpointer()
    return runtime_store, {
        CONF: {
            "thread_id": context_id,
            "checkpoint_ns": f"a2a:{uuid4()}",
            CONFIG_KEY_CHECKPOINTER: checkpointer,
            "store": runtime_store,
            "langgraph_auth_user": user.model_dump(),
        }
    }


def make_text_artifact(text: str) -> dict[str, Any]:
    return {
        "artifactId": str(uuid4()),
        "name": "Assistant Response",
        "parts": [{"kind": "text", "text": text}],
    }


def _jsonrpc_result(request_id: Any, result: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _jsonrpc_error(request_id: Any, *, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def _task_result(record: A2ATaskRecord) -> dict[str, Any]:
    status: dict[str, Any] = {"state": record.state}
    if record.status_message:
        status["message"] = {"kind": "text", "text": record.status_message}
    return {
        "id": record.task_id,
        "contextId": record.context_id,
        "kind": "task",
        "status": status,
        "artifacts": record.artifacts,
    }


def _extract_request_text(message: dict[str, Any]) -> str:
    parts = message.get("parts")
    if not isinstance(parts, list) or not parts:
        raise ValueError("message.parts must be a non-empty array.")

    texts: list[str] = []
    for part in parts:
        if not isinstance(part, dict) or part.get("kind") != "text" or not isinstance(part.get("text"), str):
            raise ValueError("Only text parts are supported.")
        texts.append(part["text"])
    return "\n".join(texts)


def _extract_output_text(extracted: Any) -> str:
    if isinstance(extracted, str):
        return extracted
    if isinstance(extracted, dict):
        final_text = extracted.get("final_text")
        if isinstance(final_text, str):
            return final_text
        text = extracted.get("text")
        if isinstance(text, str):
            return text
    return json.dumps(extracted, default=str)


async def _invoke_a2a_graph(
    *,
    entry: GraphEntry,
    user: User,
    context_id: str,
    text: str,
) -> dict[str, Any]:
    runtime_store, config = build_graph_config(user=user, context_id=context_id)
    configurable = config.get(CONF, {})
    graph = entry.build_graph(
        checkpointer=configurable.get(CONFIG_KEY_CHECKPOINTER),
        store=runtime_store,
    )
    graph_payload = {"messages": [HumanMessage(content=text)]}
    prepared = entry.prepare_input(graph_payload)
    if hasattr(graph, "ainvoke"):
        raw_result = await graph.ainvoke(prepared, config)
    else:  # pragma: no cover
        raw_result = graph.invoke(prepared, config)
    extracted = entry.extract_output(raw_result, graph_payload)
    if isinstance(extracted, dict):
        return extracted
    return {"result": extracted}


async def handle_a2a_request(
    *,
    assistant_id: str,
    payload: dict[str, Any],
    user: User,
    service,
    registry: A2ATaskRegistry,
) -> dict[str, Any]:
    request_id = payload.get("id")
    if payload.get("jsonrpc") != "2.0" or not isinstance(payload.get("method"), str):
        return _jsonrpc_error(request_id, code=-32600, message="Invalid JSON-RPC request.")

    try:
        assistant = await load_assistant(assistant_id)
    except HTTPException as exc:
        return _jsonrpc_error(request_id, code=-32004, message=str(exc.detail))

    entry = service.get_entry(assistant.graph_id)
    if not is_a2a_compatible_entry(entry):
        return _jsonrpc_error(request_id, code=-32000, message="Assistant graph is not A2A-compatible.")

    method = payload["method"]
    params = payload.get("params")
    if not isinstance(params, dict):
        return _jsonrpc_error(request_id, code=-32602, message="params must be an object.")

    if method == "tasks/get":
        task_id = params.get("id")
        if not isinstance(task_id, str) or not task_id:
            return _jsonrpc_error(request_id, code=-32602, message="tasks/get requires params.id.")
        try:
            record = registry.get(task_id)
        except ValueError as exc:
            return _jsonrpc_error(request_id, code=-32004, message=str(exc))
        if record.assistant_id != assistant_id:
            return _jsonrpc_error(request_id, code=-32004, message=f"Unknown task: {task_id}")
        return _jsonrpc_result(request_id, _task_result(record))

    if method != "message/send":
        return _jsonrpc_error(request_id, code=-32601, message=f"Unsupported method: {method}")

    message = params.get("message")
    if not isinstance(message, dict):
        return _jsonrpc_error(request_id, code=-32602, message="message/send requires params.message.")

    try:
        text = _extract_request_text(message)
    except ValueError as exc:
        return _jsonrpc_error(request_id, code=-32602, message=str(exc))

    context_id = params.get("contextId")
    if not isinstance(context_id, str) or not context_id:
        context_id = str(uuid4())
    task_id = params.get("taskId")
    if not isinstance(task_id, str) or not task_id:
        task_id = str(uuid4())

    record = A2ATaskRecord(task_id=task_id, assistant_id=assistant_id, context_id=context_id)
    registry.save(record)

    try:
        extracted = await _invoke_a2a_graph(
            entry=entry,
            user=user,
            context_id=context_id,
            text=text,
        )
    except Exception as exc:  # noqa: BLE001
        record.state = "failed"
        record.status_message = str(exc)
        registry.save(record)
        return _jsonrpc_error(request_id, code=-32000, message=str(exc))

    record.state = "completed"
    record.artifacts = [make_text_artifact(_extract_output_text(extracted))]
    registry.save(record)
    return _jsonrpc_result(request_id, _task_result(record))


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
