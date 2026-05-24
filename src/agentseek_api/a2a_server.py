from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from threading import Lock
from typing import Any
from uuid import uuid4

from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import BaseMessage, HumanMessage
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
    user_id: str
    context_id: str
    state: str = "submitted"
    status_message: str = ""
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    cancellation_requested: bool = False


class A2ATaskRegistry:
    def __init__(self, *, max_tasks: int = 1000) -> None:
        self._tasks: dict[str, A2ATaskRecord] = {}
        self._lock = Lock()
        self._max_tasks = max_tasks

    def save(self, record: A2ATaskRecord) -> None:
        with self._lock:
            self._tasks.pop(record.task_id, None)
            self._tasks[record.task_id] = record
            self._prune_locked()

    def get(self, task_id: str) -> A2ATaskRecord:
        with self._lock:
            try:
                return self._tasks[task_id]
            except KeyError as exc:
                raise ValueError(f"Unknown task: {task_id}") from exc

    def _prune_locked(self) -> None:
        while len(self._tasks) > self._max_tasks:
            evicted = False
            for task_id, record in list(self._tasks.items()):
                if _is_terminal_state(record.state):
                    del self._tasks[task_id]
                    evicted = True
                    break
            if not evicted:
                oldest_task_id = next(iter(self._tasks), None)
                if oldest_task_id is None:
                    break
                del self._tasks[oldest_task_id]


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
        "capabilities": {"streaming": True, "pushNotifications": False},
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


def make_sdk_text_artifact(text: str) -> dict[str, Any]:
    return {
        "artifactId": str(uuid4()),
        "name": "Assistant Response",
        "parts": [{"text": text}],
    }


def _jsonrpc_result(request_id: Any, result: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _jsonrpc_error(request_id: Any, *, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def _task_result(record: A2ATaskRecord, *, sdk_compatible: bool = False) -> dict[str, Any]:
    status_state = _sdk_task_state(record.state) if sdk_compatible else record.state
    status: dict[str, Any] = {"state": status_state}
    if record.status_message and not sdk_compatible:
        status["message"] = {"kind": "text", "text": record.status_message}
    artifacts = (
        [make_sdk_text_artifact(_artifact_text(artifact)) for artifact in record.artifacts]
        if sdk_compatible
        else record.artifacts
    )
    result = {
        "id": record.task_id,
        "contextId": record.context_id,
        "status": status,
        "artifacts": artifacts,
    }
    if not sdk_compatible:
        result["kind"] = "task"
    return result


def _is_terminal_state(state: str) -> bool:
    return state in {"completed", "failed", "cancelled"}


def _sdk_task_state(state: str) -> str:
    return {
        "submitted": "TASK_STATE_SUBMITTED",
        "working": "TASK_STATE_WORKING",
        "completed": "TASK_STATE_COMPLETED",
        "failed": "TASK_STATE_FAILED",
        "cancelled": "TASK_STATE_CANCELED",
    }.get(state, "TASK_STATE_UNSPECIFIED")


def _artifact_text(artifact: dict[str, Any]) -> str:
    parts = artifact.get("parts")
    if not isinstance(parts, list):
        return ""
    for part in parts:
        if not isinstance(part, dict):
            continue
        text = part.get("text")
        if isinstance(text, str):
            return text
        if part.get("kind") == "text" and isinstance(part.get("text"), str):
            return part["text"]
    return ""


def _message_content(message: BaseMessage) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    return json.dumps(content, default=str)


def _extract_chunk_messages(chunk: Any) -> list[BaseMessage]:
    messages: list[BaseMessage] = []
    if isinstance(chunk, BaseMessage):
        return [chunk]
    if isinstance(chunk, dict):
        nested_messages = chunk.get("messages")
        if isinstance(nested_messages, list):
            messages.extend(item for item in nested_messages if isinstance(item, BaseMessage))
        for value in chunk.values():
            if value is nested_messages:
                continue
            messages.extend(_extract_chunk_messages(value))
    elif isinstance(chunk, (list, tuple)):
        for item in chunk:
            messages.extend(_extract_chunk_messages(item))
    return messages


def _extract_stream_chunk_text(chunk: Any) -> str | None:
    messages = _extract_chunk_messages(chunk)
    if messages:
        texts = [_message_content(message) for message in messages if _message_content(message)]
        combined = "\n".join(texts).strip()
        return combined or None
    text = getattr(chunk, "text", None)
    if isinstance(text, str) and text:
        return text
    if isinstance(chunk, dict):
        raw_text = chunk.get("text")
        if isinstance(raw_text, str) and raw_text:
            return raw_text
    return None


def _is_root_stream_event(event: dict[str, Any]) -> bool:
    parent_ids = event.get("parent_ids")
    return isinstance(parent_ids, list) and not parent_ids


def _extract_request_text(message: dict[str, Any]) -> str:
    parts = message.get("parts")
    if not isinstance(parts, list) or not parts:
        raise ValueError("message.parts must be a non-empty array.")

    texts: list[str] = []
    for part in parts:
        if not isinstance(part, dict):
            raise ValueError("Only text parts are supported.")
        direct_text = part.get("text")
        if isinstance(direct_text, str):
            texts.append(direct_text)
            continue
        if part.get("kind") == "text" and isinstance(direct_text, str):
            texts.append(direct_text)
            continue
        if isinstance(direct_text, dict) and isinstance(direct_text.get("text"), str):
            texts.append(direct_text["text"])
            continue
        raise ValueError("Only text parts are supported.")
    return "\n".join(texts)


def _normalize_optional_id(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


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
        messages = extracted.get("messages")
        if isinstance(messages, list):
            message_texts = [_message_content(message) for message in messages if isinstance(message, BaseMessage)]
            if message_texts:
                return message_texts[-1]
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


async def _invoke_a2a_graph_stream(
    *,
    entry: GraphEntry,
    user: User,
    context_id: str,
    text: str,
) -> AsyncIterator[dict[str, Any]]:
    runtime_store, config = build_graph_config(user=user, context_id=context_id)
    configurable = config.get(CONF, {})
    graph = entry.build_graph(
        checkpointer=configurable.get(CONFIG_KEY_CHECKPOINTER),
        store=runtime_store,
    )
    graph_payload = {"messages": [HumanMessage(content=text)]}
    prepared = entry.prepare_input(graph_payload)

    if hasattr(graph, "astream_events"):
        async for stream_event in graph.astream_events(prepared, config, version="v2"):
            raw_event_name = stream_event.get("event")
            if raw_event_name in {"on_chat_model_stream", "on_llm_stream", "on_chain_stream"}:
                data = stream_event.get("data", {})
                chunk = data.get("chunk") if isinstance(data, dict) else None
                text = _extract_stream_chunk_text(chunk)
                if text:
                    yield {"text": text}
            if raw_event_name == "on_chain_end" and _is_root_stream_event(stream_event):
                data = stream_event.get("data", {})
                output = data.get("output") if isinstance(data, dict) else None
                extracted = entry.extract_output(output, graph_payload)
                if isinstance(extracted, dict):
                    yield extracted
                else:
                    yield {"result": extracted}
                return
        return

    if hasattr(graph, "astream"):
        async for raw_result in graph.astream(prepared, config):
            extracted = entry.extract_output(raw_result, graph_payload)
            if isinstance(extracted, dict):
                yield extracted
            else:
                yield {"result": extracted}
        return

    if hasattr(graph, "ainvoke"):
        raw_result = await graph.ainvoke(prepared, config)
        extracted = entry.extract_output(raw_result, graph_payload)
        if isinstance(extracted, dict):
            yield extracted
        else:
            yield {"result": extracted}
        return

    raw_result = graph.invoke(prepared, config)  # pragma: no cover
    extracted = entry.extract_output(raw_result, graph_payload)
    if isinstance(extracted, dict):
        yield extracted
    else:  # pragma: no cover
        yield {"result": extracted}


def _resolve_task_record(
    *,
    registry: A2ATaskRegistry,
    assistant_id: str,
    user: User,
    context_id: str | None,
    task_id: str,
) -> A2ATaskRecord | None:
    try:
        existing = registry.get(task_id)
    except ValueError:
        existing = None

    if existing is not None:
        if existing.user_id != user.identity or existing.assistant_id != assistant_id:
            return None
        existing.context_id = context_id or existing.context_id
        existing.state = "submitted"
        existing.status_message = ""
        existing.artifacts = []
        existing.cancellation_requested = False
        return existing

    return A2ATaskRecord(
        task_id=task_id,
        assistant_id=assistant_id,
        user_id=user.identity,
        context_id=context_id or str(uuid4()),
    )


def _cancel_task(record: A2ATaskRecord) -> A2ATaskRecord:
    if not _is_terminal_state(record.state):
        record.cancellation_requested = True
        record.state = "cancelled"
        record.status_message = "Task cancelled"
    return record


def _task_status_update_event(
    record: A2ATaskRecord,
    *,
    final: bool,
    sdk_compatible: bool = False,
) -> dict[str, Any]:
    status_state = _sdk_task_state(record.state) if sdk_compatible else record.state
    status: dict[str, Any] = {"state": status_state}
    if record.status_message and not sdk_compatible:
        status["message"] = {"kind": "text", "text": record.status_message}
    event = {
        "taskId": record.task_id,
        "contextId": record.context_id,
        "status": status,
    }
    if sdk_compatible:
        return {"statusUpdate": event}
    event["final"] = final
    event["kind"] = "status-update"
    return event


def _task_artifact_update_event(
    record: A2ATaskRecord,
    *,
    artifact: dict[str, Any],
    append: bool,
    last_chunk: bool,
    sdk_compatible: bool = False,
) -> dict[str, Any]:
    artifact_payload = make_sdk_text_artifact(_artifact_text(artifact)) if sdk_compatible else artifact
    event = {
        "taskId": record.task_id,
        "contextId": record.context_id,
        "artifact": artifact_payload,
        "append": append,
        "lastChunk": last_chunk,
    }
    if sdk_compatible:
        return {"artifactUpdate": event}
    event["kind"] = "artifact-update"
    return event


def _sdk_send_message_result(record: A2ATaskRecord) -> dict[str, Any]:
    return {"task": _task_result(record, sdk_compatible=True)}


def _canonical_a2a_method(method: str) -> tuple[str, bool]:
    mapping = {
        "message/send": ("message/send", False),
        "message/stream": ("message/stream", False),
        "tasks/get": ("tasks/get", False),
        "tasks/cancel": ("tasks/cancel", False),
        "SendMessage": ("message/send", True),
        "SendStreamingMessage": ("message/stream", True),
        "GetTask": ("tasks/get", True),
        "CancelTask": ("tasks/cancel", True),
    }
    return mapping.get(method, (method, False))


def _sse_jsonrpc_event(*, request_id: Any, result: dict[str, Any]) -> str:
    return f"event: message\ndata: {json.dumps(_jsonrpc_result(request_id, result))}\n\n"


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

    method, sdk_compatible = _canonical_a2a_method(payload["method"])
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
        if record.assistant_id != assistant_id or record.user_id != user.identity:
            return _jsonrpc_error(request_id, code=-32004, message=f"Unknown task: {task_id}")
        return _jsonrpc_result(request_id, _task_result(record, sdk_compatible=sdk_compatible))

    if method == "tasks/cancel":
        task_id = params.get("id")
        if not isinstance(task_id, str) or not task_id:
            return _jsonrpc_error(request_id, code=-32602, message="tasks/cancel requires params.id.")
        try:
            record = registry.get(task_id)
        except ValueError as exc:
            return _jsonrpc_error(request_id, code=-32004, message=str(exc))
        if record.assistant_id != assistant_id or record.user_id != user.identity:
            return _jsonrpc_error(request_id, code=-32004, message=f"Unknown task: {task_id}")
        registry.save(_cancel_task(record))
        return _jsonrpc_result(request_id, _task_result(record, sdk_compatible=sdk_compatible))

    if method not in {"message/send", "message/stream"}:
        return _jsonrpc_error(request_id, code=-32601, message=f"Unsupported method: {method}")

    message = params.get("message")
    if not isinstance(message, dict):
        return _jsonrpc_error(request_id, code=-32602, message=f"{method} requires params.message.")

    try:
        text = _extract_request_text(message)
    except ValueError as exc:
        return _jsonrpc_error(request_id, code=-32602, message=str(exc))

    context_id = _normalize_optional_id(message.get("contextId")) or _normalize_optional_id(params.get("contextId"))
    task_id = _normalize_optional_id(message.get("taskId")) or _normalize_optional_id(params.get("taskId")) or str(uuid4())
    record = _resolve_task_record(
        registry=registry,
        assistant_id=assistant_id,
        user=user,
        context_id=context_id,
        task_id=task_id,
    )
    if record is None:
        return _jsonrpc_error(request_id, code=-32004, message=f"Unknown task: {task_id}")
    registry.save(record)

    if method == "message/stream":
        async def _event_iter() -> AsyncIterator[str]:
            record.state = "working"
            registry.save(record)
            yield _sse_jsonrpc_event(
                request_id=request_id,
                result=_task_status_update_event(record, final=False, sdk_compatible=sdk_compatible),
            )

            final_extracted: dict[str, Any] | None = None
            pending_text: str | None = None
            try:
                async for extracted in _invoke_a2a_graph_stream(
                    entry=entry,
                    user=user,
                    context_id=record.context_id,
                    text=text,
                ):
                    final_extracted = extracted
                    chunk_text = _extract_stream_chunk_text(extracted)
                    if chunk_text is None:
                        continue
                    if pending_text is not None:
                        yield _sse_jsonrpc_event(
                            request_id=request_id,
                            result=_task_artifact_update_event(
                                record,
                                artifact=make_text_artifact(pending_text),
                                append=True,
                                last_chunk=False,
                                sdk_compatible=sdk_compatible,
                            ),
                        )
                    pending_text = chunk_text
            except Exception as exc:  # noqa: BLE001
                record.state = "failed"
                record.status_message = str(exc)
                registry.save(record)
                yield _sse_jsonrpc_event(
                    request_id=request_id,
                    result=_task_status_update_event(record, final=True, sdk_compatible=sdk_compatible),
                )
                return

            if record.cancellation_requested:
                registry.save(record)
                yield _sse_jsonrpc_event(
                    request_id=request_id,
                    result=_task_status_update_event(record, final=True, sdk_compatible=sdk_compatible),
                )
                return

            final_text = _extract_output_text(final_extracted or {})
            terminal_chunk_text = pending_text or final_text
            if terminal_chunk_text:
                yield _sse_jsonrpc_event(
                    request_id=request_id,
                    result=_task_artifact_update_event(
                        record,
                        artifact=make_text_artifact(terminal_chunk_text),
                        append=True,
                        last_chunk=True,
                        sdk_compatible=sdk_compatible,
                    ),
                )
            record.state = "completed"
            record.artifacts = [make_text_artifact(final_text)]
            registry.save(record)
            yield _sse_jsonrpc_event(
                request_id=request_id,
                result=_task_status_update_event(record, final=True, sdk_compatible=sdk_compatible),
            )

        return StreamingResponse(_event_iter(), media_type="text/event-stream")

    try:
        record.state = "working"
        registry.save(record)
        extracted = await _invoke_a2a_graph(
            entry=entry,
            user=user,
            context_id=record.context_id,
            text=text,
        )
    except Exception as exc:  # noqa: BLE001
        record.state = "failed"
        record.status_message = str(exc)
        registry.save(record)
        return _jsonrpc_error(request_id, code=-32000, message=str(exc))

    if record.cancellation_requested:
        registry.save(record)
        return _jsonrpc_result(
            request_id,
            _sdk_send_message_result(record) if sdk_compatible else _task_result(record),
        )

    record.state = "completed"
    record.artifacts = [make_text_artifact(_extract_output_text(extracted))]
    registry.save(record)
    return _jsonrpc_result(
        request_id,
        _sdk_send_message_result(record) if sdk_compatible else _task_result(record),
    )


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
