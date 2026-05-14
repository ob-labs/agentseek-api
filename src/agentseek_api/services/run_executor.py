from dataclasses import dataclass
from typing import Any

from langchain_core.messages import BaseMessage
from langgraph.constants import CONF, CONFIG_KEY_CHECKPOINTER
from langgraph.types import Command

from agentseek_api.core.database import db_manager
from agentseek_api.services.langgraph_service import ensure_sync_checkpoint_mode, get_langgraph_service
from agentseek_api.services.run_state import run_broker

UNSET = object()


@dataclass
class RunExecutionResult:
    output: dict[str, Any]
    interrupted: bool
    interrupts: list[dict[str, Any]]


def _normalize_stream_value(value: Any) -> Any:
    if isinstance(value, BaseMessage):
        payload: dict[str, Any] = {
            "type": type(value).__name__,
            "content": _normalize_stream_value(getattr(value, "content", None)),
        }
        tool_calls = getattr(value, "tool_calls", None)
        if tool_calls:
            payload["tool_calls"] = _normalize_stream_value(tool_calls)
        message_name = getattr(value, "name", None)
        if message_name:
            payload["name"] = message_name
        tool_call_id = getattr(value, "tool_call_id", None)
        if tool_call_id:
            payload["tool_call_id"] = tool_call_id
        return payload
    if isinstance(value, dict):
        return {str(key): _normalize_stream_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_stream_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "value") and hasattr(value, "id"):
        return {
            "value": _normalize_stream_value(getattr(value, "value")),
            "id": _normalize_stream_value(getattr(value, "id")),
        }
    return repr(value)


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


def _base_stream_payload(event: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "name": str(event.get("name", "")),
        "langgraph_event": str(event.get("event", "")),
        "langgraph_run_id": str(event.get("run_id", "")),
        "metadata": _normalize_stream_value(event.get("metadata", {})),
        "tags": _normalize_stream_value(event.get("tags", [])),
        "parent_ids": _normalize_stream_value(event.get("parent_ids", [])),
    }
    node_name = event.get("metadata", {}).get("langgraph_node") if isinstance(event.get("metadata"), dict) else None
    if node_name:
        payload["node"] = str(node_name)
    return payload


def _translate_stream_events(event: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    translated: list[tuple[str, dict[str, Any]]] = []
    event_name = event.get("event")
    metadata = event.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}

    node_name = metadata.get("langgraph_node")
    if event_name in {"on_chain_start", "on_chain_end"} and node_name and event.get("name") == node_name:
        translated.append(
            (
                "node_start" if event_name == "on_chain_start" else "node_end",
                _base_stream_payload(event),
            )
        )

    if event_name in {"on_tool_start", "on_tool_end"}:
        payload = _base_stream_payload(event)
        data = event.get("data", {})
        if isinstance(data, dict):
            payload["data"] = _normalize_stream_value(data)
            if "input" in data:
                payload["input"] = _normalize_stream_value(data["input"])
            if "output" in data:
                payload["output"] = _normalize_stream_value(data["output"])
        translated.append(("tool_start" if event_name == "on_tool_start" else "tool_end", payload))

    if event_name == "on_chain_stream":
        data = event.get("data", {})
        chunk = data.get("chunk") if isinstance(data, dict) else None
        for message in _extract_chunk_messages(chunk):
            content = _normalize_stream_value(getattr(message, "content", None))
            tool_calls = _normalize_stream_value(getattr(message, "tool_calls", []) or [])
            if content in ("", [], None) and not tool_calls:
                continue
            payload = _base_stream_payload(event)
            payload["message_type"] = type(message).__name__
            payload["content"] = content
            if tool_calls:
                payload["tool_calls"] = tool_calls
            translated.append(("message_chunk", payload))

    return translated


def _is_root_stream_event(event: dict[str, Any]) -> bool:
    parent_ids = event.get("parent_ids")
    return isinstance(parent_ids, list) and not parent_ids


async def execute_run(
    *,
    thread_id: str,
    run_id: str,
    payload: dict[str, Any],
    graph_id: str | None = None,
    resume: Any = UNSET,
) -> RunExecutionResult:
    ensure_sync_checkpoint_mode(requested_async=False)
    entry = get_langgraph_service().get_entry(graph_id)
    graph = entry.build_graph(db_manager.get_langgraph_checkpointer())

    config = {
        CONF: {
            "thread_id": thread_id,
            "checkpoint_ns": run_id,
            CONFIG_KEY_CHECKPOINTER: db_manager.get_langgraph_checkpointer(),
        }
    }
    if resume is UNSET:
        invocation = entry.prepare_input(payload)
    else:
        invocation = Command(resume=resume)

    result: Any = None
    interrupt_chunk: Any = None
    async for stream_event in graph.astream_events(invocation, config, version="v2"):
        for event_name, event_payload in _translate_stream_events(stream_event):
            run_broker.publish(run_id, event_name, **event_payload)
        if stream_event.get("event") == "on_chain_stream" and _is_root_stream_event(stream_event):
            data = stream_event.get("data", {})
            chunk = data.get("chunk") if isinstance(data, dict) else None
            if isinstance(chunk, dict) and "__interrupt__" in chunk:
                interrupt_chunk = chunk["__interrupt__"]
        if stream_event.get("event") == "on_chain_end" and _is_root_stream_event(stream_event):
            data = stream_event.get("data", {})
            if isinstance(data, dict) and "output" in data:
                result = data["output"]

    if interrupt_chunk is not None:
        if isinstance(result, dict):
            result = {**result, "__interrupt__": interrupt_chunk}
        else:
            result = {"result": result, "__interrupt__": interrupt_chunk}

    output = entry.extract_output(result, payload)
    interrupts = output.get("interrupts", []) if isinstance(output, dict) else []
    interrupted = bool(output.get("interrupted")) if isinstance(output, dict) else False

    checkpointer = db_manager.get_checkpointer()
    await db_manager.run_checkpointer_call(
        checkpointer.save_checkpoint,
        thread_id=thread_id,
        run_id=run_id,
        payload={
            "input": payload,
            "resume": None if resume is UNSET else resume,
            "output": output,
            "graph_id": graph_id or "default",
        },
    )
    return RunExecutionResult(output=output, interrupted=interrupted, interrupts=interrupts)
