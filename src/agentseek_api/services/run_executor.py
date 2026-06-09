from dataclasses import dataclass, field
import inspect
from typing import Any

from langchain_core.messages import BaseMessage, BaseMessageChunk
from langchain_core.messages.utils import message_chunk_to_message
from langgraph.constants import CONF, CONFIG_KEY_CHECKPOINTER
from langgraph.types import Command

from agentseek_api.core.database import db_manager
from agentseek_api.core.runtime_store import UserScopedStore
from agentseek_api.models.auth import User
from agentseek_api.services.langgraph_service import ensure_sync_checkpoint_mode, get_langgraph_service
from agentseek_api.services.run_state import run_broker
from agentseek_api.services.stream_persistence import next_run_stream_seq, persist_run_stream_event
from agentseek_api.services.thread_protocol import (
    apublish_content_block_delta,
    apublish_content_block_finish,
    apublish_content_block_start,
    apublish_input_requested,
    apublish_stream_mode_event,
    apublish_message_complete,
    apublish_message_start,
    apublish_message_transcript,
    apublish_messages_complete,
    apublish_messages_metadata,
    apublish_messages_partial,
    apublish_messages_tuple,
    apublish_tool_event,
    apublish_updates_event,
    apublish_values_event,
    publish_content_block_delta,
    publish_content_block_finish,
    publish_content_block_start,
    publish_message_complete,
    publish_message_start,
    publish_message_transcript,
)

UNSET = object()


@dataclass
class RunExecutionResult:
    output: dict[str, Any]
    interrupted: bool
    interrupts: list[dict[str, Any]]


def _normalize_stream_value(value: Any) -> Any:
    if isinstance(value, BaseMessage):
        # Use model_dump so the wire-level shape matches the official LangGraph
        # SDK contract: lowercase ``type`` (``"ai"``/``"human"``/...), plus
        # ``id``, ``additional_kwargs``, ``response_metadata``, ``tool_calls``.
        # Without this, clients parsing ``{"type": "AIMessage"}`` won't recognize
        # the message and ``updates`` events render as opaque dicts.
        try:
            dumped = value.model_dump()
        except Exception:  # noqa: BLE001
            dumped = {
                "type": getattr(value, "type", type(value).__name__),
                "content": getattr(value, "content", None),
            }
        return _normalize_stream_value(dumped)
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
    try:
        return value.model_dump()
    except Exception:
        pass
    return str(value)


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
    elif isinstance(chunk, tuple) and len(chunk) == 2 and isinstance(chunk[0], str):
        pass
    elif isinstance(chunk, (list, tuple)):
        for item in chunk:
            messages.extend(_extract_chunk_messages(item))
    return messages


def _extract_text_chunk(chunk: Any) -> Any:
    if isinstance(chunk, str):
        return chunk
    text = getattr(chunk, "text", None)
    if isinstance(text, str):
        return text
    return None


def _to_chunk(message: BaseMessage) -> BaseMessageChunk | None:
    """Best-effort conversion of a complete BaseMessage to its chunk type.

    Needed because chat-model providers may emit a final non-chunk frame after
    streaming chunks; ``BaseMessageChunk + BaseMessage`` raises, but two chunks
    add cleanly.
    """
    from langchain_core.messages import (
        AIMessage,
        AIMessageChunk,
        HumanMessage,
        HumanMessageChunk,
        SystemMessage,
        SystemMessageChunk,
        ToolMessage,
        ToolMessageChunk,
    )

    pairs: list[tuple[type[BaseMessage], type[BaseMessageChunk]]] = [
        (AIMessage, AIMessageChunk),
        (HumanMessage, HumanMessageChunk),
        (SystemMessage, SystemMessageChunk),
        (ToolMessage, ToolMessageChunk),
    ]
    for base_cls, chunk_cls in pairs:
        if isinstance(message, base_cls) and not isinstance(message, BaseMessageChunk):
            try:
                return chunk_cls(**message.model_dump(exclude={"type"}))
            except Exception:  # noqa: BLE001
                return None
    return None


def _protocol_role_for_message(message: BaseMessage) -> str | None:
    if isinstance(message, BaseMessage):
        message_type = type(message).__name__
        if message_type.startswith("AIMessage"):
            return "ai"
        if message_type.startswith("HumanMessage"):
            return "human"
        if message_type.startswith("SystemMessage"):
            return "system"
        if message_type.startswith("ToolMessage"):
            return "tool"
    return None


def _build_entry_graph(entry: Any, *, checkpointer: Any, store: Any) -> Any:
    build_graph = entry.build_graph
    signature = inspect.signature(build_graph)
    parameters = list(signature.parameters.values())
    has_var_kwargs = any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters)
    has_store = any(parameter.name == "store" for parameter in parameters)
    if has_var_kwargs or has_store:
        return build_graph(checkpointer, store=store)
    return build_graph(checkpointer)


class _ProtocolMessageStreamState:
    @dataclass
    class _OpenMessage:
        role: str
        namespace: list[str] | None
        open_blocks: dict[int, str] = field(default_factory=dict)
        text_contents: dict[int, str] = field(default_factory=dict)

    def __init__(self, *, thread_id: str, run_id: str) -> None:
        self.thread_id = thread_id
        self.run_id = run_id
        self._open_message_ids: dict[str, _ProtocolMessageStreamState._OpenMessage] = {}
        self.saw_live_messages = False

    def _finish_blocks(
        self,
        state: "_ProtocolMessageStreamState._OpenMessage",
        *,
        namespace: list[str] | None = None,
        before_index: int | None = None,
    ) -> None:
        effective_namespace = state.namespace or namespace
        for index in sorted(list(state.open_blocks)):
            if before_index is not None and index >= before_index:
                continue
            publish_content_block_finish(
                self.thread_id,
                index=index,
                namespace=effective_namespace,
                run_id=self.run_id,
            )
            del state.open_blocks[index]

    def _publish_text_block(
        self,
        state: "_ProtocolMessageStreamState._OpenMessage",
        *,
        index: int,
        text: str,
        namespace: list[str] | None = None,
    ) -> None:
        effective_namespace = state.namespace or namespace
        if index not in state.open_blocks:
            publish_content_block_start(
                self.thread_id,
                index=index,
                content={"type": "text", "text": ""},
                namespace=effective_namespace,
                run_id=self.run_id,
            )
            state.open_blocks[index] = "text"
        previous_text = state.text_contents.get(index, "")
        if text == previous_text:
            return
        delta_text = text[len(previous_text) :] if text.startswith(previous_text) else text
        if delta_text:
            publish_content_block_delta(
                self.thread_id,
                index=index,
                delta={"type": "text-delta", "text": delta_text},
                namespace=effective_namespace,
                run_id=self.run_id,
            )
        state.text_contents[index] = text

    def _publish_nontext_block(
        self,
        state: "_ProtocolMessageStreamState._OpenMessage",
        *,
        index: int,
        block: dict[str, Any],
        namespace: list[str] | None = None,
        final: bool = False,
    ) -> None:
        effective_namespace = state.namespace or namespace
        if index not in state.open_blocks:
            publish_content_block_start(
                self.thread_id,
                index=index,
                content=block,
                namespace=effective_namespace,
                run_id=self.run_id,
            )
            if final:
                publish_content_block_finish(
                    self.thread_id,
                    index=index,
                    content=block,
                    namespace=effective_namespace,
                    run_id=self.run_id,
                )
                return
            state.open_blocks[index] = str(block.get("type", "block"))
            return

        publish_content_block_delta(
            self.thread_id,
            index=index,
            delta=block,
            namespace=effective_namespace,
            run_id=self.run_id,
        )
        if final:
            publish_content_block_finish(
                self.thread_id,
                index=index,
                content=block,
                namespace=effective_namespace,
                run_id=self.run_id,
            )
            del state.open_blocks[index]

    def publish_blocks(
        self,
        *,
        message_id: str,
        role: str,
        blocks: list[dict[str, Any]],
        namespace: list[str] | None = None,
    ) -> None:
        state = self._open_message_ids.get(message_id)
        if state is None:
            publish_message_start(
                self.thread_id,
                message_id=message_id,
                role=role,
                namespace=namespace,
                run_id=self.run_id,
            )
            state = self._OpenMessage(
                role=role,
                namespace=list(namespace) if namespace is not None else None,
            )
            self._open_message_ids[message_id] = state
        elif state.namespace is None and namespace is not None:
            state.namespace = list(namespace)

        for index, block in enumerate(blocks):
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            self._finish_blocks(state, namespace=namespace, before_index=index)
            if block_type == "text":
                text = block.get("text")
                if not isinstance(text, str):
                    continue
                self._publish_text_block(state, index=index, text=text, namespace=namespace)
                continue

            self._publish_nontext_block(state, index=index, block=block, namespace=namespace)
        self.saw_live_messages = True

    def merge_final_messages(self, *, messages: list[dict[str, Any]], run_id: str) -> None:
        transcript_messages = [
            item
            for item in (_protocol_message_from_transcript(message) for message in messages)
            if item is not None
        ]
        open_items = list(self._open_message_ids.items())
        merged_pairs: list[
            tuple[
                tuple[str, "_ProtocolMessageStreamState._OpenMessage"],
                tuple[str, list[dict[str, Any]]],
            ]
        ] = []
        open_index = len(open_items) - 1
        transcript_index = len(transcript_messages) - 1
        while open_index >= 0 and transcript_index >= 0:
            open_item = open_items[open_index]
            transcript_item = transcript_messages[transcript_index]
            if open_item[1].role != transcript_item[0]:
                break
            merged_pairs.append((open_item, transcript_item))
            open_index -= 1
            transcript_index -= 1

        merged_pairs.reverse()
        merged_count = len(merged_pairs)

        for (_message_id, state), (_role, blocks) in merged_pairs:
            for index, block in enumerate(blocks):
                if not isinstance(block, dict):
                    continue
                self._finish_blocks(state, before_index=index)
                if block.get("type") == "text":
                    text = block.get("text")
                    if isinstance(text, str):
                        self._publish_text_block(state, index=index, text=text)
                    continue
                self._publish_nontext_block(state, index=index, block=block, final=True)

        if merged_count == 0 and transcript_messages:
            remaining = messages[-len(transcript_messages) :]
            publish_message_transcript(
                self.thread_id,
                run_id=run_id,
                messages=remaining,
                start_index=max(0, len(open_items)),
            )

    def finish_all(self, *, namespace: list[str] | None = None) -> None:
        while self._open_message_ids:
            message_id = next(iter(self._open_message_ids))
            state = self._open_message_ids.pop(message_id)
            message_namespace = state.namespace or namespace
            self._finish_blocks(state, namespace=message_namespace)
            publish_message_complete(self.thread_id, namespace=message_namespace, run_id=self.run_id)

    async def afinish_blocks(
        self,
        state: "_ProtocolMessageStreamState._OpenMessage",
        *,
        namespace: list[str] | None = None,
        before_index: int | None = None,
    ) -> None:
        effective_namespace = state.namespace or namespace
        for index in sorted(list(state.open_blocks)):
            if before_index is not None and index >= before_index:
                continue
            await apublish_content_block_finish(
                self.thread_id,
                index=index,
                namespace=effective_namespace,
                run_id=self.run_id,
            )
            del state.open_blocks[index]

    async def apublish_text_block(
        self,
        state: "_ProtocolMessageStreamState._OpenMessage",
        *,
        index: int,
        text: str,
        namespace: list[str] | None = None,
    ) -> None:
        effective_namespace = state.namespace or namespace
        if index not in state.open_blocks:
            await apublish_content_block_start(
                self.thread_id,
                index=index,
                content={"type": "text", "text": ""},
                namespace=effective_namespace,
                run_id=self.run_id,
            )
            state.open_blocks[index] = "text"
        previous_text = state.text_contents.get(index, "")
        if text == previous_text:
            return
        delta_text = text[len(previous_text) :] if text.startswith(previous_text) else text
        if delta_text:
            await apublish_content_block_delta(
                self.thread_id,
                index=index,
                delta={"type": "text-delta", "text": delta_text},
                namespace=effective_namespace,
                run_id=self.run_id,
            )
        state.text_contents[index] = text

    async def apublish_nontext_block(
        self,
        state: "_ProtocolMessageStreamState._OpenMessage",
        *,
        index: int,
        block: dict[str, Any],
        namespace: list[str] | None = None,
        final: bool = False,
    ) -> None:
        effective_namespace = state.namespace or namespace
        if index not in state.open_blocks:
            await apublish_content_block_start(
                self.thread_id,
                index=index,
                content=block,
                namespace=effective_namespace,
                run_id=self.run_id,
            )
            if final:
                await apublish_content_block_finish(
                    self.thread_id,
                    index=index,
                    content=block,
                    namespace=effective_namespace,
                    run_id=self.run_id,
                )
                return
            state.open_blocks[index] = str(block.get("type", "block"))
            return

        await apublish_content_block_delta(
            self.thread_id,
            index=index,
            delta=block,
            namespace=effective_namespace,
            run_id=self.run_id,
        )
        if final:
            await apublish_content_block_finish(
                self.thread_id,
                index=index,
                content=block,
                namespace=effective_namespace,
                run_id=self.run_id,
            )
            del state.open_blocks[index]

    async def apublish_blocks(
        self,
        *,
        message_id: str,
        role: str,
        blocks: list[dict[str, Any]],
        namespace: list[str] | None = None,
    ) -> None:
        state = self._open_message_ids.get(message_id)
        if state is None:
            await apublish_message_start(
                self.thread_id,
                message_id=message_id,
                role=role,
                namespace=namespace,
                run_id=self.run_id,
            )
            state = self._OpenMessage(
                role=role,
                namespace=list(namespace) if namespace is not None else None,
            )
            self._open_message_ids[message_id] = state
        elif state.namespace is None and namespace is not None:
            state.namespace = list(namespace)

        for index, block in enumerate(blocks):
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            await self.afinish_blocks(state, namespace=namespace, before_index=index)
            if block_type == "text":
                text = block.get("text")
                if not isinstance(text, str):
                    continue
                await self.apublish_text_block(state, index=index, text=text, namespace=namespace)
                continue

            await self.apublish_nontext_block(state, index=index, block=block, namespace=namespace)
        self.saw_live_messages = True

    async def amerge_final_messages(self, *, messages: list[dict[str, Any]], run_id: str) -> None:
        transcript_messages = [
            item
            for item in (_protocol_message_from_transcript(message) for message in messages)
            if item is not None
        ]
        open_items = list(self._open_message_ids.items())
        merged_pairs: list[
            tuple[
                tuple[str, "_ProtocolMessageStreamState._OpenMessage"],
                tuple[str, list[dict[str, Any]]],
            ]
        ] = []
        open_index = len(open_items) - 1
        transcript_index = len(transcript_messages) - 1
        while open_index >= 0 and transcript_index >= 0:
            open_item = open_items[open_index]
            transcript_item = transcript_messages[transcript_index]
            if open_item[1].role != transcript_item[0]:
                break
            merged_pairs.append((open_item, transcript_item))
            open_index -= 1
            transcript_index -= 1

        merged_pairs.reverse()
        merged_count = len(merged_pairs)

        for (_message_id, state), (_role, blocks) in merged_pairs:
            for index, block in enumerate(blocks):
                if not isinstance(block, dict):
                    continue
                await self.afinish_blocks(state, before_index=index)
                if block.get("type") == "text":
                    text = block.get("text")
                    if isinstance(text, str):
                        await self.apublish_text_block(state, index=index, text=text)
                    continue
                await self.apublish_nontext_block(state, index=index, block=block, final=True)

        if merged_count == 0 and transcript_messages:
            remaining = messages[-len(transcript_messages) :]
            await apublish_message_transcript(
                self.thread_id,
                run_id=run_id,
                messages=remaining,
                start_index=max(0, len(open_items)),
            )

    async def afinish_all(self, *, namespace: list[str] | None = None) -> None:
        while self._open_message_ids:
            message_id = next(iter(self._open_message_ids))
            state = self._open_message_ids.pop(message_id)
            message_namespace = state.namespace or namespace
            await self.afinish_blocks(state, namespace=message_namespace)
            await apublish_message_complete(self.thread_id, namespace=message_namespace, run_id=self.run_id)


def _protocol_blocks_for_message(message: BaseMessage) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    content_blocks = getattr(message, "content_blocks", None)
    saw_tool_call_block = False
    if isinstance(content_blocks, list):
        for block in content_blocks:
            if isinstance(block, dict) and isinstance(block.get("type"), str):
                normalized_block = _normalize_stream_value(block)
                if not isinstance(normalized_block, dict):
                    continue
                if normalized_block.get("type") in {"tool_call", "tool_call_chunk"}:
                    saw_tool_call_block = True
                blocks.append(normalized_block)
    else:
        content = getattr(message, "content", None)
        if isinstance(content, str) and content:
            blocks.append({"type": "text", "text": content})

    if _protocol_role_for_message(message) == "ai" and not saw_tool_call_block:
        tool_calls = getattr(message, "tool_calls", None) or []
        for tool_call in tool_calls:
            normalized_tool_call = _normalize_stream_value(tool_call)
            if not isinstance(normalized_tool_call, dict):
                continue
            blocks.append(
                {
                    "type": "tool_call",
                    "id": normalized_tool_call.get("id"),
                    "name": normalized_tool_call.get("name", "tool"),
                    "args": normalized_tool_call.get("args", {}),
                }
            )
    return blocks


def _protocol_message_from_transcript(message: dict[str, Any]) -> tuple[str, list[dict[str, Any]]] | None:
    message_type = str(message.get("type", ""))
    role_map = {
        "HumanMessage": "human",
        "AIMessage": "ai",
        "SystemMessage": "system",
    }
    role = role_map.get(message_type)
    if role is None:
        return None

    blocks: list[dict[str, Any]] = []
    content = message.get("content")
    if isinstance(content, str) and content:
        blocks.append({"type": "text", "text": content})
    elif isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and isinstance(block.get("type"), str):
                blocks.append(block)

    if role == "ai" and not any(
        isinstance(block, dict) and block.get("type") in {"tool_call", "tool_call_chunk"}
        for block in blocks
    ):
        tool_calls = message.get("tool_calls") or []
        if isinstance(tool_calls, list):
            for tool_call in tool_calls:
                if not isinstance(tool_call, dict):
                    continue
                blocks.append(
                    {
                        "type": "tool_call",
                        "id": tool_call.get("id"),
                        "name": tool_call.get("name", "tool"),
                        "args": tool_call.get("args", {}),
                    }
                )
    return (role, blocks) if blocks else None


def _extract_protocol_result_messages(normalized_result: dict[str, Any]) -> list[dict[str, Any]] | None:
    messages = normalized_result.get("messages")
    if isinstance(messages, list):
        return messages
    output = normalized_result.get("output")
    if isinstance(output, dict):
        nested_messages = output.get("messages")
        if isinstance(nested_messages, list):
            return nested_messages
    return None


def _protocol_namespace_for_event(event: dict[str, Any]) -> list[str]:
    metadata = event.get("metadata", {})
    if not isinstance(metadata, dict):
        return []

    checkpoint_ns = metadata.get("langgraph_checkpoint_ns") or metadata.get("checkpoint_ns")
    if isinstance(checkpoint_ns, str) and checkpoint_ns:
        namespace = [segment.strip() for segment in checkpoint_ns.split("|") if segment.strip()]
        if namespace:
            return namespace

    path = metadata.get("langgraph_path")
    if isinstance(path, list):
        namespace = [
            str(segment)
            for segment in path
            if isinstance(segment, str) and segment and not segment.startswith("__pregel_")
        ]
        if namespace:
            return namespace

    node_name = metadata.get("langgraph_node")
    if isinstance(node_name, str) and node_name:
        return [node_name]
    return []


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

    if event_name in {"on_chain_stream", "on_chat_model_stream", "on_llm_stream"}:
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
        if not translated and event_name == "on_llm_stream":
            content = _extract_text_chunk(chunk)
            if content not in ("", None):
                payload = _base_stream_payload(event)
                payload["message_type"] = type(chunk).__name__
                payload["content"] = content
                translated.append(("message_chunk", payload))

    return translated


def _is_root_stream_event(event: dict[str, Any]) -> bool:
    parent_ids = event.get("parent_ids")
    return isinstance(parent_ids, list) and not parent_ids


async def execute_run(
    *,
    thread_id: str,
    run_id: str,
    payload: Any,
    kwargs: dict[str, Any] | None = None,
    user_id: str,
    graph_id: str | None = None,
    resume: Any = UNSET,
) -> RunExecutionResult:
    ensure_sync_checkpoint_mode(requested_async=False)
    entry = get_langgraph_service().get_entry(graph_id)
    runtime_store = UserScopedStore(db_manager.get_store(), user_id=user_id)
    graph = _build_entry_graph(
        entry,
        checkpointer=db_manager.get_langgraph_checkpointer(),
        store=runtime_store,
    )

    run_kwargs = kwargs or {}
    user_config = dict(run_kwargs.get("config", {})) if isinstance(run_kwargs.get("config"), dict) else {}
    config = dict(user_config)
    graph_bound_config = getattr(graph, "config", None) or {}
    if "recursion_limit" not in config and "recursion_limit" in graph_bound_config:
        config["recursion_limit"] = graph_bound_config["recursion_limit"]
    configurable = dict(config.get(CONF, {})) if isinstance(config.get(CONF), dict) else {}
    configurable.update(
        {
            "thread_id": thread_id,
            "checkpoint_ns": run_id,
            "graph_id": graph_id or "default",
            "assistant_id": graph_id or "default",
            "langgraph_auth_user": User(identity=user_id),
            CONFIG_KEY_CHECKPOINTER: db_manager.get_langgraph_checkpointer(),
            "store": runtime_store,
        }
    )
    if "context" in run_kwargs:
        configurable["context"] = run_kwargs["context"]
    config[CONF] = configurable
    command_payload = run_kwargs.get("command")
    if resume is not UNSET:
        invocation = Command(resume=resume)
    elif command_payload is not None:
        cmd_kwargs: dict[str, Any] = {}
        if "resume" in command_payload:
            cmd_kwargs["resume"] = command_payload["resume"]
        if "update" in command_payload:
            cmd_kwargs["update"] = command_payload["update"]
        if "goto" in command_payload:
            cmd_kwargs["goto"] = command_payload["goto"]
        invocation = Command(**cmd_kwargs) if cmd_kwargs else entry.prepare_input(payload)
    else:
        invocation = entry.prepare_input(payload)

    result: Any = None
    interrupt_chunk: Any = None
    interrupt_namespace: list[str] | None = None
    protocol_messages = _ProtocolMessageStreamState(thread_id=thread_id, run_id=run_id)
    # Accumulators for the official LangGraph ``messages/partial`` wire format —
    # one accumulated message per id, plus a "metadata seen" set so we only emit
    # ``messages/metadata`` once per message_id.
    messages_partial_acc: dict[str, BaseMessage] = {}
    messages_metadata_seen: set[str] = set()
    _emitted_values_via_stream = False
    _requested_stream_modes = run_kwargs.get("stream_modes") or []
    _want_messages_tuple = "messages-tuple" in _requested_stream_modes
    _extra_stream_modes = [m for m in _requested_stream_modes if m not in ("messages", "messages-tuple", "updates")]
    _astream_kwargs: dict[str, Any] = {}
    if _extra_stream_modes:
        _astream_kwargs["stream_mode"] = list(set(_extra_stream_modes) | {"updates"})
    _interrupt_before = run_kwargs.get("interrupt_before")
    if _interrupt_before:
        _astream_kwargs["interrupt_before"] = _interrupt_before
    _interrupt_after = run_kwargs.get("interrupt_after")
    if _interrupt_after:
        _astream_kwargs["interrupt_after"] = _interrupt_after
    _durability = run_kwargs.get("durability")
    if _durability:
        _astream_kwargs["durability"] = _durability
    if run_kwargs.get("stream_subgraphs"):
        _astream_kwargs["subgraphs"] = True
        
    async for stream_event in graph.astream_events(invocation, config, version="v2", **_astream_kwargs):
        protocol_namespace = _protocol_namespace_for_event(stream_event)
        for event_name, event_payload in _translate_stream_events(stream_event):
            seq = await next_run_stream_seq(run_id)
            seq, published_payload = run_broker.publish(run_id, event_name, seq=seq, **event_payload)
            await persist_run_stream_event(run_id, seq=seq, payload=published_payload)
        raw_event_name = stream_event.get("event")
        if raw_event_name in {"on_chat_model_stream", "on_llm_stream", "on_chain_stream"}:
            data = stream_event.get("data", {})
            chunk = data.get("chunk") if isinstance(data, dict) else None
            extracted_messages = _extract_chunk_messages(chunk)
            for message_index, message in enumerate(extracted_messages):
                role = _protocol_role_for_message(message)
                blocks = _protocol_blocks_for_message(message)
                if role is None or not blocks:
                    continue
                explicit_message_id = getattr(message, "id", None)
                if isinstance(explicit_message_id, str) and explicit_message_id:
                    message_id = explicit_message_id
                else:
                    message_id = f"{str(stream_event.get('run_id', '')) or run_id}:message:{message_index}"
                await protocol_messages.apublish_blocks(
                    message_id=message_id,
                    role=role,
                    blocks=blocks,
                    namespace=protocol_namespace,
                )
                # Emit ``messages/metadata`` once, then accumulate the message
                # and emit ``messages/partial`` with the full accumulated payload.
                first_seen = message_id not in messages_metadata_seen
                if first_seen:
                    messages_metadata_seen.add(message_id)
                    await apublish_messages_metadata(
                        thread_id,
                        message_id=message_id,
                        metadata=_normalize_stream_value(stream_event.get("metadata", {})) or {},
                        namespace=protocol_namespace,
                        run_id=run_id,
                    )
                if role in ("tool", "human", "system"):
                    if first_seen:
                        msg_dump = _normalize_stream_value(message)
                        if isinstance(msg_dump, dict):
                            await apublish_messages_complete(
                                thread_id,
                                messages=[msg_dump],
                                namespace=protocol_namespace,
                                run_id=run_id,
                            )
                    continue
                if _want_messages_tuple:
                    chunk_dump = _normalize_stream_value(message)
                    if isinstance(chunk_dump, dict):
                        event_metadata = _normalize_stream_value(stream_event.get("metadata", {})) or {}
                        await apublish_messages_tuple(
                            thread_id,
                            chunk=chunk_dump,
                            metadata=event_metadata,
                            namespace=protocol_namespace,
                            run_id=run_id,
                        )
                existing = messages_partial_acc.get(message_id)
                if existing is None:
                    accumulated = message
                elif not isinstance(message, BaseMessageChunk):
                    # A full BaseMessage with an id we've been streaming is the
                    # node's final assembled message — replace, don't re-add.
                    accumulated = message
                else:
                    left = existing if isinstance(existing, BaseMessageChunk) else _to_chunk(existing)
                    accumulated = left + message if left is not None else message
                messages_partial_acc[message_id] = accumulated
                # Wire format mirrors official LangGraph: lowercase ``type``
                # ("ai", not "AIMessageChunk"). ``message_chunk_to_message``
                # converts the accumulated chunk to its non-chunk equivalent
                # before serialization.
                output_message = (
                    message_chunk_to_message(accumulated)
                    if isinstance(accumulated, BaseMessageChunk)
                    else accumulated
                )
                accumulated_dump = _normalize_stream_value(output_message)
                if isinstance(accumulated_dump, dict):
                    await apublish_messages_partial(
                        thread_id,
                        messages=[accumulated_dump],
                        namespace=protocol_namespace,
                        run_id=run_id,
                    )
            if raw_event_name == "on_llm_stream":
                text = _extract_text_chunk(chunk)
                if text not in ("", None):
                    await protocol_messages.apublish_blocks(
                        message_id=f"{str(stream_event.get('run_id', '')) or run_id}:message:0",
                        role="ai",
                        blocks=[{"type": "text", "text": text}],
                        namespace=protocol_namespace,
                    )
        if raw_event_name == "on_tool_start":
            metadata = stream_event.get("metadata", {})
            data = stream_event.get("data", {})
            await apublish_tool_event(
                thread_id,
                tool_event="tool-started",
                tool_call_id=str(stream_event.get("run_id", "")),
                tool_name=str(stream_event.get("name", "tool")),
                node=str(metadata.get("langgraph_node")) if isinstance(metadata, dict) and metadata.get("langgraph_node") else None,
                input_payload=_normalize_stream_value(data.get("input")) if isinstance(data, dict) and "input" in data else None,
                namespace=protocol_namespace,
            )
        if raw_event_name == "on_tool_end":
            metadata = stream_event.get("metadata", {})
            data = stream_event.get("data", {})
            await apublish_tool_event(
                thread_id,
                tool_event="tool-finished",
                tool_call_id=str(stream_event.get("run_id", "")),
                tool_name=str(stream_event.get("name", "tool")),
                node=str(metadata.get("langgraph_node")) if isinstance(metadata, dict) and metadata.get("langgraph_node") else None,
                output_payload=_normalize_stream_value(data.get("output")) if isinstance(data, dict) and "output" in data else None,
                namespace=protocol_namespace,
            )
        if raw_event_name == "on_custom_event":
            data = stream_event.get("data")
            await apublish_stream_mode_event(
                thread_id,
                method="custom",
                data=_normalize_stream_value(data),
                namespace=protocol_namespace,
                run_id=run_id,
            )
        if stream_event.get("event") == "on_chain_stream":
            data = stream_event.get("data", {})
            chunk = data.get("chunk") if isinstance(data, dict) else None
            if isinstance(chunk, tuple) and len(chunk) == 2:
                stream_mode_name, stream_mode_data = chunk
                if stream_mode_name in ("custom", "debug", "tasks", "checkpoints", "events"):
                    await apublish_stream_mode_event(
                        thread_id,
                        method=stream_mode_name,
                        data=_normalize_stream_value(stream_mode_data),
                        namespace=protocol_namespace,
                        run_id=run_id,
                    )
                elif stream_mode_name == "values":
                    normalized_values = _normalize_stream_value(stream_mode_data)
                    if normalized_values:
                        _emitted_values_via_stream = True
                        await apublish_values_event(
                            thread_id,
                            values=normalized_values,
                            namespace=protocol_namespace,
                            run_id=run_id,
                        )
                elif stream_mode_name == "updates" and isinstance(stream_mode_data, dict):
                    if "__interrupt__" in stream_mode_data:
                        interrupt_chunk = stream_mode_data["__interrupt__"]
                        interrupt_namespace = protocol_namespace
                    normalized_chunk = _normalize_stream_value(stream_mode_data)
                    if isinstance(normalized_chunk, dict):
                        normalized_chunk.pop("__interrupt__", None)
                        if normalized_chunk:
                            await apublish_updates_event(
                                thread_id,
                                values=normalized_chunk,
                                namespace=protocol_namespace,
                                run_id=run_id,
                            )
            elif isinstance(chunk, dict):
                if "__interrupt__" in chunk:
                    interrupt_chunk = chunk["__interrupt__"]
                    interrupt_namespace = protocol_namespace
                normalized_chunk = _normalize_stream_value(chunk)
                if isinstance(normalized_chunk, dict):
                    normalized_chunk.pop("__interrupt__", None)
                    if normalized_chunk:
                        await apublish_updates_event(
                            thread_id,
                            values=normalized_chunk,
                            namespace=protocol_namespace,
                            run_id=run_id,
                        )
        if stream_event.get("event") == "on_chain_end" and _is_root_stream_event(stream_event):
            data = stream_event.get("data", {})
            if isinstance(data, dict) and "output" in data:
                result = data["output"]
                normalized_result = _normalize_stream_value(result)
                if isinstance(normalized_result, dict):
                    messages = _extract_protocol_result_messages(normalized_result)
                    if isinstance(messages, list):
                        if protocol_messages.saw_live_messages:
                            await protocol_messages.amerge_final_messages(messages=messages, run_id=run_id)
                        else:
                            await apublish_message_transcript(thread_id, run_id=run_id, messages=messages)
                    await protocol_messages.afinish_all()
                    if not _emitted_values_via_stream:
                        await apublish_values_event(
                            thread_id,
                            values=normalized_result,
                            namespace=protocol_namespace,
                            run_id=run_id,
                        )

    if interrupt_chunk is not None:
        if isinstance(result, dict):
            result = {**result, "__interrupt__": interrupt_chunk}
        else:
            result = {"result": result, "__interrupt__": interrupt_chunk}
        for item in _normalize_stream_value(interrupt_chunk):
            if not isinstance(item, dict):
                continue
            await apublish_input_requested(
                thread_id,
                interrupt_id=str(item.get("id", "")),
                payload=item.get("value"),
                namespace=interrupt_namespace,
                run_id=run_id,
            )

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
