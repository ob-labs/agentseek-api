from __future__ import annotations

import asyncio
import time
from collections import defaultdict, deque
from collections.abc import AsyncIterator
from typing import Any


def protocol_timestamp_ms() -> int:
    return int(time.time() * 1000)


def protocol_channel_for_method(method: str) -> str:
    return "input" if method.startswith("input.") else method


def _namespace_matches(
    event_namespace: list[str],
    *,
    namespaces: list[list[str]] | None,
    depth: int | None,
) -> bool:
    if not namespaces:
        return depth is None or len(event_namespace) <= depth

    for prefix in namespaces:
        if event_namespace[: len(prefix)] != prefix:
            continue
        if depth is None or len(event_namespace) - len(prefix) <= depth:
            return True
    return False


class ThreadProtocolEventBroker:
    def __init__(self, *, max_events_per_thread: int = 2048, max_idle_threads: int = 1024) -> None:
        self._events: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._signals: dict[str, asyncio.Event] = defaultdict(asyncio.Event)
        self._next_seq: dict[str, int] = defaultdict(lambda: 1)
        self._active_runs: dict[str, int] = defaultdict(int)
        self._idle_threads: deque[str] = deque()
        self._idle_thread_set: set[str] = set()
        self._max_events_per_thread = max_events_per_thread
        self._max_idle_threads = max_idle_threads

    def _mark_active(self, thread_id: str) -> None:
        if thread_id not in self._idle_thread_set:
            return
        self._idle_thread_set.discard(thread_id)
        try:
            self._idle_threads.remove(thread_id)
        except ValueError:
            return

    def _prune_thread_events(self, thread_id: str) -> None:
        events = self._events.get(thread_id)
        if events is None or len(events) <= self._max_events_per_thread:
            return
        self._events[thread_id] = events[-self._max_events_per_thread :]

    def _drop_thread(self, thread_id: str) -> None:
        self._events.pop(thread_id, None)
        self._signals.pop(thread_id, None)
        self._next_seq.pop(thread_id, None)
        self._active_runs.pop(thread_id, None)
        self._idle_thread_set.discard(thread_id)

    def _prune_idle_threads(self) -> None:
        while len(self._idle_threads) > self._max_idle_threads:
            stale_thread_id = self._idle_threads.popleft()
            if stale_thread_id not in self._idle_thread_set:
                continue
            self._idle_thread_set.discard(stale_thread_id)
            self._drop_thread(stale_thread_id)

    def latest_seq(self, thread_id: str) -> int:
        return self._next_seq[thread_id] - 1

    def run_started(self, thread_id: str) -> None:
        self._mark_active(thread_id)
        self._active_runs[thread_id] += 1
        self._signals[thread_id].set()

    def run_finished(self, thread_id: str) -> None:
        self._active_runs[thread_id] = max(0, self._active_runs[thread_id] - 1)
        if self._active_runs[thread_id] == 0 and thread_id not in self._idle_thread_set:
            self._idle_thread_set.add(thread_id)
            self._idle_threads.append(thread_id)
            self._prune_idle_threads()
        self._signals[thread_id].set()

    def _record_event(self, thread_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        self._mark_active(thread_id)
        seq = self._next_seq[thread_id]
        self._next_seq[thread_id] += 1
        event = {
            "type": "event",
            "event_id": f"{thread_id}:{seq}",
            "seq": seq,
            **payload,
        }
        self._events[thread_id].append(event)
        self._prune_thread_events(thread_id)
        self._signals[thread_id].set()
        return event

    def publish(self, thread_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        event = self._record_event(thread_id, payload)
        self._persist_event(thread_id, event)
        return event

    async def apublish(self, thread_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        event = self._record_event(thread_id, payload)
        await self._persist_event_async(thread_id, event)
        return event

    def _persist_event(self, thread_id: str, event: dict[str, Any]) -> None:
        try:
            from agentseek_api.services.stream_persistence import persist_thread_stream_event
        except Exception:
            return

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            try:
                asyncio.run(persist_thread_stream_event(thread_id, event))
            except Exception:
                return
            return

        loop.create_task(persist_thread_stream_event(thread_id, event))

    async def _persist_event_async(self, thread_id: str, event: dict[str, Any]) -> None:
        try:
            from agentseek_api.services.stream_persistence import persist_thread_stream_event
        except Exception:
            return

        await persist_thread_stream_event(thread_id, event)

    def delete_thread(self, thread_id: str) -> None:
        self._drop_thread(thread_id)

    def snapshot_records(self, thread_id: str, *, after_seq: int = 0) -> list[dict[str, Any]]:
        return [
            dict(event)
            for event in self._events.get(thread_id, [])
            if int(event.get("seq", 0)) > after_seq
        ]

    async def stream(
        self,
        thread_id: str,
        *,
        channels: list[str],
        namespaces: list[list[str]] | None,
        depth: int | None,
        since: int | None,
    ) -> AsyncIterator[dict[str, Any]]:
        seen = 0
        if since is not None:
            for index, event in enumerate(self._events.get(thread_id, [])):
                if int(event.get("seq", 0)) > since:
                    seen = index
                    break
            else:
                seen = len(self._events.get(thread_id, []))

        while True:
            events = self._events.get(thread_id, [])
            while seen < len(events):
                event = dict(events[seen])
                seen += 1
                channel = protocol_channel_for_method(str(event.get("method", "")))
                namespace = event.get("params", {}).get("namespace", [])
                if not isinstance(namespace, list):
                    namespace = []
                if channel not in channels:
                    continue
                if not _namespace_matches(namespace, namespaces=namespaces, depth=depth):
                    continue
                yield event

            if self._active_runs.get(thread_id, 0) == 0:
                return

            signal = self._signals[thread_id]
            signal.clear()
            await signal.wait()


thread_protocol_broker = ThreadProtocolEventBroker()


def publish_lifecycle_event(
    thread_id: str,
    *,
    event: str,
    graph_name: str | None = None,
    error: str | None = None,
    namespace: list[str] | None = None,
) -> dict[str, Any]:
    data: dict[str, Any] = {"event": event}
    if graph_name is not None:
        data["graph_name"] = graph_name
    if error is not None:
        data["error"] = error
    return thread_protocol_broker.publish(
        thread_id,
        {
            "method": "lifecycle",
            "params": {
                "namespace": namespace or [],
                "timestamp": protocol_timestamp_ms(),
                "data": data,
            },
        },
    )


def publish_tool_event(
    thread_id: str,
    *,
    tool_event: str,
    tool_call_id: str,
    tool_name: str | None = None,
    node: str | None = None,
    input_payload: Any | None = None,
    output_payload: Any | None = None,
    error_message: str | None = None,
    namespace: list[str] | None = None,
) -> dict[str, Any]:
    data: dict[str, Any] = {"event": tool_event, "tool_call_id": tool_call_id}
    if tool_name is not None:
        data["tool_name"] = tool_name
    if input_payload is not None:
        data["input"] = input_payload
    if output_payload is not None:
        data["output"] = output_payload
    if error_message is not None:
        data["message"] = error_message
    params: dict[str, Any] = {
        "namespace": namespace or [],
        "timestamp": protocol_timestamp_ms(),
        "data": data,
    }
    if node is not None:
        params["node"] = node
    return thread_protocol_broker.publish(thread_id, {"method": "tools", "params": params})


async def apublish_tool_event(
    thread_id: str,
    *,
    tool_event: str,
    tool_call_id: str,
    tool_name: str | None = None,
    node: str | None = None,
    input_payload: Any | None = None,
    output_payload: Any | None = None,
    error_message: str | None = None,
    namespace: list[str] | None = None,
) -> dict[str, Any]:
    data: dict[str, Any] = {"event": tool_event, "tool_call_id": tool_call_id}
    if tool_name is not None:
        data["tool_name"] = tool_name
    if input_payload is not None:
        data["input"] = input_payload
    if output_payload is not None:
        data["output"] = output_payload
    if error_message is not None:
        data["message"] = error_message
    params: dict[str, Any] = {
        "namespace": namespace or [],
        "timestamp": protocol_timestamp_ms(),
        "data": data,
    }
    if node is not None:
        params["node"] = node
    return await thread_protocol_broker.apublish(thread_id, {"method": "tools", "params": params})


def publish_values_event(
    thread_id: str,
    *,
    values: Any,
    namespace: list[str] | None = None,
) -> dict[str, Any]:
    return thread_protocol_broker.publish(
        thread_id,
        {
            "method": "values",
            "params": {
                "namespace": namespace or [],
                "timestamp": protocol_timestamp_ms(),
                "data": values,
            },
        },
    )


async def apublish_values_event(
    thread_id: str,
    *,
    values: Any,
    namespace: list[str] | None = None,
) -> dict[str, Any]:
    return await thread_protocol_broker.apublish(
        thread_id,
        {
            "method": "values",
            "params": {
                "namespace": namespace or [],
                "timestamp": protocol_timestamp_ms(),
                "data": values,
            },
        },
    )


def publish_updates_event(
    thread_id: str,
    *,
    values: Any,
    namespace: list[str] | None = None,
) -> dict[str, Any]:
    return thread_protocol_broker.publish(
        thread_id,
        {
            "method": "updates",
            "params": {
                "namespace": namespace or [],
                "timestamp": protocol_timestamp_ms(),
                "data": values,
            },
        },
    )


async def apublish_updates_event(
    thread_id: str,
    *,
    values: Any,
    namespace: list[str] | None = None,
) -> dict[str, Any]:
    return await thread_protocol_broker.apublish(
        thread_id,
        {
            "method": "updates",
            "params": {
                "namespace": namespace or [],
                "timestamp": protocol_timestamp_ms(),
                "data": values,
            },
        },
    )


def publish_input_requested(
    thread_id: str,
    *,
    interrupt_id: str,
    payload: Any,
    namespace: list[str] | None = None,
) -> dict[str, Any]:
    return thread_protocol_broker.publish(
        thread_id,
        {
            "method": "input.requested",
            "params": {
                "namespace": namespace or [],
                "timestamp": protocol_timestamp_ms(),
                "data": {
                    "interrupt_id": interrupt_id,
                    "payload": payload,
                },
            },
        },
    )


async def apublish_input_requested(
    thread_id: str,
    *,
    interrupt_id: str,
    payload: Any,
    namespace: list[str] | None = None,
) -> dict[str, Any]:
    return await thread_protocol_broker.apublish(
        thread_id,
        {
            "method": "input.requested",
            "params": {
                "namespace": namespace or [],
                "timestamp": protocol_timestamp_ms(),
                "data": {
                    "interrupt_id": interrupt_id,
                    "payload": payload,
                },
            },
        },
    )


def publish_message_start(
    thread_id: str,
    *,
    message_id: str,
    role: str,
    namespace: list[str] | None = None,
) -> None:
    thread_protocol_broker.publish(
        thread_id,
        {
            "method": "messages",
            "params": {
                "namespace": namespace or [],
                "timestamp": protocol_timestamp_ms(),
                "data": {
                    "event": "message-start",
                    "role": role,
                    "id": message_id,
                },
            },
        },
    )


async def apublish_message_start(
    thread_id: str,
    *,
    message_id: str,
    role: str,
    namespace: list[str] | None = None,
) -> None:
    await thread_protocol_broker.apublish(
        thread_id,
        {
            "method": "messages",
            "params": {
                "namespace": namespace or [],
                "timestamp": protocol_timestamp_ms(),
                "data": {
                    "event": "message-start",
                    "role": role,
                    "id": message_id,
                },
            },
        },
    )


def publish_content_block_start(
    thread_id: str,
    *,
    index: int,
    content: dict[str, Any],
    namespace: list[str] | None = None,
) -> None:
    thread_protocol_broker.publish(
        thread_id,
        {
            "method": "messages",
            "params": {
                "namespace": namespace or [],
                "timestamp": protocol_timestamp_ms(),
                "data": {
                    "event": "content-block-start",
                    "index": index,
                    "content": content,
                },
            },
        },
    )


async def apublish_content_block_start(
    thread_id: str,
    *,
    index: int,
    content: dict[str, Any],
    namespace: list[str] | None = None,
) -> None:
    await thread_protocol_broker.apublish(
        thread_id,
        {
            "method": "messages",
            "params": {
                "namespace": namespace or [],
                "timestamp": protocol_timestamp_ms(),
                "data": {
                    "event": "content-block-start",
                    "index": index,
                    "content": content,
                },
            },
        },
    )


def publish_content_block_delta(
    thread_id: str,
    *,
    index: int,
    delta: dict[str, Any],
    namespace: list[str] | None = None,
) -> None:
    thread_protocol_broker.publish(
        thread_id,
        {
            "method": "messages",
            "params": {
                "namespace": namespace or [],
                "timestamp": protocol_timestamp_ms(),
                "data": {
                    "event": "content-block-delta",
                    "index": index,
                    "delta": delta,
                },
            },
        },
    )


async def apublish_content_block_delta(
    thread_id: str,
    *,
    index: int,
    delta: dict[str, Any],
    namespace: list[str] | None = None,
) -> None:
    await thread_protocol_broker.apublish(
        thread_id,
        {
            "method": "messages",
            "params": {
                "namespace": namespace or [],
                "timestamp": protocol_timestamp_ms(),
                "data": {
                    "event": "content-block-delta",
                    "index": index,
                    "delta": delta,
                },
            },
        },
    )


def publish_content_block_finish(
    thread_id: str,
    *,
    index: int,
    content: dict[str, Any] | None = None,
    namespace: list[str] | None = None,
) -> None:
    data: dict[str, Any] = {
        "event": "content-block-finish",
        "index": index,
    }
    if content is not None:
        data["content"] = content
    thread_protocol_broker.publish(
        thread_id,
        {
            "method": "messages",
            "params": {
                "namespace": namespace or [],
                "timestamp": protocol_timestamp_ms(),
                "data": data,
            },
        },
    )


async def apublish_content_block_finish(
    thread_id: str,
    *,
    index: int,
    content: dict[str, Any] | None = None,
    namespace: list[str] | None = None,
) -> None:
    data: dict[str, Any] = {
        "event": "content-block-finish",
        "index": index,
    }
    if content is not None:
        data["content"] = content
    await thread_protocol_broker.apublish(
        thread_id,
        {
            "method": "messages",
            "params": {
                "namespace": namespace or [],
                "timestamp": protocol_timestamp_ms(),
                "data": data,
            },
        },
    )


def publish_message_complete(
    thread_id: str,
    *,
    namespace: list[str] | None = None,
) -> None:
    thread_protocol_broker.publish(
        thread_id,
        {
            "method": "messages",
            "params": {
                "namespace": namespace or [],
                "timestamp": protocol_timestamp_ms(),
                "data": {
                    "event": "message-finish",
                },
            },
        },
    )


async def apublish_message_complete(
    thread_id: str,
    *,
    namespace: list[str] | None = None,
) -> None:
    await thread_protocol_broker.apublish(
        thread_id,
        {
            "method": "messages",
            "params": {
                "namespace": namespace or [],
                "timestamp": protocol_timestamp_ms(),
                "data": {
                    "event": "message-finish",
                },
            },
        },
    )


def publish_message_chunk(
    thread_id: str,
    *,
    message_id: str,
    role: str,
    text: str,
    namespace: list[str] | None = None,
) -> None:
    publish_message_start(thread_id, message_id=message_id, role=role, namespace=namespace)
    publish_content_block_start(thread_id, index=0, content={"type": "text", "text": ""}, namespace=namespace)
    publish_content_block_delta(
        thread_id,
        index=0,
        delta={"type": "text-delta", "text": text},
        namespace=namespace,
    )


def publish_message_chunk_delta(
    thread_id: str,
    *,
    text: str,
    namespace: list[str] | None = None,
) -> None:
    publish_content_block_delta(
        thread_id,
        index=0,
        delta={"type": "text-delta", "text": text},
        namespace=namespace,
    )


def publish_message_finish(
    thread_id: str,
    *,
    namespace: list[str] | None = None,
) -> None:
    publish_content_block_finish(thread_id, index=0, namespace=namespace)
    publish_message_complete(thread_id, namespace=namespace)


def _message_blocks(message: dict[str, Any]) -> tuple[str | None, list[dict[str, Any]]]:
    message_type = str(message.get("type", ""))
    content = message.get("content")
    tool_calls = message.get("tool_calls") or []

    role_map = {
        "HumanMessage": "human",
        "AIMessage": "ai",
        "SystemMessage": "system",
    }
    role = role_map.get(message_type)
    if role is None:
        return None, []

    blocks: list[dict[str, Any]] = []
    if isinstance(content, str) and content:
        blocks.append({"type": "text", "text": content})
    elif isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and isinstance(block.get("type"), str):
                blocks.append(block)
    has_tool_call_block = any(
        isinstance(block, dict) and block.get("type") in {"tool_call", "tool_call_chunk"}
        for block in blocks
    )
    if role == "ai" and isinstance(tool_calls, list) and not has_tool_call_block:
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
    return role, blocks


def publish_message_transcript(
    thread_id: str,
    *,
    run_id: str,
    messages: list[dict[str, Any]],
    namespace: list[str] | None = None,
    start_index: int = 0,
) -> None:
    for index, message in enumerate(messages):
        role, blocks = _message_blocks(message)
        if role is None or not blocks:
            continue

        message_id = f"{run_id}:{start_index + index}"
        publish_message_start(thread_id, message_id=message_id, role=role, namespace=namespace)
        for block_index, block in enumerate(blocks):
            publish_content_block_start(thread_id, index=block_index, content=block, namespace=namespace)
            publish_content_block_finish(thread_id, index=block_index, content=block, namespace=namespace)
        publish_message_complete(thread_id, namespace=namespace)


async def apublish_message_transcript(
    thread_id: str,
    *,
    run_id: str,
    messages: list[dict[str, Any]],
    namespace: list[str] | None = None,
    start_index: int = 0,
) -> None:
    for index, message in enumerate(messages):
        role, blocks = _message_blocks(message)
        if role is None or not blocks:
            continue

        message_id = f"{run_id}:{start_index + index}"
        await apublish_message_start(thread_id, message_id=message_id, role=role, namespace=namespace)
        for block_index, block in enumerate(blocks):
            await apublish_content_block_start(thread_id, index=block_index, content=block, namespace=namespace)
            await apublish_content_block_finish(thread_id, index=block_index, content=block, namespace=namespace)
        await apublish_message_complete(thread_id, namespace=namespace)
