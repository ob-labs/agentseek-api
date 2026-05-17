from __future__ import annotations

import asyncio
import time
from collections import defaultdict
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
    def __init__(self) -> None:
        self._events: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._signals: dict[str, asyncio.Event] = defaultdict(asyncio.Event)
        self._next_seq: dict[str, int] = defaultdict(lambda: 1)
        self._active_runs: dict[str, int] = defaultdict(int)

    def latest_seq(self, thread_id: str) -> int:
        return self._next_seq[thread_id] - 1

    def run_started(self, thread_id: str) -> None:
        self._active_runs[thread_id] += 1
        self._signals[thread_id].set()

    def run_finished(self, thread_id: str) -> None:
        self._active_runs[thread_id] = max(0, self._active_runs[thread_id] - 1)
        self._signals[thread_id].set()

    def publish(self, thread_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        seq = self._next_seq[thread_id]
        self._next_seq[thread_id] += 1
        event = {
            "type": "event",
            "event_id": f"{thread_id}:{seq}",
            "seq": seq,
            **payload,
        }
        self._events[thread_id].append(event)
        self._signals[thread_id].set()
        return event

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
    if role == "ai" and isinstance(tool_calls, list):
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
) -> None:
    timestamp = protocol_timestamp_ms()
    for index, message in enumerate(messages):
        role, blocks = _message_blocks(message)
        if role is None or not blocks:
            continue

        message_id = f"{run_id}:{index}"
        thread_protocol_broker.publish(
            thread_id,
            {
                "method": "messages",
                "params": {
                    "namespace": namespace or [],
                    "timestamp": timestamp,
                    "data": {
                        "event": "message-start",
                        "role": role,
                        "id": message_id,
                    },
                },
            },
        )
        for block_index, block in enumerate(blocks):
            thread_protocol_broker.publish(
                thread_id,
                {
                    "method": "messages",
                    "params": {
                        "namespace": namespace or [],
                        "timestamp": timestamp,
                        "data": {
                            "event": "content-block-start",
                            "index": block_index,
                            "content": block,
                        },
                    },
                },
            )
            thread_protocol_broker.publish(
                thread_id,
                {
                    "method": "messages",
                    "params": {
                        "namespace": namespace or [],
                        "timestamp": timestamp,
                        "data": {
                            "event": "content-block-finish",
                            "index": block_index,
                            "content": block,
                        },
                    },
                },
            )
        thread_protocol_broker.publish(
            thread_id,
            {
                "method": "messages",
                "params": {
                    "namespace": namespace or [],
                    "timestamp": timestamp,
                    "data": {
                        "event": "message-finish",
                    },
                },
            },
        )
