"""Store-backed sample graph used to verify runtime BaseStore injection."""

from __future__ import annotations

import json
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.store.base import NOT_PROVIDED


class StoreMemoryState(TypedDict, total=False):
    memory_key: str
    memory_value: dict[str, Any]
    output: dict[str, Any]


def _coerce_memory_value(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        return {"text": value}
    return {"text": json.dumps(value)}


def build_graph(checkpointer=None, store=None):
    async def persist_memory(state: StoreMemoryState) -> StoreMemoryState:
        if store is None:
            raise RuntimeError("store_memory graph requires an injected store.")
        memory_key = state.get("memory_key") or "memory"
        memory_value = _coerce_memory_value(state.get("memory_value"))
        namespace = ("graph", "memory")
        await store.aput(namespace, memory_key, memory_value, ttl=NOT_PROVIDED)
        stored = await store.aget(namespace, memory_key, refresh_ttl=False)
        if stored is None:
            raise RuntimeError("store_memory graph could not reload the stored item.")
        return {
            "memory_key": memory_key,
            "memory_value": memory_value,
            "output": {
                "namespace": list(stored.namespace),
                "key": stored.key,
                "value": dict(stored.value),
            },
        }

    builder: StateGraph[StoreMemoryState] = StateGraph(StoreMemoryState)
    builder.add_node("persist_memory", persist_memory)
    builder.add_edge(START, "persist_memory")
    builder.add_edge("persist_memory", END)
    return builder.compile(name="Store Memory", checkpointer=checkpointer, store=store)


graph = build_graph()
