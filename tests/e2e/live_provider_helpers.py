from __future__ import annotations

import json
import os
from typing import Any

from langchain_oceanbase.store import OceanBaseStore

from agentseek_api.core.runtime_store import make_user_store_namespace

_DEFAULT_CAPABILITIES = frozenset({"streaming", "store", "mcp", "hitl"})


def provider_graph_id(capability: str) -> str:
    provider = os.getenv("LIVE_PROVIDER_KIND", "").strip().lower()
    graphs = {
        "openai": {
            "stream": "live_openai_stream",
            "store_memory": "live_openai_store_memory",
            "hitl": "live_openai_hitl",
        },
        "anthropic": {
            "stream": "live_anthropic_stream",
            "store_memory": "live_anthropic_store_memory",
            "hitl": "live_anthropic_hitl",
        },
    }
    return graphs[provider][capability]


def provider_capability_enabled(name: str) -> bool:
    raw_value = os.getenv("LIVE_PROVIDER_CAPABILITIES", "").strip()
    if not raw_value:
        return name in _DEFAULT_CAPABILITIES
    return name in {item.strip() for item in raw_value.split(",") if item.strip()}


def user_headers(user_id: str) -> dict[str, str]:
    return {"x-user-id": user_id}


def parse_sse_events(stream_text: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for chunk in stream_text.strip().split("\n\n"):
        if not chunk.strip():
            continue
        event: dict[str, Any] = {}
        for line in chunk.splitlines():
            if line.startswith("id: "):
                event["id"] = line.removeprefix("id: ")
            elif line.startswith("event: "):
                event["event"] = line.removeprefix("event: ")
            elif line.startswith("data: "):
                event["data"] = json.loads(line.removeprefix("data: "))
        if event:
            events.append(event)
    return events


async def fetch_store_item_from_backend(
    *,
    user_id: str,
    namespace: list[str],
    key: str,
) -> dict[str, object] | None:
    store = OceanBaseStore(
        connection_args={
            "host": os.getenv("OCEANBASE_HOST", "127.0.0.1"),
            "port": os.getenv("OCEANBASE_PORT", "2881"),
            "user": os.getenv("OCEANBASE_USER", "root@test"),
            "password": os.getenv("OCEANBASE_PASSWORD", ""),
            "db_name": os.getenv("OCEANBASE_DB_NAME", "seekdb"),
        }
    )
    try:
        item = await store.aget(make_user_store_namespace(user_id=user_id, namespace=tuple(namespace)), key)
        if item is None:
            return None
        return {
            "namespace": list(item.namespace[2:]),
            "value": dict(item.value),
        }
    finally:
        store.obvector.engine.dispose()
