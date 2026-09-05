"""Seed one default Assistant row per graph declared in the active manifest.

LangGraph Studio expects assistants to exist for every graph in
``langgraph.json`` / ``agentseek.json`` so it can hydrate the UI on first
load. Without seeded rows ``GET /assistants/{id}`` returns 404 for IDs that
Studio derives client-side, blocking the whole graph picker.

We use ``uuid5(ASSISTANT_NAMESPACE_UUID, graph_id)`` so the same graph_id
always maps to the same assistant_id across restarts — clients can cache the
mapping and the operation is idempotent.
"""

from __future__ import annotations

from collections.abc import Iterable
from uuid import UUID, uuid5

from sqlalchemy import select

from agentseek_api.core.config_file import get_active_config_payload
from agentseek_api.core.database import db_manager
from agentseek_api.core.orm import Assistant
from agentseek_api.services.langgraph_service import get_langgraph_service

ASSISTANT_NAMESPACE_UUID = UUID("6ba7b821-9dad-11d1-80b4-00c04fd430c8")


def derive_assistant_id(graph_id: str) -> str:
    return str(uuid5(ASSISTANT_NAMESPACE_UUID, graph_id))


def resolve_assistant_id(requested_id: str, available_graphs: Iterable[str] | None = None) -> str:
    """Resolve an assistant identifier that may be an assistant UUID or a graph_id.

    If ``requested_id`` matches a known graph_id, return the deterministic
    default assistant UUID derived for that graph. Otherwise return the input
    unchanged so callers can look it up directly as an assistant_id.
    """
    if available_graphs is None:
        available_graphs = get_langgraph_service().registered_graph_ids()
    graph_set = available_graphs if isinstance(available_graphs, set) else set(available_graphs)
    if requested_id in graph_set:
        return derive_assistant_id(requested_id)
    return requested_id


def _manifest_graph_ids() -> list[str]:
    payload = get_active_config_payload()
    if payload is None:
        return []
    graphs = payload.get("graphs")
    if not isinstance(graphs, dict):
        return []
    return [gid for gid in graphs if isinstance(gid, str) and gid]


async def ensure_default_assistants() -> None:
    graph_ids = _manifest_graph_ids()
    if not graph_ids:
        return
    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        for graph_id in graph_ids:
            assistant_id = derive_assistant_id(graph_id)
            existing = await session.scalar(
                select(Assistant).where(Assistant.assistant_id == assistant_id)
            )
            if existing is not None:
                continue
            session.add(
                Assistant(
                    assistant_id=assistant_id,
                    name=graph_id,
                    graph_id=graph_id,
                    description=f"Default assistant for graph '{graph_id}'",
                    metadata_json={"created_by": "system"},
                )
            )
        await session.commit()
