"""Registry of sample LangGraph apps shipped with agentseek-api.

Each entry is keyed by ``graph_id`` and exposes three pieces:

* ``graph_factory`` — a callable that compiles a ``Pregel`` instance from
  ``examples/graphs/<name>`` with an optional injected checkpointer.
* ``prepare_input`` — converts the raw JSON payload the API receives into
  the shape the graph expects (messages, state dict, ...).
* ``extract_output`` — turns the final graph result into a JSON-serialisable
  dict that is stored as the run output and the OceanBase/SeekDB checkpoint.

Adding a new sample: drop a ``build_graph(checkpointer=None, store=None)``
function under ``examples/graphs/<name>/graph.py`` and append an entry to
``build_sample_registry``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage

_EXAMPLES_ROOT = Path(__file__).resolve().parents[3] / "examples"
if str(_EXAMPLES_ROOT) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_ROOT))


def _message_content(message: BaseMessage) -> str:
    content = getattr(message, "content", "")
    return content if isinstance(content, str) else str(content)


def _ensure_messages_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if "messages" in payload and payload["messages"]:
        return payload
    text = payload.get("message") or payload.get("content")
    if text is None:
        text = json.dumps(payload)
    if not isinstance(text, str):
        text = json.dumps(text)
    return {"messages": [HumanMessage(content=text)]}


def _extract_messages_output(result: Any, _payload: dict[str, Any]) -> dict[str, Any]:
    messages = result.get("messages", []) if isinstance(result, dict) else []
    transcript: list[dict[str, Any]] = []
    for msg in messages:
        transcript.append(
            {
                "type": type(msg).__name__,
                "content": _message_content(msg),
                "tool_calls": list(getattr(msg, "tool_calls", []) or []),
            }
        )
    final = messages[-1] if messages else None
    final_text = _message_content(final) if final is not None else ""
    parsed: Any = None
    try:
        parsed = json.loads(final_text)
    except (TypeError, ValueError):
        parsed = None
    output: dict[str, Any] = {"final_text": final_text, "transcript": transcript}
    if parsed is not None:
        output["final_json"] = parsed
    return output


def _extract_interrupt_output(result: Any, _payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(result, dict):
        return {"result": result}
    output: dict[str, Any] = {"state": {k: v for k, v in result.items() if k != "__interrupt__"}}
    interrupts = result.get("__interrupt__") or []
    output["interrupted"] = bool(interrupts)
    output["interrupts"] = [
        {"value": getattr(item, "value", None), "id": getattr(item, "id", None)}
        for item in interrupts
    ]
    return output


def _prepare_subgraph_hitl_payload(payload: dict[str, Any]) -> dict[str, Any]:
    foo = payload.get("foo")
    if isinstance(foo, str):
        return {"foo": foo}
    message = payload.get("message") or payload.get("content") or ""
    return {"foo": str(message)}


def _prepare_store_memory_payload(payload: dict[str, Any]) -> dict[str, Any]:
    memory_key = payload.get("memory_key")
    if not isinstance(memory_key, str) or not memory_key:
        memory_key = "memory"
    memory_value = payload.get("memory_value", payload.get("value"))
    if memory_value is None:
        memory_value = payload.get("message") or payload.get("content") or payload
    return {"memory_key": memory_key, "memory_value": memory_value}


def _extract_store_memory_output(result: Any, _payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(result, dict):
        return {"result": result}
    output = result.get("output")
    return output if isinstance(output, dict) else {"result": result}


def build_sample_registry() -> dict[str, dict[str, Any]]:
    """Return ``graph_id -> {graph, prepare_input, extract_output}`` mappings.

    Imports happen lazily so a broken sample doesn't take the server down
    at startup — any sample that fails to import is logged and skipped.
    """

    registry: dict[str, dict[str, Any]] = {}

    try:
        from graphs.store_memory.graph import build_graph as store_memory_graph_factory  # type: ignore[import-not-found]

        registry["store_memory"] = {
            "graph_factory": store_memory_graph_factory,
            "prepare_input": _prepare_store_memory_payload,
            "extract_output": _extract_store_memory_output,
        }
    except Exception as exc:  # noqa: BLE001
        print(f"[sample_graphs] skipped store_memory: {exc}", flush=True)

    try:
        from graphs.stress_test.graph import build_graph as stress_test_graph_factory  # type: ignore[import-not-found]

        registry["stress_test"] = {
            "graph_factory": stress_test_graph_factory,
            "prepare_input": _ensure_messages_payload,
            "extract_output": _extract_messages_output,
        }
    except Exception as exc:  # noqa: BLE001
        print(f"[sample_graphs] skipped stress_test: {exc}", flush=True)

    try:
        from graphs.subgraph_agent.graph import build_graph as subgraph_agent_graph_factory  # type: ignore[import-not-found]

        registry["subgraph_agent"] = {
            "graph_factory": subgraph_agent_graph_factory,
            "prepare_input": _ensure_messages_payload,
            "extract_output": _extract_messages_output,
        }
    except Exception as exc:  # noqa: BLE001
        print(f"[sample_graphs] skipped subgraph_agent: {exc}", flush=True)

    try:
        from graphs.react_agent.graph import build_graph as react_agent_graph_factory  # type: ignore[import-not-found]

        registry["react_agent"] = {
            "graph_factory": react_agent_graph_factory,
            "prepare_input": _ensure_messages_payload,
            "extract_output": _extract_messages_output,
        }
    except Exception as exc:  # noqa: BLE001
        print(f"[sample_graphs] skipped react_agent: {exc}", flush=True)

    try:
        from graphs.stress_tool_agent.graph import build_graph as stress_tool_agent_graph_factory  # type: ignore[import-not-found]

        registry["stress_tool_agent"] = {
            "graph_factory": stress_tool_agent_graph_factory,
            "prepare_input": _ensure_messages_payload,
            "extract_output": _extract_messages_output,
        }
    except Exception as exc:  # noqa: BLE001
        print(f"[sample_graphs] skipped stress_tool_agent: {exc}", flush=True)

    try:
        from graphs.subgraph_hitl_agent.graph import build_graph as subgraph_hitl_graph_factory  # type: ignore[import-not-found]

        registry["subgraph_hitl_agent"] = {
            "graph_factory": subgraph_hitl_graph_factory,
            "prepare_input": _prepare_subgraph_hitl_payload,
            "extract_output": _extract_interrupt_output,
        }
    except Exception as exc:  # noqa: BLE001
        print(f"[sample_graphs] skipped subgraph_hitl_agent: {exc}", flush=True)

    return registry


__all__ = ["build_sample_registry"]
