"""Deterministic stress-test graph — useful as a load-test baseline.

Shape
-----
    START -> process -> (loop) -> respond -> END

``process`` sleeps for a configurable delay and counts steps. ``respond``
emits a JSON ``AIMessage`` summarising the run. No LLM calls — outputs are
fully deterministic so the graph is safe to run at high concurrency.

Input payload
-------------
Either a plain chat message::

    {"messages": [{"role": "user", "content": "hi"}]}

or a JSON-encoded tuning knob inside the message content::

    {"messages": [{"role": "user", "content": "{\"delay\": 0.01, \"steps\": 3}"}]}

When submitted via the agentseek-api HTTP interface, the plain-dict form
``{"delay": 0.01, "steps": 3}`` also works — the registry adapter wraps it.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Annotated, Any

from langchain_core.messages import AIMessage, AnyMessage
from langgraph.graph import StateGraph, add_messages


@dataclass
class State:
    messages: Annotated[Sequence[AnyMessage], add_messages] = field(default_factory=list)
    step_count: int = 0
    total_delay: float = 0.0


def _parse_config(state: State) -> dict[str, Any]:
    defaults = {"delay": 0.1, "steps": 2, "fail": False}
    if not state.messages:
        return defaults
    content = state.messages[-1].content
    try:
        parsed = json.loads(content)
        if isinstance(parsed, dict):
            return {**defaults, **parsed}
    except (json.JSONDecodeError, TypeError):
        pass
    return defaults


async def process_step(state: State) -> dict[str, Any]:
    config = _parse_config(state)
    delay = float(config["delay"])
    await asyncio.sleep(delay)

    new_step = state.step_count + 1
    new_delay = state.total_delay + delay

    if config.get("fail") and new_step >= int(config.get("steps", 2)):
        raise RuntimeError(f"Intentional failure at step {new_step}")

    return {"step_count": new_step, "total_delay": new_delay}


async def respond(state: State) -> dict[str, Any]:
    return {
        "messages": [
            AIMessage(
                content=json.dumps(
                    {
                        "echo": state.messages[-1].content if state.messages else "",
                        "steps_completed": state.step_count,
                        "total_delay_seconds": round(state.total_delay, 2),
                        "status": "completed",
                    }
                )
            )
        ]
    }


def should_continue(state: State) -> str:
    config = _parse_config(state)
    target_steps = int(config.get("steps", 2))
    return "process" if state.step_count < target_steps else "respond"


def build_graph(checkpointer=None):
    builder = StateGraph(State)
    builder.add_node("process", process_step)
    builder.add_node("respond", respond)
    builder.set_entry_point("process")
    builder.add_conditional_edges("process", should_continue)
    builder.set_finish_point("respond")
    return builder.compile(name="Stress Test", checkpointer=checkpointer)


graph = build_graph()
