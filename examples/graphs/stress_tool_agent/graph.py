"""Deterministic stress tool agent with sequential tool calls.

Shape
-----
    START -> call_model -> tools -> call_model -> ... -> END

The graph simulates an LLM-driven ReAct loop without any provider dependency.
It emits one ``slow_process`` tool call at a time until the requested number of
steps has completed, then returns a JSON summary as the final ``AIMessage``.

Input payload
-------------
Either a plain text chat message::

    {"messages": [{"role": "user", "content": "process the data"}]}

or a JSON-encoded tuning knob inside the message content::

    {"messages": [{"role": "user", "content": "{\"delay\": 0.01, \"steps\": 3}"}]}

When submitted via the agentseek-api HTTP interface, the plain-dict form
``{"delay": 0.01, "steps": 3}`` also works — the registry adapter wraps it.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Literal

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode


def _parse_request(messages: list[Any]) -> dict[str, Any]:
    defaults = {"delay": 0.1, "steps": 3}
    human_messages = [message for message in messages if isinstance(message, HumanMessage)]
    if not human_messages:
        return defaults

    content = human_messages[-1].content
    if not isinstance(content, str):
        return defaults

    try:
        parsed = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return defaults

    if not isinstance(parsed, dict):
        return defaults

    config = {**defaults, **parsed}
    config["delay"] = float(config.get("delay", defaults["delay"]))
    config["steps"] = max(1, int(config.get("steps", defaults["steps"])))
    return config


@tool
async def slow_process(step_number: int, delay_seconds: float = 0.1) -> str:
    """Process one step of a sequential pipeline."""

    await asyncio.sleep(delay_seconds)
    return f"Step {step_number} completed successfully. Result: data_chunk_{step_number}"


TOOLS = [slow_process]


async def call_model(state: MessagesState) -> dict[str, list[AIMessage]]:
    config = _parse_request(state["messages"])
    tool_results = [message for message in state["messages"] if isinstance(message, ToolMessage)]
    completed_steps = len(tool_results)

    if completed_steps >= config["steps"]:
        results = [str(message.content) for message in tool_results]
        return {
            "messages": [
                AIMessage(
                    content=json.dumps(
                        {
                            "status": "completed",
                            "steps_completed": completed_steps,
                            "results": results,
                            "total_delay_seconds": round(config["delay"] * completed_steps, 2),
                        }
                    )
                )
            ]
        }

    next_step = completed_steps + 1
    return {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "slow_process",
                        "args": {"step_number": next_step, "delay_seconds": config["delay"]},
                        "id": f"slow-process-{next_step}",
                    }
                ],
            )
        ]
    }


def route_model_output(state: MessagesState) -> Literal["tools", "__end__"]:
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return END


def build_graph(checkpointer=None):
    builder = StateGraph(MessagesState)
    builder.add_node("call_model", call_model)
    builder.add_node("tools", ToolNode(TOOLS))
    builder.add_edge(START, "call_model")
    builder.add_conditional_edges("call_model", route_model_output, {"tools": "tools", END: END})
    builder.add_edge("tools", "call_model")
    return builder.compile(name="Stress Tool Agent", checkpointer=checkpointer)


graph = build_graph()
