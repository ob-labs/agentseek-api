"""Delegating-subgraph sample: outer prelude node then hand off to an inner graph.

A common production shape is an outer orchestrator that pre-processes the
incoming request (validation, routing, tenant resolution, ...) and then
delegates the heavy lifting to a compiled inner graph. This sample keeps
both halves fully offline by reusing the deterministic ``stress_test`` graph
as the inner worker.

Shape
-----
    START -> no_stream -> subgraph(stress_test) -> END
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Annotated, Any

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langgraph.graph import END, START, StateGraph, add_messages

from graphs.stress_test.graph import graph as inner_graph


@dataclass
class State:
    messages: Annotated[Sequence[AnyMessage], add_messages] = field(default_factory=list)
    step_count: int = 0
    total_delay: float = 0.0


async def route_to_subgraph(state: State) -> dict[str, Any]:
    """Prelude node: normalise the inbound message before the subgraph runs."""

    if not state.messages:
        prelude = HumanMessage(content=json.dumps({"delay": 0.0, "steps": 1}))
        return {"messages": [prelude]}

    last = state.messages[-1]
    content = last.content if isinstance(last.content, str) else json.dumps(last.content)
    try:
        json.loads(content)
        return {"messages": [AIMessage(content="routing to subgraph")]}
    except (TypeError, ValueError):
        wrapped = HumanMessage(content=json.dumps({"delay": 0.0, "steps": 1, "echo": content}))
        return {"messages": [wrapped]}


builder = StateGraph(State)
builder.add_node("no_stream", route_to_subgraph)
builder.add_node("subgraph", inner_graph)
builder.add_edge(START, "no_stream")
builder.add_edge("no_stream", "subgraph")
builder.add_edge("subgraph", END)

graph = builder.compile(name="Subgraph Agent")
