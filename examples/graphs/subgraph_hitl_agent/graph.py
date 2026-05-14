"""Subgraph with a human-in-the-loop ``interrupt`` — minimal HITL pattern.

Demonstrates two features a production app typically needs:

* A compiled subgraph nested inside a parent graph.
* A ``langgraph.types.interrupt`` call that pauses execution so the host
  application can gather input from a human.

The first invoke returns a state dict with ``__interrupt__`` populated; the
registry adapter in ``sample_graphs.py`` surfaces that interrupt payload in
the run output so callers can inspect it. Resuming the interrupt requires a
checkpointer wired into the graph via ``compile(checkpointer=...)``; that's
left out here to keep the sample minimal.
"""

from __future__ import annotations

from typing import TypedDict

from langgraph.graph import START, StateGraph
from langgraph.types import interrupt


class State(TypedDict):
    foo: str


def subgraph_set_state(_state: State) -> State:
    return {"foo": "Initial subgraph value."}


def subgraph_node(state: State) -> State:
    value = interrupt("Provide value:")
    return {"foo": state["foo"] + value}


def build_graph(checkpointer=None):
    subgraph_builder = StateGraph(State)
    subgraph_builder.add_node(subgraph_set_state)
    subgraph_builder.add_node(subgraph_node)
    subgraph_builder.add_edge(START, "subgraph_set_state")
    subgraph_builder.add_edge("subgraph_set_state", "subgraph_node")
    subgraph_builder.add_edge("subgraph_node", "__end__")
    subgraph = subgraph_builder.compile(checkpointer=checkpointer)

    builder = StateGraph(State)
    builder.add_node("node_1", subgraph)
    builder.add_edge(START, "node_1")
    builder.add_edge("node_1", "__end__")
    return builder.compile(name="Subgraph HITL", checkpointer=checkpointer)


graph = build_graph()
