from typing import Any

from langgraph.graph import END, START, StateGraph


def configured_node(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "message": state.get("message", "assistant config example"),
        "metadata": state.get("metadata", {}),
    }


builder = StateGraph(dict)
builder.add_node("configured", configured_node)
builder.add_edge(START, "configured")
builder.add_edge("configured", END)
graph = builder.compile()
