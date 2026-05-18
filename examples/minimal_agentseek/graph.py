from typing import Any

from langgraph.graph import END, START, StateGraph


def echo_node(state: dict[str, Any]) -> dict[str, Any]:
    return {"message": state.get("message", "hello from agentseek.json")}


builder = StateGraph(dict)
builder.add_node("echo", echo_node)
builder.add_edge(START, "echo")
builder.add_edge("echo", END)
graph = builder.compile()
