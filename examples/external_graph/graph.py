from __future__ import annotations

from langchain_core.messages import AIMessage
from langgraph.graph import END, START, MessagesState, StateGraph


async def respond(state: MessagesState) -> dict:
    text = state["messages"][-1].content if state["messages"] else ""
    return {"messages": [AIMessage(content=f"external graph heard: {text}")]}


def build_graph(checkpointer=None):
    builder = StateGraph(MessagesState)
    builder.add_node("respond", respond)
    builder.add_edge(START, "respond")
    builder.add_edge("respond", END)
    return builder.compile(name="External Hello Graph", checkpointer=checkpointer)


graph = build_graph()
