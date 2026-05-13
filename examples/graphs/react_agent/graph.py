"""Tool-calling ReAct sample agent (offline-friendly).

Shape
-----
    START -> call_model -> (if tool_calls) tools -> call_model -> ... -> END

The sample ships with a scripted ``call_model`` so it runs without API keys:
first turn emits a tool call, second turn (after the ``ToolMessage`` lands
in state) emits the final answer. To adapt this for production, replace
the body of ``call_model`` with something like::

    model = ChatOpenAI(model="gpt-4o-mini").bind_tools(TOOLS)
    response = await model.ainvoke(state["messages"])
    return {"messages": [response]}

Everything else — the router, the ``ToolNode``, the topology — is unchanged.

Sample input
------------
    {"messages": [{"role": "user", "content": "what is the meaning of life?"}]}
"""

from __future__ import annotations

from typing import Literal

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode


@tool
def lookup(query: str) -> str:
    """Stand-in for a real tool. Returns a canned answer."""

    return f"Result for {query!r}: 42"


TOOLS = [lookup]


async def call_model(state: MessagesState) -> dict:
    """Scripted "LLM" step. Replace with a real chat model in production."""

    has_tool_result = any(isinstance(m, ToolMessage) for m in state["messages"])
    if has_tool_result:
        tool_answers = [m.content for m in state["messages"] if isinstance(m, ToolMessage)]
        return {"messages": [AIMessage(content=f"Final answer: {tool_answers[-1]}")]}

    return {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "lookup", "args": {"query": "meaning of life"}, "id": "call-1"}
                ],
            )
        ]
    }


def route_model_output(state: MessagesState) -> Literal["tools", "__end__"]:
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return END


builder = StateGraph(MessagesState)
builder.add_node("call_model", call_model)
builder.add_node("tools", ToolNode(TOOLS))
builder.add_edge(START, "call_model")
builder.add_conditional_edges("call_model", route_model_output, {"tools": "tools", END: END})
builder.add_edge("tools", "call_model")

graph = builder.compile(name="ReAct Agent")
