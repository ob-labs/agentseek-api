"""Run every sample graph once, in-process, to prove they import and execute.

Useful as a local smoke test while developing a new graph. Does not require
the HTTP server or SeekDB — it invokes each compiled graph directly through
LangGraph.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

_EXAMPLES_ROOT = Path(__file__).resolve().parent
if str(_EXAMPLES_ROOT) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_ROOT))

from langchain_core.messages import HumanMessage  # noqa: E402

from graphs.react_agent.graph import graph as react_graph  # noqa: E402
from graphs.stress_test.graph import graph as stress_graph  # noqa: E402
from graphs.stress_tool_agent.graph import graph as stress_tool_agent_graph  # noqa: E402
from graphs.subgraph_agent.graph import graph as subgraph_agent_graph  # noqa: E402
from graphs.subgraph_hitl_agent.graph import graph as subgraph_hitl_graph  # noqa: E402


async def _run_stress_test() -> None:
    result = await stress_graph.ainvoke(
        {"messages": [HumanMessage(content=json.dumps({"delay": 0.01, "steps": 2}))]}
    )
    payload = json.loads(result["messages"][-1].content)
    assert payload["status"] == "completed"
    assert payload["steps_completed"] == 2
    print("stress_test:", payload)


async def _run_subgraph_agent() -> None:
    result = await subgraph_agent_graph.ainvoke(
        {"messages": [HumanMessage(content=json.dumps({"delay": 0.0, "steps": 1}))]}
    )
    final = json.loads(result["messages"][-1].content)
    assert final["status"] == "completed"
    print("subgraph_agent:", final)


async def _run_react_agent() -> None:
    result = await react_graph.ainvoke(
        {"messages": [HumanMessage(content="what is the meaning of life?")]}
    )
    final = result["messages"][-1].content
    assert "42" in final
    print("react_agent:", final)


async def _run_stress_tool_agent() -> None:
    result = await stress_tool_agent_graph.ainvoke(
        {"messages": [HumanMessage(content=json.dumps({"delay": 0.01, "steps": 3}))]}
    )
    payload = json.loads(result["messages"][-1].content)
    assert payload["status"] == "completed"
    assert payload["steps_completed"] == 3
    tool_messages = [message for message in result["messages"] if type(message).__name__ == "ToolMessage"]
    assert len(tool_messages) == 3
    print("stress_tool_agent:", payload)


async def _run_subgraph_hitl() -> None:
    result = await subgraph_hitl_graph.ainvoke({"foo": "hello "})
    assert result.get("__interrupt__"), "expected interrupt payload"
    print("subgraph_hitl_agent: interrupted ->", result["__interrupt__"])


async def main() -> None:
    await _run_stress_test()
    await _run_subgraph_agent()
    await _run_react_agent()
    await _run_stress_tool_agent()
    await _run_subgraph_hitl()
    print("All sample graphs ran successfully.")


if __name__ == "__main__":
    asyncio.run(main())
