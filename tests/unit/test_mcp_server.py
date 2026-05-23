from pathlib import Path

from agentseek_api.mcp_server import list_graph_tools
from agentseek_api.services.langgraph_service import LangGraphService


def test_list_graph_tools_registers_manifest_tools(tmp_path: Path) -> None:
    graph_file = tmp_path / "graph.py"
    graph_file.write_text(
        """
from langgraph.graph import END, START, StateGraph

builder = StateGraph(dict)
builder.add_node("node", lambda state: {"answer": state["question"]})
builder.add_edge(START, "node")
builder.add_edge("node", END)
graph = builder.compile()
""".strip(),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "langgraph.json"
    manifest_path.write_text(
        """
{
  "graphs": {
    "chat": {
      "graph": "./graph.py:graph",
      "name": "chat_tool",
      "description": "Chat tool",
      "input_schema": {
        "type": "object",
        "properties": {
          "question": { "type": "string" }
        }
      }
    }
  }
}
""".strip(),
        encoding="utf-8",
    )

    tools = list_graph_tools(LangGraphService(manifest_path=manifest_path))

    chat_tool = next(tool for tool in tools if tool.name == "chat_tool")
    assert chat_tool.description == "Chat tool"
    assert chat_tool.inputSchema["properties"]["question"]["type"] == "string"
