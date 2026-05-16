import json
from pathlib import Path

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from agentseek_api.services.langgraph_service import GraphManifestError, LangGraphService


def _write_external_graph_package(tmp_path: Path, package_name: str = "external_graph_pkg") -> str:
    package_dir = tmp_path / package_name
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "graph_module.py").write_text(
        """
from langgraph.graph import END, START, StateGraph


def build_graph(checkpointer=None):
    builder = StateGraph(dict)
    builder.add_node("echo", lambda state: {"external": state["message"], "checkpointer": checkpointer is not None})
    builder.add_edge(START, "echo")
    builder.add_edge("echo", END)
    return builder.compile(checkpointer=checkpointer)


def prepare_input(payload):
    return {"message": payload["message"].upper()}


def extract_output(result, payload):
    return {
        "external": result["external"],
        "used_payload": payload["message"],
        "checkpointer": result["checkpointer"],
    }
""".strip(),
        encoding="utf-8",
    )
    return package_name


def test_manifest_entries_override_bundled_graphs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    package_name = _write_external_graph_package(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    manifest_path = tmp_path / "graphs.json"
    manifest_path.write_text(
        json.dumps(
            {
                "graphs": {
                    "default": {
                        "graph": f"{package_name}.graph_module:build_graph",
                        "prepare_input": f"{package_name}.graph_module:prepare_input",
                        "extract_output": f"{package_name}.graph_module:extract_output",
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    service = LangGraphService(manifest_path=manifest_path)
    entry = service.get_entry("default")
    output = entry.extract_output(entry.build_graph().invoke(entry.prepare_input({"message": "hello"})), {"message": "hello"})

    assert output == {"external": "HELLO", "used_payload": "hello", "checkpointer": False}


def test_manifest_uses_default_message_adapters_when_omitted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    package_name = _write_external_graph_package(tmp_path, package_name="external_graph_pkg_defaults")
    monkeypatch.syspath_prepend(str(tmp_path))
    manifest_path = tmp_path / "graphs.json"
    manifest_path.write_text(
        json.dumps({"graphs": {"external": {"graph": f"{package_name}.graph_module:build_graph"}}}),
        encoding="utf-8",
    )

    service = LangGraphService(manifest_path=manifest_path)
    entry = service.get_entry("external")

    prepared = entry.prepare_input({"message": "hello"})
    assert prepared["messages"][0].content == "hello"

    output = entry.extract_output({"messages": []}, {"message": "hello"})
    assert output == {"final_text": "", "transcript": []}


def test_manifest_path_must_exist(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.json"
    with pytest.raises(GraphManifestError, match="AGENTSEEK_GRAPHS"):
        LangGraphService(manifest_path=missing_path)


def test_manifest_rejects_bad_symbol(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    package_name = _write_external_graph_package(tmp_path, package_name="external_graph_pkg_bad_symbol")
    monkeypatch.syspath_prepend(str(tmp_path))
    manifest_path = tmp_path / "graphs.json"
    manifest_path.write_text(
        json.dumps({"graphs": {"broken": {"graph": f"{package_name}.graph_module:missing_symbol"}}}),
        encoding="utf-8",
    )

    with pytest.raises(GraphManifestError, match="broken"):
        LangGraphService(manifest_path=manifest_path)


def test_manifest_rejects_non_callable_graph_factory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    package_name = "external_graph_pkg_non_callable"
    package_dir = tmp_path / package_name
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "graph_module.py").write_text("graph = 123\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    manifest_path = tmp_path / "graphs.json"
    manifest_path.write_text(
        json.dumps({"graphs": {"broken": {"graph": f"{package_name}.graph_module:graph"}}}),
        encoding="utf-8",
    )

    with pytest.raises(GraphManifestError, match="callable"):
        LangGraphService(manifest_path=manifest_path)


def test_manifest_supports_relative_python_file_graph_defs(tmp_path: Path) -> None:
    graph_file = tmp_path / "file_graph.py"
    graph_file.write_text(
        """
from langgraph.graph import END, START, StateGraph


def build_graph(checkpointer=None):
    builder = StateGraph(dict)
    builder.add_node("node", lambda state: {"value": state["value"]})
    builder.add_edge(START, "node")
    builder.add_edge("node", END)
    return builder.compile(checkpointer=checkpointer)
""".strip(),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "graphs.json"
    manifest_path.write_text(
        json.dumps({"graphs": {"file_graph": {"graph": "./file_graph.py:build_graph"}}}),
        encoding="utf-8",
    )

    service = LangGraphService(manifest_path=manifest_path)
    result = service.get_entry("file_graph").build_graph().invoke({"value": "ok"})
    assert result["value"] == "ok"


def test_manifest_supports_langgraph_basic_config_shorthand(tmp_path: Path) -> None:
    package_dir = tmp_path / "chat"
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "graph.py").write_text(
        """
from langgraph.graph import END, START, StateGraph

builder = StateGraph(dict)
builder.add_node("node", lambda state: {"value": "basic-config"})
builder.add_edge(START, "node")
builder.add_edge("node", END)
graph = builder.compile()
""".strip(),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "langgraph.json"
    manifest_path.write_text(
        json.dumps(
            {
                "$schema": "https://langgra.ph/schema.json",
                "dependencies": ["."],
                "graphs": {"chat": "chat.graph:graph"},
            }
        ),
        encoding="utf-8",
    )

    service = LangGraphService(manifest_path=manifest_path)
    result = service.get_entry("chat").build_graph().invoke({})
    assert result["value"] == "basic-config"


def test_manifest_supports_compiled_graph_variables(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    package_dir = tmp_path / "external_graph_pkg_compiled"
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "graph_module.py").write_text(
        """
from langgraph.graph import END, START, StateGraph

builder = StateGraph(dict)
builder.add_node("node", lambda state: {"value": state["value"]})
builder.add_edge(START, "node")
builder.add_edge("node", END)
graph = builder.compile()
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    manifest_path = tmp_path / "graphs.json"
    manifest_path.write_text(
        json.dumps({"graphs": {"compiled": {"graph": "external_graph_pkg_compiled.graph_module:graph"}}}),
        encoding="utf-8",
    )

    service = LangGraphService(manifest_path=manifest_path)
    result = service.get_entry("compiled").build_graph().invoke({"value": "compiled"})
    assert result["value"] == "compiled"


def test_manifest_supports_config_style_graph_factories(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    package_dir = tmp_path / "external_graph_pkg_config"
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "graph_module.py").write_text(
        """
from langgraph.constants import CONF, CONFIG_KEY_CHECKPOINTER
from langgraph.graph import END, START, StateGraph


def make_graph(config):
    configurable = config.get(CONF, {})
    checkpointer = configurable.get(CONFIG_KEY_CHECKPOINTER)
    builder = StateGraph(dict)
    builder.add_node(
        "node",
        lambda state: {
            "used_checkpointer": checkpointer is not None,
            "configurable_present": bool(configurable),
        },
    )
    builder.add_edge(START, "node")
    builder.add_edge("node", END)
    return builder
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    manifest_path = tmp_path / "graphs.json"
    manifest_path.write_text(
        json.dumps({"graphs": {"config_factory": {"graph": "external_graph_pkg_config.graph_module:make_graph"}}}),
        encoding="utf-8",
    )

    service = LangGraphService(manifest_path=manifest_path)
    result = service.get_entry("config_factory").build_graph(InMemorySaver()).invoke(
        {},
        config={"configurable": {"thread_id": "t1", "checkpoint_ns": "r1"}},
    )
    assert result["used_checkpointer"] is True
    assert result["configurable_present"] is True
