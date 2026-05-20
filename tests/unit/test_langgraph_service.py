from agentseek_api.services import langgraph_service as langgraph_module
from agentseek_api.services.langgraph_service import LangGraphService, _echo_node, get_langgraph_service


def test_echo_node_returns_echo_payload() -> None:
    state = {"input": {"message": "hi"}, "output": {}}
    result = _echo_node(state)
    assert result["output"]["echo"] == {"message": "hi"}


def test_default_graph_entry_invokes_echo() -> None:
    service = LangGraphService()
    entry = service.get_entry("default")
    prepared = entry.prepare_input({"message": "hello"})
    result = entry.build_graph().invoke(prepared)
    assert entry.extract_output(result, {"message": "hello"})["echo"] == {"message": "hello"}


def test_get_entry_falls_back_to_default_for_unknown_graph_id() -> None:
    service = LangGraphService()
    unknown = service.get_entry("does-not-exist")
    default = service.get_entry("default")
    assert unknown is default


def test_sample_graphs_are_registered() -> None:
    service = LangGraphService()
    ids = service.registered_graph_ids()
    for expected in (
        "default",
        "store_memory",
        "stress_test",
        "subgraph_agent",
        "react_agent",
        "stress_tool_agent",
        "subgraph_hitl_agent",
    ):
        assert expected in ids, f"missing graph_id: {expected}; have: {ids}"


def test_get_graph_builds_from_factory() -> None:
    service = LangGraphService()
    graph = service.get_graph("default")
    result = graph.invoke({"input": {"message": "factory"}})
    assert result["output"]["echo"] == {"message": "factory"}


def test_get_graph_passes_store_into_compilation() -> None:
    service = LangGraphService()
    graph = service.get_graph("default", store=object())
    result = graph.invoke({"input": {"message": "factory-store"}})
    assert result["output"]["echo"] == {"message": "factory-store"}


def test_get_langgraph_service_is_singleton() -> None:
    langgraph_module._langgraph_service = None
    first = get_langgraph_service()
    second = get_langgraph_service()
    assert first is second
