import json

from langchain_core.messages import AIMessage, HumanMessage

from agentseek_api.services.sample_graphs import (
    _ensure_messages_payload,
    _extract_interrupt_output,
    _extract_messages_output,
    _message_content,
    _prepare_subgraph_hitl_payload,
    build_sample_registry,
)


def test_ensure_messages_payload_passes_through_existing_messages() -> None:
    payload = {"messages": [HumanMessage(content="hi")]}
    prepared = _ensure_messages_payload(payload)
    assert prepared is payload


def test_ensure_messages_payload_wraps_message_field() -> None:
    prepared = _ensure_messages_payload({"message": "hello"})
    assert isinstance(prepared["messages"][0], HumanMessage)
    assert prepared["messages"][0].content == "hello"


def test_ensure_messages_payload_wraps_content_field() -> None:
    prepared = _ensure_messages_payload({"content": "hey"})
    assert prepared["messages"][0].content == "hey"


def test_ensure_messages_payload_json_encodes_other_payloads() -> None:
    prepared = _ensure_messages_payload({"delay": 0.01, "steps": 2})
    content = prepared["messages"][0].content
    assert json.loads(content) == {"delay": 0.01, "steps": 2}


def test_extract_messages_output_parses_final_json() -> None:
    final = AIMessage(content=json.dumps({"status": "completed", "steps_completed": 3}))
    result = {"messages": [HumanMessage(content="go"), final]}
    output = _extract_messages_output(result, {})
    assert output["final_text"] == final.content
    assert output["final_json"]["status"] == "completed"
    assert output["transcript"][0]["type"] == "HumanMessage"
    assert output["transcript"][1]["type"] == "AIMessage"


def test_extract_messages_output_skips_final_json_when_not_parseable() -> None:
    final = AIMessage(content="plain answer")
    result = {"messages": [final]}
    output = _extract_messages_output(result, {})
    assert output["final_text"] == "plain answer"
    assert "final_json" not in output


def test_extract_messages_output_handles_non_dict_result() -> None:
    assert _extract_messages_output("unexpected", {}) == {
        "final_text": "",
        "transcript": [],
    }


def test_extract_interrupt_output_reports_interrupts() -> None:
    class FakeInterrupt:
        value = "Provide value:"
        id = "i1"

    output = _extract_interrupt_output({"foo": "bar", "__interrupt__": [FakeInterrupt()]}, {})
    assert output["interrupted"] is True
    assert output["interrupts"][0]["value"] == "Provide value:"
    assert output["state"] == {"foo": "bar"}


def test_extract_interrupt_output_without_interrupt_is_not_interrupted() -> None:
    output = _extract_interrupt_output({"foo": "done"}, {})
    assert output["interrupted"] is False
    assert output["state"] == {"foo": "done"}


def test_extract_interrupt_output_falls_back_for_non_dict() -> None:
    assert _extract_interrupt_output(["unexpected"], {}) == {"result": ["unexpected"]}


def test_prepare_subgraph_hitl_payload_prefers_explicit_foo() -> None:
    assert _prepare_subgraph_hitl_payload({"foo": "value"}) == {"foo": "value"}


def test_prepare_subgraph_hitl_payload_falls_back_to_message_or_content() -> None:
    assert _prepare_subgraph_hitl_payload({"message": "m"}) == {"foo": "m"}
    assert _prepare_subgraph_hitl_payload({"content": "c"}) == {"foo": "c"}
    assert _prepare_subgraph_hitl_payload({}) == {"foo": ""}


def test_message_content_stringifies_non_string_content() -> None:
    assert _message_content(AIMessage(content="x")) == "x"

    class Obj:
        content = ["list", "content"]

    assert _message_content(Obj()) == "['list', 'content']"


def test_build_sample_registry_returns_all_expected_ids() -> None:
    registry = build_sample_registry()
    for expected in (
        "store_memory",
        "stress_test",
        "subgraph_agent",
        "react_agent",
        "stress_tool_agent",
        "subgraph_hitl_agent",
    ):
        assert expected in registry, f"missing {expected}: have {list(registry)}"
        entry = registry[expected]
        assert callable(entry["prepare_input"])
        assert callable(entry["extract_output"])
        assert callable(entry["graph_factory"])
        assert entry["graph_factory"]() is not None
