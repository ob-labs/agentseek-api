from datetime import UTC, datetime

from agentseek_api.models.api import AssistantRead
from agentseek_api.services.langgraph_service import GraphEntry
from agentseek_api.a2a_server import build_agent_card, is_a2a_compatible_entry


def _entry(
    *,
    tool_name: str = "stress_test",
    description: str = "",
    input_schema: dict[str, object] | None = None,
) -> GraphEntry:
    return GraphEntry(
        graph_factory=lambda: None,
        prepare_input=lambda payload: payload,
        extract_output=lambda result, payload: {"result": result, "payload": payload},
        tool_name=tool_name,
        description=description,
        input_schema=input_schema or {"type": "object"},
        output_schema={"type": "object"},
    )


def _assistant(*, name: str = "assistant-name", description: str | None = "assistant-description") -> AssistantRead:
    now = datetime.now(UTC)
    return AssistantRead(
        assistant_id="assistant-123",
        name=name,
        graph_id="stress_test",
        created_at=now,
        updated_at=now,
        metadata={"scope": "test"},
        config={},
        context={},
        version=1,
        description=description,
    )


def test_is_a2a_compatible_entry_accepts_messages_array_schema() -> None:
    entry = _entry(
        input_schema={
            "type": "object",
            "properties": {"messages": {"type": "array"}},
            "required": ["messages"],
        }
    )

    assert is_a2a_compatible_entry(entry) is True


def test_is_a2a_compatible_entry_rejects_non_message_schema() -> None:
    entry = _entry(
        input_schema={
            "type": "object",
            "properties": {"question": {"type": "string"}},
            "required": ["question"],
        }
    )

    assert is_a2a_compatible_entry(entry) is False


def test_is_a2a_compatible_entry_rejects_non_object_root_schema() -> None:
    entry = _entry(
        input_schema={
            "type": "array",
            "properties": {"messages": {"type": "array"}},
            "required": ["messages"],
        }
    )

    assert is_a2a_compatible_entry(entry) is False


def test_is_a2a_compatible_entry_rejects_message_prepare_input_without_explicit_schema() -> None:
    entry = _entry(
        input_schema={"type": "object"},
    )
    entry.prepare_input = lambda payload: {"messages": [{"role": "user", "content": payload["message"]}]}

    assert is_a2a_compatible_entry(entry) is False


def test_build_agent_card_prefers_assistant_metadata_over_graph_metadata() -> None:
    assistant = _assistant(name="Assistant Preferred", description="Assistant description")
    entry = _entry(
        tool_name="graph-tool",
        description="Graph description",
        input_schema={
            "type": "object",
            "properties": {"messages": {"type": "array"}},
            "required": ["messages"],
        },
    )

    card = build_agent_card(base_url="https://example.com", assistant=assistant, entry=entry)

    assert card["name"] == "Assistant Preferred"
    assert card["description"] == "Assistant description"
    assert card["version"]
    assert card["capabilities"] == {"streaming": False, "pushNotifications": False}
    assert card["defaultInputModes"] == ["text/plain"]
    assert card["defaultOutputModes"] == ["text/plain"]
    assert card["supportedInterfaces"] == [
        {
            "url": "https://example.com/a2a/assistant-123",
            "protocolBinding": "JSONRPC",
            "protocolVersion": "1.0",
        }
    ]
    assert "url" not in card
    assert "preferredTransport" not in card
    assert card["skills"][0]["id"] == "graph-tool"
