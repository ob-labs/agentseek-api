from datetime import UTC, datetime

import pytest
from langchain_core.messages import HumanMessage

from agentseek_api.models.api import AssistantRead
from agentseek_api.models.auth import User
from agentseek_api.services.langgraph_service import GraphEntry
from agentseek_api.a2a_server import (
    A2ATaskRecord,
    A2ATaskRegistry,
    build_agent_card,
    handle_a2a_request,
    is_a2a_compatible_entry,
)


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


def test_build_agent_card_keeps_top_level_description_strictly_assistant_first() -> None:
    assistant = _assistant(name="Assistant Preferred", description="")
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
    assert card["description"] == ""
    assert card["skills"][0]["description"] == "Graph description"


def test_build_agent_card_includes_config_auth_metadata(monkeypatch) -> None:
    assistant = _assistant()
    entry = _entry(
        input_schema={
            "type": "object",
            "properties": {"messages": {"type": "array"}},
            "required": ["messages"],
        }
    )
    monkeypatch.setattr(
        "agentseek_api.a2a_server.get_config_auth_openapi",
        lambda: {
            "securitySchemes": {
                "apiKeyAuth": {
                    "type": "apiKey",
                    "in": "header",
                    "name": "X-API-Key",
                }
            },
            "security": [{"apiKeyAuth": []}],
        },
    )

    card = build_agent_card(base_url="https://example.com", assistant=assistant, entry=entry)

    assert card["securitySchemes"] == {
        "apiKeyAuth": {
            "apiKeySecurityScheme": {
                "location": "header",
                "name": "X-API-Key",
            }
        }
    }
    assert card["securityRequirements"] == [{"apiKeyAuth": []}]


def test_build_agent_card_filters_unsupported_config_auth_metadata(monkeypatch) -> None:
    assistant = _assistant()
    entry = _entry(
        input_schema={
            "type": "object",
            "properties": {"messages": {"type": "array"}},
            "required": ["messages"],
        }
    )
    monkeypatch.setattr(
        "agentseek_api.a2a_server.get_config_auth_openapi",
        lambda: {
            "securitySchemes": {
                "apiKeyAuth": {
                    "type": "apiKey",
                    "in": "header",
                    "name": "X-API-Key",
                },
                "oauthAuth": {
                    "type": "oauth2",
                },
            },
            "security": [
                {"apiKeyAuth": []},
                {"oauthAuth": []},
                {"apiKeyAuth": [], "oauthAuth": []},
            ],
        },
    )

    card = build_agent_card(base_url="https://example.com", assistant=assistant, entry=entry)

    assert card["securitySchemes"] == {
        "apiKeyAuth": {
            "apiKeySecurityScheme": {
                "location": "header",
                "name": "X-API-Key",
            }
        }
    }
    assert card["securityRequirements"] == [{"apiKeyAuth": []}]


def test_build_agent_card_drops_mixed_supported_and_unsupported_requirement_objects(monkeypatch) -> None:
    assistant = _assistant()
    entry = _entry(
        input_schema={
            "type": "object",
            "properties": {"messages": {"type": "array"}},
            "required": ["messages"],
        }
    )
    monkeypatch.setattr(
        "agentseek_api.a2a_server.get_config_auth_openapi",
        lambda: {
            "securitySchemes": {
                "apiKeyAuth": {
                    "type": "apiKey",
                    "in": "header",
                    "name": "X-API-Key",
                },
                "oauthAuth": {
                    "type": "oauth2",
                },
            },
            "security": [
                {"apiKeyAuth": [], "oauthAuth": []},
                {"apiKeyAuth": []},
            ],
        },
    )

    card = build_agent_card(base_url="https://example.com", assistant=assistant, entry=entry)

    assert card["securityRequirements"] == [{"apiKeyAuth": []}]


def test_build_agent_card_includes_builtin_api_key_auth_metadata(monkeypatch) -> None:
    assistant = _assistant()
    entry = _entry(
        input_schema={
            "type": "object",
            "properties": {"messages": {"type": "array"}},
            "required": ["messages"],
        }
    )
    monkeypatch.setattr("agentseek_api.a2a_server.get_config_auth_openapi", lambda: None)
    monkeypatch.setattr("agentseek_api.a2a_server.settings.AUTH_TYPE", "api_key")

    card = build_agent_card(base_url="https://example.com", assistant=assistant, entry=entry)

    assert card["securitySchemes"] == {
        "apiKeyAuth": {
            "apiKeySecurityScheme": {
                "location": "header",
                "name": "x-api-key",
            }
        }
    }
    assert card["securityRequirements"] == [{"apiKeyAuth": []}]


def test_a2a_task_registry_returns_saved_record() -> None:
    registry = A2ATaskRegistry()
    record = A2ATaskRecord(
        task_id="task-1",
        assistant_id="assistant-123",
        context_id="context-1",
        status_message="done",
        artifacts=[{"artifactId": "artifact-1"}],
    )

    registry.save(record)

    assert registry.get("task-1") is record


@pytest.mark.asyncio
async def test_handle_a2a_request_message_send_returns_completed_task(monkeypatch) -> None:
    registry = A2ATaskRegistry()
    assistant = _assistant(description="A2A assistant")
    entry = _entry(
        input_schema={
            "type": "object",
            "properties": {"messages": {"type": "array"}},
            "required": ["messages"],
        }
    )
    captured: dict[str, object] = {}

    async def fake_load_assistant(_assistant_id: str) -> AssistantRead:
        return assistant

    class _Graph:
        async def ainvoke(self, prepared, config):
            captured["prepared"] = prepared
            captured["config"] = config
            return {"final_text": '{"echo":"hello from a2a"}'}

    entry.build_graph = lambda checkpointer=None, store=None: _Graph()
    entry.prepare_input = lambda payload: payload
    entry.extract_output = lambda result, payload: result

    class _Service:
        def get_entry(self, graph_id: str) -> GraphEntry:
            assert graph_id == "stress_test"
            return entry

    monkeypatch.setattr("agentseek_api.a2a_server.load_assistant", fake_load_assistant)
    monkeypatch.setattr("agentseek_api.a2a_server.build_graph_config", lambda *, user, context_id: (object(), {"thread_id": context_id}))

    response = await handle_a2a_request(
        assistant_id=assistant.assistant_id,
        payload={
            "jsonrpc": "2.0",
            "id": "1",
            "method": "message/send",
            "params": {
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": "hello from a2a"}],
                    "messageId": "msg-1",
                }
            },
        },
        user=User(identity="unit-user", is_authenticated=True),
        service=_Service(),
        registry=registry,
    )

    assert response["result"]["kind"] == "task"
    assert response["result"]["status"]["state"] == "completed"
    assert response["result"]["artifacts"][0]["parts"][0]["text"] == '{"echo":"hello from a2a"}'
    assert response["result"]["id"]
    assert response["result"]["contextId"]
    prepared = captured["prepared"]
    assert isinstance(prepared, dict)
    assert len(prepared["messages"]) == 1
    assert isinstance(prepared["messages"][0], HumanMessage)
    assert prepared["messages"][0].content == "hello from a2a"
    assert registry.get(response["result"]["id"]).artifacts == response["result"]["artifacts"]


@pytest.mark.asyncio
async def test_handle_a2a_request_tasks_get_returns_saved_snapshot(monkeypatch) -> None:
    registry = A2ATaskRegistry()
    assistant = _assistant()

    async def fake_load_assistant(_assistant_id: str) -> AssistantRead:
        return assistant

    class _Service:
        def get_entry(self, graph_id: str) -> GraphEntry:
            assert graph_id == "stress_test"
            return _entry(
                input_schema={
                    "type": "object",
                    "properties": {"messages": {"type": "array"}},
                    "required": ["messages"],
                }
            )

    record = A2ATaskRecord(
        task_id="task-lookup",
        assistant_id=assistant.assistant_id,
        context_id="context-1",
        state="completed",
        artifacts=[{"artifactId": "artifact-1", "parts": [{"kind": "text", "text": "lookup me"}]}],
    )
    registry.save(record)
    monkeypatch.setattr("agentseek_api.a2a_server.load_assistant", fake_load_assistant)

    response = await handle_a2a_request(
        assistant_id=assistant.assistant_id,
        payload={
            "jsonrpc": "2.0",
            "id": "2",
            "method": "tasks/get",
            "params": {"id": "task-lookup"},
        },
        user=User(identity="unit-user", is_authenticated=True),
        service=_Service(),
        registry=registry,
    )

    assert response["result"]["id"] == "task-lookup"
    assert response["result"]["status"]["state"] == "completed"
    assert response["result"]["artifacts"][0]["parts"][0]["text"] == "lookup me"
