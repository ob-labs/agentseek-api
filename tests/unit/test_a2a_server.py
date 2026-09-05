import asyncio
from datetime import UTC, datetime

import pytest
from langchain_core.messages import HumanMessage
from starlette.responses import StreamingResponse

from agentseek_api.models.api import AssistantConfigRead, AssistantRead
from agentseek_api.models.auth import User
from agentseek_api.services.langgraph_service import GraphEntry
from agentseek_api.a2a_server import (
    A2ATaskRecord,
    A2ATaskRegistry,
    _extract_output_text,
    _message_content,
    _sse_jsonrpc_event,
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
        config=AssistantConfigRead(),
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


def test_message_content_preserves_non_ascii_json() -> None:
    text = _message_content(HumanMessage(content=[{"name": "\u9644\u4ef64"}]))
    assert "\u9644\u4ef64" in text
    assert "\\u9644\\u4ef64" not in text


def test_extract_output_text_preserves_non_ascii_json() -> None:
    text = _extract_output_text({"name": "\u9644\u4ef64"})
    assert "\u9644\u4ef64" in text
    assert "\\u9644\\u4ef64" not in text


def test_sse_jsonrpc_event_preserves_non_ascii_json() -> None:
    event = _sse_jsonrpc_event(request_id="1", result={"artifact": {"text": "\u9644\u4ef64"}})
    assert "\u9644\u4ef64" in event
    assert "\\u9644\\u4ef64" not in event

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
    assert card["capabilities"] == {"streaming": True, "pushNotifications": False}
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




def test_a2a_task_registry_returns_saved_record() -> None:
    registry = A2ATaskRegistry()
    record = A2ATaskRecord(
        task_id="task-1",
        assistant_id="assistant-123",
        user_id="user-1",
        context_id="context-1",
        status_message="done",
        artifacts=[{"artifactId": "artifact-1"}],
    )

    registry.save(record)

    assert registry.get("task-1") is record


def test_a2a_task_registry_evicts_oldest_terminal_records_when_over_limit() -> None:
    registry = A2ATaskRegistry(max_tasks=2)
    registry.save(
        A2ATaskRecord(
            task_id="task-1",
            assistant_id="assistant-123",
            user_id="user-1",
            context_id="context-1",
            state="completed",
        )
    )
    registry.save(
        A2ATaskRecord(
            task_id="task-2",
            assistant_id="assistant-123",
            user_id="user-1",
            context_id="context-2",
            state="working",
        )
    )
    registry.save(
        A2ATaskRecord(
            task_id="task-3",
            assistant_id="assistant-123",
            user_id="user-1",
            context_id="context-3",
            state="completed",
        )
    )

    with pytest.raises(ValueError, match="Unknown task: task-1"):
        registry.get("task-1")
    assert registry.get("task-2").state == "working"
    assert registry.get("task-3").state == "completed"


def test_a2a_task_registry_evicts_oldest_record_when_active_tasks_exceed_limit() -> None:
    registry = A2ATaskRegistry(max_tasks=2)
    registry.save(
        A2ATaskRecord(
            task_id="task-1",
            assistant_id="assistant-123",
            user_id="user-1",
            context_id="context-1",
            state="working",
        )
    )
    registry.save(
        A2ATaskRecord(
            task_id="task-2",
            assistant_id="assistant-123",
            user_id="user-1",
            context_id="context-2",
            state="working",
        )
    )
    registry.save(
        A2ATaskRecord(
            task_id="task-3",
            assistant_id="assistant-123",
            user_id="user-1",
            context_id="context-3",
            state="working",
        )
    )

    with pytest.raises(ValueError, match="Unknown task: task-1"):
        registry.get("task-1")
    assert registry.get("task-2").state == "working"
    assert registry.get("task-3").state == "working"


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
async def test_handle_a2a_request_message_stream_prefers_astream_and_emits_update_events(monkeypatch) -> None:
    registry = A2ATaskRegistry()
    assistant = _assistant(description="A2A assistant")
    entry = _entry(
        input_schema={
            "type": "object",
            "properties": {"messages": {"type": "array"}},
            "required": ["messages"],
        }
    )

    async def fake_load_assistant(_assistant_id: str) -> AssistantRead:
        return assistant

    seen: dict[str, int] = {"ainvoke_calls": 0}

    class _Graph:
        async def ainvoke(self, prepared, config):
            seen["ainvoke_calls"] += 1
            return {"final_text": '{"echo":"ainvoke fallback"}'}

        async def astream(self, prepared, config):
            yield {"messages": [HumanMessage(content="chunk one")]}
            yield {"messages": [HumanMessage(content="streamed final")]}

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
            "id": "stream-1",
            "method": "message/stream",
            "params": {
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": "hello from a2a stream"}],
                    "messageId": "msg-stream-1",
                    "taskId": "stream-task",
                }
            },
        },
        user=User(identity="unit-user", is_authenticated=True),
        service=_Service(),
        registry=registry,
    )

    assert isinstance(response, StreamingResponse)
    chunks = [chunk async for chunk in response.body_iterator]
    body = b"".join(chunk if isinstance(chunk, bytes) else chunk.encode("utf-8") for chunk in chunks).decode("utf-8")

    assert 'event: message' in body
    assert '"kind": "status-update"' in body
    assert '"kind": "artifact-update"' in body
    assert '"append": true' in body
    assert '"lastChunk": true' in body
    assert '"final": true' in body
    assert 'streamed final' in body
    assert seen["ainvoke_calls"] == 0
    saved = registry.get("stream-task")
    assert saved.state == "completed"
    assert saved.artifacts[0]["parts"][0]["text"] == "streamed final"


@pytest.mark.asyncio
async def test_handle_a2a_request_message_stream_stops_after_cancellation(monkeypatch) -> None:
    registry = A2ATaskRegistry()
    assistant = _assistant(description="A2A assistant")
    entry = _entry(
        input_schema={
            "type": "object",
            "properties": {"messages": {"type": "array"}},
            "required": ["messages"],
        }
    )

    async def fake_load_assistant(_assistant_id: str) -> AssistantRead:
        return assistant

    first_chunk_processed = asyncio.Event()
    continue_stream = asyncio.Event()

    class _Graph:
        async def astream(self, prepared, config):
            yield {"messages": [HumanMessage(content="chunk one")]}
            first_chunk_processed.set()
            await continue_stream.wait()

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
            "id": "stream-cancel-1",
            "method": "message/stream",
            "params": {
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": "cancel this stream"}],
                    "messageId": "msg-stream-cancel-1",
                    "taskId": "stream-cancel-task",
                }
            },
        },
        user=User(identity="unit-user", is_authenticated=True),
        service=_Service(),
        registry=registry,
    )

    assert isinstance(response, StreamingResponse)
    iterator = response.body_iterator.__aiter__()
    first_event = await anext(iterator)
    first_text = first_event if isinstance(first_event, str) else first_event.decode("utf-8")
    assert '"kind": "status-update"' in first_text
    assert '"state": "working"' in first_text

    next_event_task = asyncio.create_task(anext(iterator))
    await first_chunk_processed.wait()

    record = registry.get("stream-cancel-task")
    record.cancellation_requested = True
    record.state = "cancelled"
    record.status_message = "Task cancelled"
    registry.save(record)
    second_event = await asyncio.wait_for(next_event_task, timeout=0.2)
    second_text = second_event if isinstance(second_event, str) else second_event.decode("utf-8")
    assert '"artifact-update"' in second_text
    assert '"lastChunk": true' in second_text
    assert "chunk one" in second_text

    third_event = await asyncio.wait_for(anext(iterator), timeout=0.2)
    third_text = third_event if isinstance(third_event, str) else third_event.decode("utf-8")
    assert '"state": "cancelled"' in third_text
    assert '"kind": "status-update"' in third_text

    with pytest.raises(StopAsyncIteration):
        await anext(iterator)

    saved = registry.get("stream-cancel-task")
    assert saved.state == "cancelled"
    assert saved.artifacts[0]["parts"][0]["text"] == "chunk one"


@pytest.mark.asyncio
async def test_handle_a2a_request_preserves_message_context_and_task_ids(monkeypatch) -> None:
    registry = A2ATaskRegistry()
    assistant = _assistant()
    entry = _entry(
        input_schema={
            "type": "object",
            "properties": {"messages": {"type": "array"}},
            "required": ["messages"],
        }
    )

    async def fake_load_assistant(_assistant_id: str) -> AssistantRead:
        return assistant

    class _Graph:
        async def ainvoke(self, prepared, config):
            return {"final_text": '{"echo":"continued"}'}

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
            "id": "preserve-1",
            "method": "message/send",
            "params": {
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": "continued"}],
                    "messageId": "msg-continued",
                    "contextId": "context-from-message",
                    "taskId": "task-from-message",
                }
            },
        },
        user=User(identity="unit-user", is_authenticated=True),
        service=_Service(),
        registry=registry,
    )

    assert response["result"]["id"] == "task-from-message"
    assert response["result"]["contextId"] == "context-from-message"
    saved = registry.get("task-from-message")
    assert saved.context_id == "context-from-message"
    assert saved.user_id == "unit-user"


@pytest.mark.asyncio
async def test_handle_a2a_request_rejects_cross_user_task_id_reuse(monkeypatch) -> None:
    registry = A2ATaskRegistry()
    assistant = _assistant()
    entry = _entry(
        input_schema={
            "type": "object",
            "properties": {"messages": {"type": "array"}},
            "required": ["messages"],
        }
    )

    async def fake_load_assistant(_assistant_id: str) -> AssistantRead:
        return assistant

    class _Graph:
        async def ainvoke(self, prepared, config):
            return {"final_text": '{"echo":"hello"}'}

    entry.build_graph = lambda checkpointer=None, store=None: _Graph()
    entry.prepare_input = lambda payload: payload
    entry.extract_output = lambda result, payload: result

    class _Service:
        def get_entry(self, graph_id: str) -> GraphEntry:
            assert graph_id == "stress_test"
            return entry

    registry.save(
        A2ATaskRecord(
            task_id="shared-task",
            assistant_id=assistant.assistant_id,
            user_id="owner-user",
            context_id="context-1",
            state="completed",
        )
    )
    monkeypatch.setattr("agentseek_api.a2a_server.load_assistant", fake_load_assistant)
    monkeypatch.setattr("agentseek_api.a2a_server.build_graph_config", lambda *, user, context_id: (object(), {"thread_id": context_id}))

    response = await handle_a2a_request(
        assistant_id=assistant.assistant_id,
        payload={
            "jsonrpc": "2.0",
            "id": "reuse-1",
            "method": "message/send",
            "params": {
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": "hello"}],
                    "taskId": "shared-task",
                }
            },
        },
        user=User(identity="other-user", is_authenticated=True),
        service=_Service(),
        registry=registry,
    )

    assert response["error"]["message"] == "Unknown task: shared-task"


@pytest.mark.asyncio
async def test_handle_a2a_request_reuses_same_user_task_id_for_same_assistant(monkeypatch) -> None:
    registry = A2ATaskRegistry()
    assistant = _assistant()
    entry = _entry(
        input_schema={
            "type": "object",
            "properties": {"messages": {"type": "array"}},
            "required": ["messages"],
        }
    )

    async def fake_load_assistant(_assistant_id: str) -> AssistantRead:
        return assistant

    class _Graph:
        async def ainvoke(self, prepared, config):
            return {"final_text": '{"echo":"continued"}'}

    entry.build_graph = lambda checkpointer=None, store=None: _Graph()
    entry.prepare_input = lambda payload: payload
    entry.extract_output = lambda result, payload: result

    class _Service:
        def get_entry(self, graph_id: str) -> GraphEntry:
            assert graph_id == "stress_test"
            return entry

    registry.save(
        A2ATaskRecord(
            task_id="shared-task",
            assistant_id=assistant.assistant_id,
            user_id="same-user",
            context_id="context-1",
            state="completed",
            status_message="prior",
            artifacts=[{"artifactId": "artifact-1"}],
        )
    )
    monkeypatch.setattr("agentseek_api.a2a_server.load_assistant", fake_load_assistant)
    monkeypatch.setattr("agentseek_api.a2a_server.build_graph_config", lambda *, user, context_id: (object(), {"thread_id": context_id}))

    response = await handle_a2a_request(
        assistant_id=assistant.assistant_id,
        payload={
            "jsonrpc": "2.0",
            "id": "reuse-same-assistant",
            "method": "message/send",
            "params": {
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": "continued"}],
                    "taskId": "shared-task",
                }
            },
        },
        user=User(identity="same-user", is_authenticated=True),
        service=_Service(),
        registry=registry,
    )

    assert response["result"]["id"] == "shared-task"
    assert response["result"]["contextId"] == "context-1"
    saved = registry.get("shared-task")
    assert saved.assistant_id == assistant.assistant_id
    assert saved.context_id == "context-1"
    assert saved.status_message == ""
    assert saved.state == "completed"


@pytest.mark.asyncio
async def test_handle_a2a_request_rejects_cross_assistant_task_id_reuse(monkeypatch) -> None:
    registry = A2ATaskRegistry()
    assistant_a = _assistant(name="assistant-a")
    assistant_b = AssistantRead(
        assistant_id="assistant-456",
        name="assistant-b",
        graph_id="stress_test",
        created_at=assistant_a.created_at,
        updated_at=assistant_a.updated_at,
        metadata={"scope": "test"},
        config=AssistantConfigRead(),
        context={},
        version=1,
        description="assistant-b",
    )
    entry = _entry(
        input_schema={
            "type": "object",
            "properties": {"messages": {"type": "array"}},
            "required": ["messages"],
        }
    )

    async def fake_load_assistant(current_assistant_id: str) -> AssistantRead:
        if current_assistant_id == assistant_a.assistant_id:
            return assistant_a
        if current_assistant_id == assistant_b.assistant_id:
            return assistant_b
        raise AssertionError(f"Unexpected assistant id: {current_assistant_id}")

    class _Graph:
        async def ainvoke(self, prepared, config):
            return {"final_text": '{"echo":"hello"}'}

    entry.build_graph = lambda checkpointer=None, store=None: _Graph()
    entry.prepare_input = lambda payload: payload
    entry.extract_output = lambda result, payload: result

    class _Service:
        def get_entry(self, graph_id: str) -> GraphEntry:
            assert graph_id == "stress_test"
            return entry

    registry.save(
        A2ATaskRecord(
            task_id="shared-task",
            assistant_id=assistant_a.assistant_id,
            user_id="same-user",
            context_id="context-1",
            state="completed",
        )
    )
    monkeypatch.setattr("agentseek_api.a2a_server.load_assistant", fake_load_assistant)
    monkeypatch.setattr("agentseek_api.a2a_server.build_graph_config", lambda *, user, context_id: (object(), {"thread_id": context_id}))

    response = await handle_a2a_request(
        assistant_id=assistant_b.assistant_id,
        payload={
            "jsonrpc": "2.0",
            "id": "reuse-cross-assistant",
            "method": "message/send",
            "params": {
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": "hello"}],
                    "taskId": "shared-task",
                }
            },
        },
        user=User(identity="same-user", is_authenticated=True),
        service=_Service(),
        registry=registry,
    )

    assert response["error"]["message"] == "Unknown task: shared-task"
    saved = registry.get("shared-task")
    assert saved.assistant_id == assistant_a.assistant_id
    assert saved.context_id == "context-1"


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
        user_id="unit-user",
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


@pytest.mark.asyncio
async def test_handle_a2a_request_tasks_cancel_marks_active_task_cancelled(monkeypatch) -> None:
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

    registry.save(
        A2ATaskRecord(
            task_id="task-cancel",
            assistant_id=assistant.assistant_id,
            user_id="unit-user",
            context_id="context-1",
            state="working",
        )
    )
    monkeypatch.setattr("agentseek_api.a2a_server.load_assistant", fake_load_assistant)

    response = await handle_a2a_request(
        assistant_id=assistant.assistant_id,
        payload={
            "jsonrpc": "2.0",
            "id": "cancel-active",
            "method": "tasks/cancel",
            "params": {"id": "task-cancel"},
        },
        user=User(identity="unit-user", is_authenticated=True),
        service=_Service(),
        registry=registry,
    )

    assert response["result"]["id"] == "task-cancel"
    assert response["result"]["status"]["state"] == "cancelled"
    assert response["result"]["status"]["message"]["text"] == "Task cancelled"
    assert registry.get("task-cancel").state == "cancelled"


@pytest.mark.asyncio
async def test_handle_a2a_request_tasks_cancel_returns_terminal_task_unchanged(monkeypatch) -> None:
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

    registry.save(
        A2ATaskRecord(
            task_id="task-done",
            assistant_id=assistant.assistant_id,
            user_id="unit-user",
            context_id="context-1",
            state="completed",
            artifacts=[{"artifactId": "artifact-1", "parts": [{"kind": "text", "text": "done"}]}],
        )
    )
    monkeypatch.setattr("agentseek_api.a2a_server.load_assistant", fake_load_assistant)

    response = await handle_a2a_request(
        assistant_id=assistant.assistant_id,
        payload={
            "jsonrpc": "2.0",
            "id": "cancel-terminal",
            "method": "tasks/cancel",
            "params": {"id": "task-done"},
        },
        user=User(identity="unit-user", is_authenticated=True),
        service=_Service(),
        registry=registry,
    )

    assert response["result"]["id"] == "task-done"
    assert response["result"]["status"]["state"] == "completed"
    assert response["result"]["artifacts"][0]["parts"][0]["text"] == "done"


@pytest.mark.asyncio
async def test_handle_a2a_request_tasks_get_rejects_cross_user_access(monkeypatch) -> None:
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

    registry.save(
        A2ATaskRecord(
            task_id="task-owned-by-a",
            assistant_id=assistant.assistant_id,
            user_id="user-a",
            context_id="context-1",
            state="completed",
        )
    )
    monkeypatch.setattr("agentseek_api.a2a_server.load_assistant", fake_load_assistant)

    response = await handle_a2a_request(
        assistant_id=assistant.assistant_id,
        payload={
            "jsonrpc": "2.0",
            "id": "cross-user-get",
            "method": "tasks/get",
            "params": {"id": "task-owned-by-a"},
        },
        user=User(identity="user-b", is_authenticated=True),
        service=_Service(),
        registry=registry,
    )

    assert response["error"]["message"] == "Unknown task: task-owned-by-a"
