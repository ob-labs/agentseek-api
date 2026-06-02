import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langgraph.constants import CONF, CONFIG_KEY_CHECKPOINTER

from agentseek_api.core.runtime_store import UserScopedStore
from agentseek_api.services.run_executor import (
    RunExecutionResult,
    _ProtocolMessageStreamState,
    _translate_stream_events,
    execute_run,
)
from agentseek_api.services.thread_protocol import ThreadProtocolEventBroker


class FakeGraph:
    def __init__(self) -> None:
        self.configs: list[dict] = []

    async def astream_events(self, prepared_input: dict, config: dict, version: str = "v2"):
        self.configs.append(config)
        yield {
            "event": "on_chain_end",
            "name": "fake-graph",
            "run_id": "langgraph-run",
            "parent_ids": [],
            "metadata": {},
            "tags": [],
            "data": {"output": {"output": {"ok": True, "received": prepared_input}}},
        }


class FakeEntry:
    graph = FakeGraph()
    last_store = None

    @staticmethod
    def build_graph(_checkpointer=None, store=None) -> FakeGraph:
        FakeEntry.last_store = store
        return FakeEntry.graph

    @staticmethod
    def prepare_input(payload: dict) -> dict:
        return {"input": payload}

    @staticmethod
    def extract_output(result: dict, _payload: dict) -> dict:
        return result.get("output", {})


class FakeLangGraphService:
    def get_entry(self, _graph_id: str | None) -> FakeEntry:
        return FakeEntry()

    def get_graph(self, _graph_id: str | None = None) -> FakeGraph:
        return FakeGraph()


class FakeCheckpointer:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def save_checkpoint(self, *, thread_id: str, run_id: str, payload: dict) -> None:
        self.calls.append({"thread_id": thread_id, "run_id": run_id, "payload": payload})


class FakeDBManager:
    def __init__(self) -> None:
        self.checkpointer = FakeCheckpointer()
        self.langgraph_checkpointer = object()
        self.store = object()

    async def run_checkpointer_call(self, func, *args, **kwargs):
        return func(*args, **kwargs)

    def get_checkpointer(self) -> FakeCheckpointer:
        return self.checkpointer

    def get_langgraph_checkpointer(self):
        return self.langgraph_checkpointer

    def get_store(self):
        return self.store


@pytest.mark.asyncio
async def test_execute_run_saves_checkpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_db = FakeDBManager()
    monkeypatch.setattr("agentseek_api.services.run_executor.get_langgraph_service", lambda: FakeLangGraphService())
    monkeypatch.setattr("agentseek_api.services.run_executor.db_manager", fake_db)

    result = await execute_run(thread_id="t1", run_id="r1", payload={"hello": "world"}, user_id="user-1")
    assert isinstance(result, RunExecutionResult)
    assert result.output["ok"] is True
    assert result.output["received"] == {"input": {"hello": "world"}}
    assert result.interrupted is False
    assert len(fake_db.checkpointer.calls) == 1
    assert fake_db.checkpointer.calls[0]["thread_id"] == "t1"
    assert fake_db.checkpointer.calls[0]["payload"]["graph_id"] == "default"


@pytest.mark.asyncio
async def test_execute_run_records_graph_id(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_db = FakeDBManager()
    monkeypatch.setattr("agentseek_api.services.run_executor.get_langgraph_service", lambda: FakeLangGraphService())
    monkeypatch.setattr("agentseek_api.services.run_executor.db_manager", fake_db)

    await execute_run(thread_id="t1", run_id="r1", payload={"a": 1}, graph_id="stress_test", user_id="user-1")
    assert fake_db.checkpointer.calls[0]["payload"]["graph_id"] == "stress_test"


@pytest.mark.asyncio
async def test_execute_run_passes_runtime_checkpointer_in_config(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_db = FakeDBManager()
    FakeEntry.graph = FakeGraph()
    FakeEntry.last_store = None
    monkeypatch.setattr("agentseek_api.services.run_executor.get_langgraph_service", lambda: FakeLangGraphService())
    monkeypatch.setattr("agentseek_api.services.run_executor.db_manager", fake_db)

    await execute_run(thread_id="t1", run_id="r1", payload={"a": 1}, user_id="scoped-user")

    config = FakeEntry.graph.configs[0]
    assert config[CONF]["thread_id"] == "t1"
    assert config[CONF]["checkpoint_ns"] == "r1"
    assert config[CONF][CONFIG_KEY_CHECKPOINTER] is fake_db.langgraph_checkpointer
    assert isinstance(config[CONF]["store"], UserScopedStore)
    assert config[CONF]["store"]._store is fake_db.store
    assert config[CONF]["store"]._user_prefix == ("__agentseek_users__", "scoped-user")
    assert isinstance(FakeEntry.last_store, UserScopedStore)
    assert FakeEntry.last_store._store is fake_db.store


@pytest.mark.asyncio
async def test_execute_run_merges_user_config_and_context_into_graph_config(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_db = FakeDBManager()
    FakeEntry.graph = FakeGraph()
    monkeypatch.setattr("agentseek_api.services.run_executor.get_langgraph_service", lambda: FakeLangGraphService())
    monkeypatch.setattr("agentseek_api.services.run_executor.db_manager", fake_db)

    await execute_run(
        thread_id="t1",
        run_id="r1",
        payload={"a": 1},
        user_id="scoped-user",
        kwargs={"config": {"recursion_limit": 7}, "context": {"tenant": "acme"}},
    )

    config = FakeEntry.graph.configs[0]
    assert config["recursion_limit"] == 7
    assert config[CONF]["context"] == {"tenant": "acme"}
    assert config[CONF]["thread_id"] == "t1"


class FakeInterruptGraph(FakeGraph):
    async def astream_events(self, prepared_input: dict, config: dict, version: str = "v2"):
        self.configs.append(config)
        yield {
            "event": "on_chain_stream",
            "name": "fake-graph",
            "run_id": "langgraph-run",
            "parent_ids": [],
            "metadata": {},
            "tags": [],
            "data": {
                "chunk": {
                    "__interrupt__": [
                        type("Interrupt", (), {"value": "Provide value:", "id": "interrupt-1"})(),
                    ]
                }
            },
        }
        yield {
            "event": "on_chain_end",
            "name": "fake-graph",
            "run_id": "langgraph-run",
            "parent_ids": [],
            "metadata": {},
            "tags": [],
            "data": {"output": {"foo": prepared_input["input"]["foo"]}},
        }


class FakeInterruptEntry(FakeEntry):
    graph = FakeInterruptGraph()

    @staticmethod
    def build_graph(_checkpointer=None) -> FakeInterruptGraph:
        return FakeInterruptEntry.graph

    @staticmethod
    def extract_output(result: dict, _payload: dict) -> dict:
        interrupts = result.get("__interrupt__", [])
        return {
            "state": {"foo": result.get("foo")},
            "interrupted": bool(interrupts),
            "interrupts": [{"value": item.value, "id": item.id} for item in interrupts],
        }


class FakeInterruptLangGraphService(FakeLangGraphService):
    def get_entry(self, _graph_id: str | None) -> FakeInterruptEntry:
        return FakeInterruptEntry()


@pytest.mark.asyncio
async def test_execute_run_preserves_interrupts_from_root_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_db = FakeDBManager()
    FakeInterruptEntry.graph = FakeInterruptGraph()
    monkeypatch.setattr(
        "agentseek_api.services.run_executor.get_langgraph_service",
        lambda: FakeInterruptLangGraphService(),
    )
    monkeypatch.setattr("agentseek_api.services.run_executor.db_manager", fake_db)

    result = await execute_run(thread_id="t1", run_id="r1", payload={"foo": "hello"}, user_id="user-1")

    assert result.interrupted is True
    assert result.interrupts == [{"value": "Provide value:", "id": "interrupt-1"}]
    assert result.output["state"]["foo"] == "hello"


def test_translate_stream_events_maps_chat_model_stream_to_message_chunk() -> None:
    translated = _translate_stream_events(
        {
            "event": "on_chat_model_stream",
            "name": "chat-model",
            "run_id": "langgraph-run",
            "parent_ids": ["parent-run"],
            "metadata": {"langgraph_node": "call_model"},
            "tags": ["graph:step:1"],
            "data": {"chunk": AIMessageChunk(content="hello")},
        }
    )

    assert translated == [
        (
            "message_chunk",
            {
                "name": "chat-model",
                "langgraph_event": "on_chat_model_stream",
                "langgraph_run_id": "langgraph-run",
                "metadata": {"langgraph_node": "call_model"},
                "tags": ["graph:step:1"],
                "parent_ids": ["parent-run"],
                "node": "call_model",
                "message_type": "AIMessageChunk",
                "content": "hello",
            },
        )
    ]


class FakeProtocolStreamingGraph(FakeGraph):
    async def astream_events(self, prepared_input: dict, config: dict, version: str = "v2"):
        self.configs.append(config)
        yield {
            "event": "on_chat_model_stream",
            "name": "chat-model",
            "run_id": "chat-run",
            "parent_ids": ["root-run"],
            "metadata": {"langgraph_node": "call_model"},
            "tags": [],
            "data": {"chunk": AIMessageChunk(content="hel")},
        }
        yield {
            "event": "on_chat_model_stream",
            "name": "chat-model",
            "run_id": "chat-run",
            "parent_ids": ["root-run"],
            "metadata": {"langgraph_node": "call_model"},
            "tags": [],
            "data": {"chunk": AIMessageChunk(content="lo")},
        }
        yield {
            "event": "on_chain_stream",
            "name": "fake-graph",
            "run_id": "root-run",
            "parent_ids": [],
            "metadata": {},
            "tags": [],
            "data": {"chunk": {"step": "partial"}},
        }
        yield {
            "event": "on_chain_end",
            "name": "fake-graph",
            "run_id": "root-run",
            "parent_ids": [],
            "metadata": {},
            "tags": [],
            "data": {"output": {"output": {"messages": [AIMessage(content="hello")], "step": "final"}}},
        }


class FakeProtocolLlmStreamingGraph(FakeGraph):
    async def astream_events(self, prepared_input: dict, config: dict, version: str = "v2"):
        self.configs.append(config)
        yield {
            "event": "on_llm_stream",
            "name": "completion-model",
            "run_id": "llm-run",
            "parent_ids": ["root-run"],
            "metadata": {"langgraph_node": "call_model"},
            "tags": [],
            "data": {"chunk": type("Chunk", (), {"text": "hel"})()},
        }
        yield {
            "event": "on_llm_stream",
            "name": "completion-model",
            "run_id": "llm-run",
            "parent_ids": ["root-run"],
            "metadata": {"langgraph_node": "call_model"},
            "tags": [],
            "data": {"chunk": type("Chunk", (), {"text": "lo"})()},
        }
        yield {
            "event": "on_chain_end",
            "name": "fake-graph",
            "run_id": "root-run",
            "parent_ids": [],
            "metadata": {},
            "tags": [],
            "data": {"output": {"output": {"text": "hello"}}},
        }


class FakeProtocolStreamingEntry(FakeEntry):
    graph = FakeProtocolStreamingGraph()

    @staticmethod
    def build_graph(_checkpointer=None) -> FakeProtocolStreamingGraph:
        return FakeProtocolStreamingEntry.graph


class FakeProtocolStreamingLangGraphService(FakeLangGraphService):
    def get_entry(self, _graph_id: str | None) -> FakeProtocolStreamingEntry:
        return FakeProtocolStreamingEntry()


class FakeProtocolLlmStreamingEntry(FakeEntry):
    graph = FakeProtocolLlmStreamingGraph()

    @staticmethod
    def build_graph(_checkpointer=None) -> FakeProtocolLlmStreamingGraph:
        return FakeProtocolLlmStreamingEntry.graph


class FakeProtocolLlmStreamingLangGraphService(FakeLangGraphService):
    def get_entry(self, _graph_id: str | None) -> FakeProtocolLlmStreamingEntry:
        return FakeProtocolLlmStreamingEntry()


class FakeProtocolNamespaceGraph(FakeGraph):
    async def astream_events(self, prepared_input: dict, config: dict, version: str = "v2"):
        self.configs.append(config)
        yield {
            "event": "on_tool_start",
            "name": "search_docs",
            "run_id": "tool-run",
            "parent_ids": ["root-run"],
            "metadata": {
                "langgraph_node": "search_docs",
                "langgraph_checkpoint_ns": "node_1:task-1|search_docs:task-2",
            },
            "tags": [],
            "data": {"input": {"query": prepared_input["input"]["hello"]}},
        }
        yield {
            "event": "on_chat_model_stream",
            "name": "chat-model",
            "run_id": "chat-run",
            "parent_ids": ["root-run"],
            "metadata": {
                "langgraph_node": "call_model",
                "langgraph_checkpoint_ns": "node_1:task-1|call_model:task-3",
            },
            "tags": [],
            "data": {"chunk": AIMessageChunk(content="hello")},
        }
        yield {
            "event": "on_chain_stream",
            "name": "node_1",
            "run_id": "subgraph-run",
            "parent_ids": ["root-run"],
            "metadata": {
                "langgraph_node": "node_1",
                "langgraph_checkpoint_ns": "node_1:task-1",
            },
            "tags": [],
            "data": {"chunk": {"step": "partial"}},
        }
        yield {
            "event": "on_tool_end",
            "name": "search_docs",
            "run_id": "tool-run",
            "parent_ids": ["root-run"],
            "metadata": {
                "langgraph_node": "search_docs",
                "langgraph_checkpoint_ns": "node_1:task-1|search_docs:task-2",
            },
            "tags": [],
            "data": {"output": {"answer": "docs"}},
        }
        yield {
            "event": "on_chain_end",
            "name": "fake-graph",
            "run_id": "root-run",
            "parent_ids": [],
            "metadata": {},
            "tags": [],
            "data": {"output": {"output": {"messages": [AIMessage(content="hello")], "step": "final"}}},
        }


class FakeProtocolNamespaceEntry(FakeEntry):
    graph = FakeProtocolNamespaceGraph()

    @staticmethod
    def build_graph(_checkpointer=None) -> FakeProtocolNamespaceGraph:
        return FakeProtocolNamespaceEntry.graph


class FakeProtocolNamespaceLangGraphService(FakeLangGraphService):
    def get_entry(self, _graph_id: str | None) -> FakeProtocolNamespaceEntry:
        return FakeProtocolNamespaceEntry()


class FakeProtocolStructuredMessageGraph(FakeGraph):
    async def astream_events(self, prepared_input: dict, config: dict, version: str = "v2"):
        self.configs.append(config)
        yield {
            "event": "on_chat_model_stream",
            "name": "chat-model",
            "run_id": "chat-run",
            "parent_ids": ["root-run"],
            "metadata": {"langgraph_node": "call_model"},
            "tags": [],
            "data": {
                "chunk": AIMessageChunk(
                    content=[
                        {"type": "text", "text": "hello"},
                        {"type": "reasoning", "summary": [{"type": "summary_text", "text": "why"}]},
                    ]
                )
            },
        }
        yield {
            "event": "on_chain_end",
            "name": "fake-graph",
            "run_id": "root-run",
            "parent_ids": [],
            "metadata": {},
            "tags": [],
            "data": {
                "output": {
                    "output": {
                        "messages": [
                            AIMessage(
                                content=[
                                    {"type": "text", "text": "hello"},
                                    {"type": "reasoning", "summary": [{"type": "summary_text", "text": "why"}]},
                                ]
                            )
                        ]
                    }
                }
            },
        }


class FakeProtocolStructuredMessageEntry(FakeEntry):
    graph = FakeProtocolStructuredMessageGraph()

    @staticmethod
    def build_graph(_checkpointer=None) -> FakeProtocolStructuredMessageGraph:
        return FakeProtocolStructuredMessageEntry.graph


class FakeProtocolStructuredMessageLangGraphService(FakeLangGraphService):
    def get_entry(self, _graph_id: str | None) -> FakeProtocolStructuredMessageEntry:
        return FakeProtocolStructuredMessageEntry()


class FakeProtocolToolCallChunkGraph(FakeGraph):
    async def astream_events(self, prepared_input: dict, config: dict, version: str = "v2"):
        self.configs.append(config)
        yield {
            "event": "on_chat_model_stream",
            "name": "chat-model",
            "run_id": "chat-run",
            "parent_ids": ["root-run"],
            "metadata": {"langgraph_node": "call_model"},
            "tags": [],
            "data": {
                "chunk": AIMessageChunk(
                    content="",
                    tool_call_chunks=[
                        {"id": "call-1", "name": "search", "args": '{"q":"hel"}', "index": 0},
                    ],
                )
            },
        }
        yield {
            "event": "on_chat_model_stream",
            "name": "chat-model",
            "run_id": "chat-run",
            "parent_ids": ["root-run"],
            "metadata": {"langgraph_node": "call_model"},
            "tags": [],
            "data": {
                "chunk": AIMessageChunk(
                    content="",
                    tool_call_chunks=[
                        {"id": "call-1", "name": None, "args": 'lo"}', "index": 0},
                    ],
                )
            },
        }
        yield {
            "event": "on_chain_end",
            "name": "fake-graph",
            "run_id": "root-run",
            "parent_ids": [],
            "metadata": {},
            "tags": [],
            "data": {"output": {"output": {"messages": []}}},
        }


class FakeProtocolToolCallChunkEntry(FakeEntry):
    graph = FakeProtocolToolCallChunkGraph()

    @staticmethod
    def build_graph(_checkpointer=None) -> FakeProtocolToolCallChunkGraph:
        return FakeProtocolToolCallChunkEntry.graph


class FakeProtocolToolCallChunkLangGraphService(FakeLangGraphService):
    def get_entry(self, _graph_id: str | None) -> FakeProtocolToolCallChunkEntry:
        return FakeProtocolToolCallChunkEntry()


class FakeProtocolMixedStructuredGraph(FakeGraph):
    async def astream_events(self, prepared_input: dict, config: dict, version: str = "v2"):
        self.configs.append(config)
        yield {
            "event": "on_chat_model_stream",
            "name": "chat-model",
            "run_id": "chat-run",
            "parent_ids": ["root-run"],
            "metadata": {"langgraph_node": "call_model"},
            "tags": [],
            "data": {"chunk": AIMessageChunk(content="hello")},
        }
        yield {
            "event": "on_chain_end",
            "name": "fake-graph",
            "run_id": "root-run",
            "parent_ids": [],
            "metadata": {},
            "tags": [],
            "data": {
                "output": {
                    "output": {
                        "messages": [
                            {
                                "type": "AIMessage",
                                "content": [
                                    {"type": "text", "text": "hello"},
                                    {"type": "reasoning", "summary": [{"type": "summary_text", "text": "why"}]},
                                ],
                            }
                        ]
                    }
                }
            },
        }


class FakeProtocolMixedStructuredEntry(FakeEntry):
    graph = FakeProtocolMixedStructuredGraph()

    @staticmethod
    def build_graph(_checkpointer=None) -> FakeProtocolMixedStructuredGraph:
        return FakeProtocolMixedStructuredEntry.graph


class FakeProtocolMixedStructuredLangGraphService(FakeLangGraphService):
    def get_entry(self, _graph_id: str | None) -> FakeProtocolMixedStructuredEntry:
        return FakeProtocolMixedStructuredEntry()


class FakeProtocolMultiMessageGraph(FakeGraph):
    async def astream_events(self, prepared_input: dict, config: dict, version: str = "v2"):
        self.configs.append(config)
        yield {
            "event": "on_chain_stream",
            "name": "fake-graph",
            "run_id": "root-run",
            "parent_ids": [],
            "metadata": {},
            "tags": [],
            "data": {"chunk": {"messages": [HumanMessage(content="hi"), AIMessage(content="hello")]}}
        }
        yield {
            "event": "on_chain_end",
            "name": "fake-graph",
            "run_id": "root-run",
            "parent_ids": [],
            "metadata": {},
            "tags": [],
            "data": {"output": {"output": {"messages": [HumanMessage(content="hi"), AIMessage(content="hello")]}}}
        }


class FakeProtocolMultiMessageEntry(FakeEntry):
    graph = FakeProtocolMultiMessageGraph()

    @staticmethod
    def build_graph(_checkpointer=None) -> FakeProtocolMultiMessageGraph:
        return FakeProtocolMultiMessageEntry.graph


class FakeProtocolMultiMessageLangGraphService(FakeLangGraphService):
    def get_entry(self, _graph_id: str | None) -> FakeProtocolMultiMessageEntry:
        return FakeProtocolMultiMessageEntry()


@pytest.mark.asyncio
async def test_execute_run_publishes_incremental_protocol_messages_and_values(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_db = FakeDBManager()
    protocol_broker = ThreadProtocolEventBroker()
    FakeProtocolStreamingEntry.graph = FakeProtocolStreamingGraph()

    monkeypatch.setattr(
        "agentseek_api.services.run_executor.get_langgraph_service",
        lambda: FakeProtocolStreamingLangGraphService(),
    )
    monkeypatch.setattr("agentseek_api.services.run_executor.db_manager", fake_db)
    monkeypatch.setattr("agentseek_api.services.thread_protocol.thread_protocol_broker", protocol_broker)

    await execute_run(thread_id="t1", run_id="r1", payload={"hello": "world"}, user_id="user-1")

    thread_events = protocol_broker._events["t1"]
    message_events = [event for event in thread_events if event["method"] == "messages"]
    update_events = [event for event in thread_events if event["method"] == "updates"]
    value_events = [event for event in thread_events if event["method"] == "values"]

    assert [event["params"]["data"]["event"] for event in message_events[:3]] == [
        "message-start",
        "content-block-start",
        "content-block-delta",
    ]
    assert message_events[2]["params"]["data"]["delta"] == {"type": "text-delta", "text": "hel"}
    assert update_events[0]["params"]["data"] == {"step": "partial"}
    assert len(value_events) == 1
    assert value_events[0]["params"]["data"]["output"]["step"] == "final"


@pytest.mark.asyncio
async def test_execute_run_publishes_incremental_protocol_messages_for_llm_text_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_db = FakeDBManager()
    protocol_broker = ThreadProtocolEventBroker()
    FakeProtocolLlmStreamingEntry.graph = FakeProtocolLlmStreamingGraph()

    monkeypatch.setattr(
        "agentseek_api.services.run_executor.get_langgraph_service",
        lambda: FakeProtocolLlmStreamingLangGraphService(),
    )
    monkeypatch.setattr("agentseek_api.services.run_executor.db_manager", fake_db)
    monkeypatch.setattr("agentseek_api.services.thread_protocol.thread_protocol_broker", protocol_broker)

    await execute_run(thread_id="t1", run_id="r1", payload={"hello": "world"}, user_id="user-1")

    message_events = [event for event in protocol_broker._events["t1"] if event["method"] == "messages"]
    assert [event["params"]["data"]["event"] for event in message_events] == [
        "message-start",
        "content-block-start",
        "content-block-delta",
        "content-block-delta",
        "content-block-finish",
        "message-finish",
    ]
    assert message_events[2]["params"]["data"]["delta"] == {"type": "text-delta", "text": "hel"}
    assert message_events[3]["params"]["data"]["delta"] == {"type": "text-delta", "text": "lo"}


@pytest.mark.asyncio
async def test_execute_run_uses_langgraph_namespaces_for_protocol_events(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_db = FakeDBManager()
    protocol_broker = ThreadProtocolEventBroker()
    FakeProtocolNamespaceEntry.graph = FakeProtocolNamespaceGraph()

    monkeypatch.setattr(
        "agentseek_api.services.run_executor.get_langgraph_service",
        lambda: FakeProtocolNamespaceLangGraphService(),
    )
    monkeypatch.setattr("agentseek_api.services.run_executor.db_manager", fake_db)
    monkeypatch.setattr("agentseek_api.services.thread_protocol.thread_protocol_broker", protocol_broker)

    await execute_run(thread_id="t1", run_id="r1", payload={"hello": "world"}, user_id="user-1")

    thread_events = protocol_broker._events["t1"]
    message_events = [event for event in thread_events if event["method"] == "messages"]
    tool_events = [event for event in thread_events if event["method"] == "tools"]
    updates_events = [event for event in thread_events if event["method"] == "updates"]

    assert message_events[0]["params"]["namespace"] == ["node_1:task-1", "call_model:task-3"]
    assert all(event["params"]["namespace"] == ["node_1:task-1", "search_docs:task-2"] for event in tool_events)
    assert updates_events[0]["params"]["namespace"] == ["node_1:task-1"]


@pytest.mark.asyncio
async def test_execute_run_publishes_structured_protocol_message_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_db = FakeDBManager()
    protocol_broker = ThreadProtocolEventBroker()
    FakeProtocolStructuredMessageEntry.graph = FakeProtocolStructuredMessageGraph()

    monkeypatch.setattr(
        "agentseek_api.services.run_executor.get_langgraph_service",
        lambda: FakeProtocolStructuredMessageLangGraphService(),
    )
    monkeypatch.setattr("agentseek_api.services.run_executor.db_manager", fake_db)
    monkeypatch.setattr("agentseek_api.services.thread_protocol.thread_protocol_broker", protocol_broker)

    await execute_run(thread_id="t1", run_id="r1", payload={"hello": "world"}, user_id="user-1")

    message_events = [event for event in protocol_broker._events["t1"] if event["method"] == "messages"]
    block_starts = [event["params"]["data"] for event in message_events if event["params"]["data"]["event"] == "content-block-start"]
    assert any(block["content"]["type"] == "reasoning" for block in block_starts)
    ordered_events = [event["params"]["data"] for event in message_events]
    text_finish_index = next(
        index
        for index, event in enumerate(ordered_events)
        if event["event"] == "content-block-finish" and event["index"] == 0
    )
    reasoning_start_index = next(
        index
        for index, event in enumerate(ordered_events)
        if event["event"] == "content-block-start" and event["content"]["type"] == "reasoning"
    )
    assert text_finish_index < reasoning_start_index


@pytest.mark.asyncio
async def test_execute_run_streams_tool_call_chunks_without_duplicate_complete_blocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_db = FakeDBManager()
    protocol_broker = ThreadProtocolEventBroker()
    FakeProtocolToolCallChunkEntry.graph = FakeProtocolToolCallChunkGraph()

    monkeypatch.setattr(
        "agentseek_api.services.run_executor.get_langgraph_service",
        lambda: FakeProtocolToolCallChunkLangGraphService(),
    )
    monkeypatch.setattr("agentseek_api.services.run_executor.db_manager", fake_db)
    monkeypatch.setattr("agentseek_api.services.thread_protocol.thread_protocol_broker", protocol_broker)

    await execute_run(thread_id="t1", run_id="r1", payload={"hello": "world"}, user_id="user-1")

    message_events = [event["params"]["data"] for event in protocol_broker._events["t1"] if event["method"] == "messages"]
    block_starts = [event for event in message_events if event["event"] == "content-block-start"]
    block_deltas = [event for event in message_events if event["event"] == "content-block-delta"]

    assert len(block_starts) == 1
    assert block_starts[0]["content"]["type"] == "tool_call_chunk"
    assert len(block_deltas) == 1
    assert block_deltas[0]["delta"]["type"] == "tool_call_chunk"


@pytest.mark.asyncio
async def test_execute_run_merges_final_structured_blocks_after_live_text(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_db = FakeDBManager()
    protocol_broker = ThreadProtocolEventBroker()
    FakeProtocolMixedStructuredEntry.graph = FakeProtocolMixedStructuredGraph()

    monkeypatch.setattr(
        "agentseek_api.services.run_executor.get_langgraph_service",
        lambda: FakeProtocolMixedStructuredLangGraphService(),
    )
    monkeypatch.setattr("agentseek_api.services.run_executor.db_manager", fake_db)
    monkeypatch.setattr("agentseek_api.services.thread_protocol.thread_protocol_broker", protocol_broker)

    await execute_run(thread_id="t1", run_id="r1", payload={"hello": "world"}, user_id="user-1")

    message_events = [event["params"]["data"] for event in protocol_broker._events["t1"] if event["method"] == "messages"]
    block_starts = [event for event in message_events if event["event"] == "content-block-start"]
    assert any(block["content"]["type"] == "reasoning" for block in block_starts)


def test_protocol_message_stream_state_merges_open_messages_against_transcript_tail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol_broker = ThreadProtocolEventBroker()
    monkeypatch.setattr("agentseek_api.services.thread_protocol.thread_protocol_broker", protocol_broker)

    state = _ProtocolMessageStreamState(thread_id="t1", run_id="r1")
    state.publish_blocks(message_id="m1", role="ai", blocks=[{"type": "text", "text": "hel"}])
    state.merge_final_messages(
        messages=[
            {"type": "HumanMessage", "content": "hi"},
            {"type": "AIMessage", "content": "hello"},
        ],
        run_id="r1",
    )
    state.finish_all()

    message_events = [event["params"]["data"] for event in protocol_broker._events["t1"] if event["method"] == "messages"]
    message_starts = [event for event in message_events if event["event"] == "message-start"]
    assert message_starts == [{"event": "message-start", "role": "ai", "id": "m1"}]
    assert {"event": "content-block-delta", "index": 0, "delta": {"type": "text-delta", "text": "lo"}} in message_events
    assert [event for event in message_events if event["event"] == "message-finish"] == [{"event": "message-finish"}]


@pytest.mark.asyncio
async def test_execute_run_keeps_multiple_messages_in_single_chunk_distinct(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_db = FakeDBManager()
    protocol_broker = ThreadProtocolEventBroker()
    FakeProtocolMultiMessageEntry.graph = FakeProtocolMultiMessageGraph()

    monkeypatch.setattr(
        "agentseek_api.services.run_executor.get_langgraph_service",
        lambda: FakeProtocolMultiMessageLangGraphService(),
    )
    monkeypatch.setattr("agentseek_api.services.run_executor.db_manager", fake_db)
    monkeypatch.setattr("agentseek_api.services.thread_protocol.thread_protocol_broker", protocol_broker)

    await execute_run(thread_id="t1", run_id="r1", payload={"hello": "world"}, user_id="user-1")

    message_starts = [
        event["params"]["data"]
        for event in protocol_broker._events["t1"]
        if event["method"] == "messages" and event["params"]["data"]["event"] == "message-start"
    ]
    assert [event["role"] for event in message_starts] == ["human", "ai"]
    assert message_starts[0]["id"] != message_starts[1]["id"]
