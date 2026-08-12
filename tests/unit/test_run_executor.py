from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from langgraph.constants import CONF, CONFIG_KEY_CHECKPOINTER

from agentseek_api.core.runtime_store import UserScopedStore
from agentseek_api.services.run_executor import (
    RunExecutionResult,
    _ProtocolMessageStreamState,
    execute_run,
)
from agentseek_api.services.thread_protocol import ThreadProtocolEventBroker


class FakeGraph:
    """Fake compiled graph: the default path streams through ``astream()`` and
    then reads the final state back via ``aget_state()``, mirroring how
    ``execute_run`` drives a real langgraph graph."""

    def __init__(self) -> None:
        self.configs: list[dict] = []

    async def astream(self, prepared_input: dict, config: dict, **kwargs):
        self.configs.append(config)
        yield ("values", {"output": {"ok": True, "received": prepared_input}})

    async def aget_state(self, config: dict):
        return SimpleNamespace(values=None)


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
    assert config[CONF]["thread_id"] == "t1"


class FakeKwargsCapturingGraph(FakeGraph):
    def __init__(self) -> None:
        super().__init__()
        self.stream_kwargs: list[dict] = []
        self.inputs: list[Any] = []

    async def astream(self, prepared_input, config: dict, **kwargs):
        self.configs.append(config)
        self.stream_kwargs.append(kwargs)
        self.inputs.append(prepared_input)
        yield ("values", {"output": {"ok": True}})


class FakeKwargsCapturingEntry:
    graph = FakeKwargsCapturingGraph()

    @staticmethod
    def build_graph(_checkpointer=None, store=None):
        return FakeKwargsCapturingEntry.graph

    @staticmethod
    def prepare_input(payload: dict) -> dict:
        return {"input": payload}

    @staticmethod
    def extract_output(result: dict, _payload: dict) -> dict:
        return result.get("output", {})


class FakeKwargsCapturingLangGraphService:
    def get_entry(self, _graph_id: str | None):
        return FakeKwargsCapturingEntry()

    def get_graph(self, _graph_id: str | None = None):
        return FakeKwargsCapturingEntry.graph


@pytest.mark.asyncio
async def test_execute_run_forwards_command_as_invocation(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_db = FakeDBManager()
    FakeKwargsCapturingEntry.graph = FakeKwargsCapturingGraph()
    monkeypatch.setattr(
        "agentseek_api.services.run_executor.get_langgraph_service",
        lambda: FakeKwargsCapturingLangGraphService(),
    )
    monkeypatch.setattr("agentseek_api.services.run_executor.db_manager", fake_db)

    from langgraph.types import Command

    result = await execute_run(
        thread_id="t1",
        run_id="r1",
        payload={"msg": "hi"},
        user_id="user-1",
        kwargs={"command": {"resume": "yes_continue", "goto": ["nodeB"]}},
    )

    assert isinstance(result, RunExecutionResult)
    invocation = FakeKwargsCapturingEntry.graph.inputs[0]
    assert isinstance(invocation, Command)
    assert invocation.resume == "yes_continue"
    assert invocation.goto == ["nodeB"]


@pytest.mark.asyncio
async def test_execute_run_forwards_interrupt_and_stream_mode_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_db = FakeDBManager()
    FakeKwargsCapturingEntry.graph = FakeKwargsCapturingGraph()
    monkeypatch.setattr(
        "agentseek_api.services.run_executor.get_langgraph_service",
        lambda: FakeKwargsCapturingLangGraphService(),
    )
    monkeypatch.setattr("agentseek_api.services.run_executor.db_manager", fake_db)

    await execute_run(
        thread_id="t1",
        run_id="r1",
        payload={"msg": "hi"},
        user_id="user-1",
        kwargs={
            "interrupt_before": ["node_a"],
            "interrupt_after": ["node_b"],
            "stream_modes": ["values", "updates", "debug"],
        },
    )

    kwargs = FakeKwargsCapturingEntry.graph.stream_kwargs[0]
    assert "node_a" in kwargs["interrupt_before"]
    assert "node_b" in kwargs["interrupt_after"]
    assert "values" in kwargs["stream_mode"]
    assert "debug" in kwargs["stream_mode"]

class FakeInterruptGraph(FakeGraph):
    async def astream(self, prepared_input: dict, config: dict, **kwargs):
        self.configs.append(config)
        yield (
            "updates",
            {
                "__interrupt__": [
                    type("Interrupt", (), {"value": "Provide value:", "id": "interrupt-1"})(),
                ]
            },
        )
        yield ("values", {"foo": prepared_input["input"]["foo"]})


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
async def test_execute_run_preserves_interrupts_from_updates_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_db = FakeDBManager()
    protocol_broker = ThreadProtocolEventBroker()
    FakeInterruptEntry.graph = FakeInterruptGraph()
    monkeypatch.setattr(
        "agentseek_api.services.run_executor.get_langgraph_service",
        lambda: FakeInterruptLangGraphService(),
    )
    monkeypatch.setattr("agentseek_api.services.run_executor.db_manager", fake_db)
    monkeypatch.setattr("agentseek_api.services.thread_protocol.thread_protocol_broker", protocol_broker)

    result = await execute_run(thread_id="t1", run_id="r1", payload={"foo": "hello"}, user_id="user-1")

    assert result.interrupted is True
    assert result.interrupts == [{"value": "Provide value:", "id": "interrupt-1"}]
    assert result.output["state"]["foo"] == "hello"

    # The interrupt rides on a ``values`` event with ``__interrupt__``
    # intact (remapped from updates because updates were not explicitly requested),
    # so the official SDK stream() parser can surface it.
    value_events = [event for event in protocol_broker._events["t1"] if event["method"] == "values"]
    interrupt_values = [event for event in value_events if "__interrupt__" in event["params"]["data"]]
    assert len(interrupt_values) == 1
    assert interrupt_values[0]["params"]["data"]["__interrupt__"] == [
        {"value": "Provide value:", "id": "interrupt-1"}
    ]


@pytest.mark.asyncio
async def test_execute_run_keeps_interrupt_in_updates_when_updates_requested(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the client explicitly requests updates, the interrupt stays on the
    updates channel with ``__interrupt__`` intact (official behavior)."""
    fake_db = FakeDBManager()
    protocol_broker = ThreadProtocolEventBroker()
    FakeInterruptEntry.graph = FakeInterruptGraph()
    monkeypatch.setattr(
        "agentseek_api.services.run_executor.get_langgraph_service",
        lambda: FakeInterruptLangGraphService(),
    )
    monkeypatch.setattr("agentseek_api.services.run_executor.db_manager", fake_db)
    monkeypatch.setattr("agentseek_api.services.thread_protocol.thread_protocol_broker", protocol_broker)

    await execute_run(
        thread_id="t1",
        run_id="r1",
        payload={"foo": "hello"},
        user_id="user-1",
        kwargs={"stream_modes": ["updates"]},
    )

    update_events = [event for event in protocol_broker._events["t1"] if event["method"] == "updates"]
    interrupt_updates = [event for event in update_events if "__interrupt__" in event["params"]["data"]]
    assert len(interrupt_updates) == 1
    assert interrupt_updates[0]["params"]["data"]["__interrupt__"] == [
        {"value": "Provide value:", "id": "interrupt-1"}
    ]


class FakeProtocolStreamingGraph(FakeGraph):
    async def astream(self, prepared_input: dict, config: dict, **kwargs):
        self.configs.append(config)
        yield ("messages", (AIMessageChunk(content="hel"), {"langgraph_node": "call_model"}))
        yield ("messages", (AIMessageChunk(content="lo"), {"langgraph_node": "call_model"}))
        yield ("updates", {"step": "partial"})
        yield ("values", {"output": {"messages": [AIMessage(content="hello")], "step": "final"}})


class FakeProtocolLlmStreamingGraph(FakeGraph):
    async def astream(self, prepared_input: dict, config: dict, **kwargs):
        self.configs.append(config)
        yield ("messages", (AIMessageChunk(content="hel"), {"langgraph_node": "call_model"}))
        yield ("messages", (AIMessageChunk(content="lo"), {"langgraph_node": "call_model"}))
        yield ("values", {"output": {"text": "hello"}})


class FakeProtocolNamespaceGraph(FakeGraph):
    """Namespaces flow through the ``(namespace, mode, chunk)`` tuples produced
    when the run requests ``stream_subgraphs``."""

    async def astream(self, prepared_input: dict, config: dict, **kwargs):
        self.configs.append(config)
        yield (
            ["node_1:task-1", "call_model:task-3"],
            "messages",
            (AIMessageChunk(content="hello"), {"langgraph_node": "call_model"}),
        )
        yield (["node_1:task-1"], "updates", {"step": "partial"})
        yield ([], "values", {"output": {"messages": [AIMessage(content="hello")], "step": "final"}})


class FakeProtocolStructuredMessageGraph(FakeGraph):
    async def astream(self, prepared_input: dict, config: dict, **kwargs):
        self.configs.append(config)
        yield (
            "messages",
            (
                AIMessageChunk(
                    content=[
                        {"type": "text", "text": "hello"},
                        {"type": "reasoning", "summary": [{"type": "summary_text", "text": "why"}]},
                    ]
                ),
                {"langgraph_node": "call_model"},
            ),
        )
        yield (
            "values",
            {
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
            },
        )


class FakeProtocolToolCallChunkGraph(FakeGraph):
    async def astream(self, prepared_input: dict, config: dict, **kwargs):
        self.configs.append(config)
        yield (
            "messages",
            (
                AIMessageChunk(
                    content="",
                    tool_call_chunks=[
                        {"id": "call-1", "name": "search", "args": '{"q":"hel"}', "index": 0},
                    ],
                ),
                {"langgraph_node": "call_model"},
            ),
        )
        yield (
            "messages",
            (
                AIMessageChunk(
                    content="",
                    tool_call_chunks=[
                        {"id": "call-1", "name": None, "args": 'lo"}', "index": 0},
                    ],
                ),
                {"langgraph_node": "call_model"},
            ),
        )
        yield ("values", {"output": {"messages": []}})


class FakeProtocolMixedStructuredGraph(FakeGraph):
    async def astream(self, prepared_input: dict, config: dict, **kwargs):
        self.configs.append(config)
        yield ("messages", (AIMessageChunk(content="hello"), {"langgraph_node": "call_model"}))
        yield (
            "values",
            {
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
            },
        )


class FakeProtocolMultiMessageGraph(FakeGraph):
    """A single state update carrying several non-LLM messages at once; each
    must surface as its own distinct protocol message (issue #48 regression:
    updates are emitted 1:1, never duplicated or collapsed into one id)."""

    async def astream(self, prepared_input: dict, config: dict, **kwargs):
        self.configs.append(config)
        yield (
            "updates",
            {
                "some_node": {
                    "messages": [
                        HumanMessage(content="hi"),
                        ToolMessage(content="tool result", tool_call_id="call-1"),
                    ]
                }
            },
        )
        yield (
            "values",
            {
                "output": {
                    "messages": [
                        HumanMessage(content="hi"),
                        ToolMessage(content="tool result", tool_call_id="call-1"),
                    ]
                }
            },
        )


class FakeProtocolToolMessageGraph(FakeGraph):
    async def astream(self, prepared_input: dict, config: dict, **kwargs):
        self.configs.append(config)
        tool_message = ToolMessage(
            content="42 characters",
            tool_call_id="call-character-count",
            id="tool-message-1",
        )
        tool_meta = {
            "langgraph_node": "tools",
            "langgraph_checkpoint_ns": "tools:task-1",
            "provider": "deterministic",
        }
        yield (["tools:task-1"], "messages", (tool_message, tool_meta))
        # The same tool message is streamed twice; it must only be published once.
        yield (["tools:task-1"], "messages", (tool_message, tool_meta))
        yield (
            ["call_model:task-3"],
            "messages",
            (AIMessageChunk(content="Final answer", id="ai-message-1"), {"langgraph_node": "call_model"}),
        )
        yield (
            [],
            "values",
            {"output": {"messages": [tool_message, AIMessage(content="Final answer", id="ai-message-1")]}},
        )


class _FakeProtocolEntry(FakeEntry):
    graph = FakeGraph()

    @staticmethod
    def build_graph(_checkpointer=None) -> FakeGraph:
        return _FakeProtocolEntry.graph


class _FakeProtocolLangGraphService(FakeLangGraphService):
    def get_entry(self, _graph_id: str | None) -> _FakeProtocolEntry:
        return _FakeProtocolEntry()


async def _run_fake_graph(
    monkeypatch: pytest.MonkeyPatch,
    graph: FakeGraph,
    *,
    stream_modes: list[str] | None = None,
    stream_subgraphs: bool = False,
) -> list[dict[str, Any]]:
    fake_db = FakeDBManager()
    protocol_broker = ThreadProtocolEventBroker()
    _FakeProtocolEntry.graph = graph
    monkeypatch.setattr(
        "agentseek_api.services.run_executor.get_langgraph_service",
        lambda: _FakeProtocolLangGraphService(),
    )
    monkeypatch.setattr("agentseek_api.services.run_executor.db_manager", fake_db)
    monkeypatch.setattr("agentseek_api.services.thread_protocol.thread_protocol_broker", protocol_broker)

    kwargs: dict[str, Any] = {}
    if stream_modes is not None:
        kwargs["stream_modes"] = stream_modes
    if stream_subgraphs:
        kwargs["stream_subgraphs"] = True
    await execute_run(thread_id="t1", run_id="r1", payload={"hello": "world"}, user_id="user-1", kwargs=kwargs)
    return protocol_broker._events["t1"]


@pytest.mark.asyncio
async def test_execute_run_publishes_incremental_protocol_messages_and_values(monkeypatch: pytest.MonkeyPatch) -> None:
    thread_events = await _run_fake_graph(
        monkeypatch, FakeProtocolStreamingGraph(), stream_modes=["updates", "values"]
    )
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
    thread_events = await _run_fake_graph(monkeypatch, FakeProtocolLlmStreamingGraph())
    message_events = [event for event in thread_events if event["method"] == "messages"]
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
    thread_events = await _run_fake_graph(
        monkeypatch, FakeProtocolNamespaceGraph(), stream_modes=["updates"], stream_subgraphs=True
    )
    message_events = [event for event in thread_events if event["method"] == "messages"]
    updates_events = [event for event in thread_events if event["method"] == "updates"]

    assert message_events[0]["params"]["namespace"] == ["node_1:task-1", "call_model:task-3"]
    assert updates_events[0]["params"]["namespace"] == ["node_1:task-1"]


@pytest.mark.asyncio
async def test_execute_run_publishes_structured_protocol_message_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    thread_events = await _run_fake_graph(monkeypatch, FakeProtocolStructuredMessageGraph())
    message_events = [event for event in thread_events if event["method"] == "messages"]
    block_starts = [
        event["params"]["data"]
        for event in message_events
        if event["params"]["data"]["event"] == "content-block-start"
    ]
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
    thread_events = await _run_fake_graph(monkeypatch, FakeProtocolToolCallChunkGraph())
    message_events = [event["params"]["data"] for event in thread_events if event["method"] == "messages"]
    block_starts = [event for event in message_events if event["event"] == "content-block-start"]
    block_deltas = [event for event in message_events if event["event"] == "content-block-delta"]

    assert len(block_starts) == 1
    assert block_starts[0]["content"]["type"] == "tool_call_chunk"
    assert len(block_deltas) == 1
    assert block_deltas[0]["delta"]["type"] == "tool_call_chunk"


@pytest.mark.asyncio
async def test_execute_run_merges_final_structured_blocks_after_live_text(monkeypatch: pytest.MonkeyPatch) -> None:
    thread_events = await _run_fake_graph(monkeypatch, FakeProtocolMixedStructuredGraph())
    message_events = [event["params"]["data"] for event in thread_events if event["method"] == "messages"]
    block_starts = [event for event in message_events if event["event"] == "content-block-start"]
    assert any(block["content"]["type"] == "reasoning" for block in block_starts)

@pytest.mark.asyncio
async def test_execute_run_mirrors_tool_message_to_requested_tuple_stream_once_and_before_final_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    thread_events = await _run_fake_graph(
        monkeypatch,
        FakeProtocolToolMessageGraph(),
        stream_modes=["messages-tuple", "values"],
        stream_subgraphs=True,
    )
    complete_events = [event for event in thread_events if event["method"] == "messages/complete"]
    tuple_events = [event for event in thread_events if event["method"] == "messages-tuple"]
    tool_tuples = [event for event in tuple_events if event["params"]["data"][0]["type"] == "tool"]

    assert len(complete_events) == 1
    assert len(tool_tuples) == 1

    tool_tuple = tool_tuples[0]
    tool_payload, metadata = tool_tuple["params"]["data"]
    assert tool_payload["content"] == "42 characters"
    assert tool_payload["id"] == "tool-message-1"
    assert tool_payload["tool_call_id"] == "call-character-count"
    assert metadata == {
        "langgraph_node": "tools",
        "langgraph_checkpoint_ns": "tools:task-1",
        "provider": "deterministic",
    }
    assert tool_tuple["params"]["namespace"] == ["tools:task-1"]
    assert tool_tuple["params"]["run_id"] == "r1"
    assert complete_events[0]["params"]["data"] == [tool_payload]

    final_answer_tuple = next(
        event for event in tuple_events if event["params"]["data"][0].get("id") == "ai-message-1"
    )
    assert thread_events.index(tool_tuple) < thread_events.index(final_answer_tuple)


@pytest.mark.asyncio
async def test_execute_run_keeps_tool_message_complete_without_unrequested_tuple_mirror(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    thread_events = await _run_fake_graph(
        monkeypatch, FakeProtocolToolMessageGraph(), stream_modes=["values"], stream_subgraphs=True
    )
    complete_events = [event for event in thread_events if event["method"] == "messages/complete"]
    tuple_events = [event for event in thread_events if event["method"] == "messages-tuple"]

    assert len(complete_events) == 1
    assert complete_events[0]["params"]["data"][0]["id"] == "tool-message-1"
    assert tuple_events == []


@pytest.mark.asyncio
async def test_execute_run_keeps_multiple_messages_in_single_chunk_distinct(monkeypatch: pytest.MonkeyPatch) -> None:
    thread_events = await _run_fake_graph(monkeypatch, FakeProtocolMultiMessageGraph(), stream_modes=["updates"])
    message_events = [event["params"]["data"] for event in thread_events if event["method"] == "messages"]
    message_starts = [event for event in message_events if event["event"] == "message-start"]

    assert [event["role"] for event in message_starts] == ["human", "tool"]
    assert message_starts[0]["id"] != message_starts[1]["id"]


@pytest.mark.asyncio
async def test_execute_run_publishes_each_astream_updates_chunk_exactly_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """The default astream path must forward every ``updates`` chunk to the wire
    exactly once. This pins the issue #48 regression: parallel (Send-style)
    worker updates used to be emitted twice (once bare, once node-wrapped)
    through the astream_events translation."""
    graph = FakeUpdatesOnceGraph()
    thread_events = await _run_fake_graph(monkeypatch, graph, stream_modes=["updates"])
    updates = [event for event in thread_events if event["method"] == "updates"]

    assert [event["params"]["data"] for event in updates] == [
        {"process_item": {"results": [{"processed_item": "a", "length": 1}]}},
        {"process_item": {"results": [{"processed_item": "b", "length": 1}]}},
        {"process_item": {"results": [{"processed_item": "c", "length": 1}]}},
    ]


class FakeUpdatesOnceGraph(FakeGraph):
    async def astream(self, prepared_input: dict, config: dict, **kwargs):
        self.configs.append(config)
        yield ("updates", {"process_item": {"results": [{"processed_item": "a", "length": 1}]}})
        yield ("updates", {"process_item": {"results": [{"processed_item": "b", "length": 1}]}})
        yield ("updates", {"process_item": {"results": [{"processed_item": "c", "length": 1}]}})
        yield (
            "values",
            {
                "output": {
                    "results": [
                        {"processed_item": "a", "length": 1},
                        {"processed_item": "b", "length": 1},
                        {"processed_item": "c", "length": 1},
                    ]
                }
            },
        )


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

    message_events = [
        event["params"]["data"] for event in protocol_broker._events["t1"] if event["method"] == "messages"
    ]
    message_starts = [event for event in message_events if event["event"] == "message-start"]
    assert message_starts == [{"event": "message-start", "role": "ai", "id": "m1"}]
    assert {"event": "content-block-delta", "index": 0, "delta": {"type": "text-delta", "text": "lo"}} in message_events
    assert [event for event in message_events if event["event"] == "message-finish"] == [{"event": "message-finish"}]


class FakeAstreamEventsGraph(FakeGraph):
    """Fake graph for the ``events`` stream mode: executes through the retained
    ``astream_events`` path (raw event stream) instead of the default astream."""

    async def astream_events(self, prepared_input: dict, config: dict, *, version: str, **kwargs):
        self.configs.append(config)
        yield {
            "event": "on_chat_model_stream",
            "data": {"chunk": AIMessageChunk(content="hi")},
            "metadata": {"langgraph_node": "call_model"},
            "parent_ids": [],
        }
        yield {
            "event": "on_custom_event",
            "data": {"custom": "payload"},
            "metadata": {},
            "parent_ids": [],
        }
        yield {
            "event": "on_chain_stream",
            "data": {"chunk": ("values", {"output": {"ok": True}})},
            "metadata": {"langgraph_node": "root"},
            "parent_ids": [],
        }
        yield {
            "event": "on_chain_end",
            "data": {"output": {"ok": True}},
            "metadata": {"langgraph_node": "root"},
            "parent_ids": [],
        }


class _FakeAstreamEventsEntry(FakeEntry):
    graph = FakeAstreamEventsGraph()

    @staticmethod
    def build_graph(_checkpointer=None, store=None) -> FakeAstreamEventsGraph:
        return _FakeAstreamEventsEntry.graph

    @staticmethod
    def extract_output(result: dict, _payload: dict) -> dict:
        return result if isinstance(result, dict) else {}


class _FakeAstreamEventsLangGraphService(FakeLangGraphService):
    def get_entry(self, _graph_id: str | None) -> _FakeAstreamEventsEntry:
        return _FakeAstreamEventsEntry()


@pytest.mark.asyncio
async def test_execute_run_events_mode_uses_astream_events_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """``stream_mode=["events"]`` keeps the raw ``astream_events`` path: message
    chunks surface as protocol messages, custom events surface on the custom
    channel, and the root on_chain_end finalizes the stream."""
    fake_db = FakeDBManager()
    protocol_broker = ThreadProtocolEventBroker()
    monkeypatch.setattr(
        "agentseek_api.services.run_executor.get_langgraph_service",
        lambda: _FakeAstreamEventsLangGraphService(),
    )
    monkeypatch.setattr("agentseek_api.services.run_executor.db_manager", fake_db)
    monkeypatch.setattr("agentseek_api.services.thread_protocol.thread_protocol_broker", protocol_broker)

    result = await execute_run(
        thread_id="t1",
        run_id="r1",
        payload={"msg": "hello"},
        user_id="user-1",
        kwargs={"stream_modes": ["events"]},
    )

    assert result.output == {"ok": True}
    message_events = [e for e in protocol_broker._events["t1"] if e["method"] == "messages"]
    assert message_events, "expected protocol message events from on_chat_model_stream"
    custom_events = [e for e in protocol_broker._events["t1"] if e["method"] == "custom"]
    assert custom_events
    assert custom_events[0]["params"]["data"] == {"custom": "payload"}
    # Each raw astream_events() item is published onto the events channel.
    raw_events = [e for e in protocol_broker._events["t1"] if e["method"] == "events"]
    assert raw_events, "expected raw astream_events() items on the events channel"
    assert len(raw_events) >= 4, "expected one events frame per astream_events() item"


class FakeSubgraphAggregateGraph(FakeGraph):
    """Graph whose root-level on_chain_end result differs from the values chunk,
    exercising the final-state capture fallback via aget_state root probe."""

    async def astream(self, prepared_input: dict, config: dict, **kwargs):
        self.configs.append(config)
        yield (["sub:1"], "values", {"output": {"partial": True}})

    async def aget_state(self, config: dict):
        if not (config.get(CONF) or {}).get("checkpoint_ns"):
            return SimpleNamespace(values={"output": {"final": True}})
        return SimpleNamespace(values=None)


class _FakeSubgraphAggregateEntry(FakeEntry):
    graph = FakeSubgraphAggregateGraph()

    @staticmethod
    def build_graph(_checkpointer=None, store=None) -> FakeSubgraphAggregateGraph:
        return _FakeSubgraphAggregateEntry.graph


class _FakeSubgraphAggregateService(FakeLangGraphService):
    def get_entry(self, _graph_id: str | None) -> _FakeSubgraphAggregateEntry:
        return _FakeSubgraphAggregateEntry()


@pytest.mark.asyncio
async def test_execute_run_subgraphs_namespace_uses_root_state_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """With ``stream_subgraphs=True`` events are (ns, mode, chunk) triples and,
    absent a root values chunk, the final state is captured from the checkpointer
    root-namespace probe."""
    fake_db = FakeDBManager()
    protocol_broker = ThreadProtocolEventBroker()
    monkeypatch.setattr(
        "agentseek_api.services.run_executor.get_langgraph_service",
        lambda: _FakeSubgraphAggregateService(),
    )
    monkeypatch.setattr("agentseek_api.services.run_executor.db_manager", fake_db)
    monkeypatch.setattr("agentseek_api.services.thread_protocol.thread_protocol_broker", protocol_broker)

    result = await execute_run(
        thread_id="t1",
        run_id="r1",
        payload={"msg": "hello"},
        user_id="user-1",
        kwargs={"stream_modes": ["values"], "stream_subgraphs": True},
    )

    assert result.output == {"final": True}
    value_events = [e for e in protocol_broker._events["t1"] if e["method"] == "values"]
    assert value_events
    assert value_events[0]["params"]["namespace"] == ["sub:1"]


class FakeInterruptResultGraph(FakeInterruptGraph):
    """HITL interrupt: the interrupt arrives in the updates stream with a
    non-empty state, so the run result must merge __interrupt__ into the final
    values and emit input.requested."""

    async def astream(self, prepared_input: dict, config: dict, **kwargs):
        self.configs.append(config)
        yield (
            "updates",
            {
                "__interrupt__": [
                    type("Interrupt", (), {"value": "Provide value:", "id": "interrupt-1"})(),
                ],
                "foo": prepared_input["input"]["foo"],
            },
        )


class _FakeInterruptResultEntry(FakeEntry):
    graph = FakeInterruptResultGraph()

    @staticmethod
    def build_graph(_checkpointer=None, store=None) -> FakeInterruptResultGraph:
        return _FakeInterruptResultEntry.graph

    @staticmethod
    def extract_output(result: dict, _payload: dict) -> dict:
        interrupts = result.get("__interrupt__", [])
        return {
            "state": result.get("foo"),
            "interrupted": bool(interrupts),
            "interrupts": [{"value": item.value, "id": item.id} for item in interrupts],
        }


class _FakeInterruptResultService(FakeLangGraphService):
    def get_entry(self, _graph_id: str | None) -> _FakeInterruptResultEntry:
        return _FakeInterruptResultEntry()


@pytest.mark.asyncio
async def test_execute_run_interrupt_merges_into_result_and_emits_input_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """HITL interrupt with non-empty state: __interrupt__ is merged into the run
    result (so extract_output sees it) and input.requested is emitted."""
    fake_db = FakeDBManager()
    protocol_broker = ThreadProtocolEventBroker()
    monkeypatch.setattr(
        "agentseek_api.services.run_executor.get_langgraph_service",
        lambda: _FakeInterruptResultService(),
    )
    monkeypatch.setattr("agentseek_api.services.run_executor.db_manager", fake_db)
    monkeypatch.setattr("agentseek_api.services.thread_protocol.thread_protocol_broker", protocol_broker)

    result = await execute_run(
        thread_id="t1",
        run_id="r1",
        payload={"foo": "hello"},
        user_id="user-1",
        kwargs={"stream_modes": ["values"]},
    )

    assert result.interrupted is True
    assert result.interrupts == [{"value": "Provide value:", "id": "interrupt-1"}]
    input_requested = [e for e in protocol_broker._events["t1"] if e["method"] == "input.requested"]
    assert len(input_requested) == 1
    assert input_requested[0]["params"]["data"]["payload"] == "Provide value:"


class FakeTupleNamespaceGraph(FakeGraph):
    """astream(subgraphs=True) yields tuple namespaces; they must be normalized
    to lists before publication so the live broker's namespace filter (which
    compares list slices against list prefixes) can match them."""

    async def astream(self, prepared_input: dict, config: dict, **kwargs):
        self.configs.append(config)
        yield (("node_1:task-1",), "updates", {"step": "partial"})
        yield (("node_1:task-1",), "values", {"output": {"step": "final"}})


@pytest.mark.asyncio
async def test_execute_run_normalizes_tuple_namespaces_for_live_filter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    thread_events = await _run_fake_graph(
        monkeypatch, FakeTupleNamespaceGraph(), stream_modes=["updates"], stream_subgraphs=True
    )
    updates_events = [event for event in thread_events if event["method"] == "updates"]
    assert updates_events, "expected updates event from tuple-namespaced subgraph"
    assert updates_events[0]["params"]["namespace"] == ["node_1:task-1"]
    assert isinstance(updates_events[0]["params"]["namespace"], list)


class FakeParallelIdlessMessagesGraph(FakeGraph):
    """Two id-less messages from different subgraph namespaces, each with one
    incremental chunk.

    ``AIMessageChunk`` has no ``id`` by default, so both go through the fallback
    identity path. The namespaces differ, so the fallback id must keep the two
    messages distinct (previously both collapsed to ``{run}:message:0`` and were
    merged by the client).
    """

    async def astream(self, prepared_input: dict, config: dict, **kwargs):
        self.configs.append(config)
        yield (("ns_a:task-1",), "messages", (AIMessageChunk(content="hello", id=None), {"langgraph_node": "ns_a"}))
        yield (("ns_b:task-1",), "messages", (AIMessageChunk(content="world", id=None), {"langgraph_node": "ns_b"}))

    async def aget_state(self, config: dict):
        return SimpleNamespace(values={"output": {"ok": True}})


@pytest.mark.asyncio
async def test_execute_run_idless_messages_from_different_namespaces_get_distinct_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    thread_events = await _run_fake_graph(
        monkeypatch, FakeParallelIdlessMessagesGraph(), stream_modes=["messages"], stream_subgraphs=True
    )
    metadata_events = [
        event for event in thread_events if event["method"] == "messages/metadata"
    ]
    assert len(metadata_events) == 2, (
        f"expected one messages/metadata per distinct id-less message, got {len(metadata_events)}"
    )
    # The two metadata events must carry distinct message ids (their wire
    # identity), so the SDK client routes them to two independent streams
    # instead of merging the second chunk into the first message.
    message_ids = [
        next(iter(event["params"]["data"].keys()))
        for event in metadata_events
    ]
    assert len(message_ids) == len(set(message_ids)), f"id-less message ids collided: {message_ids}"
    assert any("ns_a" in mid for mid in message_ids)
    assert any("ns_b" in mid for mid in message_ids)
