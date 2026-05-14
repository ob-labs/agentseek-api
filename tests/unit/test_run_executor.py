import pytest
from langchain_core.messages import AIMessageChunk
from langgraph.constants import CONF, CONFIG_KEY_CHECKPOINTER

from agentseek_api.services.run_executor import RunExecutionResult, _translate_stream_events, execute_run


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

    @staticmethod
    def build_graph(_checkpointer=None) -> FakeGraph:
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

    async def run_checkpointer_call(self, func, *args, **kwargs):
        return func(*args, **kwargs)

    def get_checkpointer(self) -> FakeCheckpointer:
        return self.checkpointer

    def get_langgraph_checkpointer(self):
        return self.langgraph_checkpointer


@pytest.mark.asyncio
async def test_execute_run_saves_checkpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_db = FakeDBManager()
    monkeypatch.setattr("agentseek_api.services.run_executor.get_langgraph_service", lambda: FakeLangGraphService())
    monkeypatch.setattr("agentseek_api.services.run_executor.db_manager", fake_db)

    result = await execute_run(thread_id="t1", run_id="r1", payload={"hello": "world"})
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

    await execute_run(thread_id="t1", run_id="r1", payload={"a": 1}, graph_id="stress_test")
    assert fake_db.checkpointer.calls[0]["payload"]["graph_id"] == "stress_test"


@pytest.mark.asyncio
async def test_execute_run_passes_runtime_checkpointer_in_config(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_db = FakeDBManager()
    FakeEntry.graph = FakeGraph()
    monkeypatch.setattr("agentseek_api.services.run_executor.get_langgraph_service", lambda: FakeLangGraphService())
    monkeypatch.setattr("agentseek_api.services.run_executor.db_manager", fake_db)

    await execute_run(thread_id="t1", run_id="r1", payload={"a": 1})

    config = FakeEntry.graph.configs[0]
    assert config[CONF]["thread_id"] == "t1"
    assert config[CONF]["checkpoint_ns"] == "r1"
    assert config[CONF][CONFIG_KEY_CHECKPOINTER] is fake_db.langgraph_checkpointer


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

    result = await execute_run(thread_id="t1", run_id="r1", payload={"foo": "hello"})

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
