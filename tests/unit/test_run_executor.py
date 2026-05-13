import pytest

from agentseek_api.services.run_executor import execute_run


class FakeGraph:
    async def ainvoke(self, prepared_input: dict, _config: dict) -> dict:
        return {"output": {"ok": True, "received": prepared_input}}


class FakeEntry:
    graph = FakeGraph()

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

    async def run_checkpointer_call(self, func, *args, **kwargs):
        return func(*args, **kwargs)

    def get_checkpointer(self) -> FakeCheckpointer:
        return self.checkpointer


@pytest.mark.asyncio
async def test_execute_run_saves_checkpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_db = FakeDBManager()
    monkeypatch.setattr("agentseek_api.services.run_executor.get_langgraph_service", lambda: FakeLangGraphService())
    monkeypatch.setattr("agentseek_api.services.run_executor.db_manager", fake_db)

    output = await execute_run(thread_id="t1", run_id="r1", payload={"hello": "world"})
    assert output["ok"] is True
    assert output["received"] == {"input": {"hello": "world"}}
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
