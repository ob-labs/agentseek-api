from collections import namedtuple
from copy import deepcopy
from datetime import UTC, datetime

import pytest

from agentseek_api.services import thread_checkpoint_store as store
from agentseek_api.services.thread_checkpoint_store import _make_serializable


class FakeCheckpointTuple:
    def __init__(
        self,
        *,
        checkpoint_id: str,
        created_at: datetime,
        parent_id: str | None = None,
        checkpoint_ns: str = "",
    ) -> None:
        self.config = {
            "configurable": {
                "thread_id": "source-thread",
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        }
        self.parent_config = None
        if parent_id is not None:
            self.parent_config = {
                "configurable": {
                    "thread_id": "source-thread",
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": parent_id,
                }
            }
        self.metadata = {}
        self.checkpoint = {
            "id": checkpoint_id,
            "ts": created_at.isoformat(),
            "channel_versions": {},
        }


class FakeSaver:
    def __init__(self, tuples: list[FakeCheckpointTuple]) -> None:
        self._tuples = tuples
        self.calls: list[dict[str, object]] = []

    async def acopy_thread(self, _source_thread_id: str, _target_thread_id: str) -> None:
        raise NotImplementedError

    async def alist(self, _config, limit=None):
        _ = limit
        for item in self._tuples:
            yield item

    async def aput(self, config, checkpoint, metadata, versions):
        next_config = {
            "configurable": {
                "thread_id": "target-thread",
                "checkpoint_ns": config["configurable"].get("checkpoint_ns", ""),
                "checkpoint_id": f"copied-{checkpoint['id']}",
            }
        }
        self.calls.append(
            {
                "input_config": deepcopy(config),
                "checkpoint_id": checkpoint["id"],
                "output_config": deepcopy(next_config),
                "metadata": deepcopy(metadata),
                "versions": deepcopy(versions),
            }
        )
        return next_config


@pytest.mark.asyncio
async def test_copy_checkpoints_fallback_copies_parents_before_children(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = FakeCheckpointTuple(
        checkpoint_id="parent",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    child = FakeCheckpointTuple(
        checkpoint_id="child",
        created_at=datetime(2026, 1, 2, tzinfo=UTC),
        parent_id="parent",
    )
    saver = FakeSaver([child, parent])
    monkeypatch.setattr(
        "agentseek_api.services.thread_checkpoint_store.db_manager.get_langgraph_checkpointer",
        lambda: saver,
    )

    await store.copy_checkpoints("source-thread", "target-thread")

    assert [call["checkpoint_id"] for call in saver.calls] == ["parent", "child"]
    assert saver.calls[0]["input_config"] == {"configurable": {"thread_id": "target-thread", "checkpoint_ns": ""}}
    assert saver.calls[1]["input_config"] == saver.calls[0]["output_config"]


class TestMakeSerializable:
    def test_primitives_pass_through(self) -> None:
        assert _make_serializable(None) is None
        assert _make_serializable("hello") == "hello"
        assert _make_serializable(42) == 42
        assert _make_serializable(3.14) == 3.14
        assert _make_serializable(True) is True

    def test_dict_recursion(self) -> None:
        assert _make_serializable({"a": {1, 2}}) == {"a": [1, 2]}

    def test_list_and_tuple(self) -> None:
        assert _make_serializable([1, (2, 3)]) == [1, [2, 3]]

    def test_set_and_frozenset(self) -> None:
        result = _make_serializable({1, 2})
        assert isinstance(result, list)
        assert set(result) == {1, 2}

        result = _make_serializable(frozenset([3]))
        assert result == [3]

    def test_send_type(self) -> None:
        from langgraph.types import Send
        s = Send(node="my_node", arg={"key": "val"})
        result = _make_serializable(s)
        assert result == {"__type__": "Send", "node": "my_node", "arg": {"key": "val"}}

    def test_pydantic_model(self) -> None:
        from pydantic import BaseModel

        class Dummy(BaseModel):
            x: int = 1

        result = _make_serializable(Dummy())
        assert result == {"x": 1}

    def test_namedtuple(self) -> None:
        Point = namedtuple("Point", ["x", "y"])
        result = _make_serializable(Point(1, 2))
        assert result == [1, 2]

    def test_object_with_asdict(self) -> None:
        class Record:
            def _asdict(self):
                return {"a": 1, "b": {2, 3}}

        result = _make_serializable(Record())
        assert result["a"] == 1
        assert set(result["b"]) == {2, 3}

    def test_non_serializable_falls_back_to_repr(self) -> None:
        class Opaque:
            def __repr__(self) -> str:
                return "Opaque()"

        result = _make_serializable(Opaque())
        assert result == "Opaque()"


def test_config_includes_checkpoint_id_when_provided() -> None:
    result = store._config("thread-1", checkpoint_id="cp-42")
    assert result == {"configurable": {"thread_id": "thread-1", "checkpoint_ns": "", "checkpoint_id": "cp-42"}}


def test_checkpoint_step_returns_zero_for_none() -> None:
    assert store._checkpoint_step(None) == 0


def test_checkpoint_step_returns_zero_for_non_dict_metadata() -> None:
    fake = type("FakeTuple", (), {"metadata": "not-a-dict"})()
    assert store._checkpoint_step(fake) == 0


def test_checkpoint_step_returns_incremented_value() -> None:
    fake = type("FakeTuple", (), {"metadata": {"step": 5}})()
    assert store._checkpoint_step(fake) == 6


def test_checkpoint_step_returns_zero_for_non_int_step() -> None:
    fake = type("FakeTuple", (), {"metadata": {"step": "abc"}})()
    assert store._checkpoint_step(fake) == 0


def test_checkpoint_created_at_fallback_on_bad_timestamp() -> None:
    result = store._checkpoint_created_at({"ts": "not-a-date"})
    assert result.tzinfo is not None

    result = store._checkpoint_created_at({})
    assert result.tzinfo is not None
