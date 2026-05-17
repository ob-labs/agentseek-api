from copy import deepcopy
from datetime import UTC, datetime

import pytest

from agentseek_api.services import thread_checkpoint_store as store


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
