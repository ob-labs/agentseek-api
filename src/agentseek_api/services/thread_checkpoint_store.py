from __future__ import annotations

import json
from copy import deepcopy
from datetime import UTC, datetime
from typing import Any

from langgraph.checkpoint.base import CheckpointTuple, create_checkpoint, empty_checkpoint
from langgraph.types import Send

from agentseek_api.core.database import db_manager


def _make_serializable(obj: Any) -> Any:
    """Recursively convert LangGraph internal types to JSON-safe representations."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, Send):
        return {"__type__": "Send", "node": obj.node, "arg": _make_serializable(obj.arg)}
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(item) for item in obj]
    if isinstance(obj, (set, frozenset)):
        return [_make_serializable(item) for item in obj]
    if hasattr(obj, "model_dump") and callable(obj.model_dump):
        return _make_serializable(obj.model_dump())
    if hasattr(obj, "_asdict") and callable(obj._asdict):
        return {k: _make_serializable(v) for k, v in obj._asdict().items()}
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        return repr(obj)


def _config(
    thread_id: str,
    *,
    checkpoint_ns: str = "",
    checkpoint_id: str | None = None,
) -> dict[str, dict[str, str]]:
    configurable: dict[str, str] = {
        "thread_id": thread_id,
        "checkpoint_ns": checkpoint_ns,
    }
    if checkpoint_id is not None:
        configurable["checkpoint_id"] = checkpoint_id
    return {"configurable": configurable}


def _checkpoint_created_at(checkpoint: dict[str, Any]) -> datetime:
    raw_ts = checkpoint.get("ts")
    if isinstance(raw_ts, str):
        try:
            created_at = datetime.fromisoformat(raw_ts)
            return created_at if created_at.tzinfo is not None else created_at.replace(tzinfo=UTC)
        except ValueError:
            pass
    return datetime.now(UTC)


def _checkpoint_step(checkpoint_tuple: CheckpointTuple | None) -> int:
    if checkpoint_tuple is None or not isinstance(checkpoint_tuple.metadata, dict):
        return 0
    raw_step = checkpoint_tuple.metadata.get("step")
    if isinstance(raw_step, int):
        return raw_step + 1
    return 0


def _checkpoint_id(checkpoint_tuple: CheckpointTuple) -> str:
    configurable = checkpoint_tuple.config.get("configurable", {})
    return str(configurable.get("checkpoint_id", checkpoint_tuple.checkpoint.get("id", "")))


def _parent_checkpoint_id(checkpoint_tuple: CheckpointTuple) -> str | None:
    if checkpoint_tuple.parent_config is None:
        return None
    configurable = checkpoint_tuple.parent_config.get("configurable", {})
    raw_parent_id = configurable.get("checkpoint_id")
    if isinstance(raw_parent_id, str) and raw_parent_id:
        return raw_parent_id
    return None


def checkpoint_to_payload(checkpoint_tuple: CheckpointTuple) -> dict[str, Any]:
    configurable = checkpoint_tuple.config.get("configurable", {})
    checkpoint = checkpoint_tuple.checkpoint
    checkpoint_id = _checkpoint_id(checkpoint_tuple)
    checkpoint_ns = str(configurable.get("checkpoint_ns", ""))
    thread_id = str(configurable.get("thread_id", ""))
    metadata = deepcopy(checkpoint_tuple.metadata) if isinstance(checkpoint_tuple.metadata, dict) else {}
    parent_checkpoint = None
    if checkpoint_tuple.parent_config is not None:
        parent_configurable = checkpoint_tuple.parent_config.get("configurable", {})
        parent_checkpoint = {
            "thread_id": str(parent_configurable.get("thread_id", thread_id)),
            "checkpoint_ns": str(parent_configurable.get("checkpoint_ns", "")),
            "checkpoint_id": str(parent_configurable.get("checkpoint_id", "")),
        }
    return {
        "values": _make_serializable(deepcopy(checkpoint.get("channel_values", {}))),
        "next": [],
        "tasks": [],
        "checkpoint": {
            "thread_id": thread_id,
            "checkpoint_ns": checkpoint_ns,
            "checkpoint_id": checkpoint_id,
        },
        "metadata": metadata,
        "created_at": _checkpoint_created_at(checkpoint),
        "parent_checkpoint": parent_checkpoint,
        "interrupts": [],
    }


async def get_latest_checkpoint(thread_id: str) -> CheckpointTuple | None:
    return await db_manager.get_langgraph_checkpointer().aget_tuple(_config(thread_id))


async def list_checkpoints(thread_id: str, *, limit: int | None = None) -> list[CheckpointTuple]:
    checkpoints: list[CheckpointTuple] = []
    async for item in db_manager.get_langgraph_checkpointer().alist(_config(thread_id), limit=limit):
        checkpoints.append(item)
    return checkpoints


async def get_checkpoint_by_id(thread_id: str, checkpoint_id: str) -> CheckpointTuple | None:
    async for item in db_manager.get_langgraph_checkpointer().alist(_config(thread_id)):
        if _checkpoint_id(item) == checkpoint_id:
            return item
    return None


async def put_checkpoint(
    thread_id: str,
    values: dict[str, Any],
    *,
    metadata: dict[str, Any] | None = None,
) -> CheckpointTuple:
    saver = db_manager.get_langgraph_checkpointer()
    latest = await get_latest_checkpoint(thread_id)
    base_checkpoint = deepcopy(latest.checkpoint) if latest is not None else empty_checkpoint()
    next_step = _checkpoint_step(latest)
    checkpoint = create_checkpoint(base_checkpoint, None, next_step)
    checkpoint["channel_values"] = deepcopy(values)
    checkpoint["updated_channels"] = sorted(values.keys()) if values else []
    existing_versions = deepcopy(checkpoint.get("channel_versions", {}))
    new_versions = {
        key: saver.get_next_version(existing_versions.get(key), None)
        for key in values
    }
    checkpoint["channel_versions"] = {
        **existing_versions,
        **new_versions,
    }
    next_metadata = {
        "source": "update",
        "step": next_step,
        "writes": deepcopy(values),
    }
    if metadata:
        next_metadata.update(metadata)
    next_config = await saver.aput(
        latest.config if latest is not None else _config(thread_id),
        checkpoint,
        next_metadata,
        new_versions,
    )
    saved = await saver.aget_tuple(next_config)
    if saved is None:
        raise RuntimeError("Checkpoint save did not return a persisted checkpoint")
    return saved


async def copy_checkpoints(source_thread_id: str, target_thread_id: str) -> None:
    saver = db_manager.get_langgraph_checkpointer()
    try:
        await saver.acopy_thread(source_thread_id, target_thread_id)
        return
    except NotImplementedError:
        pass

    checkpoints = await list_checkpoints(source_thread_id)
    if not checkpoints:
        return

    checkpoints_by_id = {_checkpoint_id(item): item for item in checkpoints}
    pending_ids = set(checkpoints_by_id)
    copied_configs: dict[str, dict[str, dict[str, str]]] = {}

    while pending_ids:
        progressed = False
        ordered_pending = sorted(
            (checkpoints_by_id[checkpoint_id] for checkpoint_id in pending_ids),
            key=lambda item: (_checkpoint_created_at(item.checkpoint), _checkpoint_id(item)),
        )

        for item in ordered_pending:
            checkpoint_id = _checkpoint_id(item)
            parent_id = _parent_checkpoint_id(item)
            if parent_id is not None and parent_id in pending_ids:
                continue

            source_configurable = item.config.get("configurable", {})
            checkpoint_ns = str(source_configurable.get("checkpoint_ns", ""))
            parent_config = copied_configs.get(parent_id) if parent_id is not None else None
            next_config = await saver.aput(
                parent_config or _config(target_thread_id, checkpoint_ns=checkpoint_ns),
                deepcopy(item.checkpoint),
                deepcopy(item.metadata) if isinstance(item.metadata, dict) else {},
                deepcopy(item.checkpoint.get("channel_versions", {})),
            )
            copied_configs[checkpoint_id] = next_config
            pending_ids.remove(checkpoint_id)
            progressed = True

        if not progressed:
            unresolved = ", ".join(sorted(pending_ids))
            raise RuntimeError(f"Unable to copy checkpoints with unresolved parent references: {unresolved}")


async def prune_checkpoints(thread_ids: list[str], *, strategy: str) -> None:
    saver = db_manager.get_langgraph_checkpointer()
    try:
        await saver.aprune(thread_ids, strategy=strategy)
        return
    except NotImplementedError:
        pass

    if strategy == "delete":
        for thread_id in thread_ids:
            await saver.adelete_thread(thread_id)
        return

    if strategy != "keep_latest":
        raise ValueError(f"Unsupported prune strategy: {strategy}")

    for thread_id in thread_ids:
        checkpoints = await list_checkpoints(thread_id)
        if not checkpoints:
            continue

        keepers: dict[str, CheckpointTuple] = {}
        for item in checkpoints:
            checkpoint_ns = str(item.config.get("configurable", {}).get("checkpoint_ns", ""))
            keepers.setdefault(checkpoint_ns, item)

        await saver.adelete_thread(thread_id)
        for checkpoint_ns, item in sorted(keepers.items()):
            await saver.aput(
                _config(thread_id, checkpoint_ns=checkpoint_ns),
                deepcopy(item.checkpoint),
                deepcopy(item.metadata) if isinstance(item.metadata, dict) else {},
                deepcopy(item.checkpoint.get("channel_versions", {})),
            )
