from datetime import UTC, datetime

import pytest
from fastapi import HTTPException

from agentseek_api.api import threads as threads_module
from agentseek_api.core.orm import Run, Thread
from agentseek_api.models.api import ThreadPatch, ThreadPruneRequest, ThreadSearchRequest
from agentseek_api.models.auth import User


class FakeScalarResult:
    def __init__(self, rows):
        self._rows = list(rows)

    def all(self):
        return list(self._rows)


class FakeSession:
    def __init__(self, *, scalar_rows=None, scalars_rows=None) -> None:
        self.scalar_rows = list(scalar_rows or [])
        self.scalars_rows = list(scalars_rows or [])
        self.added = []
        self.deleted = []
        self.executed = []
        self.commits = 0

    async def scalar(self, _query):
        return self.scalar_rows.pop(0) if self.scalar_rows else None

    async def scalars(self, _query):
        return FakeScalarResult(self.scalars_rows.pop(0) if self.scalars_rows else [])

    def add(self, obj) -> None:
        self.added.append(obj)

    async def delete(self, obj) -> None:
        self.deleted.append(obj)

    async def execute(self, query) -> None:
        self.executed.append(query)

    async def commit(self) -> None:
        self.commits += 1

    async def refresh(self, obj) -> None:
        if isinstance(obj, Thread) and not obj.thread_id:
            obj.thread_id = "copied-thread"
        now = datetime.now(UTC)
        if getattr(obj, "created_at", None) is None:
            obj.created_at = now
        if getattr(obj, "updated_at", None) is None:
            obj.updated_at = now
        if isinstance(obj, Thread) and getattr(obj, "state_updated_at", None) is None:
            obj.state_updated_at = now

    async def flush(self) -> None:
        for obj in self.added:
            if isinstance(obj, Thread) and not obj.thread_id:
                obj.thread_id = "copied-thread"


class FakeSessionContext:
    def __init__(self, session: FakeSession) -> None:
        self.session = session

    async def __aenter__(self) -> FakeSession:
        return self.session

    async def __aexit__(self, _exc_type, _exc, _tb) -> None:
        return None


class FakeSessionFactory:
    def __init__(self, sessions) -> None:
        self.sessions = list(sessions)

    def __call__(self) -> FakeSessionContext:
        return FakeSessionContext(self.sessions.pop(0))


def _thread(*, thread_id: str = "thread-1", user_id: str = "user-1", config=None, status: str = "idle") -> Thread:
    row = Thread(user_id=user_id, metadata_json={"topic": "alpha"}, config_json=config or {}, status=status)
    row.thread_id = thread_id
    row.created_at = datetime.now(UTC)
    row.updated_at = row.created_at
    row.state_updated_at = row.created_at
    return row


def _run(*, run_id: str = "run-1", thread_id: str = "thread-1", user_id: str = "user-1", status: str = "success") -> Run:
    row = Run(
        thread_id=thread_id,
        assistant_id="assistant-1",
        user_id=user_id,
        status=status,
        input_json={"message": "hello"},
        output_json={"result": "ok"},
        metadata_json={"origin": "test"},
        kwargs_json={"config": {}},
        multitask_strategy="enqueue",
    )
    row.run_id = run_id
    row.created_at = datetime.now(UTC)
    row.updated_at = row.created_at
    return row


def _checkpoint_payload(thread: Thread, checkpoint_id: str, *, values=None) -> dict[str, object]:
    return {
        "values": values or {"output": {"echo": {"message": "hello"}}},
        "next": [],
        "tasks": [],
        "checkpoint": {
            "thread_id": thread.thread_id,
            "checkpoint_ns": "",
            "checkpoint_id": checkpoint_id,
        },
        "metadata": {"user_id": thread.user_id, "status": thread.status},
        "created_at": datetime.now(UTC),
        "parent_checkpoint": None,
        "interrupts": [],
    }


@pytest.mark.asyncio
async def test_best_effort_checkpointer_call_covers_missing_and_not_implemented(monkeypatch: pytest.MonkeyPatch) -> None:
    class MissingMethodCheckpointer:
        pass

    monkeypatch.setattr(
        "agentseek_api.api.threads.db_manager.get_langgraph_checkpointer",
        lambda: MissingMethodCheckpointer(),
    )
    await threads_module._best_effort_checkpointer_call("aprune", ["t1"])

    class RaisingCheckpointer:
        async def aprune(self, *_args, **_kwargs) -> None:
            raise NotImplementedError

    monkeypatch.setattr(
        "agentseek_api.api.threads.db_manager.get_langgraph_checkpointer",
        lambda: RaisingCheckpointer(),
    )
    await threads_module._best_effort_checkpointer_call("aprune", ["t1"])


def test_thread_helper_functions_cover_public_config_and_checkpoint_lookup() -> None:
    assert threads_module._public_thread_config({"visible": True}) == {"visible": True}
    assert threads_module._public_thread_config(None) == {}

    assert threads_module._checkpoint_lookup_payload({"checkpoint_id": "cp-1"}) == "cp-1"
    assert threads_module._checkpoint_lookup_payload({"checkpoint": {"checkpoint_id": "cp-2"}}) == "cp-2"
    assert (
        threads_module._checkpoint_lookup_payload({"config": {"configurable": {"checkpoint_id": "cp-3"}}})
        == "cp-3"
    )
    assert threads_module._checkpoint_lookup_payload({}) is None


@pytest.mark.asyncio
async def test_update_thread_state_persists_checkpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    thread = _thread(config={"visible": True})
    session = FakeSession(scalar_rows=[thread])
    monkeypatch.setattr(
        "agentseek_api.api.threads.db_manager.get_session_factory",
        lambda: FakeSessionFactory([session]),
    )

    async def fake_put_checkpoint(*_args, **_kwargs):
        return "checkpoint-token"

    payload = _checkpoint_payload(thread, "cp-1", values={"manual": True})
    monkeypatch.setattr("agentseek_api.api.threads.put_checkpoint", fake_put_checkpoint)
    monkeypatch.setattr(
        "agentseek_api.api.threads.checkpoint_to_payload",
        lambda _checkpoint: payload,
    )

    response = await threads_module.update_thread_state(
        thread.thread_id,
        {"values": {"manual": True}},
        User(identity=thread.user_id, is_authenticated=True),
    )

    assert response["values"] == {"manual": True}
    assert thread.state_updated_at == payload["created_at"]
    assert session.commits == 1


@pytest.mark.asyncio
async def test_get_thread_state_at_checkpoint_returns_checkpoint_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    thread = _thread()
    session = FakeSession(scalar_rows=[thread])
    monkeypatch.setattr(
        "agentseek_api.api.threads.db_manager.get_session_factory",
        lambda: FakeSessionFactory([session]),
    )

    async def fake_get_checkpoint_by_id(*_args, **_kwargs):
        return "checkpoint-token"

    monkeypatch.setattr("agentseek_api.api.threads.get_checkpoint_by_id", fake_get_checkpoint_by_id)
    monkeypatch.setattr(
        "agentseek_api.api.threads.checkpoint_to_payload",
        lambda _checkpoint: _checkpoint_payload(thread, "cp-1", values={"manual": True}),
    )

    payload = await threads_module.get_thread_state_at_checkpoint(
        thread.thread_id,
        "cp-1",
        User(identity=thread.user_id, is_authenticated=True),
    )
    assert payload["values"] == {"manual": True}


@pytest.mark.asyncio
async def test_get_thread_state_prefers_latest_checkpoint_when_store_order_varies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    thread = _thread()
    old_payload = _checkpoint_payload(thread, "cp-old", values={"manual": "old"})
    old_payload["created_at"] = datetime(2026, 1, 1, tzinfo=UTC)
    new_payload = _checkpoint_payload(thread, "cp-new", values={"manual": "new"})
    new_payload["created_at"] = datetime(2026, 1, 2, tzinfo=UTC)
    session = FakeSession(scalar_rows=[thread], scalars_rows=[[]])
    monkeypatch.setattr(
        "agentseek_api.api.threads.db_manager.get_session_factory",
        lambda: FakeSessionFactory([session]),
    )

    async def fake_list_checkpoints(*_args, **_kwargs):
        return ["old", "new"]

    monkeypatch.setattr("agentseek_api.api.threads.list_checkpoints", fake_list_checkpoints)
    monkeypatch.setattr(
        "agentseek_api.api.threads.checkpoint_to_payload",
        lambda token: old_payload if token == "old" else new_payload,
    )

    payload = await threads_module.get_thread_state(
        thread.thread_id,
        User(identity=thread.user_id, is_authenticated=True),
    )

    assert payload["checkpoint"]["checkpoint_id"] == "cp-new"
    assert payload["values"] == {"manual": "new"}


@pytest.mark.asyncio
async def test_patch_copy_and_delete_thread_routes_cover_new_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    source = _thread()
    source_run = _run(thread_id=source.thread_id)
    patch_session = FakeSession(scalar_rows=[source])
    copy_session = FakeSession(scalar_rows=[source], scalars_rows=[[source_run]])
    delete_session = FakeSession(scalar_rows=[source], scalars_rows=[[source_run.run_id]])
    session_factory = FakeSessionFactory([patch_session, copy_session, delete_session])
    monkeypatch.setattr(
        "agentseek_api.api.threads.db_manager.get_session_factory",
        lambda: session_factory,
    )

    best_effort_calls = []

    async def fake_best_effort(method_name: str, *args, **kwargs) -> None:
        best_effort_calls.append((method_name, args, kwargs))

    copied_checkpoints = []

    async def fake_copy_checkpoints(source_thread_id: str, target_thread_id: str) -> None:
        copied_checkpoints.append((source_thread_id, target_thread_id))

    monkeypatch.setattr("agentseek_api.api.threads._best_effort_checkpointer_call", fake_best_effort)
    monkeypatch.setattr("agentseek_api.api.threads.copy_checkpoints", fake_copy_checkpoints)

    patched = await threads_module.patch_thread(
        source.thread_id,
        ThreadPatch(metadata={"tag": "patched"}),
        User(identity=source.user_id, is_authenticated=True),
    )
    assert patched.metadata == {"topic": "alpha", "tag": "patched"}

    copied = await threads_module.copy_thread(
        source.thread_id,
        User(identity=source.user_id, is_authenticated=True),
    )
    assert copied.thread_id == "copied-thread"
    copied_runs = [obj for obj in copy_session.added if isinstance(obj, Run)]
    assert len(copied_runs) == 1
    assert copied_runs[0].thread_id == "copied-thread"
    assert copied_checkpoints == [(source.thread_id, "copied-thread")]

    deleted = await threads_module.delete_thread(
        source.thread_id,
        User(identity=source.user_id, is_authenticated=True),
    )
    assert deleted.status_code == 204
    assert ("adelete_thread", (source.thread_id,), {}) in best_effort_calls
    assert ("adelete_for_runs", ([source_run.run_id],), {}) in best_effort_calls


@pytest.mark.asyncio
async def test_get_thread_state_at_checkpoint_raises_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    thread = _thread()
    session = FakeSession(scalar_rows=[thread])
    monkeypatch.setattr(
        "agentseek_api.api.threads.db_manager.get_session_factory",
        lambda: FakeSessionFactory([session]),
    )

    async def fake_get_checkpoint_by_id(*_args, **_kwargs):
        return None

    monkeypatch.setattr("agentseek_api.api.threads.get_checkpoint_by_id", fake_get_checkpoint_by_id)

    with pytest.raises(HTTPException, match="Checkpoint not found") as error:
        await threads_module.get_thread_state_at_checkpoint(
            thread.thread_id,
            "missing",
            User(identity=thread.user_id, is_authenticated=True),
        )
    assert error.value.status_code == 404


@pytest.mark.asyncio
async def test_count_threads_returns_exact_count_beyond_page_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    threads = [_thread(thread_id=f"thread-{index}") for index in range(10_001)]
    session_factory = FakeSessionFactory([FakeSession(scalars_rows=[threads])])
    monkeypatch.setattr(
        "agentseek_api.api.threads.db_manager.get_session_factory",
        lambda: session_factory,
    )

    count = await threads_module.count_threads(
        ThreadSearchRequest(),
        User(identity="user-1", is_authenticated=True),
    )

    assert count == 10_001


@pytest.mark.asyncio
async def test_prune_threads_keep_latest_uses_checkpoint_pruner(monkeypatch: pytest.MonkeyPatch) -> None:
    thread = _thread()
    latest_run = _run(run_id="run-latest", thread_id=thread.thread_id)
    older_run = _run(run_id="run-older", thread_id=thread.thread_id)
    older_run.created_at = older_run.created_at.replace(year=older_run.created_at.year - 1)
    session = FakeSession(scalars_rows=[[thread], [latest_run, older_run]])
    monkeypatch.setattr(
        "agentseek_api.api.threads.db_manager.get_session_factory",
        lambda: FakeSessionFactory([session]),
    )

    prune_calls = []

    async def fake_prune_checkpoints(thread_ids: list[str], *, strategy: str) -> None:
        prune_calls.append((thread_ids, strategy))

    monkeypatch.setattr("agentseek_api.api.threads.prune_checkpoints", fake_prune_checkpoints)

    result = await threads_module.prune_threads(
        ThreadPruneRequest(thread_ids=[thread.thread_id], strategy="keep_latest"),
        User(identity=thread.user_id, is_authenticated=True),
    )

    assert result == {"pruned_count": 1}
    assert len(session.executed) == 1
    assert prune_calls == [([thread.thread_id], "keep_latest")]
