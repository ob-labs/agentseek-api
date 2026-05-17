from datetime import UTC, datetime

import pytest
from fastapi import HTTPException

from agentseek_api.api import threads as threads_module
from agentseek_api.core.orm import Run, Thread
from agentseek_api.models.api import ThreadPatch, ThreadSearchRequest, ThreadPruneRequest
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
        output_json={"result": "ok", "interrupts": [{"value": "wait"}]},
        metadata_json={"origin": "test"},
        kwargs_json={"config": {}},
        multitask_strategy="enqueue",
    )
    row.run_id = run_id
    row.created_at = datetime.now(UTC)
    row.updated_at = row.created_at
    return row


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


def test_thread_helper_functions_cover_manual_checkpoint_paths() -> None:
    thread = _thread(
        config={
            "visible": True,
            threads_module._MANUAL_CHECKPOINTS_KEY: [
                {
                    "checkpoint_id": "manual-1",
                    "values": "not-a-dict",
                    "interrupts": "not-a-list",
                    "metadata": "not-a-dict",
                    "created_at": "not-a-datetime",
                }
            ],
        }
    )
    run = _run()

    public_config = threads_module._public_thread_config(thread.config_json)
    assert public_config == {"visible": True}
    assert threads_module._public_thread_config(None) == {}

    checkpoints = threads_module._manual_checkpoints(thread)
    assert len(checkpoints) == 1

    payload = threads_module._manual_checkpoint_payload(thread, checkpoints[0])
    assert payload["values"] == {}
    assert payload["interrupts"] == []
    assert payload["metadata"]["user_id"] == thread.user_id

    run_payload = threads_module._thread_state_payload(thread=thread, run=run)
    assert run_payload["checkpoint"]["checkpoint_id"] == run.run_id
    assert run_payload["interrupts"] == [{"value": "wait"}]

    empty_payload = threads_module._thread_state_payload(thread=thread, run=None)
    assert empty_payload["checkpoint"]["checkpoint_id"] == thread.thread_id
    assert empty_payload["values"] == {}

    latest_manual = threads_module._latest_state_payload(thread=thread, runs=[])
    assert latest_manual["checkpoint"]["checkpoint_id"] == "manual-1"

    newer_thread = _thread()
    older_run = _run(thread_id=newer_thread.thread_id)
    older_run.created_at = older_run.created_at.replace(year=older_run.created_at.year - 1)
    latest_run = threads_module._latest_state_payload(thread=newer_thread, runs=[older_run])
    assert latest_run["checkpoint"]["checkpoint_id"] == older_run.run_id

    assert threads_module._state_payload_created_at({"created_at": "invalid"}).tzinfo is UTC


@pytest.mark.asyncio
async def test_update_thread_state_persists_manual_checkpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    thread = _thread(config={"visible": True})
    session = FakeSession(scalar_rows=[thread])
    monkeypatch.setattr(
        "agentseek_api.api.threads.db_manager.get_session_factory",
        lambda: FakeSessionFactory([session]),
    )

    response = await threads_module.update_thread_state(
        thread.thread_id,
        {"values": {"manual": True}},
        User(identity=thread.user_id, is_authenticated=True),
    )

    assert response["values"] == {"manual": True}
    checkpoints = thread.config_json[threads_module._MANUAL_CHECKPOINTS_KEY]
    assert len(checkpoints) == 1
    assert checkpoints[0]["values"] == {"manual": True}
    assert thread.config_json["visible"] is True
    assert session.commits == 1


@pytest.mark.asyncio
async def test_get_thread_state_at_checkpoint_returns_manual_checkpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    thread = _thread(
        config={
            threads_module._MANUAL_CHECKPOINTS_KEY: [
                {
                    "checkpoint_id": "manual-1",
                    "values": {"manual": True},
                    "interrupts": [],
                    "metadata": {"user_id": "user-1", "status": "idle"},
                    "created_at": datetime.now(UTC).isoformat(),
                }
            ]
        }
    )
    session = FakeSession(scalar_rows=[thread, None])
    monkeypatch.setattr(
        "agentseek_api.api.threads.db_manager.get_session_factory",
        lambda: FakeSessionFactory([session]),
    )

    payload = await threads_module.get_thread_state_at_checkpoint(
        thread.thread_id,
        "manual-1",
        User(identity=thread.user_id, is_authenticated=True),
    )
    assert payload["values"] == {"manual": True}


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

    monkeypatch.setattr("agentseek_api.api.threads._best_effort_checkpointer_call", fake_best_effort)

    patched = await threads_module.patch_thread(
        source.thread_id,
        ThreadPatch(metadata={"topic": "patched"}),
        User(identity=source.user_id, is_authenticated=True),
    )
    assert patched.metadata == {"topic": "patched"}

    copied = await threads_module.copy_thread(
        source.thread_id,
        User(identity=source.user_id, is_authenticated=True),
    )
    assert copied.thread_id == "copied-thread"
    copied_runs = [obj for obj in copy_session.added if isinstance(obj, Run)]
    assert len(copied_runs) == 1
    assert copied_runs[0].thread_id == "copied-thread"

    deleted = await threads_module.delete_thread(
        source.thread_id,
        User(identity=source.user_id, is_authenticated=True),
    )
    assert deleted.status_code == 204
    assert ("acopy_thread", (source.thread_id, "copied-thread"), {}) in best_effort_calls
    assert ("adelete_thread", (source.thread_id,), {}) in best_effort_calls
    assert ("adelete_for_runs", ([source_run.run_id],), {}) in best_effort_calls


@pytest.mark.asyncio
async def test_get_thread_state_at_checkpoint_raises_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    thread = _thread()
    session = FakeSession(scalar_rows=[thread, None])
    monkeypatch.setattr(
        "agentseek_api.api.threads.db_manager.get_session_factory",
        lambda: FakeSessionFactory([session]),
    )

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
async def test_prune_threads_keep_latest_drops_older_manual_checkpoints(monkeypatch: pytest.MonkeyPatch) -> None:
    thread = _thread(
        config={
            threads_module._MANUAL_CHECKPOINTS_KEY: [
                {
                    "checkpoint_id": "manual-older",
                    "values": {"manual": 1},
                    "interrupts": [],
                    "metadata": {"user_id": "user-1", "status": "idle"},
                    "created_at": "2026-01-01T00:00:00+00:00",
                },
                {
                    "checkpoint_id": "manual-latest",
                    "values": {"manual": 2},
                    "interrupts": [],
                    "metadata": {"user_id": "user-1", "status": "idle"},
                    "created_at": "2026-01-02T00:00:00+00:00",
                },
            ]
        }
    )
    latest_run = _run(run_id="run-latest", thread_id=thread.thread_id)
    older_run = _run(run_id="run-older", thread_id=thread.thread_id)
    older_run.created_at = older_run.created_at.replace(year=older_run.created_at.year - 1)
    session = FakeSession(scalars_rows=[[thread], [latest_run, older_run]])
    monkeypatch.setattr(
        "agentseek_api.api.threads.db_manager.get_session_factory",
        lambda: FakeSessionFactory([session]),
    )

    best_effort_calls = []

    async def fake_best_effort(method_name: str, *args, **kwargs) -> None:
        best_effort_calls.append((method_name, args, kwargs))

    monkeypatch.setattr("agentseek_api.api.threads._best_effort_checkpointer_call", fake_best_effort)

    result = await threads_module.prune_threads(
        ThreadPruneRequest(thread_ids=[thread.thread_id], strategy="keep_latest"),
        User(identity=thread.user_id, is_authenticated=True),
    )

    assert result == {"pruned_count": 1}
    checkpoints = thread.config_json[threads_module._MANUAL_CHECKPOINTS_KEY]
    assert [item["checkpoint_id"] for item in checkpoints] == ["manual-latest"]
    assert len(session.executed) == 1
    assert ("aprune", ([thread.thread_id],), {"strategy": "keep_latest"}) in best_effort_calls
