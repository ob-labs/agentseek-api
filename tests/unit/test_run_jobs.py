from typing import Any

import pytest

from agentseek_api.settings import settings
from agentseek_api.services import run_jobs as run_jobs_module


class FakeSession:
    def __init__(self, scalar_values: list[object | None], operations: list[str]) -> None:
        self.scalar_values = scalar_values
        self.operations = operations
        self.commits = 0

    async def scalar(self, _query: Any) -> object | None:
        return self.scalar_values.pop(0) if self.scalar_values else None

    async def commit(self) -> None:
        self.operations.append("commit")
        self.commits += 1

    async def refresh(self, _obj: object) -> None:
        return None

    def add(self, _obj: object) -> None:
        return None


class FakeSessionContext:
    def __init__(self, session: FakeSession) -> None:
        self.session = session

    async def __aenter__(self) -> FakeSession:
        return self.session

    async def __aexit__(self, _exc_type, _exc, _tb) -> None:
        return None


class FakeSessionFactory:
    def __init__(self, sessions: list[FakeSession]) -> None:
        self.sessions = sessions

    def __call__(self) -> FakeSessionContext:
        return FakeSessionContext(self.sessions.pop(0))


def _job(*, run_id: str = "r1", thread_id: str = "t1") -> run_jobs_module.RunExecutionJob:
    return run_jobs_module.RunExecutionJob(
        run_id=run_id,
        thread_id=thread_id,
        user_id="u1",
        payload={"message": "hello"},
        graph_id="default",
    )


@pytest.mark.asyncio
async def test_publish_run_event_uses_atomic_redis_append(monkeypatch: pytest.MonkeyPatch) -> None:
    published: list[tuple[str, str, int | None, dict[str, Any]]] = []
    monkeypatch.setattr(settings, "EXECUTOR_BACKEND", "redis")

    async def fake_append(run_id: str, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        assert run_id == "run-1"
        return 7, payload

    async def unexpected_next_seq(_run_id: str) -> int:
        raise AssertionError("Redis sequence allocation must be part of the append")

    monkeypatch.setattr(run_jobs_module, "append_redis_run_stream_event", fake_append, raising=False)
    monkeypatch.setattr(run_jobs_module, "next_run_stream_seq", unexpected_next_seq)
    monkeypatch.setattr(
        run_jobs_module.run_broker,
        "publish",
        lambda run_id, event, *, seq=None, **payload: (
            published.append((run_id, event, seq, payload)) or (seq, {"event": event, **payload})
        ),
    )

    result = await run_jobs_module._publish_run_event("run-1", "message", data="hello")

    assert result == (7, {"event": "message", "data": "hello"})
    assert published == [("run-1", "message", 7, {"data": "hello"})]


@pytest.mark.asyncio
async def test_publish_lifecycle_uses_atomic_redis_append(monkeypatch: pytest.MonkeyPatch) -> None:
    published: list[tuple[str, str]] = []
    monkeypatch.setattr(settings, "EXECUTOR_BACKEND", "redis")

    async def fake_apublish(thread_id: str, **payload: Any) -> dict[str, Any]:
        published.append((thread_id, payload["event"]))
        return {"seq": 3, "method": "lifecycle"}

    async def unexpected_next_seq(_thread_id: str) -> int:
        raise AssertionError("Redis lifecycle sequence must be allocated atomically")

    monkeypatch.setattr(run_jobs_module, "apublish_lifecycle_event", fake_apublish, raising=False)
    monkeypatch.setattr(run_jobs_module, "next_thread_stream_seq", unexpected_next_seq)

    await run_jobs_module._publish_lifecycle("thread-1", event="completed", session=FakeSession([], []))

    assert published == [("thread-1", "completed")]


@pytest.mark.asyncio
async def test_terminal_run_event_uses_atomic_redis_append_without_sql_helper(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, str, bool, dict[str, Any]]] = []
    monkeypatch.setattr(settings, "EXECUTOR_BACKEND", "redis")
    publish_terminal = getattr(run_jobs_module, "_publish_terminal_run_event", None)

    async def fake_publish(run_id: str, event: str, *, persist: bool = True, **payload: Any):
        calls.append((run_id, event, persist, payload))
        return 4, {"event": event, **payload}

    async def unexpected_sql_helper(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("Redis terminal events must not use the SQL session helper")

    monkeypatch.setattr(run_jobs_module, "_publish_run_event", fake_publish)
    monkeypatch.setattr(run_jobs_module, "add_run_stream_event_to_session", unexpected_sql_helper)

    assert callable(publish_terminal)
    await publish_terminal(FakeSession([], []), "run-1", status="success")

    assert calls == [("run-1", "end", True, {"status": "success"})]


@pytest.mark.asyncio
async def test_execute_run_job_skips_terminal_runs(run_storage, monkeypatch):
    from agentseek_api.core.orm import Run
    async with run_storage() as session:
        session.add(Run(run_id="r1", thread_id="t1", assistant_id="a1", user_id="u1", status="success"))
        await session.commit()
    async def unexpected(**kwargs):
        raise AssertionError("Terminal runs must not execute")
    monkeypatch.setattr(run_jobs_module, "execute_run", unexpected)
    await run_jobs_module.execute_run_job(_job())
    assert run_jobs_module.run_broker.snapshot_records("r1") == []


@pytest.mark.parametrize("interrupted", [False, True])
async def test_execute_run_job_commits_terminal_state_and_both_logs(run_storage, monkeypatch, interrupted):
    from sqlalchemy import select
    from agentseek_api.core.orm import Run, RunStreamEvent, ThreadStreamEvent
    async with run_storage() as session:
        session.add(Run(run_id="r1", thread_id="t1", assistant_id="a1", user_id="u1", status="pending"))
        await session.commit()
    async def execute(**kwargs):
        return run_jobs_module.RunExecutionResult(output={"ok": True}, interrupted=interrupted, interrupts=[])
    monkeypatch.setattr(run_jobs_module, "execute_run", execute)
    await run_jobs_module.execute_run_job(_job())
    async with run_storage() as session:
        row = await session.get(Run, "r1")
        assert row.status == ("interrupted" if interrupted else "success")
        assert row.output_json == {"ok": True}
        records = list(await session.scalars(select(RunStreamEvent).order_by(RunStreamEvent.seq)))
        assert [record.event for record in records] == ["start", "end"]
        lifecycle = await session.scalar(select(ThreadStreamEvent))
        assert lifecycle.payload_json["params"]["data"]["event"] == ("interrupted" if interrupted else "completed")
    assert run_jobs_module.run_broker.snapshot_records("r1")[-1][1]["event"] == "end"


async def test_execute_run_job_does_not_recreate_stream_for_deleted_run(run_storage, monkeypatch):
    async def unexpected(**kwargs):
        raise AssertionError("Deleted runs must not execute")
    monkeypatch.setattr(run_jobs_module, "execute_run", unexpected)
    await run_jobs_module.execute_run_job(_job())
    assert run_jobs_module.thread_protocol_broker.snapshot_records("t1") == []
    assert run_jobs_module.run_broker.snapshot_records("r1") == []


def test_from_payload_rejects_unsupported_kind():
    with pytest.raises(ValueError, match="Unsupported run job kind"):
        run_jobs_module.RunExecutionJob.from_payload({"kind": "unknown"})
