from datetime import UTC, datetime
from typing import Any

import pytest

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
async def test_execute_run_job_skips_terminal_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    operations: list[str] = []
    db_run = type("DbRun", (), {"run_id": "r1", "status": "success", "last_error": None})()
    session_factory = FakeSessionFactory([FakeSession([db_run], operations)])

    async def unexpected_execute_run(**_kwargs: Any) -> run_jobs_module.RunExecutionResult:
        raise AssertionError("Terminal runs should not be re-executed")

    monkeypatch.setattr(run_jobs_module.db_manager, "get_session_factory", lambda: session_factory)
    monkeypatch.setattr(run_jobs_module, "execute_run", unexpected_execute_run)

    await run_jobs_module.execute_run_job(_job())

    assert operations == []


@pytest.mark.asyncio
async def test_execute_run_job_persists_terminal_lifecycle_before_final_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operations: list[str] = []
    db_run = type(
        "DbRun",
        (),
        {
            "run_id": "r1",
            "status": "pending",
            "output_json": None,
            "last_error": None,
            "updated_at": datetime.now(UTC),
        },
    )()
    fake_thread = type("FakeThread", (), {"thread_id": "t1", "status": "idle", "state_updated_at": None})()
    session_factory = FakeSessionFactory([FakeSession([db_run, fake_thread, fake_thread], operations)])

    async def successful_execute_run(**_kwargs: Any) -> run_jobs_module.RunExecutionResult:
        return run_jobs_module.RunExecutionResult(output={"ok": True}, interrupted=False, interrupts=[])

    async def fake_publish_run_event(_run_id: str, event: str, *, persist: bool = True, **payload: Any) -> tuple[int, dict[str, Any]]:
        _ = persist
        operations.append(f"publish:{event}")
        return (1 if event == "start" else 2), {"event": event, **payload}

    async def fake_persist_thread_snapshot(_thread_id: str) -> None:
        return None

    async def fake_add_run_stream_event_to_session(
        _session: FakeSession,
        _run_id: str,
        *,
        seq: int,
        payload: dict[str, Any],
    ) -> None:
        operations.append(f"persist:run:{payload['event']}:{seq}")

    def fake_publish_lifecycle_event(
        _thread_id: str,
        *,
        event: str,
        graph_name: str | None = None,
        error: str | None = None,
        namespace: list[str] | None = None,
        persist: bool = True,
    ) -> dict[str, Any]:
        _ = (graph_name, error, namespace)
        operations.append(f"publish:lifecycle:{event}:{persist}")
        return {
            "seq": 3,
            "method": "lifecycle",
            "params": {"namespace": [], "timestamp": 1, "data": {"event": event}},
        }

    async def fake_add_thread_stream_event_to_session(
        _session: FakeSession,
        _thread_id: str,
        *,
        seq: int,
        payload: dict[str, Any],
    ) -> None:
        operations.append(f"persist:thread:{payload['params']['data']['event']}:{seq}")

    monkeypatch.setattr(run_jobs_module.db_manager, "get_session_factory", lambda: session_factory)
    monkeypatch.setattr(run_jobs_module, "execute_run", successful_execute_run)
    monkeypatch.setattr(run_jobs_module, "_publish_run_event", fake_publish_run_event)
    monkeypatch.setattr(run_jobs_module, "_persist_thread_snapshot", fake_persist_thread_snapshot)
    monkeypatch.setattr(run_jobs_module, "add_run_stream_event_to_session", fake_add_run_stream_event_to_session)
    monkeypatch.setattr(run_jobs_module, "publish_lifecycle_event", fake_publish_lifecycle_event)
    monkeypatch.setattr(run_jobs_module, "add_thread_stream_event_to_session", fake_add_thread_stream_event_to_session)

    await run_jobs_module.execute_run_job(_job())

    assert operations == [
        "commit",
        "publish:start",
        "publish:end",
        "persist:run:end:2",
        "publish:lifecycle:completed:False",
        "persist:thread:completed:3",
        "commit",
    ]
