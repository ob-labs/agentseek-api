from collections.abc import Awaitable, Callable
from typing import Any

import pytest

from agentseek_api.models.auth import User
from agentseek_api.services import run_preparation as run_prep_module


class FakeSession:
    def __init__(self, scalar_values: list[object | None]) -> None:
        self.scalar_values = scalar_values
        self.added: list[object] = []
        self.commits = 0

    async def scalar(self, _query: Any) -> object | None:
        return self.scalar_values.pop(0) if self.scalar_values else None

    def add(self, obj: object) -> None:
        self.added.append(obj)

    async def commit(self) -> None:
        self.commits += 1

    async def refresh(self, _obj: object) -> None:
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


class InlineExecutor:
    async def submit(self, func: Callable[[], Awaitable[None]]) -> None:
        await func()


class DeferredExecutor:
    def __init__(self) -> None:
        self.submitted: list[Callable[[], Awaitable[None]]] = []

    async def submit(self, func: Callable[[], Awaitable[None]]) -> None:
        self.submitted.append(func)


@pytest.mark.asyncio
async def test_prepare_run_raises_when_thread_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    session_factory = FakeSessionFactory([FakeSession([None])])
    monkeypatch.setattr("agentseek_api.services.run_preparation.db_manager.get_session_factory", lambda: session_factory)

    with pytest.raises(ValueError, match="Thread not found"):
        await run_prep_module.prepare_and_submit_run(
            thread_id="t1",
            assistant_id="a1",
            payload={"x": 1},
            user=User(identity="u1", is_authenticated=True),
        )


@pytest.mark.asyncio
async def test_prepare_run_raises_when_assistant_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    session_factory = FakeSessionFactory([FakeSession([object(), None])])
    monkeypatch.setattr("agentseek_api.services.run_preparation.db_manager.get_session_factory", lambda: session_factory)

    with pytest.raises(ValueError, match="Assistant not found"):
        await run_prep_module.prepare_and_submit_run(
            thread_id="t1",
            assistant_id="a1",
            payload={"x": 1},
            user=User(identity="u1", is_authenticated=True),
        )


@pytest.mark.asyncio
async def test_prepare_run_sets_error_status_when_execute_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_assistant = type("FakeAssistant", (), {"graph_id": "stress_test"})()
    create_session = FakeSession([object(), fake_assistant])
    db_run = type("DbRun", (), {"run_id": "r1", "status": "pending", "output_json": None, "last_error": None})()
    exec_session = FakeSession([db_run])
    reload_session = FakeSession([db_run])
    session_factory = FakeSessionFactory([create_session, exec_session, reload_session])

    monkeypatch.setattr("agentseek_api.services.run_preparation.db_manager.get_session_factory", lambda: session_factory)
    monkeypatch.setattr("agentseek_api.services.run_preparation.get_executor", lambda: InlineExecutor())

    captured: dict[str, Any] = {}

    async def failing_execute_run(
        *,
        thread_id: str,
        run_id: str,
        payload: dict,
        graph_id: str | None = None,
        resume: Any = None,
    ) -> dict:
        captured["graph_id"] = graph_id
        _ = (thread_id, run_id, payload, resume)
        raise RuntimeError("boom")

    events: list[tuple[str, str, dict[str, Any]]] = []
    monkeypatch.setattr("agentseek_api.services.run_preparation.execute_run", failing_execute_run)
    monkeypatch.setattr(
        "agentseek_api.services.run_preparation.run_broker.publish",
        lambda run_id, event, **payload: events.append((run_id, event, payload)),
    )

    run = await run_prep_module.prepare_and_submit_run(
        thread_id="t1",
        assistant_id="a1",
        payload={"x": 1},
        user=User(identity="u1", is_authenticated=True),
    )

    assert run.status == "error"
    assert db_run.status == "error"
    assert db_run.last_error == "boom"
    assert events[-1][1] == "end"
    assert events[-1][2]["status"] == "error"
    assert captured["graph_id"] == "stress_test"


@pytest.mark.asyncio
async def test_resume_run_marks_row_pending_before_background_execution(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_assistant = type("FakeAssistant", (), {"assistant_id": "a1", "graph_id": "subgraph_hitl_agent"})()
    db_run = type(
        "DbRun",
        (),
        {
            "run_id": "r1",
            "thread_id": "t1",
            "assistant_id": "a1",
            "user_id": "u1",
            "status": "interrupted",
            "input_json": {"foo": "hello "},
            "output_json": {"interrupts": [{"value": "Provide value:"}], "interrupted": True},
            "last_error": None,
        },
    )()
    load_session = FakeSession([db_run, fake_assistant])
    reload_session = FakeSession([db_run])
    session_factory = FakeSessionFactory([load_session, reload_session])
    executor = DeferredExecutor()

    monkeypatch.setattr("agentseek_api.services.run_preparation.db_manager.get_session_factory", lambda: session_factory)
    monkeypatch.setattr("agentseek_api.services.run_preparation.get_executor", lambda: executor)

    run = await run_prep_module.resume_run(
        thread_id="t1",
        run_id="r1",
        resume="world",
        user=User(identity="u1", is_authenticated=True),
    )

    assert run.status == "pending"
    assert db_run.status == "pending"
    assert load_session.commits == 1
    assert len(executor.submitted) == 1
