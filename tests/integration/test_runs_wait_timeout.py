import pytest


class FastForwardLoop:
    def __init__(self) -> None:
        self._current = 0.0

    def time(self) -> float:
        self._current += 31.0
        return self._current


async def _noop_sleep(_seconds: float) -> None:
    return None


@pytest.mark.asyncio
async def test_wait_timeout_returns_408(monkeypatch: pytest.MonkeyPatch) -> None:
    from fastapi import HTTPException

    from agentseek_api.api import runs as runs_module
    from agentseek_api.models.auth import User

    class FakeRun:
        status = "pending"

    class FakeSession:
        async def scalar(self, _query) -> FakeRun:
            return FakeRun()

    class FakeSessionContext:
        async def __aenter__(self) -> FakeSession:
            return FakeSession()

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

    loop = FastForwardLoop()
    monkeypatch.setattr("agentseek_api.api.runs.db_manager.get_session_factory", lambda: (lambda: FakeSessionContext()))
    monkeypatch.setattr("agentseek_api.api.runs.asyncio.get_event_loop", lambda: loop)
    monkeypatch.setattr("agentseek_api.api.runs.asyncio.sleep", _noop_sleep)

    with pytest.raises(HTTPException) as exc:
        await runs_module.wait_run("thread-timeout", "run-timeout", User(identity="u1"))
    assert exc.value.status_code == 408
