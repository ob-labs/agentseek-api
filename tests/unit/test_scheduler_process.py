from __future__ import annotations

import asyncio

import pytest


class _FakeQueue:
    def __init__(self, *, acquire_result: bool = True, renew_results: list[bool] | None = None) -> None:
        self.acquire_result = acquire_result
        self.renew_results = list(renew_results or [])
        self.acquire_calls: list[tuple[str, int]] = []
        self.renew_calls: list[tuple[str, int]] = []
        self.release_calls: list[str] = []
        self.closed = False

    async def acquire_scheduler_lock(self, scheduler_id: str, *, ttl_seconds: int) -> bool:
        self.acquire_calls.append((scheduler_id, ttl_seconds))
        return self.acquire_result

    async def renew_scheduler_lock(self, scheduler_id: str, *, ttl_seconds: int) -> bool:
        self.renew_calls.append((scheduler_id, ttl_seconds))
        if self.renew_results:
            return self.renew_results.pop(0)
        return True

    async def release_scheduler_lock(self, scheduler_id: str) -> None:
        self.release_calls.append(scheduler_id)

    async def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_maintain_scheduler_lock_sets_event_when_renewal_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    from agentseek_api import scheduler as scheduler_module

    async def no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(scheduler_module.asyncio, "sleep", no_sleep)
    queue = _FakeQueue(renew_results=[False])
    lock_lost = asyncio.Event()

    await scheduler_module._maintain_scheduler_lock(
        queue,
        scheduler_id="scheduler-1",
        ttl_seconds=9,
        lock_lost=lock_lost,
    )

    assert lock_lost.is_set()
    assert queue.renew_calls == [("scheduler-1", 9)]


@pytest.mark.asyncio
async def test_run_scheduler_dispatches_until_shutdown_and_releases_lock(monkeypatch: pytest.MonkeyPatch) -> None:
    from agentseek_api import scheduler as scheduler_module
    from agentseek_api.settings import settings

    queue = _FakeQueue()
    shutdown_event = asyncio.Event()
    observed: dict[str, object] = {"released": False, "closed": False}

    async def fake_initialize() -> None:
        return None

    async def fake_close() -> None:
        observed["db_closed"] = True

    class _FakeLoop:
        def __init__(self) -> None:
            self.added: list[object] = []
            self.removed: list[object] = []

        def add_signal_handler(self, signum, callback) -> None:
            self.added.append(signum)

        def remove_signal_handler(self, signum) -> None:
            self.removed.append(signum)

    loop = _FakeLoop()
    dispatch_calls: list[tuple[int, str]] = []

    async def fake_dispatch_due_crons(*, limit: int, scheduler_id: str):
        dispatch_calls.append((limit, scheduler_id))
        shutdown_event.set()
        return [object(), object()]

    async def no_sleep(_: float) -> None:
        return None

    async def fake_maintain_scheduler_lock(*args, **kwargs) -> None:
        await shutdown_event.wait()

    monkeypatch.setattr(scheduler_module.db_manager, "initialize", fake_initialize)
    monkeypatch.setattr(scheduler_module.db_manager, "close", fake_close)
    monkeypatch.setattr(scheduler_module, "dispatch_due_crons", fake_dispatch_due_crons)
    monkeypatch.setattr(scheduler_module, "_maintain_scheduler_lock", fake_maintain_scheduler_lock)
    monkeypatch.setattr(scheduler_module.asyncio, "sleep", no_sleep)
    monkeypatch.setattr(scheduler_module.asyncio, "get_running_loop", lambda: loop)
    monkeypatch.setattr(settings, "REDIS_SCHEDULER_LOCK_TTL_SECONDS", 15)
    monkeypatch.setattr(settings, "SCHEDULER_POLL_INTERVAL_SECONDS", 5)
    monkeypatch.setattr(settings, "SCHEDULER_CLAIM_LIMIT", 7)

    dispatched = await scheduler_module.run_scheduler(queue=queue, shutdown_event=shutdown_event)

    assert dispatched == 2
    assert len(queue.acquire_calls) == 1
    scheduler_id, ttl_seconds = queue.acquire_calls[0]
    assert ttl_seconds == 15
    assert dispatch_calls == [(7, scheduler_id)]
    assert queue.release_calls == [scheduler_id]
    assert queue.closed is True
    assert observed["db_closed"] is True
    assert loop.added == [scheduler_module.signal.SIGINT, scheduler_module.signal.SIGTERM]
    assert loop.removed == [scheduler_module.signal.SIGINT, scheduler_module.signal.SIGTERM]


@pytest.mark.asyncio
async def test_run_scheduler_raises_when_lock_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    from agentseek_api import scheduler as scheduler_module

    queue = _FakeQueue(acquire_result=False)
    closed = {"db": False}

    async def fake_initialize() -> None:
        return None

    async def fake_close() -> None:
        closed["db"] = True

    class _FakeLoop:
        def add_signal_handler(self, signum, callback) -> None:
            _ = (signum, callback)

        def remove_signal_handler(self, signum) -> None:
            _ = signum

    monkeypatch.setattr(scheduler_module.db_manager, "initialize", fake_initialize)
    monkeypatch.setattr(scheduler_module.db_manager, "close", fake_close)
    monkeypatch.setattr(scheduler_module.asyncio, "get_running_loop", lambda: _FakeLoop())

    with pytest.raises(RuntimeError, match="Another scheduler is already active"):
        await scheduler_module.run_scheduler(queue=queue)

    assert queue.release_calls == []
    assert queue.closed is True
    assert closed["db"] is True


def test_scheduler_main_runs_async_entrypoint(monkeypatch: pytest.MonkeyPatch) -> None:
    from agentseek_api import scheduler as scheduler_module

    observed: dict[str, object] = {}

    async def fake_run_scheduler() -> int:
        return 13

    def fake_asyncio_run(awaitable) -> int:
        observed["awaitable"] = awaitable
        awaitable.close()
        return 13

    monkeypatch.setattr(scheduler_module, "run_scheduler", fake_run_scheduler)
    monkeypatch.setattr(scheduler_module.asyncio, "run", fake_asyncio_run)

    assert scheduler_module.main() == 13
    assert observed["awaitable"] is not None
