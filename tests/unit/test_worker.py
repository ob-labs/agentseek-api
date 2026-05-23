import asyncio

import pytest

from agentseek_api.services.run_jobs import RunExecutionJob
from agentseek_api import worker as worker_module


class FakeQueue:
    def __init__(self, reservations: list[tuple[RunExecutionJob, str] | None], *, acquire_lock: bool = True) -> None:
        self.reservations = reservations
        self.acked: list[str] = []
        self.requeue_calls = 0
        self.closed = False
        self.acquire_lock = acquire_lock
        self.lock_events: list[tuple[str, str, int]] = []

    async def acquire_worker_lock(self, worker_id: str, *, ttl_seconds: int) -> bool:
        self.lock_events.append(("acquire", worker_id, ttl_seconds))
        return self.acquire_lock

    async def renew_worker_lock(self, worker_id: str, *, ttl_seconds: int) -> bool:
        self.lock_events.append(("renew", worker_id, ttl_seconds))
        return True

    async def release_worker_lock(self, worker_id: str) -> None:
        self.lock_events.append(("release", worker_id, 0))

    async def requeue_inflight(self) -> int:
        self.requeue_calls += 1
        return 0

    async def reserve(self, *, timeout_seconds: int) -> tuple[RunExecutionJob, str] | None:
        _ = timeout_seconds
        return self.reservations.pop(0) if self.reservations else None

    async def ack(self, token: str) -> None:
        self.acked.append(token)

    async def close(self) -> None:
        self.closed = True


class BlockingQueue(FakeQueue):
    def __init__(self, release_event) -> None:
        super().__init__([])
        self.release_event = release_event

    async def reserve(self, *, timeout_seconds: int) -> tuple[RunExecutionJob, str] | None:
        _ = timeout_seconds
        await self.release_event.wait()
        return None


@pytest.mark.asyncio
async def test_run_worker_requeues_inflight_and_processes_reserved_job(monkeypatch: pytest.MonkeyPatch) -> None:
    job = RunExecutionJob(
        run_id="r1",
        thread_id="t1",
        user_id="u1",
        payload={"message": "hello"},
        graph_id="default",
    )
    queue = FakeQueue([(job, "token-1")])
    observed: list[RunExecutionJob] = []
    lifecycle: list[str] = []

    async def fake_initialize() -> None:
        lifecycle.append("initialize")

    async def fake_close() -> None:
        lifecycle.append("close")

    async def fake_execute_run_job(received: RunExecutionJob) -> None:
        observed.append(received)

    monkeypatch.setattr(worker_module.settings, "EXECUTOR_BACKEND", "redis")
    monkeypatch.setattr(worker_module.db_manager, "initialize", fake_initialize)
    monkeypatch.setattr(worker_module.db_manager, "close", fake_close)
    monkeypatch.setattr(worker_module, "execute_run_job", fake_execute_run_job)

    processed = await worker_module.run_worker(queue=queue, stop_after_jobs=1, poll_timeout_seconds=0)

    assert processed == 1
    assert queue.requeue_calls == 1
    assert observed == [job]
    assert queue.acked == ["token-1"]
    assert queue.closed is True
    assert lifecycle == ["initialize", "close"]
    assert [event[0] for event in queue.lock_events] == ["acquire", "release"]


@pytest.mark.asyncio
async def test_run_worker_requires_redis_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(worker_module.settings, "EXECUTOR_BACKEND", "inline")

    with pytest.raises(RuntimeError, match="EXECUTOR_BACKEND=redis"):
        await worker_module.run_worker(queue=FakeQueue([]), stop_after_jobs=0, poll_timeout_seconds=0)


@pytest.mark.asyncio
async def test_run_worker_rejects_second_live_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    queue = FakeQueue([], acquire_lock=False)

    async def fake_initialize() -> None:
        return None

    async def fake_close() -> None:
        return None

    monkeypatch.setattr(worker_module.settings, "EXECUTOR_BACKEND", "redis")
    monkeypatch.setattr(worker_module.db_manager, "initialize", fake_initialize)
    monkeypatch.setattr(worker_module.db_manager, "close", fake_close)

    with pytest.raises(RuntimeError, match="Another Redis worker is already active"):
        await worker_module.run_worker(queue=queue, stop_after_jobs=0, poll_timeout_seconds=0)

    assert [event[0] for event in queue.lock_events] == ["acquire"]
    assert queue.requeue_calls == 0
    assert queue.closed is True


@pytest.mark.asyncio
async def test_run_worker_releases_lock_on_graceful_shutdown(monkeypatch: pytest.MonkeyPatch) -> None:
    shutdown_event = asyncio.Event()
    queue = BlockingQueue(shutdown_event)
    lifecycle: list[str] = []

    async def fake_initialize() -> None:
        lifecycle.append("initialize")

    async def fake_close() -> None:
        lifecycle.append("close")

    monkeypatch.setattr(worker_module.settings, "EXECUTOR_BACKEND", "redis")
    monkeypatch.setattr(worker_module.db_manager, "initialize", fake_initialize)
    monkeypatch.setattr(worker_module.db_manager, "close", fake_close)

    task = asyncio.create_task(
        worker_module.run_worker(
            queue=queue,
            poll_timeout_seconds=0,
            shutdown_event=shutdown_event,
        )
    )
    await asyncio.sleep(0)
    shutdown_event.set()

    processed = await task

    assert processed == 0
    assert queue.requeue_calls == 1
    assert queue.closed is True
    assert lifecycle == ["initialize", "close"]
    assert [event[0] for event in queue.lock_events] == ["acquire", "release"]
