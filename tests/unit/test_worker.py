import asyncio
from types import SimpleNamespace

import pytest

from agentseek_api.settings import Settings
from agentseek_api.services.run_jobs import RunExecutionJob
from agentseek_api import worker as worker_module


class FakeQueue:
    def __init__(
        self,
        reservations: list[tuple[RunExecutionJob, str] | None],
        *,
        acquire_lock: bool = True,
        ack_allowed: bool = True,
    ) -> None:
        self.reservations = reservations
        self.acked: list[str] = []
        self.requeue_calls = 0
        self.closed = False
        self.acquire_lock = acquire_lock
        self.ack_allowed = ack_allowed
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

    async def reserve(
        self, *, timeout_seconds: int
    ) -> tuple[RunExecutionJob, str] | None:
        _ = timeout_seconds
        return self.reservations.pop(0) if self.reservations else None

    async def ack(self, token: str) -> None:
        self.acked.append(token)

    async def ack_if_worker_lock_owner(self, worker_id: str, token: str) -> bool:
        _ = worker_id
        if not self.ack_allowed:
            return False
        self.acked.append(token)
        return True

    async def close(self) -> None:
        self.closed = True


class BlockingQueue(FakeQueue):
    def __init__(self, release_event) -> None:
        super().__init__([])
        self.release_event = release_event

    async def reserve(
        self, *, timeout_seconds: int
    ) -> tuple[RunExecutionJob, str] | None:
        _ = timeout_seconds
        await self.release_event.wait()
        return None


class ControlledReservationQueue(FakeQueue):
    def __init__(self, reservation: tuple[RunExecutionJob, str]) -> None:
        super().__init__([])
        self.reservation = reservation
        self.reserve_started = asyncio.Event()
        self.release_reservation = asyncio.Event()

    async def reserve(self, *, timeout_seconds: int) -> tuple[RunExecutionJob, str]:
        _ = timeout_seconds
        self.reserve_started.set()
        await self.release_reservation.wait()
        return self.reservation


class BlockingAfterReservationsQueue(FakeQueue):
    def __init__(self, reservations: list[tuple[RunExecutionJob, str]]) -> None:
        super().__init__(reservations)
        self.blocked_reserve_started = asyncio.Event()
        self.blocked_reserve_cancelled = asyncio.Event()

    async def reserve(
        self, *, timeout_seconds: int
    ) -> tuple[RunExecutionJob, str] | None:
        _ = timeout_seconds
        if self.reservations:
            return self.reservations.pop(0)
        self.blocked_reserve_started.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            self.blocked_reserve_cancelled.set()
            raise
        return None


def _job(index: str) -> RunExecutionJob:
    return RunExecutionJob(
        run_id=f"r{index}",
        thread_id=f"t{index}",
        user_id="u1",
        payload={"message": index},
        graph_id="default",
    )


def _worker_settings(concurrent_jobs: int) -> SimpleNamespace:
    return SimpleNamespace(
        EXECUTOR_BACKEND="redis",
        REDIS_WORKER_LOCK_TTL_SECONDS=30,
        REDIS_WORKER_POLL_TIMEOUT_SECONDS=1,
        WORKER_CONCURRENT_JOBS=concurrent_jobs,
    )


def test_worker_concurrent_jobs_defaults_to_ten_and_reads_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("WORKER_CONCURRENT_JOBS", raising=False)
    assert Settings().WORKER_CONCURRENT_JOBS == 10

    monkeypatch.setenv("WORKER_CONCURRENT_JOBS", "3")
    assert Settings().WORKER_CONCURRENT_JOBS == 3


@pytest.mark.asyncio
async def test_run_worker_requeues_inflight_and_processes_reserved_job(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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

    processed = await worker_module.run_worker(
        queue=queue, stop_after_jobs=1, poll_timeout_seconds=0
    )

    assert processed == 1
    assert queue.requeue_calls == 1
    assert observed == [job]
    assert queue.acked == ["token-1"]
    assert queue.closed is True
    assert lifecycle == ["initialize", "close"]
    assert [event[0] for event in queue.lock_events] == ["acquire", "release"]


@pytest.mark.asyncio
async def test_run_worker_processes_jobs_up_to_configured_concurrency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queue = FakeQueue(
        [(_job("1"), "token-1"), (_job("2"), "token-2"), (_job("3"), "token-3")]
    )
    release_jobs = asyncio.Event()
    two_started = asyncio.Event()
    started: list[str] = []
    active = 0
    peak_active = 0

    async def fake_execute_run_job(job: RunExecutionJob) -> None:
        nonlocal active, peak_active
        active += 1
        peak_active = max(peak_active, active)
        started.append(job.run_id)
        if len(started) == 2:
            two_started.set()
        await release_jobs.wait()
        active -= 1

    async def noop() -> None:
        return None

    monkeypatch.setattr(worker_module, "settings", _worker_settings(2))
    monkeypatch.setattr(worker_module.db_manager, "initialize", noop)
    monkeypatch.setattr(worker_module.db_manager, "close", noop)
    monkeypatch.setattr(worker_module, "execute_run_job", fake_execute_run_job)

    task = asyncio.create_task(
        worker_module.run_worker(queue=queue, stop_after_jobs=3, poll_timeout_seconds=0)
    )
    processed: int | None = None
    try:
        await asyncio.wait_for(two_started.wait(), timeout=1)
        await asyncio.sleep(0)
        assert started == ["r1", "r2"]
    finally:
        release_jobs.set()
        processed = await task

    assert processed == 3
    assert peak_active == 2
    assert sorted(queue.acked) == ["token-1", "token-2", "token-3"]


@pytest.mark.asyncio
async def test_run_worker_starts_queued_job_when_one_slot_becomes_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queue = FakeQueue(
        [(_job("1"), "token-1"), (_job("2"), "token-2"), (_job("3"), "token-3")]
    )
    first_two_started = asyncio.Event()
    release_short_job = asyncio.Event()
    release_long_job = asyncio.Event()
    queued_job_started = asyncio.Event()
    started: list[str] = []

    async def fake_execute_run_job(job: RunExecutionJob) -> None:
        started.append(job.run_id)
        if len(started) == 2:
            first_two_started.set()
        if job.run_id == "r1":
            await release_short_job.wait()
        elif job.run_id == "r2":
            await release_long_job.wait()
        else:
            queued_job_started.set()

    async def noop() -> None:
        return None

    monkeypatch.setattr(worker_module, "settings", _worker_settings(2))
    monkeypatch.setattr(worker_module.db_manager, "initialize", noop)
    monkeypatch.setattr(worker_module.db_manager, "close", noop)
    monkeypatch.setattr(worker_module, "execute_run_job", fake_execute_run_job)

    task = asyncio.create_task(
        worker_module.run_worker(queue=queue, stop_after_jobs=3, poll_timeout_seconds=0)
    )
    try:
        await asyncio.wait_for(first_two_started.wait(), timeout=1)
        release_short_job.set()
        await asyncio.wait_for(queued_job_started.wait(), timeout=1)
        assert release_long_job.is_set() is False
        assert started == ["r1", "r2", "r3"]
    finally:
        release_short_job.set()
        release_long_job.set()
        processed = await asyncio.wait_for(task, timeout=1)

    assert processed == 3
    assert sorted(queue.acked) == ["token-1", "token-2", "token-3"]


@pytest.mark.asyncio
async def test_run_worker_requires_redis_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(worker_module.settings, "EXECUTOR_BACKEND", "inline")

    with pytest.raises(RuntimeError, match="EXECUTOR_BACKEND=redis"):
        await worker_module.run_worker(
            queue=FakeQueue([]), stop_after_jobs=0, poll_timeout_seconds=0
        )


@pytest.mark.asyncio
async def test_run_worker_rejects_second_live_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queue = FakeQueue([], acquire_lock=False)

    async def fake_initialize() -> None:
        return None

    async def fake_close() -> None:
        return None

    monkeypatch.setattr(worker_module.settings, "EXECUTOR_BACKEND", "redis")
    monkeypatch.setattr(worker_module.db_manager, "initialize", fake_initialize)
    monkeypatch.setattr(worker_module.db_manager, "close", fake_close)

    with pytest.raises(RuntimeError, match="Another Redis worker is already active"):
        await worker_module.run_worker(
            queue=queue, stop_after_jobs=0, poll_timeout_seconds=0
        )

    assert [event[0] for event in queue.lock_events] == ["acquire"]
    assert queue.requeue_calls == 0
    assert queue.closed is True


@pytest.mark.asyncio
async def test_run_worker_releases_lock_on_graceful_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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


@pytest.mark.asyncio
async def test_run_worker_drains_active_jobs_before_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queue = FakeQueue([(_job("1"), "token-1"), (_job("2"), "token-2")])
    shutdown_event = asyncio.Event()
    release_jobs = asyncio.Event()
    two_started = asyncio.Event()
    started = 0
    lifecycle: list[str] = []

    async def fake_execute_run_job(_job: RunExecutionJob) -> None:
        nonlocal started
        started += 1
        if started == 2:
            two_started.set()
        await release_jobs.wait()

    async def fake_initialize() -> None:
        lifecycle.append("initialize")

    async def fake_close() -> None:
        lifecycle.append("close")

    monkeypatch.setattr(worker_module, "settings", _worker_settings(2))
    monkeypatch.setattr(worker_module.db_manager, "initialize", fake_initialize)
    monkeypatch.setattr(worker_module.db_manager, "close", fake_close)
    monkeypatch.setattr(worker_module, "execute_run_job", fake_execute_run_job)

    task = asyncio.create_task(
        worker_module.run_worker(
            queue=queue,
            poll_timeout_seconds=0,
            shutdown_event=shutdown_event,
        )
    )
    processed: int | None = None
    try:
        await asyncio.wait_for(two_started.wait(), timeout=1)
        shutdown_event.set()
        await asyncio.sleep(0)
        assert task.done() is False
        assert queue.closed is False
    finally:
        shutdown_event.set()
        release_jobs.set()
        processed = await asyncio.wait_for(task, timeout=1)

    assert processed == 2
    assert sorted(queue.acked) == ["token-1", "token-2"]
    assert lifecycle == ["initialize", "close"]


@pytest.mark.asyncio
async def test_run_worker_cancels_sibling_jobs_when_one_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queue = FakeQueue([(_job("1"), "token-1"), (_job("2"), "token-2")])
    first_started = asyncio.Event()
    first_cancelled = asyncio.Event()
    lifecycle: list[str] = []

    async def fake_execute_run_job(job: RunExecutionJob) -> None:
        if job.run_id == "r1":
            first_started.set()
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                first_cancelled.set()
                raise
        await first_started.wait()
        raise RuntimeError("job failed")

    async def fake_initialize() -> None:
        lifecycle.append("initialize")

    async def fake_close() -> None:
        lifecycle.append("close")

    monkeypatch.setattr(worker_module, "settings", _worker_settings(2))
    monkeypatch.setattr(worker_module.db_manager, "initialize", fake_initialize)
    monkeypatch.setattr(worker_module.db_manager, "close", fake_close)
    monkeypatch.setattr(worker_module, "execute_run_job", fake_execute_run_job)

    with pytest.raises(RuntimeError, match="job failed"):
        await asyncio.wait_for(
            worker_module.run_worker(
                queue=queue, stop_after_jobs=2, poll_timeout_seconds=0
            ),
            timeout=1,
        )

    assert first_cancelled.is_set()
    assert queue.acked == []
    assert queue.closed is True
    assert lifecycle == ["initialize", "close"]


@pytest.mark.asyncio
async def test_run_worker_does_not_ack_after_lease_transfer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queue = FakeQueue([(_job("1"), "token-1")], ack_allowed=False)

    async def noop() -> None:
        return None

    monkeypatch.setattr(worker_module, "settings", _worker_settings(1))
    monkeypatch.setattr(worker_module.db_manager, "initialize", noop)
    monkeypatch.setattr(worker_module.db_manager, "close", noop)
    monkeypatch.setattr(worker_module, "execute_run_job", lambda _job: noop())

    with pytest.raises(RuntimeError, match="lost its active lease"):
        await worker_module.run_worker(
            queue=queue, stop_after_jobs=1, poll_timeout_seconds=0
        )

    assert queue.acked == []


@pytest.mark.asyncio
async def test_worker_lock_heartbeat_surfaces_renewal_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queue = FakeQueue([])
    lock_lost = asyncio.Event()

    async def immediate_sleep(_seconds: float) -> None:
        return None

    async def fail_renewal(worker_id: str, *, ttl_seconds: int) -> bool:
        _ = worker_id, ttl_seconds
        raise ConnectionError("redis unavailable")

    monkeypatch.setattr(worker_module.asyncio, "sleep", immediate_sleep)
    monkeypatch.setattr(queue, "renew_worker_lock", fail_renewal)

    with pytest.raises(RuntimeError, match="lease renewal failed"):
        await worker_module._maintain_worker_lock(
            queue,
            worker_id="worker-1",
            ttl_seconds=30,
            lock_lost=lock_lost,
        )

    assert lock_lost.is_set()


@pytest.mark.asyncio
async def test_run_worker_cancels_long_job_promptly_when_heartbeat_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queue = FakeQueue([(_job("1"), "token-1")])
    job_started = asyncio.Event()
    job_cancelled = asyncio.Event()

    async def fake_execute_run_job(_job: RunExecutionJob) -> None:
        job_started.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            job_cancelled.set()
            raise

    async def failing_heartbeat(
        _queue: FakeQueue,
        *,
        worker_id: str,
        ttl_seconds: int,
        lock_lost: asyncio.Event,
    ) -> None:
        _ = worker_id, ttl_seconds
        await job_started.wait()
        lock_lost.set()
        raise RuntimeError("Redis worker lease renewal failed.")

    async def noop() -> None:
        return None

    monkeypatch.setattr(worker_module, "settings", _worker_settings(1))
    monkeypatch.setattr(worker_module.db_manager, "initialize", noop)
    monkeypatch.setattr(worker_module.db_manager, "close", noop)
    monkeypatch.setattr(worker_module, "execute_run_job", fake_execute_run_job)
    monkeypatch.setattr(worker_module, "_maintain_worker_lock", failing_heartbeat)

    task = asyncio.create_task(
        worker_module.run_worker(queue=queue, poll_timeout_seconds=0)
    )
    try:
        with pytest.raises(RuntimeError, match="lease renewal failed"):
            await asyncio.wait_for(asyncio.shield(task), timeout=0.2)
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    assert job_cancelled.is_set()
    assert queue.acked == []


@pytest.mark.asyncio
async def test_run_worker_does_not_start_job_reserved_during_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queue = ControlledReservationQueue((_job("1"), "token-1"))
    shutdown_event = asyncio.Event()
    executed: list[str] = []

    async def fake_execute_run_job(job: RunExecutionJob) -> None:
        executed.append(job.run_id)

    async def noop() -> None:
        return None

    monkeypatch.setattr(worker_module, "settings", _worker_settings(1))
    monkeypatch.setattr(worker_module.db_manager, "initialize", noop)
    monkeypatch.setattr(worker_module.db_manager, "close", noop)
    monkeypatch.setattr(worker_module, "execute_run_job", fake_execute_run_job)

    task = asyncio.create_task(
        worker_module.run_worker(
            queue=queue,
            poll_timeout_seconds=0,
            shutdown_event=shutdown_event,
        )
    )
    await asyncio.wait_for(queue.reserve_started.wait(), timeout=1)
    queue.release_reservation.set()
    shutdown_event.set()

    assert await asyncio.wait_for(task, timeout=1) == 0
    assert executed == []
    assert queue.acked == []


@pytest.mark.asyncio
async def test_run_worker_stops_blocked_reservation_when_active_job_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queue = BlockingAfterReservationsQueue(
        [(_job("1"), "token-1"), (_job("2"), "token-2")]
    )
    sibling_cancelled = asyncio.Event()

    async def fake_execute_run_job(job: RunExecutionJob) -> None:
        if job.run_id == "r1":
            await queue.blocked_reserve_started.wait()
            raise RuntimeError("job failed while reserving")
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            sibling_cancelled.set()
            raise

    async def noop() -> None:
        return None

    monkeypatch.setattr(worker_module, "settings", _worker_settings(3))
    monkeypatch.setattr(worker_module.db_manager, "initialize", noop)
    monkeypatch.setattr(worker_module.db_manager, "close", noop)
    monkeypatch.setattr(worker_module, "execute_run_job", fake_execute_run_job)

    task = asyncio.create_task(
        worker_module.run_worker(queue=queue, poll_timeout_seconds=0)
    )
    try:
        with pytest.raises(RuntimeError, match="job failed while reserving"):
            await asyncio.wait_for(asyncio.shield(task), timeout=0.2)
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    assert queue.blocked_reserve_cancelled.is_set()
    assert sibling_cancelled.is_set()
    assert queue.acked == []


@pytest.mark.asyncio
async def test_run_worker_rejects_non_positive_concurrency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initialized = False

    async def fake_initialize() -> None:
        nonlocal initialized
        initialized = True

    monkeypatch.setattr(worker_module, "settings", _worker_settings(0))
    monkeypatch.setattr(worker_module.db_manager, "initialize", fake_initialize)

    with pytest.raises(RuntimeError, match="WORKER_CONCURRENT_JOBS must be at least 1"):
        await worker_module.run_worker(
            queue=FakeQueue([]), stop_after_jobs=0, poll_timeout_seconds=0
        )

    assert initialized is False
