import pytest

from agentseek_api.services.run_jobs import RunExecutionJob
from agentseek_api import worker as worker_module


class FakeQueue:
    def __init__(self, reservations: list[tuple[RunExecutionJob, str] | None]) -> None:
        self.reservations = reservations
        self.acked: list[str] = []
        self.requeue_calls = 0
        self.closed = False

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


@pytest.mark.asyncio
async def test_run_worker_requires_redis_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(worker_module.settings, "EXECUTOR_BACKEND", "inline")

    with pytest.raises(RuntimeError, match="EXECUTOR_BACKEND=redis"):
        await worker_module.run_worker(queue=FakeQueue([]), stop_after_jobs=0, poll_timeout_seconds=0)
