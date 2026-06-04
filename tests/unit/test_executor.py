import asyncio

import pytest

from agentseek_api.services import executor as executor_module
from agentseek_api.services.executor import InlineExecutor, RedisExecutor, get_executor
from agentseek_api.services.run_jobs import RunExecutionJob


class FakeQueue:
    def __init__(self) -> None:
        self.enqueued: list[RunExecutionJob] = []

    async def enqueue(self, job: RunExecutionJob) -> None:
        self.enqueued.append(job)


@pytest.mark.asyncio
async def test_submit_schedules_task_without_waiting_for_completion(monkeypatch: pytest.MonkeyPatch) -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    finished = asyncio.Event()
    observed: list[RunExecutionJob] = []

    async def fake_execute_run_job(job: RunExecutionJob) -> None:
        observed.append(job)
        started.set()
        await release.wait()
        finished.set()

    monkeypatch.setattr(executor_module, "execute_run_job", fake_execute_run_job)
    facade = InlineExecutor()
    job = RunExecutionJob(
        run_id="r1",
        thread_id="t1",
        user_id="u1",
        payload={"message": "hello"},
        graph_id="default",
    )
    await asyncio.wait_for(facade.submit(job), timeout=0.1)

    await asyncio.wait_for(started.wait(), timeout=0.1)
    assert not finished.is_set()
    assert observed == [job]

    release.set()
    await asyncio.wait_for(finished.wait(), timeout=0.1)


def test_get_executor_returns_singleton() -> None:
    executor_module._executor = None
    executor_module.settings.EXECUTOR_BACKEND = "inline"
    first = get_executor()
    second = get_executor()
    assert first is second


@pytest.mark.asyncio
async def test_redis_executor_enqueues_job() -> None:
    queue = FakeQueue()
    executor = RedisExecutor(queue=queue)
    job = RunExecutionJob(
        run_id="r1",
        thread_id="t1",
        user_id="u1",
        payload={"message": "hello"},
        graph_id="default",
    )

    await executor.submit(job)

    assert queue.enqueued == [job]


@pytest.mark.asyncio
async def test_executor_facade_not_implemented() -> None:
    from agentseek_api.services.executor import ExecutorFacade

    with pytest.raises(NotImplementedError):
        await ExecutorFacade().submit(None)  # type: ignore[arg-type]


def test_get_executor_invalid_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    executor_module._executor = None
    monkeypatch.setattr(executor_module.settings, "EXECUTOR_BACKEND", "bogus")
    with pytest.raises(ValueError, match="Unsupported EXECUTOR_BACKEND"):
        get_executor()
    executor_module._executor = None


def test_get_executor_redis_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    executor_module._executor = None
    monkeypatch.setattr(executor_module.settings, "EXECUTOR_BACKEND", "redis")
    ex = get_executor()
    assert isinstance(ex, RedisExecutor)
    executor_module._executor = None
