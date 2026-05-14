import asyncio

import pytest

from agentseek_api.services import executor as executor_module
from agentseek_api.services.executor import ExecutorFacade, get_executor


@pytest.mark.asyncio
async def test_submit_schedules_task_without_waiting_for_completion() -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    finished = asyncio.Event()

    async def sample_task() -> None:
        started.set()
        await release.wait()
        finished.set()

    facade = ExecutorFacade()
    await asyncio.wait_for(facade.submit(sample_task), timeout=0.1)

    await asyncio.wait_for(started.wait(), timeout=0.1)
    assert not finished.is_set()

    release.set()
    await asyncio.wait_for(finished.wait(), timeout=0.1)


def test_get_executor_returns_singleton() -> None:
    executor_module._executor = None
    first = get_executor()
    second = get_executor()
    assert first is second
