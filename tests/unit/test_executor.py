import asyncio

import pytest

from agentseek_api.services import executor as executor_module
from agentseek_api.services.executor import ExecutorFacade, get_executor


@pytest.mark.asyncio
async def test_submit_delegates_to_asyncio_create_task(monkeypatch: pytest.MonkeyPatch) -> None:
    called = {"count": 0}
    real_create_task = asyncio.create_task

    async def sample_task() -> None:
        called["count"] += 1

    async def run_immediately(coro) -> None:
        await coro

    monkeypatch.setattr(
        "agentseek_api.services.executor.asyncio.create_task",
        lambda coro: real_create_task(run_immediately(coro)),
    )

    facade = ExecutorFacade()
    await facade.submit(sample_task)
    await asyncio.sleep(0)
    assert called["count"] == 1


def test_get_executor_returns_singleton() -> None:
    executor_module._executor = None
    first = get_executor()
    second = get_executor()
    assert first is second
