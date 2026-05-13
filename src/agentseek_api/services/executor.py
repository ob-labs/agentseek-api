import asyncio
from collections.abc import Awaitable, Callable


class ExecutorFacade:
    async def submit(self, func: Callable[[], Awaitable[None]]) -> None:
        asyncio.create_task(func())


_executor: ExecutorFacade | None = None


def get_executor() -> ExecutorFacade:
    global _executor
    if _executor is None:
        _executor = ExecutorFacade()
    return _executor
