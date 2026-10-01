import asyncio
import logging

from agentseek_api.services.redis_queue import RedisRunQueue
from agentseek_api.services.run_jobs import RunExecutionJob, execute_run_job
from agentseek_api.settings import settings


class ExecutorFacade:
    async def submit(self, job: RunExecutionJob) -> None:
        raise NotImplementedError


class InlineExecutor(ExecutorFacade):
    def __init__(self) -> None:
        self.tasks: set[asyncio.Task] = set()

    async def submit(self, job: RunExecutionJob) -> None:
        task = asyncio.create_task(execute_run_job(job), name=f"run:{job.run_id}")
        self.tasks.add(task)
        def finished(task):
            self.tasks.discard(task)
            if not task.cancelled() and task.exception() is not None:
                logging.getLogger(__name__).error("Run execution left recoverable work", exc_info=task.exception())
        task.add_done_callback(finished)

    async def close(self) -> None:
        tasks = list(self.tasks)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


class RedisExecutor(ExecutorFacade):
    def __init__(self, *, queue: RedisRunQueue | None = None) -> None:
        self.queue = queue or RedisRunQueue()

    async def submit(self, job: RunExecutionJob) -> None:
        await self.queue.enqueue(job)


_executor: ExecutorFacade | None = None


async def close_executor() -> None:
    global _executor
    executor, _executor = _executor, None
    if isinstance(executor, InlineExecutor):
        await executor.close()
    elif isinstance(executor, RedisExecutor):
        await executor.queue.close()


def get_executor() -> ExecutorFacade:
    global _executor
    if _executor is None:
        backend = settings.EXECUTOR_BACKEND.strip().lower()
        if backend == "inline":
            _executor = InlineExecutor()
        elif backend == "redis":
            _executor = RedisExecutor()
        else:
            raise ValueError(f"Unsupported EXECUTOR_BACKEND: {settings.EXECUTOR_BACKEND}")
    return _executor
