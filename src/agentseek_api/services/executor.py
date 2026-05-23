import asyncio

from agentseek_api.services.redis_queue import RedisRunQueue
from agentseek_api.services.run_jobs import RunExecutionJob, execute_run_job
from agentseek_api.settings import settings


class ExecutorFacade:
    async def submit(self, job: RunExecutionJob) -> None:
        raise NotImplementedError


class InlineExecutor(ExecutorFacade):
    async def submit(self, job: RunExecutionJob) -> None:
        asyncio.create_task(execute_run_job(job))


class RedisExecutor(ExecutorFacade):
    def __init__(self, *, queue: RedisRunQueue | None = None) -> None:
        self.queue = queue or RedisRunQueue()

    async def submit(self, job: RunExecutionJob) -> None:
        await self.queue.enqueue(job)


_executor: ExecutorFacade | None = None


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
