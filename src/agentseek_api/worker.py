from __future__ import annotations

import asyncio

from agentseek_api.core.database import db_manager
from agentseek_api.services.redis_queue import RedisRunQueue
from agentseek_api.services.run_jobs import execute_run_job
from agentseek_api.settings import settings


async def run_worker(
    *,
    queue: RedisRunQueue | None = None,
    stop_after_jobs: int | None = None,
    poll_timeout_seconds: int | None = None,
) -> int:
    if settings.EXECUTOR_BACKEND.strip().lower() != "redis":
        raise RuntimeError("The worker requires EXECUTOR_BACKEND=redis.")

    await db_manager.initialize()
    run_queue = queue or RedisRunQueue()
    processed = 0

    try:
        await run_queue.requeue_inflight()
        timeout_seconds = poll_timeout_seconds if poll_timeout_seconds is not None else settings.REDIS_WORKER_POLL_TIMEOUT_SECONDS
        while stop_after_jobs is None or processed < stop_after_jobs:
            reserved = await run_queue.reserve(timeout_seconds=timeout_seconds)
            if reserved is None:
                continue
            job, token = reserved
            await execute_run_job(job)
            await run_queue.ack(token)
            processed += 1
    finally:
        await run_queue.close()
        await db_manager.close()

    return processed


def main() -> int:
    return asyncio.run(run_worker())


if __name__ == "__main__":
    raise SystemExit(main())
