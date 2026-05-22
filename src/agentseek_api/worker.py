from __future__ import annotations

import asyncio
from contextlib import suppress
from uuid import uuid4

from agentseek_api.core.database import db_manager
from agentseek_api.services.redis_queue import RedisRunQueue
from agentseek_api.services.run_jobs import execute_run_job
from agentseek_api.settings import settings


async def _maintain_worker_lock(
    queue: RedisRunQueue,
    *,
    worker_id: str,
    ttl_seconds: int,
    lock_lost: asyncio.Event,
) -> None:
    interval_seconds = max(1, ttl_seconds // 3)
    while not lock_lost.is_set():
        await asyncio.sleep(interval_seconds)
        if not await queue.renew_worker_lock(worker_id, ttl_seconds=ttl_seconds):
            lock_lost.set()
            return


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
    worker_id = str(uuid4())
    worker_lock_ttl_seconds = settings.REDIS_WORKER_LOCK_TTL_SECONDS
    lock_lost = asyncio.Event()
    heartbeat_task: asyncio.Task[None] | None = None
    acquired_lock = False

    try:
        acquired_lock = await run_queue.acquire_worker_lock(worker_id, ttl_seconds=worker_lock_ttl_seconds)
        if not acquired_lock:
            raise RuntimeError("Another Redis worker is already active.")
        heartbeat_task = asyncio.create_task(
            _maintain_worker_lock(
                run_queue,
                worker_id=worker_id,
                ttl_seconds=worker_lock_ttl_seconds,
                lock_lost=lock_lost,
            )
        )
        await run_queue.requeue_inflight()
        timeout_seconds = poll_timeout_seconds if poll_timeout_seconds is not None else settings.REDIS_WORKER_POLL_TIMEOUT_SECONDS
        while stop_after_jobs is None or processed < stop_after_jobs:
            if lock_lost.is_set():
                raise RuntimeError("Redis worker lost its active lease.")
            reserved = await run_queue.reserve(timeout_seconds=timeout_seconds)
            if reserved is None:
                continue
            job, token = reserved
            await execute_run_job(job)
            if lock_lost.is_set():
                raise RuntimeError("Redis worker lost its active lease.")
            await run_queue.ack(token)
            processed += 1
    finally:
        if heartbeat_task is not None:
            heartbeat_task.cancel()
            with suppress(asyncio.CancelledError):
                await heartbeat_task
        if acquired_lock:
            await run_queue.release_worker_lock(worker_id)
        await run_queue.close()
        await db_manager.close()

    return processed


def main() -> int:
    return asyncio.run(run_worker())


if __name__ == "__main__":
    raise SystemExit(main())
