from __future__ import annotations

import asyncio
import signal
from contextlib import suppress
from uuid import uuid4

from agentseek_api.core.database import db_manager
from agentseek_api.services.redis_queue import RedisRunQueue
from agentseek_api.services.run_jobs import RunExecutionJob, execute_run_job
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


async def _execute_reserved_job(
    queue: RedisRunQueue,
    *,
    job: RunExecutionJob,
    token: str,
    lock_lost: asyncio.Event,
    job_slots: asyncio.Semaphore,
) -> None:
    try:
        await execute_run_job(job)
        if lock_lost.is_set():
            raise RuntimeError("Redis worker lost its active lease.")
        await queue.ack(token)
    finally:
        job_slots.release()


async def _reap_jobs(active_jobs: set[asyncio.Task[None]], *, wait: bool) -> int:
    if not active_jobs:
        return 0
    if wait:
        completed, _ = await asyncio.wait(
            active_jobs, return_when=asyncio.FIRST_COMPLETED
        )
    else:
        completed = {task for task in active_jobs if task.done()}
    active_jobs.difference_update(completed)

    first_error: BaseException | None = None
    for task in completed:
        try:
            task.result()
        except BaseException as exc:
            if first_error is None:
                first_error = exc
    if first_error is not None:
        raise first_error
    return len(completed)


async def run_worker(
    *,
    queue: RedisRunQueue | None = None,
    stop_after_jobs: int | None = None,
    poll_timeout_seconds: int | None = None,
    shutdown_event: asyncio.Event | None = None,
) -> int:
    if settings.EXECUTOR_BACKEND.strip().lower() != "redis":
        raise RuntimeError("The worker requires EXECUTOR_BACKEND=redis.")
    concurrent_jobs = settings.WORKER_CONCURRENT_JOBS
    if concurrent_jobs < 1:
        raise RuntimeError("WORKER_CONCURRENT_JOBS must be at least 1.")

    await db_manager.initialize()
    run_queue = queue or RedisRunQueue()
    processed = 0
    scheduled = 0
    worker_id = str(uuid4())
    worker_lock_ttl_seconds = settings.REDIS_WORKER_LOCK_TTL_SECONDS
    lock_lost = asyncio.Event()
    stop_requested = shutdown_event or asyncio.Event()
    heartbeat_task: asyncio.Task[None] | None = None
    acquired_lock = False
    registered_signals: list[signal.Signals] = []
    loop = asyncio.get_running_loop()
    job_slots = asyncio.Semaphore(concurrent_jobs)
    active_jobs: set[asyncio.Task[None]] = set()

    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(signum, stop_requested.set)
            except (NotImplementedError, RuntimeError, ValueError):
                continue
            registered_signals.append(signum)
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
        while not stop_requested.is_set() and (
            stop_after_jobs is None or scheduled < stop_after_jobs
        ):
            processed += await _reap_jobs(active_jobs, wait=False)
            if lock_lost.is_set():
                raise RuntimeError("Redis worker lost its active lease.")

            await job_slots.acquire()
            slot_transferred = False
            try:
                processed += await _reap_jobs(active_jobs, wait=False)
                if stop_requested.is_set() or (
                    stop_after_jobs is not None and scheduled >= stop_after_jobs
                ):
                    break
                if lock_lost.is_set():
                    raise RuntimeError("Redis worker lost its active lease.")

                reserved = await run_queue.reserve(timeout_seconds=timeout_seconds)
                if reserved is None:
                    continue
                if lock_lost.is_set():
                    raise RuntimeError("Redis worker lost its active lease.")
                job, token = reserved
                task = asyncio.create_task(
                    _execute_reserved_job(
                        run_queue,
                        job=job,
                        token=token,
                        lock_lost=lock_lost,
                        job_slots=job_slots,
                    )
                )
                active_jobs.add(task)
                scheduled += 1
                slot_transferred = True
            finally:
                if not slot_transferred:
                    job_slots.release()

        while active_jobs:
            processed += await _reap_jobs(active_jobs, wait=True)
    finally:
        for signum in registered_signals:
            loop.remove_signal_handler(signum)
        if active_jobs:
            for task in active_jobs:
                task.cancel()
            await asyncio.gather(*active_jobs, return_exceptions=True)
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
