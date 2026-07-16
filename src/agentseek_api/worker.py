from __future__ import annotations

import asyncio
import signal
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
    try:
        while not lock_lost.is_set():
            await asyncio.sleep(interval_seconds)
            try:
                renewed = await queue.renew_worker_lock(
                    worker_id, ttl_seconds=ttl_seconds
                )
            except Exception as exc:
                raise RuntimeError("Redis worker lease renewal failed.") from exc
            if not renewed:
                raise RuntimeError("Redis worker lost its active lease.")
    except asyncio.CancelledError:
        raise
    except Exception:
        lock_lost.set()
        raise


async def _execute_reserved_job(
    queue: RedisRunQueue,
    *,
    job: RunExecutionJob,
    token: str,
    worker_id: str,
    lock_lost: asyncio.Event,
    job_slots: asyncio.Semaphore,
    job_failed: asyncio.Event,
) -> None:
    try:
        await execute_run_job(job)
        if lock_lost.is_set() or not await queue.ack_if_worker_lock_owner(
            worker_id, token
        ):
            lock_lost.set()
            raise RuntimeError("Redis worker lost its active lease.")
    except BaseException:
        job_failed.set()
        raise
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


def _raise_if_worker_lock_lost(
    lock_lost: asyncio.Event,
    heartbeat_task: asyncio.Task[None] | None,
) -> None:
    if not lock_lost.is_set():
        return
    if heartbeat_task is not None and heartbeat_task.done():
        heartbeat_task.result()
    raise RuntimeError("Redis worker lost its active lease.")


async def _raise_if_job_failed(
    active_jobs: set[asyncio.Task[None]],
    job_failed: asyncio.Event,
) -> int:
    if not job_failed.is_set():
        return 0

    reaped = 0
    while active_jobs:
        reaped += await _reap_jobs(active_jobs, wait=True)
    raise RuntimeError("A worker job failed without surfacing its exception.")


async def _acquire_job_slot_or_signal(
    job_slots: asyncio.Semaphore,
    *,
    stop_requested: asyncio.Event,
    lock_lost: asyncio.Event,
    job_failed: asyncio.Event,
) -> bool:
    acquire_task = asyncio.create_task(job_slots.acquire())
    slot_returned = False
    signal_tasks = {
        asyncio.create_task(stop_requested.wait()),
        asyncio.create_task(lock_lost.wait()),
        asyncio.create_task(job_failed.wait()),
    }
    try:
        await asyncio.wait(
            {acquire_task, *signal_tasks}, return_when=asyncio.FIRST_COMPLETED
        )
        signal_received = (
            stop_requested.is_set() or lock_lost.is_set() or job_failed.is_set()
        )
        if signal_received:
            return False

        await acquire_task
        slot_returned = True
        return True
    finally:
        for task in signal_tasks:
            task.cancel()
        await asyncio.gather(*signal_tasks, return_exceptions=True)
        if not slot_returned:
            if not acquire_task.done():
                acquire_task.cancel()
            await asyncio.gather(acquire_task, return_exceptions=True)
            if not acquire_task.cancelled() and acquire_task.exception() is None:
                job_slots.release()


async def _reserve_or_signal(
    queue: RedisRunQueue,
    *,
    timeout_seconds: int,
    stop_requested: asyncio.Event,
    lock_lost: asyncio.Event,
    job_failed: asyncio.Event,
) -> tuple[RunExecutionJob, str] | None:
    reserve_task = asyncio.create_task(queue.reserve(timeout_seconds=timeout_seconds))
    signal_tasks = {
        asyncio.create_task(stop_requested.wait()),
        asyncio.create_task(lock_lost.wait()),
        asyncio.create_task(job_failed.wait()),
    }
    try:
        await asyncio.wait(
            {reserve_task, *signal_tasks}, return_when=asyncio.FIRST_COMPLETED
        )
        if stop_requested.is_set() or lock_lost.is_set() or job_failed.is_set():
            if not reserve_task.done():
                reserve_task.cancel()
            await asyncio.gather(reserve_task, return_exceptions=True)
            return None
        return await reserve_task
    finally:
        if not reserve_task.done():
            reserve_task.cancel()
            await asyncio.gather(reserve_task, return_exceptions=True)
        for task in signal_tasks:
            task.cancel()
        await asyncio.gather(*signal_tasks, return_exceptions=True)


async def _reap_job_or_lock_loss(
    active_jobs: set[asyncio.Task[None]],
    *,
    lock_lost: asyncio.Event,
    heartbeat_task: asyncio.Task[None] | None,
) -> int:
    lock_wait_task = asyncio.create_task(lock_lost.wait())
    try:
        await asyncio.wait(
            {*active_jobs, lock_wait_task}, return_when=asyncio.FIRST_COMPLETED
        )
        _raise_if_worker_lock_lost(lock_lost, heartbeat_task)
        return await _reap_jobs(active_jobs, wait=False)
    finally:
        lock_wait_task.cancel()
        await asyncio.gather(lock_wait_task, return_exceptions=True)


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
    job_failed = asyncio.Event()

    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(signum, stop_requested.set)
            except (NotImplementedError, RuntimeError, ValueError):
                continue
            registered_signals.append(signum)
        acquired_lock = await run_queue.acquire_worker_lock(
            worker_id, ttl_seconds=worker_lock_ttl_seconds
        )
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
        timeout_seconds = (
            poll_timeout_seconds
            if poll_timeout_seconds is not None
            else settings.REDIS_WORKER_POLL_TIMEOUT_SECONDS
        )
        while not stop_requested.is_set() and (
            stop_after_jobs is None or scheduled < stop_after_jobs
        ):
            processed += await _reap_jobs(active_jobs, wait=False)
            _raise_if_worker_lock_lost(lock_lost, heartbeat_task)
            processed += await _raise_if_job_failed(active_jobs, job_failed)

            slot_acquired = await _acquire_job_slot_or_signal(
                job_slots,
                stop_requested=stop_requested,
                lock_lost=lock_lost,
                job_failed=job_failed,
            )
            _raise_if_worker_lock_lost(lock_lost, heartbeat_task)
            processed += await _raise_if_job_failed(active_jobs, job_failed)
            if stop_requested.is_set():
                break
            if not slot_acquired:
                continue

            slot_transferred = False
            try:
                processed += await _reap_jobs(active_jobs, wait=False)
                if stop_requested.is_set() or (
                    stop_after_jobs is not None and scheduled >= stop_after_jobs
                ):
                    break
                _raise_if_worker_lock_lost(lock_lost, heartbeat_task)
                processed += await _raise_if_job_failed(active_jobs, job_failed)

                reserved = await _reserve_or_signal(
                    run_queue,
                    timeout_seconds=timeout_seconds,
                    stop_requested=stop_requested,
                    lock_lost=lock_lost,
                    job_failed=job_failed,
                )
                _raise_if_worker_lock_lost(lock_lost, heartbeat_task)
                processed += await _raise_if_job_failed(active_jobs, job_failed)
                if stop_requested.is_set():
                    break
                if reserved is None:
                    continue
                job, token = reserved
                task = asyncio.create_task(
                    _execute_reserved_job(
                        run_queue,
                        job=job,
                        token=token,
                        worker_id=worker_id,
                        lock_lost=lock_lost,
                        job_slots=job_slots,
                        job_failed=job_failed,
                    )
                )
                active_jobs.add(task)
                scheduled += 1
                slot_transferred = True
            finally:
                if not slot_transferred:
                    job_slots.release()

        while active_jobs:
            processed += await _reap_job_or_lock_loss(
                active_jobs,
                lock_lost=lock_lost,
                heartbeat_task=heartbeat_task,
            )
        _raise_if_worker_lock_lost(lock_lost, heartbeat_task)
    finally:
        for signum in registered_signals:
            loop.remove_signal_handler(signum)
        if active_jobs:
            for task in active_jobs:
                task.cancel()
            await asyncio.gather(*active_jobs, return_exceptions=True)
        if heartbeat_task is not None:
            heartbeat_task.cancel()
            await asyncio.gather(heartbeat_task, return_exceptions=True)
        if acquired_lock:
            await run_queue.release_worker_lock(worker_id)
        await run_queue.close()
        await db_manager.close()

    return processed


def main() -> int:
    return asyncio.run(run_worker())


if __name__ == "__main__":
    raise SystemExit(main())
