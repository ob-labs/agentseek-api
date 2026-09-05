from __future__ import annotations

import asyncio
import signal
from contextlib import suppress
from uuid import uuid4

from agentseek_api.core.database import db_manager
from agentseek_api.services.cron_scheduler import dispatch_due_crons
from agentseek_api.services.redis_queue import RedisRunQueue
from agentseek_api.settings import settings


async def _maintain_scheduler_lock(
    queue: RedisRunQueue,
    *,
    scheduler_id: str,
    ttl_seconds: int,
    lock_lost: asyncio.Event,
) -> None:
    interval_seconds = max(1, ttl_seconds // 3)
    while not lock_lost.is_set():
        await asyncio.sleep(interval_seconds)
        if not await queue.renew_scheduler_lock(scheduler_id, ttl_seconds=ttl_seconds):
            lock_lost.set()
            return


async def run_scheduler(
    *,
    queue: RedisRunQueue | None = None,
    shutdown_event: asyncio.Event | None = None,
) -> int:
    await db_manager.initialize()
    run_queue = queue or RedisRunQueue()
    scheduler_id = str(uuid4())
    ttl_seconds = settings.REDIS_SCHEDULER_LOCK_TTL_SECONDS
    poll_interval = settings.SCHEDULER_POLL_INTERVAL_SECONDS
    claim_limit = settings.SCHEDULER_CLAIM_LIMIT
    lock_lost = asyncio.Event()
    stop_requested = shutdown_event or asyncio.Event()
    heartbeat_task: asyncio.Task[None] | None = None
    acquired_lock = False
    dispatched = 0
    registered_signals: list[signal.Signals] = []
    loop = asyncio.get_running_loop()

    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(signum, stop_requested.set)
            except (NotImplementedError, RuntimeError, ValueError):
                continue
            registered_signals.append(signum)

        acquired_lock = await run_queue.acquire_scheduler_lock(scheduler_id, ttl_seconds=ttl_seconds)
        if not acquired_lock:
            raise RuntimeError("Another scheduler is already active.")

        heartbeat_task = asyncio.create_task(
            _maintain_scheduler_lock(
                run_queue,
                scheduler_id=scheduler_id,
                ttl_seconds=ttl_seconds,
                lock_lost=lock_lost,
            )
        )

        while not stop_requested.is_set():
            if lock_lost.is_set():
                raise RuntimeError("Scheduler lost its active lease.")
            results = await dispatch_due_crons(limit=claim_limit, scheduler_id=scheduler_id)
            dispatched += len(results)
            if stop_requested.is_set():
                break
            await asyncio.sleep(poll_interval)
    finally:
        for signum in registered_signals:
            loop.remove_signal_handler(signum)
        if heartbeat_task is not None:
            heartbeat_task.cancel()
            with suppress(asyncio.CancelledError):
                await heartbeat_task
        if acquired_lock:
            await run_queue.release_scheduler_lock(scheduler_id)
        await run_queue.close()
        await db_manager.close()

    return dispatched


def main() -> int:
    return asyncio.run(run_scheduler())


if __name__ == "__main__":
    raise SystemExit(main())
