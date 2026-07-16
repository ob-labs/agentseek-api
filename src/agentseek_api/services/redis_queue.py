from __future__ import annotations

import json

from redis.asyncio import Redis, from_url

from agentseek_api.services.run_jobs import RunExecutionJob
from agentseek_api.settings import settings

_RENEW_WORKER_LOCK_SCRIPT = """
if redis.call("GET", KEYS[1]) == ARGV[1] then
  return redis.call("EXPIRE", KEYS[1], tonumber(ARGV[2]))
end
return 0
"""

_RELEASE_WORKER_LOCK_SCRIPT = """
if redis.call("GET", KEYS[1]) == ARGV[1] then
  return redis.call("DEL", KEYS[1])
end
return 0
"""

_ACK_IF_WORKER_LOCK_OWNER_SCRIPT = """
if redis.call("GET", KEYS[1]) == ARGV[1] then
  return redis.call("LREM", KEYS[2], 1, ARGV[2])
end
return 0
"""


class RedisRunQueue:
    def __init__(
        self,
        *,
        client: Redis | None = None,
        queue_key: str | None = None,
        processing_key: str | None = None,
        worker_lock_key: str | None = None,
        scheduler_lock_key: str | None = None,
    ) -> None:
        self.client = client or from_url(settings.REDIS_URL, decode_responses=True)
        self.queue_key = queue_key or settings.REDIS_RUN_QUEUE_KEY
        self.processing_key = processing_key or settings.REDIS_RUN_PROCESSING_KEY
        self.worker_lock_key = worker_lock_key or settings.REDIS_WORKER_LOCK_KEY
        self.scheduler_lock_key = scheduler_lock_key or settings.REDIS_SCHEDULER_LOCK_KEY

    async def enqueue(self, job: RunExecutionJob) -> None:
        await self.client.lpush(self.queue_key, self._serialize(job))

    async def reserve(self, *, timeout_seconds: int) -> tuple[RunExecutionJob, str] | None:
        raw = await self.client.brpoplpush(self.queue_key, self.processing_key, timeout=timeout_seconds)
        if raw is None:
            return None
        payload = json.loads(raw)
        return RunExecutionJob.from_payload(payload), raw

    async def ack(self, token: str) -> None:
        await self.client.lrem(self.processing_key, 1, token)

    async def ack_if_worker_lock_owner(self, worker_id: str, token: str) -> bool:
        acknowledged = await self.client.eval(
            _ACK_IF_WORKER_LOCK_OWNER_SCRIPT,
            2,
            self.worker_lock_key,
            self.processing_key,
            worker_id,
            token,
        )
        return bool(acknowledged)

    async def contains_run(self, *, run_id: str) -> bool:
        for key in (self.queue_key, self.processing_key):
            items = await self.client.lrange(key, 0, -1)
            for raw in items:
                try:
                    payload = json.loads(raw)
                except Exception:  # noqa: BLE001
                    continue
                if str(payload.get("run_id", "")) == run_id:
                    return True
        return False

    async def requeue_inflight(self) -> int:
        moved = 0
        while True:
            token = await self.client.rpoplpush(self.processing_key, self.queue_key)
            if token is None:
                return moved
            moved += 1

    async def acquire_lease(self, lease_key: str, owner_id: str, *, ttl_seconds: int) -> bool:
        acquired = await self.client.set(lease_key, owner_id, ex=ttl_seconds, nx=True)
        return bool(acquired)

    async def renew_lease(self, lease_key: str, owner_id: str, *, ttl_seconds: int) -> bool:
        renewed = await self.client.eval(
            _RENEW_WORKER_LOCK_SCRIPT,
            1,
            lease_key,
            owner_id,
            str(ttl_seconds),
        )
        return bool(renewed)

    async def release_lease(self, lease_key: str, owner_id: str) -> None:
        await self.client.eval(
            _RELEASE_WORKER_LOCK_SCRIPT,
            1,
            lease_key,
            owner_id,
        )

    async def acquire_worker_lock(self, worker_id: str, *, ttl_seconds: int) -> bool:
        return await self.acquire_lease(self.worker_lock_key, worker_id, ttl_seconds=ttl_seconds)

    async def renew_worker_lock(self, worker_id: str, *, ttl_seconds: int) -> bool:
        return await self.renew_lease(self.worker_lock_key, worker_id, ttl_seconds=ttl_seconds)

    async def release_worker_lock(self, worker_id: str) -> None:
        await self.release_lease(self.worker_lock_key, worker_id)

    async def acquire_scheduler_lock(self, scheduler_id: str, *, ttl_seconds: int) -> bool:
        return await self.acquire_lease(self.scheduler_lock_key, scheduler_id, ttl_seconds=ttl_seconds)

    async def renew_scheduler_lock(self, scheduler_id: str, *, ttl_seconds: int) -> bool:
        return await self.renew_lease(self.scheduler_lock_key, scheduler_id, ttl_seconds=ttl_seconds)

    async def release_scheduler_lock(self, scheduler_id: str) -> None:
        await self.release_lease(self.scheduler_lock_key, scheduler_id)

    async def close(self) -> None:
        close = getattr(self.client, "aclose", None)
        if callable(close):
            await close()

    @staticmethod
    def _serialize(job: RunExecutionJob) -> str:
        return json.dumps(job.to_payload(), sort_keys=True, separators=(",", ":"))
