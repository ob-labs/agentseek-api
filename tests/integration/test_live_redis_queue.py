import os
from uuid import uuid4

import pytest
from redis.asyncio import from_url

from agentseek_api.services.redis_queue import RedisRunQueue
from agentseek_api.services.run_jobs import RunExecutionJob


_TEST_REDIS_URL = os.getenv("AGENTSEEK_TEST_REDIS_URL")
pytestmark = pytest.mark.skipif(
    not _TEST_REDIS_URL,
    reason="AGENTSEEK_TEST_REDIS_URL is required for live Redis queue tests",
)


@pytest.mark.asyncio
async def test_stale_worker_cannot_ack_token_re_reserved_by_new_lock_owner() -> None:
    assert _TEST_REDIS_URL is not None
    client = from_url(_TEST_REDIS_URL, decode_responses=True)
    key_prefix = f"agentseek:test:redis-queue:{uuid4().hex}"
    queue = RedisRunQueue(
        client=client,
        queue_key=f"{key_prefix}:pending",
        processing_key=f"{key_prefix}:processing",
        worker_lock_key=f"{key_prefix}:worker-lock",
        scheduler_lock_key=f"{key_prefix}:scheduler-lock",
    )
    job = RunExecutionJob(
        run_id=f"run-{uuid4().hex}",
        thread_id=f"thread-{uuid4().hex}",
        user_id="live-redis-queue-test",
        payload={"message": "conditional-ack"},
        graph_id="default",
    )

    try:
        assert await queue.acquire_worker_lock("worker-a", ttl_seconds=30) is True
        assert await client.get(queue.worker_lock_key) == "worker-a"
        await queue.enqueue(job)
        reserved = await queue.reserve(timeout_seconds=1)
        assert reserved is not None
        _, token = reserved

        assert await client.set(queue.worker_lock_key, "worker-b", ex=30) is True
        assert await client.get(queue.worker_lock_key) == "worker-b"
        assert await queue.requeue_inflight() == 1
        re_reserved = await queue.reserve(timeout_seconds=1)
        assert re_reserved is not None
        _, re_reserved_token = re_reserved
        assert re_reserved_token == token

        assert await queue.ack_if_worker_lock_owner("worker-a", token) is False
        assert await client.lrange(queue.processing_key, 0, -1) == [token]

        assert await queue.ack_if_worker_lock_owner("worker-b", token) is True
        assert await client.lrange(queue.processing_key, 0, -1) == []
        assert await client.lrange(queue.queue_key, 0, -1) == []
    finally:
        try:
            await client.delete(
                queue.queue_key,
                queue.processing_key,
                queue.worker_lock_key,
                queue.scheduler_lock_key,
            )
        finally:
            await queue.close()
