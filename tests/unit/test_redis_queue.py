import pytest

from agentseek_api.services.redis_queue import RedisRunQueue
from agentseek_api.services.run_jobs import RunExecutionJob
from agentseek_api.services.run_executor import UNSET


class FakeRedis:
    def __init__(self) -> None:
        self.lists: dict[str, list[str]] = {}
        self.values: dict[str, str] = {}
        self.expirations: dict[str, int] = {}

    async def lpush(self, key: str, value: str) -> int:
        items = self.lists.setdefault(key, [])
        items.insert(0, value)
        return len(items)

    async def rpush(self, key: str, value: str) -> int:
        items = self.lists.setdefault(key, [])
        items.append(value)
        return len(items)

    async def brpoplpush(self, source: str, destination: str, timeout: int) -> str | None:
        _ = timeout
        source_items = self.lists.setdefault(source, [])
        if not source_items:
            return None
        value = source_items.pop()
        self.lists.setdefault(destination, []).insert(0, value)
        return value

    async def rpoplpush(self, source: str, destination: str) -> str | None:
        source_items = self.lists.setdefault(source, [])
        if not source_items:
            return None
        value = source_items.pop()
        self.lists.setdefault(destination, []).insert(0, value)
        return value

    async def lrem(self, key: str, count: int, value: str) -> int:
        items = self.lists.setdefault(key, [])
        removed = 0
        remaining: list[str] = []
        for item in items:
            if item == value and removed < count:
                removed += 1
                continue
            remaining.append(item)
        self.lists[key] = remaining
        return removed

    async def lrange(self, key: str, start: int, end: int) -> list[str]:
        items = self.lists.setdefault(key, [])
        normalized_end = None if end == -1 else end + 1
        return items[start:normalized_end]

    async def set(self, key: str, value: str, *, ex: int, nx: bool) -> bool:
        if nx and key in self.values:
            return False
        self.values[key] = value
        self.expirations[key] = ex
        return True

    async def eval(self, script: str, numkeys: int, *args: str) -> int:
        if "LREM" in script:
            assert numkeys == 2
            worker_key, processing_key, owner, token = args
            if self.values.get(worker_key) != owner:
                return 0
            return await self.lrem(processing_key, 1, token)

        assert numkeys == 1
        key = args[0]
        owner = args[1]
        if "EXPIRE" in script:
            ttl_seconds = int(args[2])
            if self.values.get(key) != owner:
                return 0
            self.expirations[key] = ttl_seconds
            return 1
        if "DEL" in script:
            if self.values.get(key) != owner:
                return 0
            self.values.pop(key, None)
            self.expirations.pop(key, None)
            return 1
        raise AssertionError(f"Unexpected Lua script: {script}")


def _job(run_id: str) -> RunExecutionJob:
    return RunExecutionJob(
        run_id=run_id,
        thread_id="thread-1",
        user_id="user-1",
        payload={"message": run_id},
        graph_id="default",
    )


@pytest.mark.asyncio
async def test_redis_queue_reserves_jobs_in_fifo_order() -> None:
    queue = RedisRunQueue(client=FakeRedis(), queue_key="pending", processing_key="processing")
    first = _job("run-1")
    second = _job("run-2")

    await queue.enqueue(first)
    await queue.enqueue(second)

    reserved_first = await queue.reserve(timeout_seconds=0)
    reserved_second = await queue.reserve(timeout_seconds=0)

    assert reserved_first is not None
    assert reserved_second is not None
    assert reserved_first[0].run_id == "run-1"
    assert reserved_second[0].run_id == "run-2"


@pytest.mark.asyncio
async def test_redis_queue_renews_worker_lock_only_for_current_owner() -> None:
    client = FakeRedis()
    queue = RedisRunQueue(client=client, worker_lock_key="worker")

    assert await queue.acquire_worker_lock("worker-a", ttl_seconds=30) is True
    client.values["worker"] = "worker-b"

    assert await queue.renew_worker_lock("worker-a", ttl_seconds=60) is False
    assert client.values["worker"] == "worker-b"
    assert client.expirations["worker"] == 30


@pytest.mark.asyncio
async def test_redis_queue_releases_worker_lock_only_for_current_owner() -> None:
    client = FakeRedis()
    queue = RedisRunQueue(client=client, worker_lock_key="worker")

    assert await queue.acquire_worker_lock("worker-a", ttl_seconds=30) is True
    client.values["worker"] = "worker-b"

    await queue.release_worker_lock("worker-a")

    assert client.values["worker"] == "worker-b"


@pytest.mark.asyncio
async def test_redis_queue_acknowledges_only_for_current_worker_lock_owner() -> None:
    client = FakeRedis()
    queue = RedisRunQueue(
        client=client,
        processing_key="processing",
        worker_lock_key="worker",
    )
    token = RedisRunQueue._serialize(_job("run-1"))
    client.lists["processing"] = [token]
    client.values["worker"] = "worker-b"

    assert await queue.ack_if_worker_lock_owner("worker-a", token) is False
    assert client.lists["processing"] == [token]

    assert await queue.ack_if_worker_lock_owner("worker-b", token) is True
    assert client.lists["processing"] == []


@pytest.mark.asyncio
async def test_redis_queue_round_trips_non_dict_payloads() -> None:
    queue = RedisRunQueue(client=FakeRedis(), queue_key="pending", processing_key="processing")
    job = RunExecutionJob(
        run_id="run-list",
        thread_id="thread-1",
        user_id="user-1",
        payload=["cron", {"kind": "list-payload"}],
        graph_id="default",
    )

    await queue.enqueue(job)
    reserved = await queue.reserve(timeout_seconds=0)

    assert reserved is not None
    assert reserved[0].payload == ["cron", {"kind": "list-payload"}]


@pytest.mark.asyncio
async def test_redis_queue_round_trips_non_resume_jobs_without_serializing_unset_resume() -> None:
    queue = RedisRunQueue(client=FakeRedis(), queue_key="pending", processing_key="processing")
    job = RunExecutionJob(
        run_id="run-normal",
        thread_id="thread-1",
        user_id="user-1",
        payload={"message": "hello"},
        graph_id="default",
        resume=UNSET,
        is_resume=False,
    )

    await queue.enqueue(job)
    reserved = await queue.reserve(timeout_seconds=0)

    assert reserved is not None
    assert reserved[0].run_id == "run-normal"
    assert reserved[0].resume is None
    assert reserved[0].is_resume is False


@pytest.mark.asyncio
async def test_redis_queue_contains_run_checks_pending_and_processing_lists() -> None:
    client = FakeRedis()
    queue = RedisRunQueue(client=client, queue_key="pending", processing_key="processing")
    first = _job("run-1")
    second = _job("run-2")

    await queue.enqueue(first)
    await queue.enqueue(second)
    reserved = await queue.reserve(timeout_seconds=0)

    assert reserved is not None
    assert await queue.contains_run(run_id="run-1") is True
    assert await queue.contains_run(run_id="run-2") is True
    assert await queue.contains_run(run_id="missing") is False
