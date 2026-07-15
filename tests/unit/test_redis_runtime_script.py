from pathlib import Path


def test_redis_runtime_runs_live_queue_ownership_tests() -> None:
    script = Path("scripts/test-redis-runtime.sh").read_text()

    assert "tests/integration/test_live_redis_queue.py" in script
