from pathlib import Path


def test_redis_runtime_runs_live_queue_ownership_tests() -> None:
    script = Path("scripts/test-redis-runtime.sh").read_text()

    assert "tests/integration/test_live_redis_queue.py" in script


def test_redis_runtime_wires_real_concurrency_and_recovery_probes() -> None:
    script = Path("scripts/test-redis-runtime.sh").read_text()

    assert '-e WORKER_CONCURRENT_JOBS="${WORKER_CONCURRENT_JOBS}"' in script
    for mode in ("bounded", "fanout", "failure", "shutdown-seed", "shutdown-check"):
        assert f"--mode {mode}" in script


def test_redis_runtime_restarts_workers_for_each_concurrency_and_forced_recovery() -> None:
    script = Path("scripts/test-redis-runtime.sh").read_text()

    expected_orchestration = """\
WORKER_CONCURRENT_JOBS=2
start_worker
run_probe --mode bounded

WORKER_CONCURRENT_JOBS=10
start_worker
run_probe --mode fanout

WORKER_CONCURRENT_JOBS=2
start_worker
run_probe --mode failure

WORKER_CONCURRENT_JOBS=2
start_worker
run_probe --mode shutdown-seed >"$SHUTDOWN_STATE_FILE"
stop_worker 1
start_worker
run_probe --mode shutdown-check --state-file "$SHUTDOWN_STATE_FILE"
"""
    assert expected_orchestration in script
    assert 'local timeout="${1:-10}"' in script
    assert 'docker stop -t "$timeout" "$WORKER_CONTAINER"' in script
    assert 'SHUTDOWN_STATE_FILE="$(mktemp ' in script
    assert 'rm -f "$SHUTDOWN_STATE_FILE"' in script
    assert 'redis-cli DEL "$REDIS_WORKER_LOCK_KEY"' in script
    assert "print_logs >&2" in script
