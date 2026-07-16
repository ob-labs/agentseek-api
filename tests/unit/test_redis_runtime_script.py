from pathlib import Path


def test_redis_runtime_runs_live_queue_ownership_tests() -> None:
    script = Path("scripts/test-redis-runtime.sh").read_text()

    assert "tests/integration/test_live_redis_queue.py" in script


def test_redis_runtime_wires_real_concurrency_and_recovery_probes() -> None:
    script = Path("scripts/test-redis-runtime.sh").read_text()

    assert '-e WORKER_CONCURRENT_JOBS="${WORKER_CONCURRENT_JOBS}"' in script
    for mode in ("bounded", "fanout", "failure", "shutdown-seed", "shutdown-check"):
        assert f"--mode {mode}" in script


def test_redis_runtime_orders_probes_with_only_required_worker_restarts() -> None:
    script = Path("scripts/test-redis-runtime.sh").read_text()

    lines = script.splitlines()
    suite_start = lines.index("WORKER_CONCURRENCY_SUITE_STARTED_SECONDS=$SECONDS")
    suite_end = lines.index(
        'echo "worker concurrency probe suite completed in '
        '$((SECONDS - WORKER_CONCURRENCY_SUITE_STARTED_SECONDS))s" >&2'
    )
    suite = lines[suite_start : suite_end + 1]
    assert suite == [
        "WORKER_CONCURRENCY_SUITE_STARTED_SECONDS=$SECONDS",
        'echo "worker concurrency probe suite started" >&2',
        "",
        "WORKER_CONCURRENT_JOBS=10",
        "start_worker",
        "run_probe --mode fanout",
        "",
        "WORKER_CONCURRENT_JOBS=2",
        "start_worker",
        "run_probe --mode bounded",
        "run_probe --mode failure",
        'run_probe --mode shutdown-seed >"$SHUTDOWN_STATE_FILE"',
        "stop_worker 1",
        "start_worker",
        'run_probe --mode shutdown-check --state-file "$SHUTDOWN_STATE_FILE"',
        "",
        'echo "worker concurrency probe suite completed in '
        '$((SECONDS - WORKER_CONCURRENCY_SUITE_STARTED_SECONDS))s" >&2',
    ]
    assert suite.count("start_worker") == 3
    assert suite.count("WORKER_CONCURRENT_JOBS=10") == 1
    assert suite.count("WORKER_CONCURRENT_JOBS=2") == 1
    assert 'local timeout="${1:-10}"' in script
    assert 'docker stop -t "$timeout" "$WORKER_CONTAINER"' in script
    assert 'SHUTDOWN_STATE_FILE="$(mktemp ' in script
    assert 'rm -f "$SHUTDOWN_STATE_FILE"' in script
    assert 'redis-cli DEL "$REDIS_WORKER_LOCK_KEY"' in script
    assert "print_logs >&2" in script


def test_redis_runtime_logs_concurrency_suite_timing_without_polluting_probe_json() -> None:
    script = Path("scripts/test-redis-runtime.sh").read_text()

    assert "WORKER_CONCURRENCY_SUITE_STARTED_SECONDS=$SECONDS" in script
    assert 'echo "worker concurrency probe suite started" >&2' in script
    assert (
        'echo "worker concurrency probe suite completed in '
        '$((SECONDS - WORKER_CONCURRENCY_SUITE_STARTED_SECONDS))s" >&2'
    ) in script
    assert 'run_probe --mode shutdown-seed >"$SHUTDOWN_STATE_FILE"' in script
