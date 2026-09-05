from pathlib import Path


def test_redis_runtime_builds_image_from_exact_candidate_wheel() -> None:
    script = Path("scripts/test-redis-runtime.sh").read_text()

    assert "uv run agentseek-api build" not in script
    assert 'mktemp -d "$ROOT_DIR/.tmp/agentseek-redis-candidate.XXXXXX"' in script
    assert "${TMPDIR:-/tmp}/agentseek-redis-candidate" not in script
    assert 'uv build --wheel --out-dir "$CANDIDATE_DIR"' in script
    assert "agentseek_api-0.3.2-*.whl" in script
    assert "candidate_runtime_artifact" in script
    assert "runtime_artifact=artifact" in script
    assert 'rm -rf -- "$CANDIDATE_DIR"' in script


def test_redis_runtime_launches_api_and_worker_in_preloaded_mode() -> None:
    script = Path("scripts/test-redis-runtime.sh").read_text()

    assert script.count('-e AGENTSEEK_GRAPHS="/opt/agentseek/manifest.v1.json"') == 2
    assert (
        'AUTH_MODULE_PATH="${AUTH_MODULE_PATH:-/deps/agent/examples/'
        'docker_ci_auth/auth_backend.py:HeaderAuthBackend}"' in script
    )
    assert script.count('-e AUTH_MODULE_PATH="${AUTH_MODULE_PATH}"') == 2
    assert (
        "python -I -m agentseek_api.cli worker --environment-mode preloaded-v1"
        in script
    )


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


def test_redis_runtime_logs_concurrency_suite_timing_without_polluting_probe_json() -> (
    None
):
    script = Path("scripts/test-redis-runtime.sh").read_text()

    assert "WORKER_CONCURRENCY_SUITE_STARTED_SECONDS=$SECONDS" in script
    assert 'echo "worker concurrency probe suite started" >&2' in script
    assert (
        'echo "worker concurrency probe suite completed in '
        '$((SECONDS - WORKER_CONCURRENCY_SUITE_STARTED_SECONDS))s" >&2'
    ) in script
    assert 'run_probe --mode shutdown-seed >"$SHUTDOWN_STATE_FILE"' in script
