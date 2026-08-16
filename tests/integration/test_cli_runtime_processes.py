from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest


PROBE_SITE_DIR = (
    Path(__file__).resolve().parents[1] / "fixtures" / "runtime_settings_probe"
)
TERMINATION_TREE_FIXTURE = (
    Path(__file__).resolve().parents[1] / "fixtures" / "termination_tree.py"
)
VALIDATION_CHILD_PID_PATH_ENV = "AGENTSEEK_VALIDATION_CHILD_PID_PATH"
TERMINATION_PROBE_PATH_ENV = "AGENTSEEK_TERMINATION_PROBE_PATH"


def _probe_pythonpath() -> str:
    existing_pythonpath = os.environ.get("PYTHONPATH")
    pythonpath = str(PROBE_SITE_DIR)
    if existing_pythonpath:
        pythonpath = os.pathsep.join((pythonpath, existing_pythonpath))
    return pythonpath


def _pid_is_alive(pid: int) -> bool:
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        open_process = kernel32.OpenProcess
        open_process.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        open_process.restype = wintypes.HANDLE
        get_exit_code = kernel32.GetExitCodeProcess
        get_exit_code.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        get_exit_code.restype = wintypes.BOOL
        close_handle = kernel32.CloseHandle
        close_handle.argtypes = [wintypes.HANDLE]
        close_handle.restype = wintypes.BOOL
        handle = open_process(0x1000, False, pid)
        if not handle:
            return False
        try:
            exit_code = wintypes.DWORD()
            if not get_exit_code(handle, ctypes.byref(exit_code)):
                return False
            return exit_code.value == 259
        finally:
            close_handle(handle)

    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _read_observed_pid(path: Path, *, timeout_seconds: float = 2.0) -> int:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            return int(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            time.sleep(0.01)
    raise AssertionError("Runtime validation child PID was not observed.")


def _terminate_observed_pid(pid: int, *, timeout_seconds: float = 2.0) -> None:
    if not _pid_is_alive(pid):
        return
    os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if not _pid_is_alive(pid):
            return
        time.sleep(0.01)
    kill_signal = getattr(signal, "SIGKILL", signal.SIGTERM)
    os.kill(pid, kill_signal)
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if not _pid_is_alive(pid):
            return
        time.sleep(0.01)
    raise AssertionError(f"Runtime validation child PID {pid} did not exit.")


def _read_tree_pids(
    path: Path,
    *,
    timeout_seconds: float = 5.0,
) -> tuple[int, int]:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return int(payload["parent"]), int(payload["grandchild"])
        except (FileNotFoundError, KeyError, TypeError, ValueError):
            time.sleep(0.01)
    raise AssertionError("The supervised parent/grandchild PIDs were not observed.")


def _wait_for_pids_gone(
    pids: tuple[int, ...],
    *,
    timeout_seconds: float = 8.0,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if not any(_pid_is_alive(pid) for pid in pids):
            return
        time.sleep(0.02)
    live_pids = [pid for pid in pids if _pid_is_alive(pid)]
    raise AssertionError(f"Supervised process IDs remained alive: {live_pids}")


def _stop_test_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.communicate(timeout=3)
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate(timeout=3)


def _cleanup_recorded_pids(pids: tuple[int, ...]) -> None:
    for pid in pids:
        _terminate_observed_pid(pid, timeout_seconds=3.0)


def _run_python(
    *arguments: str,
    cwd: Path,
    extra_env: dict[str, str] | None = None,
    removed_env: tuple[str, ...] = (),
) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env.update(extra_env or {})
    for field in removed_env:
        env.pop(field, None)
    return subprocess.run(
        [sys.executable, *arguments],
        cwd=cwd,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )


def _write_runtime_config(
    root: Path,
    name: str,
    env_mapping: dict[str, object],
) -> Path:
    config_path = root / f"{name}.json"
    config_path.write_text(
        json.dumps(
            {
                "graphs": {"chat": "chat.graph:graph"},
                "env": env_mapping,
            }
        ),
        encoding="utf-8",
    )
    return config_path


def _settings_probe_environment(
    *,
    output_path: Path,
    fields: tuple[str, ...],
    exit_code: int,
) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONPATH": _probe_pythonpath(),
            "AGENTSEEK_SETTINGS_PROBE_PATH": str(output_path),
            "AGENTSEEK_SETTINGS_PROBE_FIELDS": ",".join(fields),
            "AGENTSEEK_SETTINGS_PROBE_EXIT_CODE": str(exit_code),
        }
    )
    for field in fields:
        environment.pop(field, None)
    return environment


def _run_role_probe(
    *,
    role: str,
    config_path: Path,
    environment: dict[str, str],
) -> tuple[int, int, str]:
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "agentseek_api.cli",
            role,
            "--config",
            str(config_path),
        ],
        cwd=config_path.parent,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        _stdout, stderr = process.communicate(timeout=20)
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()
        raise
    return process.returncode, process.pid, stderr


def test_cli_import_does_not_import_runtime_settings(tmp_path: Path) -> None:
    result = _run_python(
        "-c",
        (
            "import sys; "
            "import agentseek_api.cli; "
            "assert 'agentseek_api.settings' not in sys.modules"
        ),
        cwd=tmp_path,
        extra_env={"PORT": "not-an-integer"},
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "arguments",
    [
        ("-m", "agentseek_api.cli", "version"),
        ("-m", "agentseek_api.cli", "--help"),
    ],
    ids=["version", "help"],
)
def test_non_runtime_commands_ignore_invalid_runtime_settings(
    tmp_path: Path,
    arguments: tuple[str, ...],
) -> None:
    result = _run_python(
        *arguments,
        cwd=tmp_path,
        extra_env={"PORT": "not-an-integer"},
    )

    assert result.returncode == 0, result.stderr
    assert "ValidationError" not in result.stderr


def test_dockerfile_rendering_ignores_invalid_runtime_settings(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "langgraph.json"
    config_path.write_text(
        '{"graphs":{"chat":"chat.graph:graph"}}',
        encoding="utf-8",
    )
    output_path = tmp_path / "Dockerfile"

    result = _run_python(
        "-m",
        "agentseek_api.cli",
        "dockerfile",
        "--config",
        str(config_path),
        str(output_path),
        cwd=tmp_path,
        extra_env={"PORT": "not-an-integer"},
    )

    assert result.returncode == 0, result.stderr
    assert output_path.exists()
    assert "ValidationError" not in result.stderr


@pytest.mark.parametrize(
    ("role", "invalid_field", "invalid_value", "error_type"),
    [
        ("dev", "PORT", "invalid-port-canary", "int_parsing"),
        ("serve", "PORT", "invalid-port-canary", "int_parsing"),
        (
            "worker",
            "WORKER_CONCURRENT_JOBS",
            "invalid-jobs-canary",
            "int_parsing",
        ),
        (
            "scheduler",
            "WORKER_CONCURRENT_JOBS",
            "invalid-jobs-canary",
            "int_parsing",
        ),
    ],
)
def test_invalid_runtime_setting_is_redacted_and_fresh_child_exits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    role: str,
    invalid_field: str,
    invalid_value: str,
    error_type: str,
) -> None:
    monkeypatch.setenv(
        invalid_field,
        "2024" if invalid_field == "PORT" else "10",
    )
    validation_child_pid_path = tmp_path / f"{role}-validation-child.pid"
    config_path = _write_runtime_config(
        tmp_path,
        f"invalid-{role}",
        {
            invalid_field: invalid_value,
            VALIDATION_CHILD_PID_PATH_ENV: str(validation_child_pid_path),
        },
    )
    arguments = [
        "-m",
        "agentseek_api.cli",
        role,
        "--config",
        str(config_path),
    ]
    if role == "dev":
        arguments.append("--no-reload")

    observed_pid: int | None = None
    child_alive_after_cli: bool | None = None
    try:
        result = _run_python(
            *arguments,
            cwd=tmp_path,
            extra_env={"PYTHONPATH": _probe_pythonpath()},
            removed_env=(invalid_field, VALIDATION_CHILD_PID_PATH_ENV),
        )
        observed_pid = _read_observed_pid(validation_child_pid_path)
        child_alive_after_cli = _pid_is_alive(observed_pid)
    finally:
        if observed_pid is None and validation_child_pid_path.exists():
            observed_pid = int(validation_child_pid_path.read_text(encoding="utf-8"))
        if observed_pid is not None:
            _terminate_observed_pid(observed_pid)

    assert result.returncode == 2
    assert result.stderr == (
        f"Invalid runtime setting(s): {invalid_field} ({error_type}).\n"
    )
    assert invalid_value not in result.stderr
    assert "ValidationError" not in result.stderr
    assert "input_value" not in result.stderr
    assert "Traceback" not in result.stderr
    assert child_alive_after_cli is False


def test_invalid_internal_runtime_target_returns_fixed_error(
    tmp_path: Path,
) -> None:
    result = _run_python(
        "-m",
        "agentseek_api.runtime_entrypoint",
        "invalid-target",
        cwd=tmp_path,
    )

    assert result.returncode == 2
    assert result.stderr == "Invalid internal runtime target.\n"


@pytest.mark.parametrize(
    ("role", "env_mapping", "fields", "expected", "exit_code"),
    [
        (
            "worker",
            {
                "EXECUTOR_BACKEND": "redis",
                "WORKER_CONCURRENT_JOBS": 3,
                "REDIS_URL": "redis://worker.example:6379/1",
            },
            (
                "EXECUTOR_BACKEND",
                "WORKER_CONCURRENT_JOBS",
                "REDIS_URL",
            ),
            {
                "EXECUTOR_BACKEND": "redis",
                "WORKER_CONCURRENT_JOBS": 3,
                "REDIS_URL": "redis://worker.example:6379/1",
            },
            17,
        ),
        (
            "scheduler",
            {
                "SCHEDULER_CLAIM_LIMIT": 23,
                "SCHEDULER_POLL_INTERVAL_SECONDS": 0.25,
                "REDIS_URL": "redis://scheduler.example:6379/2",
            },
            (
                "SCHEDULER_CLAIM_LIMIT",
                "SCHEDULER_POLL_INTERVAL_SECONDS",
                "REDIS_URL",
            ),
            {
                "SCHEDULER_CLAIM_LIMIT": 23,
                "SCHEDULER_POLL_INTERVAL_SECONDS": 0.25,
                "REDIS_URL": "redis://scheduler.example:6379/2",
            },
            19,
        ),
    ],
    ids=["worker", "scheduler"],
)
def test_runtime_role_default_path_observes_settings_in_fresh_child(
    tmp_path: Path,
    role: str,
    env_mapping: dict[str, object],
    fields: tuple[str, ...],
    expected: dict[str, object],
    exit_code: int,
) -> None:
    config_path = _write_runtime_config(
        tmp_path,
        f"{role}-config",
        env_mapping,
    )
    probe_output = tmp_path / f"{role}-settings.json"
    environment = _settings_probe_environment(
        output_path=probe_output,
        fields=fields,
        exit_code=exit_code,
    )

    actual_exit_code, cli_pid, stderr = _run_role_probe(
        role=role,
        config_path=config_path,
        environment=environment,
    )

    assert actual_exit_code == exit_code
    assert stderr == ""
    observation = json.loads(probe_output.read_text(encoding="utf-8"))
    assert observation["pid"] != cli_pid
    assert observation["settings"] == expected


def test_sequential_worker_invocations_do_not_reuse_settings_singleton(
    tmp_path: Path,
) -> None:
    observed: list[int] = []
    for index, concurrent_jobs in enumerate((2, 7), start=1):
        config_path = _write_runtime_config(
            tmp_path,
            f"worker-{index}",
            {
                "EXECUTOR_BACKEND": "redis",
                "WORKER_CONCURRENT_JOBS": concurrent_jobs,
            },
        )
        probe_output = tmp_path / f"worker-{index}.json"
        environment = _settings_probe_environment(
            output_path=probe_output,
            fields=("WORKER_CONCURRENT_JOBS",),
            exit_code=0,
        )

        exit_code, cli_pid, stderr = _run_role_probe(
            role="worker",
            config_path=config_path,
            environment=environment,
        )
        assert exit_code == 0
        assert stderr == ""
        observation = json.loads(probe_output.read_text(encoding="utf-8"))
        assert observation["pid"] != cli_pid
        observed.append(observation["settings"]["WORKER_CONCURRENT_JOBS"])

    assert observed == [2, 7]


def _start_supervisor_wrapper(
    *,
    command: list[str],
    cwd: Path,
) -> subprocess.Popen[str]:
    wrapper = (
        "import os, sys; "
        "from agentseek_api.cli import _default_runner; "
        "raise SystemExit(_default_runner(sys.argv[1:], "
        "env=dict(os.environ), cwd=None))"
    )
    return subprocess.Popen(
        [sys.executable, "-c", wrapper, *command],
        cwd=cwd,
        env=dict(os.environ),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def test_external_sigterm_forwards_and_leaves_no_runtime_child(
    tmp_path: Path,
) -> None:
    result_path = tmp_path / "termination-tree.json"
    process = _start_supervisor_wrapper(
        command=[
            sys.executable,
            str(TERMINATION_TREE_FIXTURE),
            str(result_path),
        ],
        cwd=tmp_path,
    )
    recorded_pids: tuple[int, ...] = ()
    try:
        parent_pid, grandchild_pid = _read_tree_pids(result_path)
        recorded_pids = (parent_pid, grandchild_pid)
        if os.name == "nt":
            process.terminate()
        else:
            os.kill(process.pid, signal.SIGTERM)
        stdout, stderr = process.communicate(timeout=15)

        if os.name != "nt":
            assert process.returncode == 128 + signal.SIGTERM
        assert stdout == ""
        assert stderr == ""
        _wait_for_pids_gone(recorded_pids)
    finally:
        _stop_test_process(process)
        _cleanup_recorded_pids(recorded_pids)


def test_normal_child_return_reaps_remaining_grandchild(
    tmp_path: Path,
) -> None:
    result_path = tmp_path / "normal-return-tree.json"
    sentinel_exit_code = 37
    process = _start_supervisor_wrapper(
        command=[
            sys.executable,
            str(TERMINATION_TREE_FIXTURE),
            str(result_path),
            "--parent-exit",
            str(sentinel_exit_code),
        ],
        cwd=tmp_path,
    )
    recorded_pids: tuple[int, ...] = ()
    try:
        recorded_pids = _read_tree_pids(result_path)
        stdout, stderr = process.communicate(timeout=15)

        assert process.returncode == sentinel_exit_code
        assert stdout == ""
        assert stderr == ""
        _wait_for_pids_gone(recorded_pids)
    finally:
        _stop_test_process(process)
        _cleanup_recorded_pids(recorded_pids)


@pytest.mark.skipif(os.name == "nt", reason="POSIX signal masks only")
def test_supervised_posix_child_starts_with_forwarded_signals_unblocked(
    tmp_path: Path,
) -> None:
    result_path = tmp_path / "startup-mask-tree.json"
    process = _start_supervisor_wrapper(
        command=[
            sys.executable,
            str(TERMINATION_TREE_FIXTURE),
            str(result_path),
            "--parent-exit",
            "0",
        ],
        cwd=tmp_path,
    )
    recorded_pids: tuple[int, ...] = ()
    try:
        recorded_pids = _read_tree_pids(result_path)
        observation = json.loads(result_path.read_text(encoding="utf-8"))
        stdout, stderr = process.communicate(timeout=15)

        assert process.returncode == 0
        assert observation["blocked_signals"] == []
        assert stdout == ""
        assert stderr == ""
        _wait_for_pids_gone(recorded_pids)
    finally:
        _stop_test_process(process)
        _cleanup_recorded_pids(recorded_pids)


def test_supervised_child_preserves_captured_stdout(
    tmp_path: Path,
) -> None:
    result_path = tmp_path / "captured-output-tree.json"
    output_marker = "captured-child-output"
    process = _start_supervisor_wrapper(
        command=[
            sys.executable,
            str(TERMINATION_TREE_FIXTURE),
            str(result_path),
            "--parent-exit",
            "0",
            "--output-marker",
            output_marker,
        ],
        cwd=tmp_path,
    )
    recorded_pids: tuple[int, ...] = ()
    try:
        recorded_pids = _read_tree_pids(result_path)
        stdout, stderr = process.communicate(timeout=15)

        assert process.returncode == 0
        assert stdout == f"{output_marker}\n"
        assert stderr == ""
        _wait_for_pids_gone(recorded_pids)
    finally:
        _stop_test_process(process)
        _cleanup_recorded_pids(recorded_pids)


@pytest.mark.parametrize("role", ["worker", "scheduler"])
def test_public_runtime_role_sigterm_reaps_role_tree(
    tmp_path: Path,
    role: str,
) -> None:
    config_path = _write_runtime_config(tmp_path, f"{role}-termination", {})
    result_path = tmp_path / f"{role}-termination-tree.json"
    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONPATH": _probe_pythonpath(),
            TERMINATION_PROBE_PATH_ENV: str(result_path),
        }
    )
    for field in (
        "AGENTSEEK_SETTINGS_PROBE_PATH",
        "AGENTSEEK_SETTINGS_PROBE_FIELDS",
        "AGENTSEEK_SETTINGS_PROBE_EXIT_CODE",
        VALIDATION_CHILD_PID_PATH_ENV,
    ):
        environment.pop(field, None)
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "agentseek_api.cli",
            role,
            "--config",
            str(config_path),
        ],
        cwd=tmp_path,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    recorded_pids: tuple[int, ...] = ()
    try:
        parent_pid, grandchild_pid = _read_tree_pids(result_path)
        recorded_pids = (parent_pid, grandchild_pid)
        assert parent_pid != process.pid
        if os.name == "nt":
            process.terminate()
        else:
            os.kill(process.pid, signal.SIGTERM)
        stdout, stderr = process.communicate(timeout=15)

        if os.name != "nt":
            assert process.returncode == 128 + signal.SIGTERM
        assert stdout == ""
        assert stderr == ""
        _wait_for_pids_gone(recorded_pids)
    finally:
        _stop_test_process(process)
        _cleanup_recorded_pids(recorded_pids)
