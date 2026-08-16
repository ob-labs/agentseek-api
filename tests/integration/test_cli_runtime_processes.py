from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


PROBE_SITE_DIR = (
    Path(__file__).resolve().parents[1] / "fixtures" / "runtime_settings_probe"
)


def _run_python(
    *arguments: str,
    cwd: Path,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env.update(extra_env or {})
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
    existing_pythonpath = environment.get("PYTHONPATH")
    pythonpath = str(PROBE_SITE_DIR)
    if existing_pythonpath:
        pythonpath = os.pathsep.join((pythonpath, existing_pythonpath))
    environment.update(
        {
            "PYTHONPATH": pythonpath,
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
def test_invalid_runtime_setting_is_redacted_by_fresh_child(
    tmp_path: Path,
    role: str,
    invalid_field: str,
    invalid_value: str,
    error_type: str,
) -> None:
    config_path = _write_runtime_config(
        tmp_path,
        f"invalid-{role}",
        {invalid_field: invalid_value},
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

    result = _run_python(*arguments, cwd=tmp_path)

    assert result.returncode == 2
    assert result.stderr == (
        f"Invalid runtime setting(s): {invalid_field} ({error_type}).\n"
    )
    assert invalid_value not in result.stderr
    assert "ValidationError" not in result.stderr
    assert "input_value" not in result.stderr
    assert "Traceback" not in result.stderr


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
