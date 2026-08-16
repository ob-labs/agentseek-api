from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


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
    )


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
