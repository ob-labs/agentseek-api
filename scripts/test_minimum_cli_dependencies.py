"""Verify host environment resolution with all direct dependency floors."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path


def main() -> None:
    repository = Path(__file__).resolve().parents[1]
    project = tomllib.loads((repository / "pyproject.toml").read_text(encoding="utf-8"))
    requirements = list(project["project"]["dependencies"])
    requirements.append("pytest>=8.0.0")
    requirements.append("pytest-asyncio>=0.23.5")

    with tempfile.TemporaryDirectory(prefix="agentseek-minimum-") as directory:
        environment = Path(directory) / ".venv"
        subprocess.run(
            [
                "uv",
                "venv",
                "--python",
                "3.12",
                str(environment),
            ],
            check=True,
        )
        python = environment / (
            "Scripts/python.exe" if sys.platform == "win32" else "bin/python"
        )
        subprocess.run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(python),
                "--resolution",
                "lowest-direct",
                *requirements,
            ],
            check=True,
        )
        subprocess.run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(python),
                "--no-deps",
                "-e",
                str(repository),
            ],
            check=True,
        )

        version = subprocess.run(
            [
                str(python),
                "-c",
                (
                    "import importlib.metadata; "
                    "print(importlib.metadata.version('python-dotenv'))"
                ),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        assert version.stdout.strip() == "1.0.0"

        cli_env = dict(os.environ)
        cli_env.pop("PYTHONPATH", None)
        cli_env["PORT"] = "not-an-integer"
        version_result = subprocess.run(
            [str(python), "-m", "agentseek_api.cli", "version"],
            cwd=repository,
            env=cli_env,
            check=True,
            capture_output=True,
            text=True,
        )
        expected_version = project["project"]["version"]
        assert version_result.stdout.strip() == f"agentseek-api {expected_version}"

        test_env = dict(os.environ)
        test_env.pop("PYTHONPATH", None)
        subprocess.run(
            [
                str(python),
                "-m",
                "pytest",
                "tests/unit/test_dotenv_adapter.py",
                "tests/unit/test_runtime_environment.py",
                "-q",
            ],
            cwd=repository,
            env=test_env,
            check=True,
        )


if __name__ == "__main__":
    main()
