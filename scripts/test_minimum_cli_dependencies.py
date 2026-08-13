"""Verify the installed CLI runs with the declared minimum dotenv stack."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> None:
    repository = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="agentseek-minimum-") as directory:
        environment = Path(directory) / ".venv"
        subprocess.run(["uv", "venv", "--python", sys.executable, str(environment)], check=True)
        python = environment / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
        subprocess.run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(python),
                "pydantic-settings==2.4.0",
                "pydantic==2.8.0",
                "python-dotenv==1.0.0",
            ],
            check=True,
        )
        subprocess.run(["uv", "pip", "install", "--python", str(python), "--no-deps", "-e", str(repository)], check=True)
        cli = environment / ("Scripts/agentseek-api.exe" if sys.platform == "win32" else "bin/agentseek-api")
        result = subprocess.run([str(cli), "version"], check=True, capture_output=True, text=True)
        assert result.stdout.strip() == "agentseek-api 0.2.1"
        conformance = subprocess.run(
            [
                str(python),
                "-c",
                (
                    "import sys; "
                    f"sys.path.insert(0, {str(repository / 'scripts')!r}); "
                    "from dotenv_conformance import assert_runtime_conformance; "
                    "assert_runtime_conformance(expected_dotenv_version='1.0.0')"
                ),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if conformance.returncode != 0:
            raise RuntimeError(f"Minimum dependency dotenv conformance failed: {conformance.stderr.strip()}")


if __name__ == "__main__":
    main()
