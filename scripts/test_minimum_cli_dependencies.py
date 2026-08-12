"""Verify the CLI imports and runs with the declared minimum dotenv stack."""

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
            ["uv", "pip", "install", "--python", str(python), "pydantic-settings==2.4.0", "python-dotenv>=1.0"],
            check=True,
        )
        subprocess.run(["uv", "pip", "install", "--python", str(python), "--no-deps", "-e", str(repository)], check=True)
        subprocess.run(
            [str(python), "-c", "from agentseek_api.cli import main; raise SystemExit(main(['version']))"],
            check=True,
        )


if __name__ == "__main__":
    main()
