"""Verify the container handoff does not expand secrets from the host shell."""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path


def main() -> None:
    from agentseek_api.cli import build_container_env

    with tempfile.TemporaryDirectory(prefix="agentseek-container-env-") as directory:
        root = Path(directory)
        package = root / "chat"
        package.mkdir()
        (package / "__init__.py").write_text("", encoding="utf-8")
        (package / "graph.py").write_text("graph = object()\n", encoding="utf-8")
        config = root / "langgraph.json"
        config.write_text('{"graphs":{"chat":"chat.graph:graph"}}\n', encoding="utf-8")
        env_file = root / ".env"
        env_file.write_text("OPENAI_API_KEY=${PR69_DISALLOWED_SECRET}\n", encoding="utf-8")

        previous = os.environ.get("PR69_DISALLOWED_SECRET")
        os.environ["PR69_DISALLOWED_SECRET"] = "host-sensitive-value"
        try:
            container_env = build_container_env(config_path=config, env_file=str(env_file), cwd=root)
        finally:
            if previous is None:
                os.environ.pop("PR69_DISALLOWED_SECRET", None)
            else:
                os.environ["PR69_DISALLOWED_SECRET"] = previous

        assert container_env["OPENAI_API_KEY"] == "${PR69_DISALLOWED_SECRET}"
        assert "PR69_DISALLOWED_SECRET" not in container_env
        result = subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "-e",
                f"OPENAI_API_KEY={container_env['OPENAI_API_KEY']}",
                "python:3.12-slim",
                "python",
                "-c",
                "import os; print(os.environ['OPENAI_API_KEY'])",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Docker container smoke failed: {result.stderr.strip()}")
        assert result.stdout.strip() == "${PR69_DISALLOWED_SECRET}"
        assert "host-sensitive-value" not in result.stdout
        print(result.stdout.strip())


if __name__ == "__main__":
    main()
