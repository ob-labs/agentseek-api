from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="agentseek-cli-autodiscovery-") as tmp_dir_text:
        tmp_dir = Path(tmp_dir_text)
        (tmp_dir / "agentseek.json").write_text(
            """
{
  "graphs": {
    "agentseek": "chat.graph:graph"
  }
}
""".strip(),
            encoding="utf-8",
        )
        (tmp_dir / "langgraph.json").write_text(
            """
{
  "graphs": {
    "langgraph": "chat.graph:graph"
  }
}
""".strip(),
            encoding="utf-8",
        )

        output_path = tmp_dir / "Dockerfile.agentseek"
        completed = subprocess.run(
            [sys.executable, "-m", "agentseek_api.cli", "dockerfile", str(output_path)],
            cwd=str(tmp_dir),
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise SystemExit(
                "agentseek-api dockerfile failed without --config:\n"
                f"stdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}"
            )

        content = output_path.read_text(encoding="utf-8")
        assert "ENV AGENTSEEK_GRAPHS=/deps/agent/agentseek.json" in content
        assert "ENV AGENTSEEK_GRAPHS=/deps/agent/langgraph.json" not in content
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
