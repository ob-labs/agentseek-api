from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path


_URL_USERINFO = re.compile(r"(?P<scheme>https?://)[^/@\s]+@", re.IGNORECASE)


def _report_github_failure(message: str) -> None:
    if os.environ.get("GITHUB_ACTIONS") != "true":
        return
    normalized = " | ".join(
        line.strip() for line in message.splitlines() if line.strip()
    )
    normalized = _URL_USERINFO.sub(r"\g<scheme><redacted>@", normalized)[:800]
    if not normalized:
        normalized = "dockerfile subprocess failed without a diagnostic"
    escaped = normalized.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
    print(f"::error title=AgentSeek dockerfile smoke::{escaped}", file=sys.stderr)


def _verify_bundle(output: Path) -> tuple[Path, dict[str, object]]:
    context = output / "context"
    dockerfile = context / "Dockerfile"
    manifest_path = context / "manifest.v1.json"
    inventory_path = output / "inventory.json"
    if not all(path.is_file() for path in (dockerfile, manifest_path, inventory_path)):
        raise SystemExit("dockerfile bundle contract was incomplete")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise SystemExit("dockerfile bundle metadata was invalid") from None
    if not isinstance(manifest, dict) or not isinstance(inventory, list):
        raise SystemExit("dockerfile bundle metadata shape was invalid")
    matches = [
        item
        for item in inventory
        if isinstance(item, dict) and item.get("relative_path") == "Dockerfile"
    ]
    payload = dockerfile.read_bytes()
    if (
        len(matches) != 1
        or matches[0].get("size") != len(payload)
        or matches[0].get("sha256") != hashlib.sha256(payload).hexdigest()
    ):
        raise SystemExit("dockerfile inventory binding was invalid")
    return dockerfile, manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path)
    args = parser.parse_args(argv)
    with tempfile.TemporaryDirectory(
        prefix="agentseek-cli-autodiscovery-"
    ) as tmp_dir_text:
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

        output_path = tmp_dir / "agentseek-build-bundle"
        command = [sys.executable, "-m", "agentseek_api.cli", "dockerfile"]
        command_cwd = tmp_dir
        if args.config is not None:
            resolved_config = args.config.resolve()
            command.extend(("--config", str(resolved_config)))
            command_cwd = resolved_config.parent
        command.append(str(output_path))
        completed = subprocess.run(
            command,
            cwd=str(command_cwd),
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            _report_github_failure(completed.stderr)
            raise SystemExit("agentseek-api dockerfile bundle generation failed")

        _dockerfile, manifest = _verify_bundle(output_path)
        if args.config is None:
            assert manifest["graphs"] == {"agentseek": "chat.graph:graph"}
            assert "langgraph" not in json.dumps(manifest["graphs"])
        else:
            assert manifest.get("graphs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
