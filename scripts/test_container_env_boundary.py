"""Verify the shared dotenv matrix through the real container handoff."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from dotenv_conformance import (
    CROSS_LAYER_TOMBSTONE_CASES,
    DOTENV_CONFORMANCE_CASES,
    DOTENV_CONFORMANCE_ENV_KEYS,
)


def _inspect_real_up(
    *,
    root: Path,
    config: Path,
    env_file: Path,
    port: int,
    process_env: dict[str, str],
) -> dict[str, str]:
    container_name = f"agentseek-up-{port}"
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "agentseek_api.cli",
                "up",
                "--config",
                str(config),
                "--image",
                "python:3.12-slim",
                "--port",
                str(port),
                "--env-file",
                str(env_file),
                "--recreate",
            ],
            cwd=root,
            env=process_env,
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"agentseek-api up smoke failed: {result.stderr.strip()}")
        inspected = subprocess.run(
            ["docker", "inspect", container_name, "--format", "{{json .Config.Env}}"],
            check=True,
            capture_output=True,
            text=True,
        )
        assert "host-sensitive-value" not in inspected.stdout
        return dict(entry.split("=", maxsplit=1) for entry in json.loads(inspected.stdout))
    finally:
        subprocess.run(
            ["docker", "rm", "-f", container_name],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def main() -> None:
    from agentseek_api.cli import _CONTAINER_ENV_PREFIXES

    inherited_allowlisted = {
        key: value
        for key, value in os.environ.items()
        if key.startswith(_CONTAINER_ENV_PREFIXES)
    }
    clean_env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(_CONTAINER_ENV_PREFIXES) and key not in DOTENV_CONFORMANCE_ENV_KEYS
    }

    with tempfile.TemporaryDirectory(prefix="agentseek-container-env-") as directory:
        root = Path(directory)
        package = root / "chat"
        package.mkdir()
        (package / "__init__.py").write_text("", encoding="utf-8")
        (package / "graph.py").write_text("graph = object()\n", encoding="utf-8")

        port = 18125
        for index, case in enumerate(DOTENV_CONFORMANCE_CASES):
            config = root / f"matrix-{index}.json"
            config.write_text('{"graphs":{"chat":"chat.graph:graph"}}\n', encoding="utf-8")
            env_file = root / f"matrix-{index}.env"
            env_file.write_text(case["contents"], encoding="utf-8")
            process_env = {**clean_env, **case["ambient"]}

            actual = _inspect_real_up(
                root=root,
                config=config,
                env_file=env_file,
                port=port,
                process_env=process_env,
            )

            expected = case["container_expected"]
            assert {key: actual[key] for key in expected} == expected, case["name"]
            assert all(key not in actual for key in case["container_absent"]), case["name"]
            port += 1

        for index, case in enumerate(CROSS_LAYER_TOMBSTONE_CASES):
            config = root / f"tombstone-{index}.json"
            config.write_text(
                json.dumps({"graphs": {"chat": "chat.graph:graph"}, "env": case["config_env"]}),
                encoding="utf-8",
            )
            if case["config_dotenv"] is not None:
                (root / "config.env").write_text(case["config_dotenv"], encoding="utf-8")
            env_file = root / f"tombstone-{index}.env"
            tombstone_key = next(iter(case["expected"]), "CONF_TOMBSTONE")
            env_file.write_text(f"{tombstone_key}\n", encoding="utf-8")
            process_env = {**clean_env, **case["shell_env"]}

            actual = _inspect_real_up(
                root=root,
                config=config,
                env_file=env_file,
                port=port,
                process_env=process_env,
            )

            assert {key: actual[key] for key in case["expected"]} == case["expected"], case["name"]
            if not case["expected"]:
                assert "CONF_TOMBSTONE" not in actual, case["name"]
            port += 1

    os.environ.update(inherited_allowlisted)
    print("container dotenv conformance passed")


if __name__ == "__main__":
    main()
