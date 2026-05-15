from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import pytest

from agentseek_api.services.langgraph_service import LangGraphService


@dataclass
class _RunCapture:
    calls: list[list[str]] | None = None
    command: list[str] | None = None
    env: dict[str, str] | None = None
    cwd: str | None = None

    def __call__(self, command: list[str], *, env: dict[str, str], cwd: str | None = None) -> int:
        if self.calls is None:
            self.calls = []
        self.calls.append(command)
        self.command = command
        self.env = env
        self.cwd = cwd
        return 0


def _write_basic_langgraph_config(root: Path) -> Path:
    package_dir = root / "chat"
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "graph.py").write_text(
        """
from langgraph.graph import END, START, StateGraph

builder = StateGraph(dict)
builder.add_node("node", lambda state: {"value": "basic-config"})
builder.add_edge(START, "node")
builder.add_edge("node", END)
graph = builder.compile()
""".strip(),
        encoding="utf-8",
    )
    config_path = root / "langgraph.json"
    config_path.write_text(
        """
{
  "$schema": "https://langgra.ph/schema.json",
  "dependencies": ["."],
  "graphs": {
    "chat": "chat.graph:graph"
  }
}
""".strip(),
        encoding="utf-8",
    )
    return config_path


def test_dev_command_prefers_agentseek_json_over_langgraph_json(tmp_path: Path) -> None:
    from agentseek_api.cli import main

    config_path = tmp_path / "agentseek.json"
    config_path.write_text('{"graphs":{"agentseek":"chat.graph:graph"}}', encoding="utf-8")
    _write_basic_langgraph_config(tmp_path)
    capture = _RunCapture()

    exit_code = main(["dev", "--no-reload"], runner=capture, cwd=tmp_path)

    assert exit_code == 0
    assert capture.command == ["uvicorn", "agentseek_api.main:app", "--host", "127.0.0.1", "--port", "2026"]
    assert capture.env is not None
    assert capture.env["AGENTSEEK_GRAPHS"] == str(config_path.resolve())


def test_serve_command_falls_back_to_langgraph_json_and_runs_graph(tmp_path: Path) -> None:
    from agentseek_api.cli import main

    config_path = _write_basic_langgraph_config(tmp_path)
    capture = _RunCapture()

    exit_code = main(["serve", "--host", "0.0.0.0", "--port", "3030"], runner=capture, cwd=tmp_path)

    assert exit_code == 0
    assert capture.command == ["uvicorn", "agentseek_api.main:app", "--host", "0.0.0.0", "--port", "3030"]
    assert capture.env is not None
    assert capture.env["AGENTSEEK_GRAPHS"] == str(config_path.resolve())

    service = LangGraphService(manifest_path=capture.env["AGENTSEEK_GRAPHS"])
    result = service.get_entry("chat").build_graph().invoke({})
    assert result["value"] == "basic-config"


def test_dev_command_accepts_langgraph_cli_flags_and_env_file(tmp_path: Path) -> None:
    from agentseek_api.cli import main

    config_path = _write_basic_langgraph_config(tmp_path)
    env_file = tmp_path / ".env"
    env_file.write_text("AUTH_TYPE=custom\nAUTH_MODULE_PATH=test.module:backend\n", encoding="utf-8")
    capture = _RunCapture()

    exit_code = main(
        [
            "dev",
            "--config",
            str(config_path),
            "--host",
            "0.0.0.0",
            "--port",
            "9999",
            "--no-reload",
            "--env-file",
            str(env_file),
        ],
        runner=capture,
        cwd=tmp_path,
    )

    assert exit_code == 0
    assert capture.command == ["uvicorn", "agentseek_api.main:app", "--host", "0.0.0.0", "--port", "9999"]
    assert capture.env is not None
    assert capture.env["AGENTSEEK_GRAPHS"] == str(config_path.resolve())
    assert capture.env["AUTH_TYPE"] == "custom"
    assert capture.env["AUTH_MODULE_PATH"] == "test.module:backend"


def test_dev_command_rejects_unsupported_langgraph_flags(tmp_path: Path) -> None:
    from agentseek_api.cli import main

    _write_basic_langgraph_config(tmp_path)
    stderr = io.StringIO()

    exit_code = main(["dev", "--tunnel"], cwd=tmp_path, stderr=stderr)

    assert exit_code == 2
    assert "Unsupported option(s) for 'agentseek dev': --tunnel" in stderr.getvalue()


def test_version_reports_cli_and_package_versions() -> None:
    from agentseek_api import __version__
    from agentseek_api.cli import main

    stdout = io.StringIO()

    exit_code = main(["version"], stdout=stdout)

    assert exit_code == 0
    assert stdout.getvalue().strip().splitlines() == [
        f"agentseek {__version__}",
        f"agentseek-api {__version__}",
    ]


def test_dockerfile_command_writes_langgraph_compatible_runtime_file(tmp_path: Path) -> None:
    from agentseek_api.cli import main

    _write_basic_langgraph_config(tmp_path)
    dockerfile_path = tmp_path / "Dockerfile.agentseek"

    exit_code = main(["dockerfile", str(dockerfile_path)], cwd=tmp_path)

    assert exit_code == 0
    content = dockerfile_path.read_text(encoding="utf-8")
    assert 'FROM python:3.12-slim' in content
    assert 'RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*' in content
    assert 'WORKDIR /deps/agent' in content
    assert 'COPY . /deps/agent' in content
    assert 'ENV PYTHONPATH=/deps/agent' in content
    assert 'ENV AGENTSEEK_GRAPHS=/deps/agent/langgraph.json' in content
    assert 'CMD ["agentseek", "serve", "--host", "0.0.0.0", "--port", "2026"]' in content


def test_build_command_plans_docker_build_from_generated_dockerfile(tmp_path: Path) -> None:
    from agentseek_api.cli import main

    _write_basic_langgraph_config(tmp_path)
    capture = _RunCapture()

    exit_code = main(
        [
            "build",
            "-t",
            "agentseek:test",
            "--platform",
            "linux/amd64,linux/arm64",
            "--no-pull",
        ],
        runner=capture,
        cwd=tmp_path,
    )

    assert exit_code == 0
    assert capture.command is not None
    assert capture.command[:8] == [
        "docker",
        "build",
        "--platform",
        "linux/amd64,linux/arm64",
        "-t",
        "agentseek:test",
        "-f",
        str((tmp_path / ".agentseek" / "Dockerfile").resolve()),
    ]
    assert capture.command[-1] == "."
    generated = (tmp_path / ".agentseek" / "Dockerfile").read_text(encoding="utf-8")
    assert 'RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*' in generated
    assert 'ENV PYTHONPATH=/deps/agent' in generated
    assert 'ENV AGENTSEEK_GRAPHS=/deps/agent/langgraph.json' in generated
    assert 'CMD ["agentseek", "serve", "--host", "0.0.0.0", "--port", "2026"]' in generated


def test_build_runtime_env_rejects_invalid_env_lines(tmp_path: Path) -> None:
    from agentseek_api.cli import build_runtime_env

    env_file = tmp_path / ".env"
    env_file.write_text("BROKEN_LINE\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="invalid line 1"):
        build_runtime_env(config_path=None, env_file=str(env_file), cwd=tmp_path, base_env={})


def test_build_runtime_env_parses_exported_values(tmp_path: Path) -> None:
    from agentseek_api.cli import build_runtime_env

    config_path = _write_basic_langgraph_config(tmp_path)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# comment\nexport TOKEN='quoted-value'\nPLAIN=value\n",
        encoding="utf-8",
    )

    env = build_runtime_env(config_path=config_path, env_file=str(env_file), cwd=tmp_path, base_env={})

    assert env["TOKEN"] == "quoted-value"
    assert env["PLAIN"] == "value"
    assert env["AGENTSEEK_GRAPHS"] == str(config_path.resolve())


def test_dockerfile_command_requires_valid_config_object(tmp_path: Path) -> None:
    from agentseek_api.cli import main

    config_path = tmp_path / "langgraph.json"
    config_path.write_text("[]", encoding="utf-8")
    stderr = io.StringIO()

    exit_code = main(["dockerfile", "--config", str(config_path), "Dockerfile"], cwd=tmp_path, stderr=stderr)

    assert exit_code == 2
    assert "must contain a top-level JSON object" in stderr.getvalue()


def test_build_command_requires_config_file(tmp_path: Path) -> None:
    from agentseek_api.cli import main

    stderr = io.StringIO()

    exit_code = main(["build", "-t", "agentseek:test"], cwd=tmp_path, stderr=stderr)

    assert exit_code == 2
    assert "No config file found" in stderr.getvalue()


def test_up_command_plans_docker_run_with_recreate_and_env_file(tmp_path: Path) -> None:
    from agentseek_api.cli import main

    config_path = _write_basic_langgraph_config(tmp_path)
    env_file = tmp_path / "docker.env"
    env_file.write_text(
        "METADATA_DB_URL=sqlite+aiosqlite:////tmp/agentseek.db\nOCEANBASE_HOST=host.docker.internal\n",
        encoding="utf-8",
    )
    capture = _RunCapture()

    exit_code = main(
        [
            "up",
            "--config",
            str(config_path),
            "--image",
            "agentseek:test",
            "--port",
            "8123",
            "--env-file",
            str(env_file),
            "--recreate",
        ],
        runner=capture,
        cwd=tmp_path,
    )

    assert exit_code == 0
    assert capture.calls is not None
    assert capture.calls[0] == ["docker", "rm", "-f", "agentseek-up-8123"]
    assert capture.calls[1] == [
        "docker",
        "run",
        "--detach",
        "--rm",
        "--name",
        "agentseek-up-8123",
        "--add-host",
        "host.docker.internal:host-gateway",
        "-p",
        "8123:2026",
        "--env-file",
        str(env_file.resolve()),
        "-e",
        "AGENTSEEK_GRAPHS=/deps/agent/langgraph.json",
        "agentseek:test",
    ]


def test_up_command_builds_image_when_missing_and_passes_postgres_uri(tmp_path: Path) -> None:
    from agentseek_api.cli import main

    _write_basic_langgraph_config(tmp_path)
    capture = _RunCapture()

    exit_code = main(
        [
            "up",
            "--port",
            "8124",
            "--no-pull",
            "--postgres-uri",
            "postgresql://postgres:postgres@db/agentseek",
        ],
        runner=capture,
        cwd=tmp_path,
    )

    assert exit_code == 0
    assert capture.calls is not None
    assert capture.calls[0] == [
        "docker",
        "build",
        "-t",
        "agentseek-up:8124",
        "-f",
        str((tmp_path / ".agentseek" / "Dockerfile").resolve()),
        ".",
    ]
    assert capture.calls[1] == [
        "docker",
        "run",
        "--detach",
        "--rm",
        "--name",
        "agentseek-up-8124",
        "--add-host",
        "host.docker.internal:host-gateway",
        "-p",
        "8124:2026",
        "-e",
        "AGENTSEEK_GRAPHS=/deps/agent/langgraph.json",
        "-e",
        "METADATA_DB_URL=postgresql://postgres:postgres@db/agentseek",
        "-e",
        "METADATA_DB_BACKEND=postgresql",
        "agentseek-up:8124",
    ]


def test_up_command_returns_build_failure_without_running_container(tmp_path: Path) -> None:
    from agentseek_api.cli import main

    _write_basic_langgraph_config(tmp_path)
    capture = _RunCapture()

    def fail_build(command: list[str], *, env: dict[str, str], cwd: str | None = None) -> int:
        capture(command, env=env, cwd=cwd)
        return 9 if command[:2] == ["docker", "build"] else 0

    exit_code = main(["up", "--port", "8125"], runner=fail_build, cwd=tmp_path)

    assert exit_code == 9
    assert capture.calls == [["docker", "build", "--pull", "-t", "agentseek-up:8125", "-f", str((tmp_path / ".agentseek" / "Dockerfile").resolve()), "."]]


def test_up_command_waits_for_http_health_when_requested(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from agentseek_api import cli as cli_module

    config_path = _write_basic_langgraph_config(tmp_path)
    capture = _RunCapture()
    waited: list[tuple[str, float]] = []

    def fake_wait(url: str, *, timeout_seconds: float) -> None:
        waited.append((url, timeout_seconds))

    monkeypatch.setattr(cli_module, "_wait_for_http_ready", fake_wait)

    exit_code = cli_module.main(
        [
            "up",
            "--config",
            str(config_path),
            "--image",
            "agentseek:test",
            "--port",
            "8123",
            "--wait",
        ],
        runner=capture,
        cwd=tmp_path,
    )

    assert exit_code == 0
    assert waited == [("http://127.0.0.1:8123/health", 30.0)]


def test_up_command_rejects_unsupported_langgraph_flags(tmp_path: Path) -> None:
    from agentseek_api.cli import main

    config_path = _write_basic_langgraph_config(tmp_path)
    stderr = io.StringIO()

    exit_code = main(
        ["up", "--config", str(config_path), "--image", "agentseek:test", "--watch"],
        cwd=tmp_path,
        stderr=stderr,
    )

    assert exit_code == 2
    assert "Unsupported option(s) for 'agentseek up': --watch" in stderr.getvalue()


@pytest.mark.parametrize(
    ("argv", "expected_name"),
    [
        (["deploy"], "deploy"),
    ],
)
def test_unimplemented_langgraph_commands_fail_clearly(argv: list[str], expected_name: str) -> None:
    from agentseek_api.cli import main

    stderr = io.StringIO()

    exit_code = main(argv, stderr=stderr)

    assert exit_code == 2
    assert stderr.getvalue().strip() == f"'agentseek {expected_name}' is not implemented yet in this milestone slice."
