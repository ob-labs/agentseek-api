from __future__ import annotations

import argparse
import importlib
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


def test_dev_command_loads_config_env_mapping_and_auth_path(tmp_path: Path) -> None:
    from agentseek_api.cli import main

    package_dir = tmp_path / "chat"
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "graph.py").write_text("graph = object()\n", encoding="utf-8")
    config_path = tmp_path / "langgraph.json"
    config_path.write_text(
        """
{
  "dependencies": ["."],
  "graphs": {
    "chat": "chat.graph:graph"
  },
  "env": {
    "OPENAI_API_KEY": "test-key",
    "FEATURE_FLAG": true
  },
  "auth": {
    "path": "./auth.py:auth"
  }
}
""".strip(),
        encoding="utf-8",
    )
    capture = _RunCapture()

    exit_code = main(["dev", "--config", str(config_path), "--no-reload"], runner=capture, cwd=tmp_path)

    assert exit_code == 0
    assert capture.env is not None
    assert capture.env["OPENAI_API_KEY"] == "test-key"
    assert capture.env["FEATURE_FLAG"] == "True"
    assert capture.env["AUTH_TYPE"] == "custom"
    assert capture.env["AUTH_MODULE_PATH"] == f"{(tmp_path / 'auth.py').resolve()}:auth"


def test_dev_command_merges_config_env_file_before_cli_env_file(tmp_path: Path) -> None:
    from agentseek_api.cli import main

    config_path = _write_basic_langgraph_config(tmp_path)
    config_env = tmp_path / "config.env"
    config_env.write_text("TOKEN=from-config\nSHARED=config\n", encoding="utf-8")
    config_path.write_text(
        """
{
  "$schema": "https://langgra.ph/schema.json",
  "dependencies": ["."],
  "graphs": {
    "chat": "chat.graph:graph"
  },
  "env": "./config.env"
}
""".strip(),
        encoding="utf-8",
    )
    cli_env = tmp_path / "override.env"
    cli_env.write_text("SHARED=override\n", encoding="utf-8")
    capture = _RunCapture()

    exit_code = main(
        ["dev", "--config", str(config_path), "--env-file", str(cli_env), "--no-reload"],
        runner=capture,
        cwd=tmp_path,
    )

    assert exit_code == 0
    assert capture.env is not None
    assert capture.env["TOKEN"] == "from-config"
    assert capture.env["SHARED"] == "override"


def test_dev_command_rejects_unsupported_langgraph_flags(tmp_path: Path) -> None:
    from agentseek_api.cli import main

    _write_basic_langgraph_config(tmp_path)
    stderr = io.StringIO()

    exit_code = main(["dev", "--tunnel"], cwd=tmp_path, stderr=stderr)

    assert exit_code == 2
    assert "Unsupported option(s) for 'agentseek dev': --tunnel" in stderr.getvalue()


def test_dev_command_rejects_missing_explicit_config(tmp_path: Path) -> None:
    from agentseek_api.cli import main

    stderr = io.StringIO()

    exit_code = main(["dev", "--config", str(tmp_path / "missing.json")], cwd=tmp_path, stderr=stderr)

    assert exit_code == 2
    assert "does not exist" in stderr.getvalue()


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


def test_cli_module_is_importable_with_embeddable_entrypoints() -> None:
    cli_module = importlib.import_module("agentseek_api.cli")

    assert cli_module.main is not None
    assert cli_module.create_parser is not None
    assert cli_module.register_subcommands is not None
    assert cli_module.run_namespace is not None


def test_register_subcommands_supports_embedding_under_parent_parser() -> None:
    from agentseek_api import cli as cli_module

    parser = argparse.ArgumentParser(prog="parent")
    subparsers = parser.add_subparsers(dest="tool", required=True)
    cli_module.register_subcommands(subparsers, command_name="agentseek")

    parsed = parser.parse_args(["agentseek", "version"])

    assert parsed.tool == "agentseek"
    assert parsed.command == "version"


def test_run_namespace_allows_parent_cli_dispatch(tmp_path: Path) -> None:
    from agentseek_api import cli as cli_module

    parser = argparse.ArgumentParser(prog="parent")
    subparsers = parser.add_subparsers(dest="tool", required=True)
    cli_module.register_subcommands(subparsers, command_name="agentseek")
    parsed = parser.parse_args(["agentseek", "serve", "--host", "0.0.0.0", "--port", "3030"])
    capture = _RunCapture()

    exit_code = cli_module.run_namespace(parsed, runner=capture, cwd=tmp_path)

    assert exit_code == 0
    assert capture.command == ["uvicorn", "agentseek_api.main:app", "--host", "0.0.0.0", "--port", "3030"]


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


def test_dockerfile_command_honors_base_image_python_and_custom_lines(tmp_path: Path) -> None:
    from agentseek_api.cli import main

    package_dir = tmp_path / "chat"
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "graph.py").write_text("graph = object()\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text(
        """
[project]
name = "sample-project"
version = "0.1.0"
""".strip(),
        encoding="utf-8",
    )
    pip_conf = tmp_path / "pip.conf"
    pip_conf.write_text("[global]\nindex-url = https://pypi.org/simple\n", encoding="utf-8")
    config_path = tmp_path / "langgraph.json"
    config_path.write_text(
        """
{
  "dependencies": ["."],
  "graphs": {
    "chat": "chat.graph:graph"
  },
  "python_version": "3.13",
  "image_distro": "bookworm",
  "pip_config_file": "./pip.conf",
  "dockerfile_lines": [
    "RUN echo custom-step"
  ]
}
""".strip(),
        encoding="utf-8",
    )
    dockerfile_path = tmp_path / "Dockerfile.agentseek"

    exit_code = main(["dockerfile", "--config", str(config_path), str(dockerfile_path)], cwd=tmp_path)

    assert exit_code == 0
    content = dockerfile_path.read_text(encoding="utf-8")
    assert "FROM python:3.13-slim-bookworm" in content
    assert "RUN echo custom-step" in content
    assert "RUN PIP_CONFIG_FILE=/deps/agent/pip.conf pip install --no-cache-dir /deps/agent" in content


def test_dockerfile_command_translates_manifest_dependencies(tmp_path: Path) -> None:
    from agentseek_api.cli import main

    project_dir = tmp_path / "sample_project"
    project_dir.mkdir()
    local_pkg_dir = project_dir / "local_pkg"
    local_pkg_dir.mkdir()
    (local_pkg_dir / "pyproject.toml").write_text(
        """
[project]
name = "local-pkg"
version = "0.1.0"
""".strip(),
        encoding="utf-8",
    )
    requirements_dir = project_dir / "reqs"
    requirements_dir.mkdir()
    (requirements_dir / "requirements.txt").write_text("httpx==0.28.1\n", encoding="utf-8")
    config_path = project_dir / "langgraph.json"
    config_path.write_text(
        """
{
  "dependencies": [".", "./local_pkg", "./reqs", "httpx"],
  "graphs": {
    "chat": "chat.graph:graph"
  }
}
""".strip(),
        encoding="utf-8",
    )
    dockerfile_path = tmp_path / "Dockerfile.agentseek"

    exit_code = main(["dockerfile", "--config", str(config_path), str(dockerfile_path)], cwd=tmp_path)

    assert exit_code == 0
    content = dockerfile_path.read_text(encoding="utf-8")
    assert "ENV PYTHONPATH=/deps/agent:/deps/agent/sample_project:/deps/agent/sample_project/local_pkg:/deps/agent/sample_project/reqs" in content
    assert "RUN pip install --no-cache-dir /deps/agent/sample_project/local_pkg" in content
    assert "RUN pip install --no-cache-dir -r /deps/agent/sample_project/reqs/requirements.txt" in content
    assert "RUN pip install --no-cache-dir httpx" in content
    assert "RUN pip install --no-cache-dir ." not in content


def test_dockerfile_command_skips_root_install_when_root_is_not_installable(tmp_path: Path) -> None:
    from agentseek_api.cli import main

    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "graph.py").write_text("graph = object()\n", encoding="utf-8")
    config_path = tmp_path / "langgraph.json"
    config_path.write_text(
        """
{
  "dependencies": ["./src"],
  "graphs": {
    "chat": "./src/graph.py:graph"
  }
}
""".strip(),
        encoding="utf-8",
    )
    dockerfile_path = tmp_path / "Dockerfile.agentseek"

    exit_code = main(["dockerfile", "--config", str(config_path), str(dockerfile_path)], cwd=tmp_path)

    assert exit_code == 0
    content = dockerfile_path.read_text(encoding="utf-8")
    assert "ENV PYTHONPATH=/deps/agent:/deps/agent/src" in content
    assert "RUN pip install --no-cache-dir ." not in content


def test_dockerfile_command_uses_manifest_project_root_not_invocation_root(tmp_path: Path) -> None:
    from agentseek_api.cli import main

    (tmp_path / "pyproject.toml").write_text(
        """
[project]
name = "workspace-root"
version = "0.1.0"
""".strip(),
        encoding="utf-8",
    )
    app_dir = tmp_path / "apps" / "agent"
    app_dir.mkdir(parents=True)
    (app_dir / "pyproject.toml").write_text(
        """
[project]
name = "nested-agent"
version = "0.1.0"
""".strip(),
        encoding="utf-8",
    )
    (app_dir / "graph.py").write_text("graph = object()\n", encoding="utf-8")
    config_path = app_dir / "langgraph.json"
    config_path.write_text(
        """
{
  "graphs": {
    "chat": "./graph.py:graph"
  }
}
""".strip(),
        encoding="utf-8",
    )
    dockerfile_path = tmp_path / "Dockerfile.agentseek"

    exit_code = main(["dockerfile", "--config", str(config_path), str(dockerfile_path)], cwd=tmp_path)

    assert exit_code == 0
    content = dockerfile_path.read_text(encoding="utf-8")
    assert "RUN pip install --no-cache-dir /deps/agent/apps/agent" in content
    assert "RUN pip install --no-cache-dir /deps/agent\n" not in content


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


def test_build_runtime_env_rejects_invalid_config_env_shape(tmp_path: Path) -> None:
    from agentseek_api.cli import build_runtime_env

    config_path = tmp_path / "langgraph.json"
    config_path.write_text('{"graphs":{"chat":"chat.graph:graph"},"env":["bad"]}', encoding="utf-8")

    with pytest.raises(RuntimeError, match="must set 'env' to a path string or key/value object"):
        build_runtime_env(config_path=config_path, env_file=None, cwd=tmp_path, base_env={})


def test_build_runtime_env_rejects_non_scalar_config_env_value(tmp_path: Path) -> None:
    from agentseek_api.cli import build_runtime_env

    config_path = tmp_path / "langgraph.json"
    config_path.write_text('{"graphs":{"chat":"chat.graph:graph"},"env":{"BAD":[]}}', encoding="utf-8")

    with pytest.raises(RuntimeError, match="env mapping values must be scalar"):
        build_runtime_env(config_path=config_path, env_file=None, cwd=tmp_path, base_env={})


def test_containerize_symbol_reference_supports_windows_drive_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agentseek_api import cli as cli_module

    expected_path = tmp_path / "auth.py"
    monkeypatch.setattr(cli_module, "_resolve_path", lambda path_text, *, cwd: expected_path)
    monkeypatch.setattr(
        cli_module,
        "_container_config_path",
        lambda *, config_path, cwd: "/deps/agent/auth.py",
    )

    result = cli_module._containerize_symbol_reference(r"C:\workspace\auth.py:backend", cwd=tmp_path)

    assert result == "/deps/agent/auth.py:backend"


def test_dockerfile_command_requires_valid_config_object(tmp_path: Path) -> None:
    from agentseek_api.cli import main

    config_path = tmp_path / "langgraph.json"
    config_path.write_text("[]", encoding="utf-8")
    stderr = io.StringIO()

    exit_code = main(["dockerfile", "--config", str(config_path), "Dockerfile"], cwd=tmp_path, stderr=stderr)

    assert exit_code == 2
    assert "must contain a top-level JSON object" in stderr.getvalue()


def test_dockerfile_command_rejects_invalid_auth_and_missing_pip_config(tmp_path: Path) -> None:
    from agentseek_api.cli import main

    config_path = tmp_path / "langgraph.json"
    config_path.write_text(
        """
{
  "graphs": {
    "chat": "chat.graph:graph"
  },
  "auth": [],
  "pip_config_file": "./missing.conf"
}
""".strip(),
        encoding="utf-8",
    )
    stderr = io.StringIO()

    exit_code = main(["dockerfile", "--config", str(config_path), "Dockerfile"], cwd=tmp_path, stderr=stderr)

    assert exit_code == 2
    assert "field 'auth' must be an object" in stderr.getvalue()


def test_dockerfile_command_rejects_missing_pip_config_file(tmp_path: Path) -> None:
    from agentseek_api.cli import main

    config_path = tmp_path / "langgraph.json"
    config_path.write_text(
        """
{
  "graphs": {
    "chat": "chat.graph:graph"
  },
  "pip_config_file": "./missing.conf"
}
""".strip(),
        encoding="utf-8",
    )
    stderr = io.StringIO()

    exit_code = main(["dockerfile", "--config", str(config_path), "Dockerfile"], cwd=tmp_path, stderr=stderr)

    assert exit_code == 2
    assert "Pip config file" in stderr.getvalue()


def test_dockerfile_command_rejects_unsupported_image_distro(tmp_path: Path) -> None:
    from agentseek_api.cli import main

    config_path = tmp_path / "langgraph.json"
    config_path.write_text(
        """
{
  "graphs": {
    "chat": "chat.graph:graph"
  },
  "image_distro": "wolfi"
}
""".strip(),
        encoding="utf-8",
    )
    stderr = io.StringIO()

    exit_code = main(["dockerfile", "--config", str(config_path), "Dockerfile"], cwd=tmp_path, stderr=stderr)

    assert exit_code == 2
    assert "not supported without an explicit base_image" in stderr.getvalue()


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
        "-e",
        "AGENTSEEK_GRAPHS=/deps/agent/langgraph.json",
        "-e",
        "METADATA_DB_URL=sqlite+aiosqlite:////tmp/agentseek.db",
        "-e",
        "OCEANBASE_HOST=host.docker.internal",
        "agentseek:test",
    ]


def test_up_command_supports_docker_compose_sidecars(tmp_path: Path) -> None:
    from agentseek_api.cli import main

    config_path = _write_basic_langgraph_config(tmp_path)
    compose_path = tmp_path / "docker-compose.yml"
    compose_path.write_text("services: {}\n", encoding="utf-8")
    capture = _RunCapture()

    exit_code = main(
        [
            "up",
            "--config",
            str(config_path),
            "--image",
            "agentseek:test",
            "--docker-compose",
            str(compose_path),
            "--recreate",
        ],
        runner=capture,
        cwd=tmp_path,
    )

    assert exit_code == 0
    assert capture.calls is not None
    assert capture.calls[0] == ["docker", "rm", "-f", "agentseek-up-8123"]
    assert capture.calls[1] == ["docker", "compose", "-f", str(compose_path.resolve()), "up", "-d", "--force-recreate"]
    assert capture.calls[2][-1] == "agentseek:test"


def test_up_command_rejects_missing_docker_compose_file(tmp_path: Path) -> None:
    from agentseek_api.cli import main

    config_path = _write_basic_langgraph_config(tmp_path)
    stderr = io.StringIO()

    exit_code = main(
        [
            "up",
            "--config",
            str(config_path),
            "--image",
            "agentseek:test",
            "--docker-compose",
            str(tmp_path / "missing-compose.yml"),
        ],
        cwd=tmp_path,
        stderr=stderr,
    )

    assert exit_code == 2
    assert "Docker compose file" in stderr.getvalue()


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


def test_up_command_passes_config_auth_env_and_containerizes_file_paths(tmp_path: Path) -> None:
    from agentseek_api.cli import main

    package_dir = tmp_path / "chat"
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "graph.py").write_text("graph = object()\n", encoding="utf-8")
    config_path = tmp_path / "langgraph.json"
    config_path.write_text(
        """
{
  "dependencies": ["."],
  "graphs": {
    "chat": "chat.graph:graph"
  },
  "env": {
    "FEATURE_FLAG": true
  },
  "auth": {
    "path": "./auth.py:backend"
  }
}
""".strip(),
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
        ],
        runner=capture,
        cwd=tmp_path,
    )

    assert exit_code == 0
    assert capture.calls is not None
    assert capture.calls[0] == [
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
        "-e",
        "AGENTSEEK_GRAPHS=/deps/agent/langgraph.json",
        "-e",
        "AUTH_MODULE_PATH=/deps/agent/auth.py:backend",
        "-e",
        "AUTH_TYPE=custom",
        "-e",
        "FEATURE_FLAG=True",
        "agentseek:test",
    ]


def test_up_command_uses_base_image_override_when_building(tmp_path: Path) -> None:
    from agentseek_api.cli import main

    _write_basic_langgraph_config(tmp_path)
    capture = _RunCapture()

    exit_code = main(
        [
            "up",
            "--port",
            "8126",
            "--base-image",
            "python:3.13-slim-bookworm",
            "--no-pull",
        ],
        runner=capture,
        cwd=tmp_path,
    )

    assert exit_code == 0
    dockerfile = (tmp_path / ".agentseek" / "Dockerfile").read_text(encoding="utf-8")
    assert "FROM python:3.13-slim-bookworm" in dockerfile


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


def test_wait_for_http_ready_times_out() -> None:
    from agentseek_api.cli import _wait_for_http_ready

    with pytest.raises(RuntimeError, match="Timed out waiting"):
        _wait_for_http_ready("http://127.0.0.1:9/health", timeout_seconds=0.01)


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
