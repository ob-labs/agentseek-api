from __future__ import annotations

import argparse
import hashlib
import importlib
import io
import json
import signal
import tarfile
import tomllib
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pytest
from pydantic import ValidationError

from agentseek_api import __version__
from agentseek_api.docker_runtime import (
    BuildImageInvocation,
    ControlQueryInvocation,
    DockerRunInvocation,
    ProcessInvocation,
    ProcessResult,
)
from agentseek_api.services.langgraph_service import LangGraphService


def test_python_dotenv_dependency_is_available() -> None:
    from dotenv import dotenv_values

    assert callable(dotenv_values)


def test_up_parser_collects_explicit_container_and_compose_names() -> None:
    from agentseek_api.cli import create_parser

    args = create_parser().parse_args(
        [
            "up",
            "--image",
            "agentseek:test",
            "--pass-env",
            "TOKEN",
            "--pass-env",
            "OTHER",
            "--compose-pass-env",
            "TOKEN",
        ]
    )

    assert args.pass_env == ["TOKEN", "OTHER"]
    assert args.compose_pass_env == ["TOKEN"]


def test_up_pass_env_reaches_only_the_live_docker_carrier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agentseek_api.cli import main

    config_path = _write_basic_langgraph_config(tmp_path)
    monkeypatch.setenv("ARBITRARY_HOST_SECRET", "must-not-reach-docker-run")
    capture = _ProcessCapture()

    exit_code = main(
        [
            "up",
            "--config",
            str(config_path),
            "--image",
            "agentseek:test",
            "--pass-env",
            "ARBITRARY_HOST_SECRET",
        ],
        process_transport=capture,
        cwd=tmp_path,
    )

    assert exit_code == 0
    assert capture.calls is not None
    run = next(call for call in capture.calls if isinstance(call, DockerRunInvocation))
    assert run.environment["ARBITRARY_HOST_SECRET"] == "must-not-reach-docker-run"
    assert "must-not-reach-docker-run" not in " ".join(run.argv)


def test_container_selection_combines_normalized_config_and_cli_compose_names(
    tmp_path: Path,
) -> None:
    from agentseek_api.cli import build_container_selection

    config_path = tmp_path / "langgraph.json"
    config_path.write_text(
        '{"graphs":{"chat":"chat.graph:graph"},"compose_env":[" TOKEN ","FROM_CONFIG"]}',
        encoding="utf-8",
    )

    selection = build_container_selection(
        config_path=config_path,
        pass_env=[" PASS_ENV "],
        compose_pass_env=[" TOKEN ", "FROM_CLI"],
    )

    assert selection.pass_env == frozenset({"PASS_ENV"})
    assert selection.compose_env == frozenset({"TOKEN", "FROM_CONFIG", "FROM_CLI"})


@dataclass
class _RunCapture:
    calls: list[list[str]] | None = None
    command: list[str] | None = None
    env: dict[str, str] | None = None
    cwd: str | None = None

    def __call__(
        self, command: list[str], *, env: dict[str, str], cwd: str | None = None
    ) -> int:
        if self.calls is None:
            self.calls = []
        self.calls.append(command)
        self.command = command
        self.env = env
        self.cwd = cwd
        if command[:3] == ["docker", "container", "inspect"]:
            return 1
        return 0


@dataclass
class _ProcessCapture:
    calls: list[ProcessInvocation] | None = None
    container_exists: bool = False
    return_codes: dict[tuple[str, ...], int] | None = None

    def __call__(self, invocation: ProcessInvocation) -> ProcessResult:
        if self.calls is None:
            self.calls = []
        self.calls.append(invocation)
        if invocation.argv == ("docker", "compose", "version", "--short"):
            return ProcessResult(returncode=0, stdout=b"2.40.3\n")
        if invocation.argv == ("docker", "buildx", "version"):
            return ProcessResult(
                returncode=0,
                stdout=b"github.com/docker/buildx v0.14.0 deadbeef\n",
            )
        return_code = 0
        if invocation.argv[:3] == ("docker", "container", "inspect"):
            return_code = 0 if self.container_exists else 1
        if self.return_codes is not None:
            return_code = self.return_codes.get(invocation.argv, return_code)
        return ProcessResult(returncode=return_code)


def _captured_image_build(capture: _ProcessCapture) -> BuildImageInvocation:
    assert capture.calls is not None
    return next(
        invocation
        for invocation in capture.calls
        if isinstance(invocation, BuildImageInvocation)
    )


class _EncodingTextStream:
    def __init__(self, encoding: str) -> None:
        self.encoding = encoding
        self.writes: list[str] = []
        self.flush_count = 0

    def write(self, value: str) -> int:
        value.encode(self.encoding, errors="strict")
        self.writes.append(value)
        return len(value)

    def flush(self) -> None:
        self.flush_count += 1


class _RecordingTextStream:
    def __init__(self, encoding: str) -> None:
        self.encoding = encoding
        self.writes: list[str] = []
        self.flush_count = 0

    def write(self, value: str) -> int:
        self.writes.append(value)
        return len(value)

    def flush(self) -> None:
        self.flush_count += 1


class _PartialWriteFailureStream:
    encoding = "utf-8"

    def __init__(self) -> None:
        self.write_calls = 0
        self.value = ""
        self.flush_count = 0

    def write(self, value: str) -> int:
        self.write_calls += 1
        self.value += value[:8]
        raise UnicodeEncodeError("utf-8", value, 8, 9, "write-canary")

    def flush(self) -> None:
        self.flush_count += 1


class _FakeForegroundSupervisor:
    def __init__(
        self,
        *,
        wait_result: int | BaseException,
        escalates: bool = False,
    ) -> None:
        self.wait_result = wait_result
        self.escalates = escalates
        self.terminated = False
        self.killed = False
        self.wait_calls = 0
        self.close_remaining_tree_calls: list[float] = []
        self.forward_and_reap_calls: list[tuple[int, float]] = []
        self.terminate_and_reap_calls: list[float] = []
        self.ensure_closed_calls: list[float] = []
        self.close_calls = 0

    def wait(self) -> int:
        self.wait_calls += 1
        if isinstance(self.wait_result, BaseException):
            raise self.wait_result
        return self.wait_result

    def close_remaining_tree(self, *, timeout: float) -> None:
        self.close_remaining_tree_calls.append(timeout)

    def forward_signal(self, signum: int) -> None:
        self.forward_and_reap_calls.append((signum, 0.0))

    def forward_and_reap(self, signum: int, *, timeout: float) -> None:
        self.forward_and_reap_calls.append((signum, timeout))
        self.terminated = True
        self.killed = self.escalates

    def terminate_and_reap(self, *, timeout: float) -> None:
        self.terminate_and_reap_calls.append(timeout)
        self.terminated = True

    def ensure_closed(self, *, timeout: float) -> None:
        self.ensure_closed_calls.append(timeout)

    def close(self) -> None:
        self.close_calls += 1


def test_default_runner_propagates_child_exit_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agentseek_api import cli as cli_module

    child = _FakeForegroundSupervisor(wait_result=23)
    observed: dict[str, object] = {}

    def fake_start(command, *, env, cwd):
        observed.update(command=command, env=env, cwd=cwd)
        return child

    monkeypatch.setattr(
        cli_module.ForegroundChildSupervisor,
        "start",
        fake_start,
    )

    exit_code = cli_module._default_runner(
        ["python", "-m", "agentseek_api.worker"],
        env={"TOKEN": "value"},
        cwd="/runtime",
    )

    assert exit_code == 23
    assert child.terminated is False
    assert child.close_remaining_tree_calls == [5.0]
    assert child.ensure_closed_calls == [5.0]
    assert child.close_calls == 1
    assert observed == {
        "command": ["python", "-m", "agentseek_api.worker"],
        "env": {"TOKEN": "value"},
        "cwd": "/runtime",
    }


def test_default_runner_terminates_and_reaps_child_on_keyboard_interrupt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agentseek_api import cli as cli_module

    child = _FakeForegroundSupervisor(wait_result=KeyboardInterrupt())
    monkeypatch.setattr(
        cli_module.ForegroundChildSupervisor,
        "start",
        lambda command, *, env, cwd: child,
    )

    exit_code = cli_module._default_runner(
        ["python", "-m", "agentseek_api.scheduler"],
        env={},
        cwd="/runtime",
    )

    assert exit_code == 130
    assert child.terminated is True
    assert child.killed is False
    assert child.wait_calls == 1
    assert child.forward_and_reap_calls == [(signal.SIGINT, 5.0)]
    assert child.ensure_closed_calls == [5.0]
    assert child.close_calls == 1


def test_default_runner_delegates_bounded_escalation_for_interrupt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agentseek_api import cli as cli_module

    child = _FakeForegroundSupervisor(
        wait_result=KeyboardInterrupt(),
        escalates=True,
    )
    monkeypatch.setattr(
        cli_module.ForegroundChildSupervisor,
        "start",
        lambda command, *, env, cwd: child,
    )

    assert cli_module._default_runner(["child"], env={}, cwd=None) == 130
    assert child.terminated is True
    assert child.killed is True
    assert child.forward_and_reap_calls == [(signal.SIGINT, 5.0)]
    assert child.ensure_closed_calls == [5.0]
    assert child.close_calls == 1


@pytest.mark.parametrize(
    "failure_point",
    ["guard-entry", "child-start", "guard-attach", "native-cleanup"],
)
def test_public_worker_redacts_process_supervision_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_point: str,
) -> None:
    from agentseek_api import cli as cli_module
    from agentseek_api.process_supervisor import ProcessSupervisionError

    setup_canary = "setup-canary"
    command_canary = "command-canary"
    environment_canary = "environment-canary"
    config_path = tmp_path / "langgraph.json"
    config_path.write_text(
        '{"graphs":{"chat":"chat.graph:graph"},'
        f'"env":{{"SUPERVISION_SECRET":"{environment_canary}"}}}}',
        encoding="utf-8",
    )
    monkeypatch.delenv("SUPERVISION_SECRET", raising=False)

    child = _FakeForegroundSupervisor(wait_result=0)
    child.live = True

    def fail_native_cleanup(*, timeout: float) -> None:
        raise ProcessSupervisionError(setup_canary)

    if failure_point == "native-cleanup":
        child.close_remaining_tree = fail_native_cleanup  # type: ignore[method-assign]

    original_terminate_and_reap = child.terminate_and_reap

    def terminate_and_reap(*, timeout: float) -> None:
        original_terminate_and_reap(timeout=timeout)
        child.live = False

    child.terminate_and_reap = terminate_and_reap  # type: ignore[method-assign]

    original_close = child.close

    def close() -> None:
        original_close()
        child.live = False

    child.close = close  # type: ignore[method-assign]

    class _FakeGuard:
        def __enter__(self):
            if failure_point == "guard-entry":
                raise ProcessSupervisionError(setup_canary)
            return self

        def __exit__(self, exc_type, exc, traceback) -> bool:
            return False

        def attach(self, attached_child) -> None:
            assert attached_child is child
            if failure_point == "guard-attach":
                raise ProcessSupervisionError(setup_canary)

        def begin_cleanup(self) -> None:
            return None

    def start(command, *, env, cwd):
        assert command == [command_canary]
        assert env["SUPERVISION_SECRET"] == environment_canary
        assert cwd == str(tmp_path)
        if failure_point == "child-start":
            raise ProcessSupervisionError(setup_canary)
        return child

    monkeypatch.setattr(cli_module, "ForwardingSignalGuard", _FakeGuard)
    monkeypatch.setattr(
        cli_module.ForegroundChildSupervisor,
        "start",
        start,
    )
    monkeypatch.setattr(
        cli_module,
        "build_worker_command",
        lambda: [command_canary],
    )
    stdout = io.StringIO()
    stderr = io.StringIO()

    exit_code = cli_module.main(
        ["worker", "--config", str(config_path)],
        cwd=tmp_path,
        stdout=stdout,
        stderr=stderr,
    )

    combined_output = stdout.getvalue() + stderr.getvalue()
    assert exit_code == 2
    assert stderr.getvalue() == "Could not supervise the runtime child safely.\n"
    assert "Traceback" not in combined_output
    assert setup_canary not in combined_output
    assert command_canary not in combined_output
    assert environment_canary not in combined_output
    if failure_point in {"guard-attach", "native-cleanup"}:
        assert child.live is False
        assert child.ensure_closed_calls == [5.0]
        assert child.close_calls == 1


def _application_environment(capture: _ProcessCapture) -> dict[str, str]:
    assert capture.calls is not None
    run = next(call for call in capture.calls if isinstance(call, DockerRunInvocation))
    return {name: run.environment[name] for name in run.application_names}


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


def _write_basic_manifest_config(root: Path) -> Path:
    config_path = _write_basic_langgraph_config(root)
    manifest_path = root / "manifest.json"
    manifest_path.write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
    config_path.unlink()
    return manifest_path


def test_onboard_banner_preserves_unicode_for_stringio(tmp_path: Path) -> None:
    from agentseek_api.cli import main

    _write_basic_langgraph_config(tmp_path)
    stdout = io.StringIO()

    exit_code = main(
        ["serve"],
        runner=_RunCapture(),
        stdout=stdout,
        cwd=tmp_path,
    )

    assert exit_code == 0
    assert stdout.getvalue() == (
        "\n"
        "        Welcome to\n"
        "\n"
        "╔═╗┌─┐┌─┐┌┐┌┌┬┐╔═╗┌─┐┌─┐┬┌─\n"
        "╠═╣│ ┬├┤ │││ │ ╚═╗├┤ ├┤ ├┴┐\n"
        "╩ ╩└─┘└─┘┘└┘ ┴ ╚═╝└─┘└─┘┴ ┴\n"
        "\n"
        f"     AgentSeek v{__version__}\n"
        "\n"
    )


def test_onboard_banner_uses_one_write_and_flush_for_utf8_stream(
    tmp_path: Path,
) -> None:
    from agentseek_api.cli import main

    _write_basic_langgraph_config(tmp_path)
    stdout = _EncodingTextStream("utf-8")

    exit_code = main(
        ["serve"],
        runner=_RunCapture(),
        stdout=stdout,
        cwd=tmp_path,
    )

    assert exit_code == 0
    assert stdout.writes == [
        "\n"
        "        Welcome to\n"
        "\n"
        "╔═╗┌─┐┌─┐┌┐┌┌┬┐╔═╗┌─┐┌─┐┬┌─\n"
        "╠═╣│ ┬├┤ │││ │ ╚═╗├┤ ├┤ ├┴┐\n"
        "╩ ╩└─┘└─┘┘└┘ ┴ ╚═╝└─┘└─┘┴ ┴\n"
        "\n"
        f"     AgentSeek v{__version__}\n"
        "\n"
    ]
    assert stdout.flush_count == 1


@pytest.mark.parametrize("role", ["dev", "serve"])
def test_onboard_banner_falls_back_before_writing_to_cp1252_stream(
    tmp_path: Path,
    role: str,
) -> None:
    from agentseek_api.cli import main

    _write_basic_langgraph_config(tmp_path)
    stdout = _EncodingTextStream("cp1252")
    arguments = [role]
    if role == "dev":
        arguments.append("--no-reload")

    exit_code = main(
        arguments,
        runner=_RunCapture(),
        stdout=stdout,
        cwd=tmp_path,
    )

    assert exit_code == 0
    assert stdout.writes == [
        "\n"
        "        Welcome to\n"
        "\n"
        "========================\n"
        f"     AgentSeek v{__version__}\n"
        "========================\n"
        "\n"
    ]
    assert stdout.flush_count == 1


def test_onboard_banner_uses_ascii_fallback_for_unknown_named_encoding(
    tmp_path: Path,
) -> None:
    from agentseek_api.cli import main

    _write_basic_langgraph_config(tmp_path)
    stdout = _RecordingTextStream("unknown-codec-canary")

    exit_code = main(
        ["serve"],
        runner=_RunCapture(),
        stdout=stdout,
        cwd=tmp_path,
    )

    assert exit_code == 0
    assert stdout.writes == [
        "\n"
        "        Welcome to\n"
        "\n"
        "========================\n"
        f"     AgentSeek v{__version__}\n"
        "========================\n"
        "\n"
    ]
    assert stdout.flush_count == 1


def test_onboard_banner_does_not_retry_or_flush_after_partial_write() -> None:
    from agentseek_api import cli as cli_module

    stdout = _PartialWriteFailureStream()

    with pytest.raises(UnicodeEncodeError, match="write-canary"):
        cli_module._write_onboard_banner(stdout)

    assert stdout.write_calls == 1
    assert stdout.value == "\n       "
    assert stdout.flush_count == 0


def test_dev_command_prefers_agentseek_json_over_langgraph_json(tmp_path: Path) -> None:
    from agentseek_api.cli import main

    config_path = tmp_path / "agentseek.json"
    config_path.write_text(
        '{"graphs":{"agentseek":"chat.graph:graph"}}', encoding="utf-8"
    )
    _write_basic_langgraph_config(tmp_path)
    capture = _RunCapture()

    exit_code = main(["dev", "--no-reload"], runner=capture, cwd=tmp_path)

    assert exit_code == 0
    assert capture.command[1:] == [
        "-m",
        "agentseek_api.runtime_entrypoint",
        "uvicorn",
        "--",
        "agentseek_api.main:app",
        "--host",
        "127.0.0.1",
        "--port",
        "2024",
    ]
    assert capture.env is not None
    assert capture.env["AGENTSEEK_GRAPHS"] == str(config_path.resolve())


def test_serve_command_falls_back_to_langgraph_json_and_runs_graph(
    tmp_path: Path,
) -> None:
    from agentseek_api.cli import main

    config_path = _write_basic_langgraph_config(tmp_path)
    capture = _RunCapture()

    exit_code = main(
        ["serve", "--host", "0.0.0.0", "--port", "3030"], runner=capture, cwd=tmp_path
    )

    assert exit_code == 0
    assert capture.command[1:] == [
        "-m",
        "agentseek_api.runtime_entrypoint",
        "uvicorn",
        "--",
        "agentseek_api.main:app",
        "--host",
        "0.0.0.0",
        "--port",
        "3030",
    ]
    assert capture.env is not None
    assert capture.env["AGENTSEEK_GRAPHS"] == str(config_path.resolve())

    service = LangGraphService(manifest_path=capture.env["AGENTSEEK_GRAPHS"])
    result = service.get_entry("chat").build_graph().invoke({})
    assert result["value"] == "basic-config"


def test_serve_command_uses_agentseek_graphs_env_for_manifest_named_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agentseek_api.cli import main

    config_path = _write_basic_manifest_config(tmp_path)
    monkeypatch.setenv("AGENTSEEK_GRAPHS", str(config_path.resolve()))
    capture = _RunCapture()

    exit_code = main(
        ["serve", "--host", "0.0.0.0", "--port", "3030"], runner=capture, cwd=tmp_path
    )

    assert exit_code == 0
    assert capture.command[1:] == [
        "-m",
        "agentseek_api.runtime_entrypoint",
        "uvicorn",
        "--",
        "agentseek_api.main:app",
        "--host",
        "0.0.0.0",
        "--port",
        "3030",
    ]
    assert capture.env is not None
    assert capture.env["AGENTSEEK_GRAPHS"] == str(config_path.resolve())


def test_worker_command_uses_runtime_env_and_worker_module(tmp_path: Path) -> None:
    from agentseek_api.cli import main

    config_path = _write_basic_langgraph_config(tmp_path)
    capture = _RunCapture()

    exit_code = main(
        ["worker", "--config", str(config_path)], runner=capture, cwd=tmp_path
    )

    assert exit_code == 0
    assert capture.command is not None
    assert capture.command[1:] == [
        "-m",
        "agentseek_api.runtime_entrypoint",
        "worker",
    ]
    assert capture.env is not None
    assert capture.env["AGENTSEEK_GRAPHS"] == str(config_path.resolve())


def test_scheduler_command_uses_runtime_env_and_scheduler_module(
    tmp_path: Path,
) -> None:
    from agentseek_api.cli import main

    config_path = _write_basic_langgraph_config(tmp_path)
    capture = _RunCapture()

    exit_code = main(
        ["scheduler", "--config", str(config_path)], runner=capture, cwd=tmp_path
    )

    assert exit_code == 0
    assert capture.command is not None
    assert capture.command[1:] == [
        "-m",
        "agentseek_api.runtime_entrypoint",
        "scheduler",
    ]
    assert capture.env is not None
    assert capture.env["AGENTSEEK_GRAPHS"] == str(config_path.resolve())


def test_settings_validation_formatter_omits_input_values() -> None:
    from agentseek_api.runtime_entrypoint import (
        _format_settings_validation_error,
    )
    from agentseek_api.settings import Settings

    with pytest.raises(ValidationError) as captured:
        Settings.model_validate({"PORT": "invalid-port-canary"})

    message = _format_settings_validation_error(captured.value)

    assert message == "Invalid runtime setting(s): PORT (int_parsing)."
    assert "invalid-port-canary" not in message


def test_dev_command_accepts_langgraph_cli_flags_and_env_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agentseek_api.cli import main

    monkeypatch.delenv("AUTH_MODULE_PATH", raising=False)
    config_path = _write_basic_langgraph_config(tmp_path)
    env_file = tmp_path / ".env"
    env_file.write_text("AUTH_MODULE_PATH=test.module:backend\n", encoding="utf-8")
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
    assert capture.command[1:] == [
        "-m",
        "agentseek_api.runtime_entrypoint",
        "uvicorn",
        "--",
        "agentseek_api.main:app",
        "--host",
        "0.0.0.0",
        "--port",
        "9999",
    ]
    assert capture.env is not None
    assert capture.env["AGENTSEEK_GRAPHS"] == str(config_path.resolve())
    assert capture.env["AUTH_MODULE_PATH"] == "test.module:backend"


def test_dev_command_loads_config_env_mapping_and_auth_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agentseek_api.cli import main

    for key in ("OPENAI_API_KEY", "FEATURE_FLAG", "AUTH_MODULE_PATH"):
        monkeypatch.delenv(key, raising=False)
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

    exit_code = main(
        ["dev", "--config", str(config_path), "--no-reload"],
        runner=capture,
        cwd=tmp_path,
    )

    assert exit_code == 0
    assert capture.env is not None
    assert capture.env["OPENAI_API_KEY"] == "test-key"
    assert capture.env["FEATURE_FLAG"] == "True"
    assert capture.env["AUTH_MODULE_PATH"] == f"{(tmp_path / 'auth.py').resolve()}:auth"


def test_dev_command_merges_config_env_file_before_cli_env_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agentseek_api.cli import main

    for key in ("TOKEN", "SHARED"):
        monkeypatch.delenv(key, raising=False)
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
        [
            "dev",
            "--config",
            str(config_path),
            "--env-file",
            str(cli_env),
            "--no-reload",
        ],
        runner=capture,
        cwd=tmp_path,
    )

    assert exit_code == 0
    assert capture.env is not None
    assert capture.env["TOKEN"] == "from-config"
    assert capture.env["SHARED"] == "override"


def test_dev_command_preserves_dotenv_default_and_bare_variable_syntax(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agentseek_api.cli import main

    monkeypatch.delenv("API_ORIGIN", raising=False)
    config_path = _write_basic_langgraph_config(tmp_path)
    env_file = tmp_path / "defaults.env"
    env_file.write_text(
        "OPENAI_BASE_URL=${API_ORIGIN:-https://default.example.test}/v1\n"
        "BARE_REFERENCE=$API_ORIGIN\n",
        encoding="utf-8",
    )
    capture = _RunCapture()

    exit_code = main(
        [
            "dev",
            "--config",
            str(config_path),
            "--env-file",
            str(env_file),
            "--no-reload",
        ],
        runner=capture,
        cwd=tmp_path,
    )

    assert exit_code == 0
    assert capture.env is not None
    assert capture.env["OPENAI_BASE_URL"] == "https://default.example.test/v1"
    assert capture.env["BARE_REFERENCE"] == "$API_ORIGIN"


def test_dev_command_rejects_unsupported_langgraph_flags(tmp_path: Path) -> None:
    from agentseek_api.cli import main

    _write_basic_langgraph_config(tmp_path)
    stderr = io.StringIO()

    exit_code = main(["dev", "--tunnel"], cwd=tmp_path, stderr=stderr)

    assert exit_code == 2
    assert (
        "Unsupported option(s) for 'agentseek-api dev': --tunnel" in stderr.getvalue()
    )
    assert (
        "Use 'langgraph dev' for mocked or tunneled local workflows."
        in stderr.getvalue()
    )


def test_dev_command_forces_local_studio_auth_after_inherited_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agentseek_api.cli import main

    _write_basic_langgraph_config(tmp_path)
    monkeypatch.setenv("STUDIO_AUTH_LOCAL_DEV", "false")
    capture = _RunCapture()

    exit_code = main(["dev", "--no-reload"], runner=capture, cwd=tmp_path)

    assert exit_code == 0
    assert capture.env is not None
    assert capture.env["STUDIO_AUTH_LOCAL_DEV"] == "true"


def test_serve_port_flag_does_not_rewrite_inherited_port_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agentseek_api.cli import main

    _write_basic_langgraph_config(tmp_path)
    monkeypatch.setenv("PORT", "7777")
    capture = _RunCapture()

    exit_code = main(
        ["serve", "--port", "3030"],
        runner=capture,
        cwd=tmp_path,
    )

    assert exit_code == 0
    assert capture.command is not None
    assert capture.command[-2:] == ["--port", "3030"]
    assert capture.env is not None
    assert capture.env["PORT"] == "7777"


def test_resolve_dev_urls_use_localhost_display_and_loopback_base_url() -> None:
    from agentseek_api.cli import _resolve_dev_urls

    urls = _resolve_dev_urls(host="0.0.0.0", port=2024, studio_url=None)

    assert urls.api_url == "http://localhost:2024"
    assert urls.docs_url == "http://localhost:2024/docs"
    assert (
        urls.studio_url
        == "https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024"
    )


def test_resolve_dev_urls_preserve_explicit_host_and_override_studio_origin() -> None:
    from agentseek_api.cli import _resolve_dev_urls

    urls = _resolve_dev_urls(
        host="devbox.local", port=3030, studio_url="https://smith.example.com"
    )

    assert urls.api_url == "http://devbox.local:3030"
    assert urls.docs_url == "http://devbox.local:3030/docs"
    assert (
        urls.studio_url
        == "https://smith.example.com/studio/?baseUrl=http://devbox.local:3030"
    )


def test_run_managed_dev_server_prints_banner_and_opens_browser(tmp_path: Path) -> None:
    from agentseek_api import cli as cli_module

    class FakeProcess:
        def __init__(self) -> None:
            self.returncode: int | None = None
            self.terminated = False

        def poll(self) -> int | None:
            return self.returncode

        def wait(self) -> int:
            self.returncode = 0 if self.returncode is None else self.returncode
            return self.returncode

        def terminate(self) -> None:
            self.terminated = True
            self.returncode = 0

    fake_process = FakeProcess()
    opened: list[str] = []
    stdout = io.StringIO()

    exit_code = cli_module._run_managed_dev_server(
        command=[
            "uvicorn",
            "agentseek_api.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            "2024",
        ],
        env={"A": "B"},
        cwd=tmp_path,
        urls=cli_module._resolve_dev_urls(host="127.0.0.1", port=2024, studio_url=None),
        stdout=stdout,
        process_factory=lambda command, *, env, cwd: fake_process,
        wait_for_ready=lambda *_args, **_kwargs: None,
        browser_opener=opened.append,
        sleep=lambda _seconds: None,
    )

    assert exit_code == 0
    output = stdout.getvalue()
    assert "API: http://localhost:2024" in output
    assert "Docs: http://localhost:2024/docs" in output
    assert "https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024" in output
    assert opened == [
        "https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024"
    ]


def test_managed_dev_ascii_fallback_normalizes_non_ascii_urls_before_write(
    tmp_path: Path,
) -> None:
    from agentseek_api import cli as cli_module

    class FakeProcess:
        def __init__(self) -> None:
            self.returncode: int | None = None
            self.wait_calls = 0
            self.terminate_calls = 0

        def poll(self) -> int | None:
            return self.returncode

        def wait(self) -> int:
            self.wait_calls += 1
            self.returncode = 23
            return self.returncode

        def terminate(self) -> None:
            self.terminate_calls += 1
            self.returncode = -1

    process = FakeProcess()
    stdout = _EncodingTextStream("cp1252")

    exit_code = cli_module._run_managed_dev_server(
        command=["uvicorn", "agentseek_api.main:app"],
        env={},
        cwd=tmp_path,
        urls=cli_module._resolve_dev_urls(
            host="例子",
            port=2024,
            studio_url="https://例子.test",
        ),
        stdout=stdout,
        process_factory=lambda command, *, env, cwd: process,
        wait_for_ready=lambda *_args, **_kwargs: None,
        open_browser=False,
        sleep=lambda _seconds: None,
    )

    assert exit_code == 23
    assert process.wait_calls == 1
    assert process.terminate_calls == 0
    assert stdout.writes == [
        "- API: http://??:2024\n"
        "- Docs: http://??:2024/docs\n"
        "- Studio UI: https://??.test/studio/?baseUrl=http://??:2024\n"
        "\n\n"
    ]
    assert stdout.flush_count == 1


def test_run_managed_dev_server_honors_no_browser(tmp_path: Path) -> None:
    from agentseek_api import cli as cli_module

    class FakeProcess:
        def __init__(self) -> None:
            self.returncode: int | None = None

        def poll(self) -> int | None:
            return self.returncode

        def wait(self) -> int:
            self.returncode = 0 if self.returncode is None else self.returncode
            return self.returncode

        def terminate(self) -> None:
            self.returncode = 0

    opened: list[str] = []

    exit_code = cli_module._run_managed_dev_server(
        command=[
            "uvicorn",
            "agentseek_api.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            "2024",
        ],
        env={},
        cwd=tmp_path,
        urls=cli_module._resolve_dev_urls(host="127.0.0.1", port=2024, studio_url=None),
        stdout=io.StringIO(),
        process_factory=lambda command, *, env, cwd: FakeProcess(),
        wait_for_ready=lambda *_args, **_kwargs: None,
        open_browser=False,
        browser_opener=opened.append,
        sleep=lambda _seconds: None,
    )

    assert exit_code == 0
    assert opened == []


def test_dev_command_rejects_missing_explicit_config(tmp_path: Path) -> None:
    from agentseek_api.cli import main

    stderr = io.StringIO()

    exit_code = main(
        ["dev", "--config", str(tmp_path / "missing.json")], cwd=tmp_path, stderr=stderr
    )

    assert exit_code == 2
    assert "does not exist" in stderr.getvalue()


def test_version_reports_cli_and_package_versions() -> None:
    from agentseek_api import __version__
    from agentseek_api.cli import main

    stdout = io.StringIO()

    exit_code = main(["version"], stdout=stdout)

    assert exit_code == 0
    assert stdout.getvalue().strip().splitlines() == [f"agentseek-api {__version__}"]


def test_release_versions_are_consistent() -> None:
    from agentseek_api import __version__

    project_config = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]
    lock_packages = tomllib.loads(Path("uv.lock").read_text(encoding="utf-8"))[
        "package"
    ]
    root_package = next(
        package
        for package in lock_packages
        if package["name"] == "agentseek-api"
        and package.get("source") == {"editable": "."}
    )

    assert __version__ == "0.3.0"
    assert project_config["version"] == __version__
    assert root_package["version"] == __version__


def test_container_planning_uses_published_runtime_artifact_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import agentseek_api.cli as cli
    from agentseek_api.container_build import PUBLISHED_RUNTIME_ARTIFACT

    config_path = _write_basic_langgraph_config(tmp_path)
    captured: list[object] = []
    original = cli.plan_container_image

    def capture(**kwargs: object) -> object:
        captured.append(kwargs["runtime_artifact"])
        return original(**kwargs)

    monkeypatch.setattr(cli, "plan_container_image", capture)
    output = io.StringIO()

    exit_code = cli.main(
        ["dockerfile", "--config", str(config_path), "Dockerfile"],
        cwd=tmp_path,
        stdout=output,
    )

    assert exit_code == 0
    assert captured == [PUBLISHED_RUNTIME_ARTIFACT]


def test_internal_candidate_runtime_artifact_is_forwarded_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import agentseek_api.cli as cli
    from agentseek_api.container_build import RuntimeArtifactSource, RuntimeArtifactV1

    config_path = _write_basic_langgraph_config(tmp_path)
    wheel = tmp_path / "candidate.whl"
    artifact = RuntimeArtifactV1(
        distribution="agentseek-api",
        extra="embedded",
        version="0.3.0",
        source=RuntimeArtifactSource.CANDIDATE_WHEEL,
        candidate_wheel=wheel,
        candidate_sha256="0" * 64,
        candidate_identity=(0, 0, 0, 0),
    )
    captured: list[object] = []

    def capture(**kwargs: object) -> object:
        captured.append(kwargs["runtime_artifact"])
        raise cli.ContainerBuildError("candidate boundary reached")

    monkeypatch.setattr(cli, "plan_container_image", capture)
    error = io.StringIO()

    exit_code = cli.main(
        ["dockerfile", "--config", str(config_path), "Dockerfile"],
        cwd=tmp_path,
        stderr=error,
        runtime_artifact=artifact,
    )

    assert exit_code == 2
    assert captured == [artifact]
    assert error.getvalue() == "candidate boundary reached\n"


def test_package_exposes_library_and_cli_entrypoints() -> None:
    project_config = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]

    assert project_config["name"] == "agentseek-api"
    assert project_config["scripts"]["agentseek-api"] == "agentseek_api.cli:main"
    assert "packaging>=24.0" in project_config["dependencies"]
    assert project_config["optional-dependencies"]["embedded"]
    assert any(
        "langchain-oceanbase" in dep
        for dep in project_config["optional-dependencies"]["embedded"]
    )


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
    cli_module.register_subcommands(subparsers, command_name="agentseek-api")

    parsed = parser.parse_args(["agentseek-api", "version"])

    assert parsed.tool == "agentseek-api"
    assert parsed.command == "version"


def test_parser_does_not_expose_deploy_command() -> None:
    from agentseek_api.cli import create_parser

    parser = create_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["deploy"])


def test_embedded_subcommand_errors_use_registered_command_name(tmp_path: Path) -> None:
    from agentseek_api import cli as cli_module

    parser = argparse.ArgumentParser(prog="parent")
    subparsers = parser.add_subparsers(dest="tool", required=True)
    cli_module.register_subcommands(subparsers, command_name="agentseek-api")
    parsed = parser.parse_args(["agentseek-api", "dev", "--tunnel"])
    stderr = io.StringIO()

    exit_code = cli_module.run_namespace(parsed, cwd=tmp_path, stderr=stderr)

    assert exit_code == 2
    assert (
        "Unsupported option(s) for 'agentseek-api dev': --tunnel" in stderr.getvalue()
    )


def test_run_namespace_allows_parent_cli_dispatch(tmp_path: Path) -> None:
    from agentseek_api import cli as cli_module

    parser = argparse.ArgumentParser(prog="parent")
    subparsers = parser.add_subparsers(dest="tool", required=True)
    cli_module.register_subcommands(subparsers, command_name="agentseek")
    parsed = parser.parse_args(
        ["agentseek", "serve", "--host", "0.0.0.0", "--port", "3030"]
    )
    capture = _RunCapture()

    exit_code = cli_module.run_namespace(parsed, runner=capture, cwd=tmp_path)

    assert exit_code == 0
    assert capture.command[1:] == [
        "-m",
        "agentseek_api.runtime_entrypoint",
        "uvicorn",
        "--",
        "agentseek_api.main:app",
        "--host",
        "0.0.0.0",
        "--port",
        "3030",
    ]


def test_dockerfile_command_writes_langgraph_compatible_runtime_file(
    tmp_path: Path,
) -> None:
    from agentseek_api.cli import main

    _write_basic_langgraph_config(tmp_path)
    output_root = tmp_path / "docker-build-bundle"

    exit_code = main(["dockerfile", str(output_root)], cwd=tmp_path)

    assert exit_code == 0
    dockerfile_path = output_root / "context" / "Dockerfile"
    content = dockerfile_path.read_text(encoding="utf-8")
    assert (output_root / "inventory.json").is_file()
    assert (output_root / "context" / "manifest.v1.json").is_file()
    assert (output_root / "context" / "app" / "chat" / "graph.py").is_file()
    assert "FROM python:3.12-slim" in content
    assert "WORKDIR /deps/agent" in content
    assert "COPY app /deps/agent" in content
    assert "COPY manifest.v1.json /opt/agentseek/manifest.v1.json" in content
    assert "LABEL org.agentseek.environment-contract=preloaded-v1" in content
    assert (
        '"serve", "--environment-mode", "preloaded-v1", "--host", "0.0.0.0"' in content
    )


def test_dockerfile_command_prefers_agentseek_json_without_explicit_flag(
    tmp_path: Path,
) -> None:
    from agentseek_api.cli import main

    (tmp_path / "agentseek.json").write_text(
        """
{
  "graphs": {
    "chat": "chat.graph:graph"
  }
}
""".strip(),
        encoding="utf-8",
    )
    _write_basic_langgraph_config(tmp_path)
    dockerfile_path = tmp_path / "Dockerfile.agentseek"

    exit_code = main(["dockerfile", str(dockerfile_path)], cwd=tmp_path)

    assert exit_code == 0
    manifest = json.loads(
        (dockerfile_path / "context" / "manifest.v1.json").read_text(encoding="utf-8")
    )
    assert manifest["graphs"] == {"chat": "chat.graph:graph"}


def test_dockerfile_command_honors_base_image_python_and_custom_lines(
    tmp_path: Path,
) -> None:
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
    pip_conf.write_text(
        "[global]\nindex-url = https://pypi.org/simple\n", encoding="utf-8"
    )
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

    exit_code = main(
        ["dockerfile", "--config", str(config_path), str(dockerfile_path)], cwd=tmp_path
    )

    assert exit_code == 0
    content = (dockerfile_path / "context" / "Dockerfile").read_text(encoding="utf-8")
    assert "FROM python:3.13-slim-bookworm" in content
    assert "RUN echo custom-step" in content
    assert "--mount=type=secret,id=pip_config,target=/etc/pip.conf" in content
    assert '"/deps/agent"' in content
    assert "pip.conf" not in "\n".join(
        line for line in content.splitlines() if line.startswith("COPY")
    )


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
    (requirements_dir / "requirements.txt").write_text(
        "httpx==0.28.1\n", encoding="utf-8"
    )
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

    exit_code = main(
        ["dockerfile", "--config", str(config_path), str(dockerfile_path)], cwd=tmp_path
    )

    assert exit_code == 0
    content = (dockerfile_path / "context" / "Dockerfile").read_text(encoding="utf-8")
    assert '"/deps/agent/sample_project/local_pkg"' in content
    assert (
        '"--requirement", "/deps/agent/sample_project/reqs/requirements.txt"' in content
    )
    assert '"httpx"' in content
    assert '"."' not in content


def test_dockerfile_command_skips_root_install_when_root_is_not_installable(
    tmp_path: Path,
) -> None:
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

    exit_code = main(
        ["dockerfile", "--config", str(config_path), str(dockerfile_path)], cwd=tmp_path
    )

    assert exit_code == 0
    content = (dockerfile_path / "context" / "Dockerfile").read_text(encoding="utf-8")
    assert "COPY app /deps/agent" in content
    assert '"/deps/agent/src"' not in content


def test_dockerfile_command_uses_manifest_project_root_not_invocation_root(
    tmp_path: Path,
) -> None:
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

    exit_code = main(
        ["dockerfile", "--config", str(config_path), str(dockerfile_path)], cwd=tmp_path
    )

    assert exit_code == 0
    content = (dockerfile_path / "context" / "Dockerfile").read_text(encoding="utf-8")
    assert "COPY app /deps/agent" in content
    assert '"/deps/agent/apps/agent"' not in content


def test_dockerfile_command_installs_nearest_ancestor_project_for_nested_manifest(
    tmp_path: Path,
) -> None:
    from agentseek_api.cli import main

    (tmp_path / "pyproject.toml").write_text(
        """
[project]
name = "workspace-root"
version = "0.1.0"
""".strip(),
        encoding="utf-8",
    )
    manifest_dir = tmp_path / "examples" / "docker_ci_auth"
    manifest_dir.mkdir(parents=True)
    graph_dir = tmp_path / "examples" / "graphs"
    graph_dir.mkdir()
    (graph_dir / "chat.py").write_text("graph = object()\n", encoding="utf-8")
    config_path = manifest_dir / "manifest.json"
    config_path.write_text(
        """
{
  "dependencies": [".."],
  "graphs": {
    "chat": "../graphs/chat.py:graph"
  }
}
""".strip(),
        encoding="utf-8",
    )
    dockerfile_path = tmp_path / "Dockerfile.agentseek"

    exit_code = main(
        ["dockerfile", "--config", str(config_path), str(dockerfile_path)], cwd=tmp_path
    )

    assert exit_code == 0
    content = (dockerfile_path / "context" / "Dockerfile").read_text(encoding="utf-8")
    assert '"/deps/agent"' not in content
    manifest = json.loads(
        (dockerfile_path / "context" / "manifest.v1.json").read_text(encoding="utf-8")
    )
    assert manifest["dependencies"] == ["/deps/agent/examples"]


def test_build_command_plans_docker_build_from_generated_dockerfile(
    tmp_path: Path,
) -> None:
    from agentseek_api.cli import main

    _write_basic_langgraph_config(tmp_path)
    capture = _ProcessCapture()

    exit_code = main(
        [
            "build",
            "-t",
            "agentseek:test",
            "--platform",
            "linux/amd64,linux/arm64",
            "--no-pull",
        ],
        process_transport=capture,
        cwd=tmp_path,
    )

    assert exit_code == 0
    assert capture.calls is not None
    assert [call.argv for call in capture.calls[:2]] == [
        ("docker", "buildx", "version"),
        ("docker", "buildx", "inspect"),
    ]
    invocation = capture.calls[2]
    assert isinstance(invocation, BuildImageInvocation)
    assert "AGENTSEEK_GRAPHS" not in invocation.environment
    assert invocation.argv == (
        "docker",
        "buildx",
        "build",
        "--load",
        "--file",
        "Dockerfile",
        "--platform",
        "linux/amd64,linux/arm64",
        "--tag",
        "agentseek:test",
        "-",
    )
    assert invocation.stdin_bytes is not None
    assert str(tmp_path) not in " ".join(invocation.argv)
    with tarfile.open(fileobj=io.BytesIO(invocation.stdin_bytes), mode="r:") as archive:
        members = set(archive.getnames())
        generated = archive.extractfile("Dockerfile").read().decode()  # type: ignore[union-attr]
    assert {
        "Dockerfile",
        "manifest.v1.json",
        "runtime-constraints.txt",
        "app/chat/__init__.py",
        "app/chat/graph.py",
    } <= members
    assert "COPY app /deps/agent" in generated
    assert "agentseek-api[embedded]==0.3.0" in generated
    assert "org.agentseek.environment-contract=preloaded-v1" in generated
    assert (
        'CMD ["python", "-m", "agentseek_api.cli", "serve", "--environment-mode", '
        '"preloaded-v1", "--host", "0.0.0.0", "--port", "2024"]' in generated
    )
    assert not (tmp_path / ".agentseek").exists()


def test_build_command_rejects_unavailable_buildx_before_build(tmp_path: Path) -> None:
    from agentseek_api.cli import main

    _write_basic_langgraph_config(tmp_path)
    capture = _ProcessCapture(return_codes={("docker", "buildx", "inspect"): 1})
    stderr = io.StringIO()

    exit_code = main(
        ["build", "-t", "agentseek:test"],
        process_transport=capture,
        cwd=tmp_path,
        stderr=stderr,
    )

    assert exit_code == 2
    assert capture.calls is not None
    assert [call.argv for call in capture.calls] == [
        ("docker", "buildx", "version"),
        ("docker", "buildx", "inspect"),
    ]
    assert "builder is unavailable" in stderr.getvalue()


def test_build_command_carries_pip_config_only_as_buildkit_secret(
    tmp_path: Path,
) -> None:
    from agentseek_api.cli import main

    config = _write_basic_langgraph_config(tmp_path)
    pip_config = tmp_path / "pip.conf"
    pip_config.write_text("password=cli-pip-canary\n", encoding="utf-8")
    payload = json.loads(config.read_text(encoding="utf-8"))
    payload["pip_config_file"] = "./pip.conf"
    config.write_text(json.dumps(payload), encoding="utf-8")
    capture = _ProcessCapture()

    exit_code = main(
        ["build", "-t", "agentseek:test"],
        process_transport=capture,
        cwd=tmp_path,
    )

    assert exit_code == 0
    build = _captured_image_build(capture)
    assert build.argv[-3:] == (
        "--secret",
        f"id=pip_config,src={pip_config}",
        "-",
    )
    assert "cli-pip-canary" not in " ".join(build.argv)
    assert b"cli-pip-canary" not in build.stdin_bytes


def test_build_excludes_cli_dotenv_even_through_local_dependency_tree(
    tmp_path: Path,
) -> None:
    from agentseek_api.cli import main

    _write_basic_langgraph_config(tmp_path)
    env_file = tmp_path / "build.env"
    env_file.write_text("TOKEN=cli-dotenv-canary\n", encoding="utf-8")
    capture = _ProcessCapture()

    exit_code = main(
        ["build", "-t", "agentseek:test", "--env-file", str(env_file)],
        process_transport=capture,
        cwd=tmp_path,
    )

    assert exit_code == 0
    assert capture.calls is not None
    archive_bytes = _captured_image_build(capture).stdin_bytes
    assert archive_bytes is not None
    assert b"cli-dotenv-canary" not in archive_bytes
    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:") as archive:
        assert "app/build.env" not in archive.getnames()


def test_candidate_runtime_injection_changes_copied_build_artifact(
    tmp_path: Path,
) -> None:
    from agentseek_api.cli import main
    from agentseek_api.container_build import candidate_runtime_artifact

    _write_basic_langgraph_config(tmp_path)
    wheel = tmp_path / "agentseek_api-0.3.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(
            "agentseek_api-0.3.0.dist-info/METADATA",
            "Metadata-Version: 2.1\nName: agentseek-api\nVersion: 0.3.0\n",
        )
    artifact = candidate_runtime_artifact(
        wheel, hashlib.sha256(wheel.read_bytes()).hexdigest()
    )
    capture = _ProcessCapture()

    exit_code = main(
        ["build", "-t", "agentseek:candidate"],
        process_transport=capture,
        cwd=tmp_path,
        runtime_artifact=artifact,
    )

    assert exit_code == 0
    assert capture.calls is not None
    archive_bytes = _captured_image_build(capture).stdin_bytes
    assert archive_bytes is not None
    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:") as archive:
        copied = archive.extractfile(f"runtime/{wheel.name}")
        assert copied is not None
        assert copied.read() == wheel.read_bytes()


def test_generated_up_uses_final_auth_selection_and_sanitized_build_stdin(
    tmp_path: Path,
) -> None:
    from agentseek_api.cli import main

    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname="fixture"\nversion="1.0"\n', encoding="utf-8"
    )
    lower_auth = tmp_path / "lower_auth.py"
    lower_auth.write_text("auth = 'lower'\n", encoding="utf-8")
    winning_auth = tmp_path / "winning_auth.py"
    winning_auth.write_text("auth = 'winner'\n", encoding="utf-8")
    config = tmp_path / "agentseek.json"
    config.write_text(
        '{"graphs":{"chat":"installed.graph:graph"},'
        '"auth":{"path":"./lower_auth.py:auth"}}',
        encoding="utf-8",
    )
    env_file = tmp_path / "up.env"
    env_file.write_text(
        "AUTH_MODULE_PATH=winning_auth.py:auth\n"
        "OPENAI_API_KEY=provider-secret-canary\n",
        encoding="utf-8",
    )
    capture = _ProcessCapture()

    exit_code = main(
        ["up", "--env-file", str(env_file), "--no-pull"],
        process_transport=capture,
        cwd=tmp_path,
    )

    assert exit_code == 0
    assert capture.calls is not None
    build = _captured_image_build(capture)
    assert build.argv == (
        "docker",
        "buildx",
        "build",
        "--load",
        "--file",
        "Dockerfile",
        "--tag",
        "agentseek-up:8123",
        "-",
    )
    assert build.stdin_bytes is not None
    assert b"provider-secret-canary" not in build.stdin_bytes
    assert "provider-secret-canary" not in " ".join(build.argv)
    assert "provider-secret-canary" not in repr(build)
    with tarfile.open(fileobj=io.BytesIO(build.stdin_bytes), mode="r:") as archive:
        members = set(archive.getnames())
        assert "app/winning_auth.py" in members
        assert "app/lower_auth.py" not in members
        assert "app/up.env" not in members
    container_env = _application_environment(capture)
    assert container_env["AUTH_MODULE_PATH"] == "/deps/agent/winning_auth.py:auth"
    assert container_env["OPENAI_API_KEY"] == "provider-secret-canary"


@pytest.mark.parametrize("reference", ["auth.py:auth", "/host/auth.py:auth"])
def test_up_with_custom_image_rejects_host_file_auth_references(
    tmp_path: Path, reference: str
) -> None:
    from agentseek_api.cli import main

    config = _write_basic_langgraph_config(tmp_path)
    env_file = tmp_path / "up.env"
    env_file.write_text(f"AUTH_MODULE_PATH={reference}\n", encoding="utf-8")
    capture = _ProcessCapture()
    stderr = io.StringIO()

    exit_code = main(
        [
            "up",
            "--config",
            str(config),
            "--image",
            "agentseek:test",
            "--env-file",
            str(env_file),
        ],
        process_transport=capture,
        cwd=tmp_path,
        stderr=stderr,
    )

    assert exit_code == 2
    assert capture.calls is None
    assert "package" in stderr.getvalue()
    assert "host" in stderr.getvalue().lower()


@pytest.mark.parametrize("reference", ["", "installed.auth:auth"])
def test_up_with_custom_image_preserves_empty_or_package_auth(
    tmp_path: Path, reference: str
) -> None:
    from agentseek_api.cli import main

    config = _write_basic_langgraph_config(tmp_path)
    env_file = tmp_path / "up.env"
    env_file.write_text(f"AUTH_MODULE_PATH={reference}\n", encoding="utf-8")
    capture = _ProcessCapture()

    exit_code = main(
        [
            "up",
            "--config",
            str(config),
            "--image",
            "agentseek:test",
            "--env-file",
            str(env_file),
        ],
        process_transport=capture,
        cwd=tmp_path,
    )

    assert exit_code == 0
    assert _application_environment(capture)["AUTH_MODULE_PATH"] == reference


def test_build_runtime_env_parses_exported_values(tmp_path: Path) -> None:
    from agentseek_api.cli import build_runtime_env

    config_path = _write_basic_langgraph_config(tmp_path)
    env_file = tmp_path / ".env"
    env_file.write_text(
        '# comment\nexport TOKEN="quoted # value\nnext"\nPLAIN=value # inline comment\n',
        encoding="utf-8",
    )

    env = build_runtime_env(
        config_path=config_path, env_file=str(env_file), cwd=tmp_path, base_env={}
    )

    assert env["TOKEN"] == "quoted # value\nnext"
    assert env["PLAIN"] == "value"
    assert env["AGENTSEEK_GRAPHS"] == str(config_path.resolve())


def test_build_runtime_env_ignores_dotenv_entries_without_values(
    tmp_path: Path,
) -> None:
    from agentseek_api.cli import build_runtime_env

    config_path = _write_basic_langgraph_config(tmp_path)
    env_file = tmp_path / ".env"
    env_file.write_text("MALFORMED_LINE\nTOKEN=present\n", encoding="utf-8")

    env = build_runtime_env(
        config_path=config_path, env_file=str(env_file), cwd=tmp_path, base_env={}
    )

    assert "MALFORMED_LINE" not in env
    assert env["TOKEN"] == "present"


def test_build_runtime_env_shell_values_override_config_and_cli_dotenv(
    tmp_path: Path,
) -> None:
    from agentseek_api.cli import build_runtime_env

    config_path = _write_basic_langgraph_config(tmp_path)
    config_env = tmp_path / "config.env"
    config_env.write_text("TOKEN=from-config\n", encoding="utf-8")
    config_path.write_text(
        """
{
  "graphs": {"chat": "chat.graph:graph"},
  "env": "./config.env"
}
""".strip(),
        encoding="utf-8",
    )
    cli_env = tmp_path / "override.env"
    cli_env.write_text("TOKEN=from-cli-file\n", encoding="utf-8")

    env = build_runtime_env(
        config_path=config_path,
        env_file=str(cli_env),
        cwd=tmp_path,
        base_env={"TOKEN": "from-shell"},
    )

    assert env["TOKEN"] == "from-shell"


def test_higher_precedence_valueless_binding_keeps_lower_export(tmp_path: Path) -> None:
    from agentseek_api.cli import build_runtime_env

    config_env = tmp_path / "config.env"
    config_env.write_text("TOKEN=from-config\n", encoding="utf-8")
    config_path = tmp_path / "langgraph.json"
    config_path.write_text(
        '{"graphs":{"chat":"chat.graph:graph"},"env":"./config.env"}',
        encoding="utf-8",
    )
    cli_env = tmp_path / "cli.env"
    cli_env.write_text("TOKEN\nRESULT=${TOKEN:-fallback}\n", encoding="utf-8")

    env = build_runtime_env(
        config_path=config_path, env_file=str(cli_env), cwd=tmp_path, base_env={}
    )

    assert env["TOKEN"] == "from-config"
    assert env["RESULT"] == ""


def test_build_runtime_env_rejects_invalid_config_env_shape(tmp_path: Path) -> None:
    from agentseek_api.cli import build_runtime_env

    config_path = tmp_path / "langgraph.json"
    config_path.write_text(
        '{"graphs":{"chat":"chat.graph:graph"},"env":["bad"]}', encoding="utf-8"
    )

    with pytest.raises(
        RuntimeError, match="must set 'env' to a path string or key/value object"
    ):
        build_runtime_env(
            config_path=config_path, env_file=None, cwd=tmp_path, base_env={}
        )


def test_build_runtime_env_rejects_non_scalar_config_env_value(tmp_path: Path) -> None:
    from agentseek_api.cli import build_runtime_env

    config_path = tmp_path / "langgraph.json"
    config_path.write_text(
        '{"graphs":{"chat":"chat.graph:graph"},"env":{"BAD":[]}}', encoding="utf-8"
    )

    with pytest.raises(RuntimeError, match="env mapping values must be scalar"):
        build_runtime_env(
            config_path=config_path, env_file=None, cwd=tmp_path, base_env={}
        )


def test_containerize_symbol_reference_supports_windows_drive_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agentseek_api import cli as cli_module

    expected_path = tmp_path / "auth.py"
    monkeypatch.setattr(
        cli_module, "_resolve_path", lambda path_text, *, cwd: expected_path
    )
    monkeypatch.setattr(
        cli_module,
        "_container_config_path",
        lambda *, config_path, cwd: "/deps/agent/auth.py",
    )

    result = cli_module._containerize_symbol_reference(
        r"C:\workspace\auth.py:backend", cwd=tmp_path
    )

    assert result == "/deps/agent/auth.py:backend"


def test_dockerfile_command_requires_valid_config_object(tmp_path: Path) -> None:
    from agentseek_api.cli import main

    config_path = tmp_path / "langgraph.json"
    config_path.write_text("[]", encoding="utf-8")
    stderr = io.StringIO()

    exit_code = main(
        ["dockerfile", "--config", str(config_path), "Dockerfile"],
        cwd=tmp_path,
        stderr=stderr,
    )

    assert exit_code == 2
    assert "must contain a top-level JSON object" in stderr.getvalue()


def test_dockerfile_command_rejects_invalid_auth_and_missing_pip_config(
    tmp_path: Path,
) -> None:
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

    exit_code = main(
        ["dockerfile", "--config", str(config_path), "Dockerfile"],
        cwd=tmp_path,
        stderr=stderr,
    )

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

    exit_code = main(
        ["dockerfile", "--config", str(config_path), "Dockerfile"],
        cwd=tmp_path,
        stderr=stderr,
    )

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

    exit_code = main(
        ["dockerfile", "--config", str(config_path), "Dockerfile"],
        cwd=tmp_path,
        stderr=stderr,
    )

    assert exit_code == 2
    assert "not supported without an explicit base_image" in stderr.getvalue()


def test_dockerfile_command_allows_python_alpine_base_without_apt(
    tmp_path: Path,
) -> None:
    from agentseek_api.cli import main

    config_path = tmp_path / "langgraph.json"
    config_path.write_text(
        """
{
  "graphs": {
    "chat": "chat.graph:graph"
  },
  "base_image": "python:3.12-alpine"
}
""".strip(),
        encoding="utf-8",
    )
    dockerfile_path = tmp_path / "Dockerfile.agentseek"

    exit_code = main(
        ["dockerfile", "--config", str(config_path), str(dockerfile_path)],
        cwd=tmp_path,
    )

    assert exit_code == 0
    content = (dockerfile_path / "context" / "Dockerfile").read_text(encoding="utf-8")
    assert "FROM python:3.12-alpine" in content
    assert "apt-get" not in content


def test_dockerfile_command_allows_explicit_python_runtime_base_image(
    tmp_path: Path,
) -> None:
    from agentseek_api.cli import main

    config_path = tmp_path / "langgraph.json"
    config_path.write_text(
        """
{
  "graphs": {
    "chat": "chat.graph:graph"
  },
  "base_image": "registry.access.redhat.com/ubi9/python-312"
}
""".strip(),
        encoding="utf-8",
    )
    dockerfile_path = tmp_path / "Dockerfile.agentseek"

    exit_code = main(
        ["dockerfile", "--config", str(config_path), str(dockerfile_path)],
        cwd=tmp_path,
    )

    assert exit_code == 0
    content = (dockerfile_path / "context" / "Dockerfile").read_text(encoding="utf-8")
    assert "FROM registry.access.redhat.com/ubi9/python-312" in content


def test_dockerfile_command_rejects_syntactically_invalid_base_image(
    tmp_path: Path,
) -> None:
    from agentseek_api.cli import main

    config_path = tmp_path / "langgraph.json"
    config_path.write_text(
        json.dumps(
            {
                "graphs": {"chat": "chat.graph:graph"},
                "base_image": "python:3.12 slim",
            }
        ),
        encoding="utf-8",
    )
    stderr = io.StringIO()

    exit_code = main(
        ["dockerfile", "--config", str(config_path), "Dockerfile"],
        cwd=tmp_path,
        stderr=stderr,
    )

    assert exit_code == 2
    assert "base image is invalid" in stderr.getvalue()


def test_dockerfile_command_allows_supported_explicit_langgraph_base_image(
    tmp_path: Path,
) -> None:
    from agentseek_api.cli import main

    config_path = tmp_path / "langgraph.json"
    config_path.write_text(
        """
{
  "graphs": {
    "chat": "chat.graph:graph"
  },
  "base_image": "langchain/langgraph-api:0.2"
}
""".strip(),
        encoding="utf-8",
    )
    dockerfile_path = tmp_path / "Dockerfile.agentseek"

    exit_code = main(
        ["dockerfile", "--config", str(config_path), str(dockerfile_path)], cwd=tmp_path
    )

    assert exit_code == 0
    content = (dockerfile_path / "context" / "Dockerfile").read_text(encoding="utf-8")
    assert "FROM langchain/langgraph-api:0.2" in content


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
        "METADATA_DB_URL=sqlite+aiosqlite:////tmp/agentseek.db\n"
        "OCEANBASE_HOST=host.docker.internal\n"
        "API_ORIGIN=https://api.example.test\n"
        "OPENAI_BASE_URL=${API_ORIGIN}/v1\n",
        encoding="utf-8",
    )
    capture = _ProcessCapture()

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
        process_transport=capture,
        cwd=tmp_path,
    )

    assert exit_code == 0
    assert capture.calls is not None
    assert capture.calls[0].argv == (
        "docker",
        "rm",
        "-f",
        "agentseek-up-8123",
    )
    assert capture.calls[1].argv[:9] == (
        "docker",
        "run",
        "--detach",
        "--name",
        "agentseek-up-8123",
        "--add-host",
        "host.docker.internal:host-gateway",
        "-p",
        "8123:2024",
    )
    assert capture.calls[1].argv[-1] == "agentseek:test"
    container_env = _application_environment(capture)
    assert container_env["AGENTSEEK_GRAPHS"] == "/deps/agent/langgraph.json"
    assert container_env["METADATA_DB_URL"] == "sqlite+aiosqlite:////tmp/agentseek.db"
    assert container_env["OCEANBASE_HOST"] == "host.docker.internal"
    assert container_env["OPENAI_BASE_URL"] == "https://api.example.test/v1"


def test_up_keeps_application_values_only_in_final_run_carrier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agentseek_api.cli import main

    config_path = _write_basic_langgraph_config(tmp_path)
    env_file = tmp_path / "docker.env"
    env_file.write_text(
        "OPENAI_API_KEY=sk-$#=雪\nEMPTY=\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("DOCKER_HOST", "unix:///private/docker.sock")
    monkeypatch.setenv("UNSELECTED_CANARY", "must-not-cross")
    capture = _ProcessCapture()

    exit_code = main(
        [
            "up",
            "--config",
            str(config_path),
            "--image",
            "agentseek:test",
            "--env-file",
            str(env_file),
            "--pass-env",
            "EMPTY",
            "--recreate",
        ],
        process_transport=capture,
        cwd=tmp_path,
    )

    assert exit_code == 0
    assert capture.calls is not None
    assert len(capture.calls) == 2
    remove, run = capture.calls
    assert type(remove) is ProcessInvocation
    assert dict(remove.environment)["DOCKER_HOST"] == "unix:///private/docker.sock"
    assert "OPENAI_API_KEY" not in remove.environment
    assert isinstance(run, DockerRunInvocation)
    assert run.environment["OPENAI_API_KEY"] == "sk-$#=雪"
    assert run.environment["EMPTY"] == ""
    assert "UNSELECTED_CANARY" not in run.environment
    joined_argv = " ".join(run.argv)
    assert "sk-$#=雪" not in joined_argv
    assert "OPENAI_API_KEY=sk-$#=雪" not in joined_argv
    assert ("-e", "OPENAI_API_KEY") in tuple(zip(run.argv, run.argv[1:], strict=False))


def test_up_container_existence_probe_is_bounded_and_control_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agentseek_api.cli import main

    config_path = _write_basic_langgraph_config(tmp_path)
    monkeypatch.setenv("DOCKER_HOST", "unix:///private/docker.sock")
    monkeypatch.setenv("OPENAI_API_KEY", "application-canary")
    capture = _ProcessCapture()

    exit_code = main(
        ["up", "--config", str(config_path), "--image", "agentseek:test"],
        process_transport=capture,
        cwd=tmp_path,
    )

    assert exit_code == 0
    assert capture.calls is not None
    probe = capture.calls[0]
    assert isinstance(probe, ControlQueryInvocation)
    assert probe.timeout_seconds > 0
    assert probe.argv == (
        "docker",
        "container",
        "inspect",
        "agentseek-up-8123",
    )
    assert probe.environment["DOCKER_HOST"] == "unix:///private/docker.sock"
    assert "OPENAI_API_KEY" not in probe.environment
    assert capture.calls[1].environment["OPENAI_API_KEY"] == "application-canary"


def test_up_command_supports_docker_compose_sidecars(tmp_path: Path) -> None:
    from agentseek_api.cli import main

    config_path = _write_basic_langgraph_config(tmp_path)
    compose_path = tmp_path / "docker-compose.yml"
    compose_path.write_text("services: {}\n", encoding="utf-8")
    capture = _ProcessCapture()

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
        process_transport=capture,
        cwd=tmp_path,
    )

    assert exit_code == 0
    assert capture.calls is not None
    assert capture.calls[0].argv == (
        "docker",
        "compose",
        "version",
        "--short",
    )
    assert capture.calls[1].argv == (
        "docker",
        "rm",
        "-f",
        "agentseek-up-8123",
    )
    compose_invocation = capture.calls[2]
    assert compose_invocation.argv[:3] == (
        "docker",
        "compose",
        "--env-file",
    )
    assert compose_invocation.argv[4:] == (
        "-f",
        str(compose_path.resolve()),
        "up",
        "-d",
        "--force-recreate",
    )
    assert capture.calls[3].argv[-1] == "agentseek:test"
    for invocation in capture.calls:
        assert "AGENTSEEK_GRAPHS" not in invocation.environment or isinstance(
            invocation, DockerRunInvocation
        )


def test_up_compose_uses_selected_literal_artifact_and_ignores_ambient_controls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agentseek_api.cli import main

    config_path = tmp_path / "langgraph.json"
    config_path.write_text(
        '{"graphs":{"chat":"chat.graph:graph"},'
        '"env":{"TOKEN":"${HOSTILE} # literal"},'
        '"compose_env":["TOKEN"]}',
        encoding="utf-8",
    )
    compose_path = tmp_path / "compose.yaml"
    compose_path.write_text(
        'services:\n  db:\n    image: busybox\n    environment:\n      TOKEN: "${TOKEN}"\n',
        encoding="utf-8",
    )
    (tmp_path / ".env").write_text("TOKEN=project-dotenv-canary\n", encoding="utf-8")
    monkeypatch.setenv("COMPOSE_FILE", "hostile.yaml")
    monkeypatch.setenv("COMPOSE_ENV_FILES", "hostile.env")
    monkeypatch.setenv("HOSTILE", "ambient-canary")

    class ContentCapture(_ProcessCapture):
        compose_contents: bytes | None = None

        def __call__(self, invocation: ProcessInvocation) -> ProcessResult:
            if invocation.argv[:3] == ("docker", "compose", "--env-file"):
                self.compose_contents = Path(invocation.argv[3]).read_bytes()
            return super().__call__(invocation)

    capture = ContentCapture()

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
        process_transport=capture,
        cwd=tmp_path,
    )

    assert exit_code == 0
    assert capture.calls is not None
    compose = next(
        call
        for call in capture.calls
        if call.argv[:3] == ("docker", "compose", "--env-file")
    )
    artifact = Path(compose.argv[3])
    assert not artifact.exists()
    assert "COMPOSE_FILE" not in compose.environment
    assert "COMPOSE_ENV_FILES" not in compose.environment
    assert "HOSTILE" not in compose.environment
    assert "project-dotenv-canary" not in " ".join(compose.argv)
    assert capture.compose_contents == b'TOKEN="$${HOSTILE} # literal"\n'


def test_up_compose_artifact_is_removed_after_compose_failure(tmp_path: Path) -> None:
    from agentseek_api.cli import main

    config_path = _write_basic_langgraph_config(tmp_path)
    compose_path = tmp_path / "compose.yaml"
    compose_path.write_text("services: {}\n", encoding="utf-8")
    artifact_paths: list[Path] = []

    class FailureCapture(_ProcessCapture):
        def __call__(self, invocation: ProcessInvocation) -> ProcessResult:
            result = super().__call__(invocation)
            if invocation.argv[:3] == ("docker", "compose", "--env-file"):
                artifact = Path(invocation.argv[3])
                assert artifact.exists()
                artifact_paths.append(artifact)
                return ProcessResult(returncode=19)
            return result

    capture = FailureCapture()
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
        process_transport=capture,
        cwd=tmp_path,
    )

    assert exit_code == 19
    assert len(artifact_paths) == 1
    assert not artifact_paths[0].exists()
    assert capture.calls is not None
    assert not any(isinstance(call, DockerRunInvocation) for call in capture.calls)


def test_up_rejects_missing_compose_selection_before_build_or_artifact(
    tmp_path: Path,
) -> None:
    from agentseek_api.cli import main

    config_path = tmp_path / "langgraph.json"
    config_path.write_text(
        '{"graphs":{"chat":"chat.graph:graph"},"compose_env":["MISSING"]}',
        encoding="utf-8",
    )
    compose_path = tmp_path / "compose.yaml"
    compose_path.write_text("services: {}\n", encoding="utf-8")
    capture = _ProcessCapture()
    stderr = io.StringIO()

    exit_code = main(
        [
            "up",
            "--config",
            str(config_path),
            "--docker-compose",
            str(compose_path),
        ],
        process_transport=capture,
        cwd=tmp_path,
        stderr=stderr,
    )

    assert exit_code == 2
    assert capture.calls is None
    assert "not present in application payload" in stderr.getvalue()


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


def test_up_command_rejects_existing_container_before_starting_compose_sidecars(
    tmp_path: Path,
) -> None:
    from agentseek_api.cli import main

    config_path = _write_basic_langgraph_config(tmp_path)
    compose_path = tmp_path / "docker-compose.yml"
    compose_path.write_text("services: {}\n", encoding="utf-8")
    stderr = io.StringIO()
    capture = _ProcessCapture(container_exists=True)

    exit_code = main(
        [
            "up",
            "--config",
            str(config_path),
            "--image",
            "agentseek:test",
            "--docker-compose",
            str(compose_path),
        ],
        process_transport=capture,
        cwd=tmp_path,
        stderr=stderr,
    )

    assert exit_code == 2
    assert capture.calls is not None
    assert [call.argv for call in capture.calls] == [
        ("docker", "compose", "version", "--short"),
        ("docker", "container", "inspect", "agentseek-up-8123"),
    ]
    assert "already exists" in stderr.getvalue()
    assert "--recreate" in stderr.getvalue()


def test_up_command_builds_image_when_missing_and_passes_postgres_uri(
    tmp_path: Path,
) -> None:
    from agentseek_api.cli import main

    _write_basic_langgraph_config(tmp_path)
    capture = _ProcessCapture()

    exit_code = main(
        [
            "up",
            "--port",
            "8124",
            "--no-pull",
            "--postgres-uri",
            "postgresql://postgres:postgres@db/agentseek",
        ],
        process_transport=capture,
        cwd=tmp_path,
    )

    assert exit_code == 0
    assert capture.calls is not None
    assert [call.argv for call in capture.calls[:2]] == [
        ("docker", "buildx", "version"),
        ("docker", "buildx", "inspect"),
    ]
    assert capture.calls[2].argv == (
        "docker",
        "buildx",
        "build",
        "--load",
        "--file",
        "Dockerfile",
        "--tag",
        "agentseek-up:8124",
        "-",
    )
    assert capture.calls[2].stdin_bytes is not None
    assert capture.calls[3].argv == (
        "docker",
        "container",
        "inspect",
        "agentseek-up-8124",
    )
    assert capture.calls[4].argv[:9] == (
        "docker",
        "run",
        "--detach",
        "--name",
        "agentseek-up-8124",
        "--add-host",
        "host.docker.internal:host-gateway",
        "-p",
        "8124:2024",
    )
    assert capture.calls[4].argv[-1] == "agentseek-up:8124"
    for invocation in capture.calls[:4]:
        assert "METADATA_DB_URL" not in invocation.environment
        assert "postgresql://postgres:postgres@db/agentseek" not in " ".join(
            invocation.argv
        )
    container_env = _application_environment(capture)
    assert container_env["AGENTSEEK_GRAPHS"] == "/deps/agent/langgraph.json"
    assert (
        container_env["METADATA_DB_URL"]
        == "postgresql://postgres:postgres@db/agentseek"
    )
    assert container_env["METADATA_DB_BACKEND"] == "postgresql"


def test_up_command_passes_config_auth_env_and_containerizes_file_paths(
    tmp_path: Path,
) -> None:
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
    (tmp_path / "auth.py").write_text("backend = object()\n", encoding="utf-8")
    capture = _ProcessCapture()

    exit_code = main(
        [
            "up",
            "--config",
            str(config_path),
            "--no-pull",
        ],
        process_transport=capture,
        cwd=tmp_path,
    )

    assert exit_code == 0
    assert capture.calls is not None
    assert capture.calls[3].argv == (
        "docker",
        "container",
        "inspect",
        "agentseek-up-8123",
    )
    assert capture.calls[4].argv[:9] == (
        "docker",
        "run",
        "--detach",
        "--name",
        "agentseek-up-8123",
        "--add-host",
        "host.docker.internal:host-gateway",
        "-p",
        "8123:2024",
    )
    assert capture.calls[4].argv[-1] == "agentseek-up:8123"
    container_env = _application_environment(capture)
    assert container_env["AGENTSEEK_GRAPHS"] == "/deps/agent/langgraph.json"
    assert container_env["AUTH_MODULE_PATH"] == "/deps/agent/auth.py:backend"
    assert container_env["FEATURE_FLAG"] == "True"


def test_up_command_passes_ambient_env_into_container(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agentseek_api.cli import main

    config_path = _write_basic_langgraph_config(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "ambient-key")
    capture = _ProcessCapture()

    exit_code = main(
        [
            "up",
            "--config",
            str(config_path),
            "--image",
            "agentseek:test",
        ],
        process_transport=capture,
        cwd=tmp_path,
    )

    assert exit_code == 0
    assert capture.calls is not None
    container_env = _application_environment(capture)
    assert container_env["OPENAI_API_KEY"] == "ambient-key"


def test_up_command_prefers_agentseek_json_without_explicit_flag(
    tmp_path: Path,
) -> None:
    from agentseek_api.cli import main

    (tmp_path / "agentseek.json").write_text(
        """
{
  "graphs": {
    "chat": "chat.graph:graph"
  }
}
""".strip(),
        encoding="utf-8",
    )
    _write_basic_langgraph_config(tmp_path)
    capture = _ProcessCapture()

    exit_code = main(
        [
            "up",
            "--image",
            "agentseek:test",
        ],
        process_transport=capture,
        cwd=tmp_path,
    )

    assert exit_code == 0
    assert capture.calls is not None
    container_env = _application_environment(capture)
    assert container_env["AGENTSEEK_GRAPHS"] == "/deps/agent/agentseek.json"


def test_up_command_does_not_pass_shell_runtime_env_into_container(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agentseek_api.cli import main

    config_path = _write_basic_langgraph_config(tmp_path)
    monkeypatch.setenv("PATH", "/tmp/bad-path")
    monkeypatch.setenv("PWD", "/tmp/host-pwd")
    capture = _ProcessCapture()

    exit_code = main(
        [
            "up",
            "--config",
            str(config_path),
            "--image",
            "agentseek:test",
        ],
        process_transport=capture,
        cwd=tmp_path,
    )

    assert exit_code == 0
    assert capture.calls is not None
    container_env = _application_environment(capture)
    assert "PATH" not in container_env
    assert "PWD" not in container_env


def test_up_command_uses_base_image_override_when_building(tmp_path: Path) -> None:
    from agentseek_api.cli import main

    _write_basic_langgraph_config(tmp_path)
    capture = _ProcessCapture()

    exit_code = main(
        [
            "up",
            "--port",
            "8126",
            "--base-image",
            "python:3.13-slim-bookworm",
            "--no-pull",
        ],
        process_transport=capture,
        cwd=tmp_path,
    )

    assert exit_code == 0
    assert capture.calls is not None
    archive_bytes = _captured_image_build(capture).stdin_bytes
    assert archive_bytes is not None
    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:") as archive:
        dockerfile_file = archive.extractfile("Dockerfile")
        assert dockerfile_file is not None
        dockerfile = dockerfile_file.read().decode()
    assert "FROM python:3.13-slim-bookworm" in dockerfile


def test_up_command_allows_python_alpine_base_without_apt(
    tmp_path: Path,
) -> None:
    from agentseek_api.cli import main

    _write_basic_langgraph_config(tmp_path)
    capture = _ProcessCapture()

    exit_code = main(
        [
            "up",
            "--base-image",
            "python:3.12-alpine",
        ],
        process_transport=capture,
        cwd=tmp_path,
    )

    assert exit_code == 0
    archive_bytes = _captured_image_build(capture).stdin_bytes
    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:") as archive:
        dockerfile_file = archive.extractfile("Dockerfile")
        assert dockerfile_file is not None
        dockerfile = dockerfile_file.read().decode()
    assert "FROM python:3.12-alpine" in dockerfile
    assert "apt-get" not in dockerfile


def test_up_command_returns_build_failure_without_running_container(
    tmp_path: Path,
) -> None:
    from agentseek_api.cli import main

    _write_basic_langgraph_config(tmp_path)
    build_argv = (
        "docker",
        "buildx",
        "build",
        "--load",
        "--file",
        "Dockerfile",
        "--pull",
        "--tag",
        "agentseek-up:8125",
        "-",
    )
    capture = _ProcessCapture(return_codes={build_argv: 9})

    exit_code = main(
        ["up", "--port", "8125"],
        process_transport=capture,
        cwd=tmp_path,
    )

    assert exit_code == 9
    assert capture.calls is not None
    assert [call.argv for call in capture.calls] == [
        ("docker", "buildx", "version"),
        ("docker", "buildx", "inspect"),
        build_argv,
    ]


def test_up_command_rejects_existing_container_without_recreate(tmp_path: Path) -> None:
    from agentseek_api.cli import main

    config_path = _write_basic_langgraph_config(tmp_path)
    stderr = io.StringIO()
    capture = _ProcessCapture(container_exists=True)

    exit_code = main(
        [
            "up",
            "--config",
            str(config_path),
            "--image",
            "agentseek:test",
        ],
        process_transport=capture,
        cwd=tmp_path,
        stderr=stderr,
    )

    assert exit_code == 2
    assert capture.calls is not None
    assert [call.argv for call in capture.calls] == [
        ("docker", "container", "inspect", "agentseek-up-8123")
    ]
    assert "already exists" in stderr.getvalue()
    assert "--recreate" in stderr.getvalue()


def test_container_exists_uses_a_private_bounded_query(tmp_path: Path) -> None:
    from agentseek_api import cli as cli_module

    capture = _ProcessCapture(container_exists=True)

    exists = cli_module._container_exists(
        "agentseek-up-8123",
        process_transport=capture,
        docker_control={},
        cwd=tmp_path,
    )

    assert exists is True
    assert capture.calls is not None
    invocation = capture.calls[0]
    assert isinstance(invocation, ControlQueryInvocation)
    assert invocation.timeout_seconds > 0
    assert invocation.argv == (
        "docker",
        "container",
        "inspect",
        "agentseek-up-8123",
    )


def test_up_command_waits_for_http_health_when_requested(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agentseek_api import cli as cli_module

    config_path = _write_basic_langgraph_config(tmp_path)
    capture = _ProcessCapture()
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
        process_transport=capture,
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
    assert "Unsupported option(s) for 'agentseek-api up': --watch" in stderr.getvalue()
