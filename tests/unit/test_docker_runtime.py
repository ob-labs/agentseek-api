from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from agentseek_api.docker_runtime import (
    ControlQueryInvocation,
    IMAGE_COMPATIBILITY_FORMAT,
    DockerRuntimeError,
    LegacyRunnerAdapter,
    ProcessInvocation,
    ProcessResult,
    SubprocessTransport,
    build_docker_control_invocation,
    build_docker_query_invocation,
    build_docker_run_invocation,
    parse_buildx_version_result,
    parse_compose_version_result,
    parse_image_compatibility_result,
    require_buildx_available,
)
from agentseek_api.environment import ContainerPolicyError


def test_docker_run_uses_names_in_argv_and_values_only_in_carrier(
    tmp_path: Path,
) -> None:
    invocation = build_docker_run_invocation(
        base_argv=("docker", "run", "--rm"),
        image="agentseek:test",
        docker_control={"PATH": "/usr/bin"},
        application_payload={"OPENAI_API_KEY": "sk-$#=雪", "EMPTY": ""},
        container_argv=(
            "agentseek-api",
            "serve",
            "--environment-mode=preloaded-v1",
        ),
        cwd=tmp_path,
    )

    assert invocation.argv == (
        "docker",
        "run",
        "--rm",
        "-e",
        "EMPTY",
        "-e",
        "OPENAI_API_KEY",
        "agentseek:test",
        "agentseek-api",
        "serve",
        "--environment-mode=preloaded-v1",
    )
    assert invocation.environment["OPENAI_API_KEY"] == "sk-$#=雪"
    assert invocation.application_names == frozenset({"EMPTY", "OPENAI_API_KEY"})
    assert "sk-$#=雪" not in " ".join(invocation.argv)


def test_non_run_docker_invocation_has_only_docker_control(tmp_path: Path) -> None:
    invocation = build_docker_control_invocation(
        argv=("docker", "image", "inspect", "agentseek:test"),
        docker_control={"PATH": "/usr/bin"},
        cwd=tmp_path,
    )

    assert dict(invocation.environment) == {"PATH": "/usr/bin"}
    with pytest.raises(TypeError):
        invocation.environment["OPENAI_API_KEY"] = "not-allowed"  # type: ignore[index]


@pytest.mark.parametrize(
    ("docker_control", "application_payload"),
    [
        ({"OPENAI_API_KEY": "control"}, {"OPENAI_API_KEY": "application"}),
        ({"PATH": "bad\0path"}, {}),
        ({}, {"BAD\0NAME": "value"}),
        ({}, {"TOKEN": "bad\0value"}),
    ],
)
def test_docker_run_rejects_collisions_and_nul(
    tmp_path: Path,
    docker_control: dict[str, str],
    application_payload: dict[str, str],
) -> None:
    with pytest.raises(ContainerPolicyError):
        build_docker_run_invocation(
            base_argv=("docker", "run"),
            image="agentseek:test",
            docker_control=docker_control,
            application_payload=application_payload,
            container_argv=(),
            cwd=tmp_path,
        )


def test_query_invocation_is_bounded_and_redacted(tmp_path: Path) -> None:
    canary = "baked-env-canary"
    invocation = build_docker_query_invocation(
        argv=(
            "docker",
            "image",
            "inspect",
            "--format",
            "[{{json .Config.Labels}},{{json .Config.Entrypoint}},{{json .Config.Cmd}}]",
            "hostile:test",
        ),
        docker_control={"DOCKER_HOST": "unix:///private/docker.sock"},
        cwd=tmp_path,
        timeout_seconds=3.0,
    )
    result = ProcessResult(
        returncode=0,
        stdout=b'[{"contract":"preloaded-v1"},[],[]]',
        stderr=canary.encode(),
    )

    assert isinstance(invocation, ControlQueryInvocation)
    assert invocation.timeout_seconds == 3.0
    assert ".Config.Env" not in " ".join(invocation.argv)
    assert "DOCKER_HOST" not in repr(invocation)
    assert canary not in repr(result)
    assert "preloaded-v1" not in repr(result)


def test_query_timeout_is_value_free(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def timeout(*args: object, **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        raise subprocess.TimeoutExpired(
            cmd=["docker", "image", "inspect", "secret-image-name"],
            timeout=1.0,
            output=b"private-output",
            stderr=b"private-error",
        )

    monkeypatch.setattr(subprocess, "run", timeout)
    invocation = build_docker_query_invocation(
        argv=("docker", "image", "inspect", "secret-image-name"),
        docker_control={},
        cwd=tmp_path,
        timeout_seconds=1.0,
    )

    with pytest.raises(DockerRuntimeError) as exc_info:
        SubprocessTransport()(invocation)

    message = str(exc_info.value)
    assert message == "Docker control query timed out."
    assert "secret-image-name" not in message
    assert "private-output" not in message
    assert "private-error" not in message


def test_subprocess_transport_captures_only_queries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_run(
        argv: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[bytes]:
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout=b"captured" if kwargs["stdout"] is subprocess.PIPE else None,
            stderr=b"private" if kwargs["stderr"] is subprocess.PIPE else None,
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    query = build_docker_query_invocation(
        argv=("docker", "compose", "version", "--short"),
        docker_control={"PATH": "/usr/bin"},
        cwd=tmp_path,
        timeout_seconds=2.0,
    )
    control = build_docker_control_invocation(
        argv=("docker", "rm", "-f", "agentseek-up-8123"),
        docker_control={"PATH": "/usr/bin"},
        cwd=tmp_path,
    )

    query_result = SubprocessTransport()(query)
    control_result = SubprocessTransport()(control)

    assert query_result.stdout == b"captured"
    assert query_result.stderr == b"private"
    assert control_result.stdout == b""
    assert control_result.stderr == b""
    assert calls[0][1]["stdout"] is subprocess.PIPE
    assert calls[0][1]["stderr"] is subprocess.PIPE
    assert calls[0][1]["timeout"] == 2.0
    assert calls[1][1]["stdout"] is None
    assert calls[1][1]["stderr"] is None
    assert calls[1][1]["timeout"] is None
    for _, kwargs in calls:
        assert kwargs["shell"] is False
        assert kwargs["check"] is False
        assert kwargs["env"] == {"PATH": "/usr/bin"}
        assert kwargs["cwd"] == tmp_path


def test_legacy_runner_adapter_rejects_queries_and_stdin(tmp_path: Path) -> None:
    calls: list[tuple[list[str], dict[str, str], str | None]] = []

    def runner(
        command: list[str], *, env: dict[str, str], cwd: str | None = None
    ) -> int:
        calls.append((command, env, cwd))
        return 7

    adapter = LegacyRunnerAdapter(runner)
    control = build_docker_control_invocation(
        argv=("docker", "rm", "-f", "test"), docker_control={}, cwd=tmp_path
    )
    query = build_docker_query_invocation(
        argv=("docker", "image", "inspect", "test"),
        docker_control={},
        cwd=tmp_path,
    )
    stdin_invocation = ProcessInvocation(
        argv=("docker", "build", "-"),
        environment={},
        cwd=tmp_path,
        stdin_bytes=b"archive",
    )

    assert adapter(control).returncode == 7
    assert calls == [(["docker", "rm", "-f", "test"], {}, str(tmp_path))]
    assert "archive" not in repr(stdin_invocation)
    with pytest.raises(DockerRuntimeError, match="control queries"):
        adapter(query)
    with pytest.raises(DockerRuntimeError, match="standard input"):
        adapter(stdin_invocation)


def test_narrow_version_and_availability_parsers_are_value_free() -> None:
    assert (
        parse_compose_version_result(ProcessResult(returncode=0, stdout=b"v2.27.1\n"))
        == "2.27.1"
    )
    assert (
        parse_buildx_version_result(
            ProcessResult(
                returncode=0,
                stdout=b"github.com/docker/buildx v0.14.0 171fcbe\n",
            )
        )
        == "0.14.0"
    )
    require_buildx_available(ProcessResult(returncode=0))

    canary = "private-version-output"
    for parser, result in (
        (
            parse_compose_version_result,
            ProcessResult(returncode=0, stdout=f"2.27.1\n{canary}\n".encode()),
        ),
        (
            parse_buildx_version_result,
            ProcessResult(returncode=1, stderr=canary.encode()),
        ),
        (require_buildx_available, ProcessResult(returncode=1, stderr=canary.encode())),
    ):
        with pytest.raises(DockerRuntimeError) as exc_info:
            parser(result)
        assert canary not in str(exc_info.value)


def test_image_compatibility_query_never_requests_hostile_baked_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    baked_environment_canary = "hostile-baked-env-canary"
    hostile_config = {
        "Env": [f"PRIVATE_TOKEN={baked_environment_canary}"],
        "Labels": {"org.agentseek.environment-contract": "preloaded-v1"},
        "Entrypoint": ["agentseek-api"],
        "Cmd": ["serve"],
    }

    def fake_run(
        argv: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[bytes]:
        assert argv == [
            "docker",
            "image",
            "inspect",
            "--format",
            IMAGE_COMPATIBILITY_FORMAT,
            "hostile:test",
        ]
        assert ".Config.Env" not in " ".join(argv)
        selected = [
            hostile_config["Labels"],
            hostile_config["Entrypoint"],
            hostile_config["Cmd"],
        ]
        return subprocess.CompletedProcess(
            argv, 0, stdout=(json.dumps(selected) + "\n").encode(), stderr=b""
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    invocation = build_docker_query_invocation(
        argv=(
            "docker",
            "image",
            "inspect",
            "--format",
            IMAGE_COMPATIBILITY_FORMAT,
            "hostile:test",
        ),
        docker_control={},
        cwd=tmp_path,
    )

    result = SubprocessTransport()(invocation)
    config = parse_image_compatibility_result(result)

    assert config.labels["org.agentseek.environment-contract"] == "preloaded-v1"
    assert config.entrypoint == ("agentseek-api",)
    assert config.command == ("serve",)
    assert baked_environment_canary.encode() not in result.stdout
    assert baked_environment_canary not in repr(config)


@pytest.mark.parametrize(
    "result",
    [
        ProcessResult(returncode=1, stderr=b"private-error"),
        ProcessResult(returncode=0, stdout=b"not-json"),
        ProcessResult(returncode=0, stdout=b"[{},[]]"),
        ProcessResult(returncode=0, stdout=b'[{"label": 1},[],[]]'),
        ProcessResult(returncode=0, stdout=b"[{},[],[]]\n[{},[],[]]"),
    ],
)
def test_image_compatibility_parser_rejects_nonexact_private_output(
    result: ProcessResult,
) -> None:
    with pytest.raises(DockerRuntimeError) as exc_info:
        parse_image_compatibility_result(result)

    message = str(exc_info.value)
    assert message == "Docker image compatibility query returned an invalid result."
    assert "private-error" not in message
    assert "not-json" not in message
