from __future__ import annotations

import hashlib
import json
import os
import subprocess
import uuid
import zipfile
from dataclasses import replace
from pathlib import Path

import pytest

from agentseek_api.docker_runtime import (
    MINIMUM_BUILDX_VERSION,
    MINIMUM_COMPOSE_VERSION,
    BuildImageInvocation,
    ControlQueryInvocation,
    IMAGE_COMPATIBILITY_FORMAT,
    DockerRuntimeError,
    LegacyRunnerAdapter,
    ProcessInvocation,
    ProcessResult,
    SubprocessTransport,
    build_image_invocation,
    build_docker_control_invocation,
    build_compose_invocation,
    build_docker_query_invocation,
    build_docker_run_invocation,
    encode_compose_environment,
    parse_buildx_version_result,
    parse_compose_version_result,
    parse_image_compatibility_result,
    require_buildx_available,
    require_supported_buildx,
    require_supported_compose,
)
from agentseek_api.container_build import (
    PUBLISHED_RUNTIME_ARTIFACT,
    candidate_runtime_artifact,
    materialize_build_bundle,
    plan_container_image,
    render_build_dockerfile,
)
from agentseek_api.environment import ContainerPolicyError
from tests.container_plan_helpers import (
    build_plan_fixture,
    decode_with_supported_compose,
    docker_compose_available,
    docker_daemon_available,
)


def test_build_image_invocation_uses_stdin_buildx_and_secret(tmp_path: Path) -> None:
    plan = build_plan_fixture(tmp_path)
    bundle = materialize_build_bundle(
        plan,
        dockerfile_bytes=render_build_dockerfile(plan),
        output_root=tmp_path / "bundle",
    )
    invocation = build_image_invocation(
        bundle,
        plan=plan,
        docker_control={"PATH": "/usr/bin", "DOCKER_BUILDKIT": "0"},
        tag="agentseek:test",
        platform="linux/amd64",
        pull=True,
    )
    assert isinstance(invocation, BuildImageInvocation)
    assert invocation.argv == (
        "docker",
        "buildx",
        "build",
        "--load",
        "--file",
        "Dockerfile",
        "--platform",
        "linux/amd64",
        "--pull",
        "--tag",
        "agentseek:test",
        "--secret",
        f"id=pip_config,src={plan.pip_config_file}",
        "-",
    )
    assert invocation.environment == {"PATH": "/usr/bin"}
    assert invocation.stdin_bytes == bundle.archive_bytes()
    assert b"packages.example.invalid" not in invocation.stdin_bytes
    assert "packages.example.invalid" not in repr(invocation)


def test_dockerfile_and_build_omit_pip_secret_when_not_configured(
    tmp_path: Path,
) -> None:
    plan = replace(build_plan_fixture(tmp_path), pip_config_file=None)
    dockerfile = render_build_dockerfile(plan)
    bundle = materialize_build_bundle(
        plan,
        dockerfile_bytes=dockerfile,
        output_root=tmp_path / "bundle-no-pip-secret",
    )
    invocation = build_image_invocation(bundle, plan=plan, docker_control={})
    assert b"type=secret,id=pip_config" not in dockerfile
    assert "--secret" not in invocation.argv
    assert "None" not in invocation.argv


@pytest.mark.parametrize("mismatch", ["pip-secret", "runtime", "plan"])
def test_build_image_invocation_rejects_bundle_plan_mismatch(
    tmp_path: Path, mismatch: str
) -> None:
    plan = build_plan_fixture(tmp_path)
    bundle = materialize_build_bundle(
        plan,
        dockerfile_bytes=render_build_dockerfile(plan),
        output_root=tmp_path / "bundle",
    )
    if mismatch == "pip-secret":
        supplied_plan = replace(plan, pip_config_file=None, pip_config_identity=None)
    elif mismatch == "runtime":
        wheel = plan.project_root / "candidate.whl"
        with zipfile.ZipFile(wheel, "w") as archive:
            archive.writestr(
                "agentseek_api-0.3.0.dist-info/METADATA",
                "Metadata-Version: 2.1\nName: agentseek-api\nVersion: 0.3.0\n",
            )
        artifact = candidate_runtime_artifact(
            wheel, hashlib.sha256(wheel.read_bytes()).hexdigest()
        )
        assert artifact != PUBLISHED_RUNTIME_ARTIFACT
        supplied_plan = replace(plan, runtime_artifact=artifact)
    else:
        supplied_plan = replace(plan, base_image="python:3.13-alpine")

    with pytest.raises(DockerRuntimeError, match="bundle does not match.*plan"):
        build_image_invocation(bundle, plan=supplied_plan, docker_control={})


def test_build_image_invocation_requires_stdin_bytes(tmp_path: Path) -> None:
    with pytest.raises(TypeError):
        BuildImageInvocation(argv=("docker",), environment={}, cwd=tmp_path)  # type: ignore[call-arg]


def test_require_supported_buildx_uses_two_bounded_queries(tmp_path: Path) -> None:
    plan = build_plan_fixture(tmp_path)
    calls: list[ProcessInvocation] = []

    def transport(invocation: ProcessInvocation) -> ProcessResult:
        calls.append(invocation)
        if invocation.argv == ("docker", "buildx", "version"):
            return ProcessResult(
                returncode=0,
                stdout=b"github.com/docker/buildx v0.12.0 deadbeef\n",
            )
        return ProcessResult(returncode=0)

    assert MINIMUM_BUILDX_VERSION == (0, 12, 0)
    assert require_supported_buildx(
        transport=transport,
        docker_control={"PATH": "/usr/bin"},
        cwd=tmp_path,
        plan=plan,
    ) == (0, 12, 0)
    assert [call.argv for call in calls] == [
        ("docker", "buildx", "version"),
        ("docker", "buildx", "inspect"),
    ]
    assert all(isinstance(call, ControlQueryInvocation) for call in calls)
    assert all("--bootstrap" not in call.argv for call in calls)


@pytest.mark.parametrize(
    ("version_result", "inspect_result", "match"),
    [
        (ProcessResult(returncode=1), ProcessResult(returncode=0), "version query"),
        (
            ProcessResult(
                returncode=0, stdout=b"github.com/docker/buildx v0.11.2 deadbeef\n"
            ),
            ProcessResult(returncode=0),
            "0.12.0 or newer",
        ),
        (
            ProcessResult(
                returncode=0, stdout=b"github.com/docker/buildx v0.12.0-rc.1 deadbeef\n"
            ),
            ProcessResult(returncode=0),
            "0.12.0 or newer",
        ),
        (
            ProcessResult(
                returncode=0, stdout=b"github.com/docker/buildx v0.14.0 deadbeef\n"
            ),
            ProcessResult(returncode=1),
            "builder is unavailable",
        ),
    ],
)
def test_require_supported_buildx_fails_before_build(
    tmp_path: Path,
    version_result: ProcessResult,
    inspect_result: ProcessResult,
    match: str,
) -> None:
    calls: list[ProcessInvocation] = []

    def transport(invocation: ProcessInvocation) -> ProcessResult:
        calls.append(invocation)
        return version_result if len(calls) == 1 else inspect_result

    with pytest.raises(DockerRuntimeError, match=match):
        require_supported_buildx(transport=transport, docker_control={}, cwd=tmp_path)
    assert all(call.argv[:3] != ("docker", "buildx", "build") for call in calls)


def test_buildx_without_secret_support_fails_bounded_and_value_free(
    tmp_path: Path,
) -> None:
    plan = build_plan_fixture(tmp_path)
    assert plan.pip_config_file is not None
    canary = "private-index-password-canary"
    plan.pip_config_file.write_text(canary, encoding="utf-8")
    calls: list[ProcessInvocation] = []

    def transport(invocation: ProcessInvocation) -> ProcessResult:
        calls.append(invocation)
        return ProcessResult(
            returncode=0,
            stdout=b"github.com/docker/buildx v0.11.2 deadbeef\n",
        )

    with pytest.raises(DockerRuntimeError, match="secret") as caught:
        require_supported_buildx(
            transport=transport,
            docker_control={},
            cwd=tmp_path,
            plan=plan,
        )

    assert len(calls) == 1
    assert isinstance(calls[0], ControlQueryInvocation)
    assert canary not in str(caught.value)
    assert canary not in repr(calls)


def test_pip_config_swap_before_build_invocation_is_rejected(tmp_path: Path) -> None:
    plan = build_plan_fixture(tmp_path)
    bundle = materialize_build_bundle(
        plan,
        dockerfile_bytes=render_build_dockerfile(plan),
        output_root=tmp_path / "bundle",
    )
    assert plan.pip_config_file is not None
    replacement = tmp_path / "replacement.conf"
    replacement.write_text("password=swapped\n", encoding="utf-8")
    replacement.replace(plan.pip_config_file)
    with pytest.raises(DockerRuntimeError, match="pip config identity changed"):
        build_image_invocation(bundle, plan=plan, docker_control={})


def test_pip_config_swap_during_buildx_probes_is_rejected(tmp_path: Path) -> None:
    plan = build_plan_fixture(tmp_path)
    assert plan.pip_config_file is not None
    calls: list[ProcessInvocation] = []

    def transport(invocation: ProcessInvocation) -> ProcessResult:
        calls.append(invocation)
        if len(calls) == 1:
            replacement = tmp_path / "replacement.conf"
            replacement.write_text("password=swapped\n", encoding="utf-8")
            replacement.replace(plan.pip_config_file)
            return ProcessResult(
                returncode=0,
                stdout=b"github.com/docker/buildx v0.14.0 deadbeef\n",
            )
        return ProcessResult(returncode=0)

    with pytest.raises(DockerRuntimeError, match="pip config identity changed"):
        require_supported_buildx(
            transport=transport,
            docker_control={},
            cwd=tmp_path,
            plan=plan,
        )
    assert [call.argv for call in calls] == [
        ("docker", "buildx", "version"),
        ("docker", "buildx", "inspect"),
    ]


@pytest.mark.docker
@pytest.mark.parametrize("valid_secret", [True, False])
def test_buildx_secret_mount_consumes_canary_without_disclosure(
    tmp_path: Path, valid_secret: bool
) -> None:
    if not docker_daemon_available(cwd=tmp_path):
        pytest.skip("requires a Docker daemon")
    plan = build_plan_fixture(tmp_path)
    assert plan.pip_config_file is not None
    expected_canary = f"agentseek-buildx-secret-{uuid.uuid4().hex}"
    actual_secret = (
        expected_canary
        if valid_secret
        else f"agentseek-buildx-wrong-secret-{uuid.uuid4().hex}"
    )
    plan.pip_config_file.write_text(actual_secret, encoding="utf-8")
    plan = plan_container_image(config_path=plan.config_path)
    digest = hashlib.sha256(expected_canary.encode()).hexdigest()
    dockerfile = (
        "# syntax=docker/dockerfile:1.7\n"
        "FROM python:3.12-alpine\n"
        "RUN --mount=type=secret,id=pip_config,target=/run/secrets/pip_config "
        + json.dumps(
            [
                "python",
                "-c",
                (
                    "import hashlib,pathlib,sys;"
                    "data=pathlib.Path('/run/secrets/pip_config').read_bytes();"
                    f"sys.exit(1) if hashlib.sha256(data).hexdigest()!='{digest}' else None"
                ),
            ]
        )
        + "\n"
    ).encode()
    bundle = materialize_build_bundle(
        plan,
        dockerfile_bytes=dockerfile,
        output_root=tmp_path / f"secret-smoke-bundle-{valid_secret}",
    )
    allowed = {
        "PATH",
        "HOME",
        "USERPROFILE",
        "SYSTEMROOT",
        "DOCKER_HOST",
        "DOCKER_CONTEXT",
        "DOCKER_CONFIG",
        "DOCKER_TLS_VERIFY",
        "DOCKER_CERT_PATH",
    }
    docker_control = {
        name: value for name, value in os.environ.items() if name in allowed
    }
    transport = SubprocessTransport()
    require_supported_buildx(
        transport=transport,
        docker_control=docker_control,
        cwd=tmp_path,
        plan=plan,
    )
    tag = f"agentseek-buildx-secret-smoke:{uuid.uuid4().hex}"
    invocation = build_image_invocation(
        bundle,
        plan=plan,
        docker_control=docker_control,
        tag=tag,
    )

    try:
        completed = subprocess.run(
            list(invocation.argv),
            cwd=invocation.cwd,
            env=dict(invocation.environment),
            input=invocation.stdin_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            shell=False,
            timeout=240,
        )
        combined = completed.stdout + completed.stderr
        assert (completed.returncode == 0) is valid_secret, combined.decode(
            errors="replace"
        )
        history = subprocess.run(
            ["docker", "image", "history", "--no-trunc", tag],
            cwd=tmp_path,
            env=docker_control,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            shell=False,
            timeout=30,
        )
        assert (history.returncode == 0) is valid_secret
        observed = combined + history.stdout + history.stderr + invocation.stdin_bytes
        for secret in (expected_canary, actual_secret):
            assert secret.encode() not in observed
            assert all(secret not in argument for argument in invocation.argv)
            assert secret not in repr(invocation)
    finally:
        subprocess.run(
            ["docker", "image", "rm", "--force", tag],
            cwd=tmp_path,
            env=docker_control,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            shell=False,
            timeout=30,
        )


SPECIAL_VALUES = {
    "DOLLAR": "${DOCKER_HOST}",
    "LONE_DOLLAR": "$",
    "HASH": "value # literal",
    "EQUALS": "a=b",
    "QUOTES": "single' double\" slash\\",
    "TRAILING_SLASH": "trailing\\",
    "SLASH_QUOTE": 'slash-before-\\"quote',
    "MULTILINE": "line one\nline two\rline three\tend",
    "UNICODE": "雪",
    "EMPTY": "",
}


def test_compose_encoder_uses_one_literal_double_quoted_codec() -> None:
    encoded = encode_compose_environment(SPECIAL_VALUES)

    assert 'DOLLAR="$${DOCKER_HOST}"' in encoded
    assert 'LONE_DOLLAR="$$"' in encoded
    assert 'HASH="value # literal"' in encoded
    assert 'EQUALS="a=b"' in encoded
    assert 'TRAILING_SLASH="trailing\\\\"' in encoded
    assert 'SLASH_QUOTE="slash-before-\\\\\\"quote"' in encoded
    assert 'MULTILINE="line one\\nline two\\rline three\\tend"' in encoded
    assert encoded.endswith("\n")
    assert "line one\nline two" not in encoded


@pytest.mark.docker
@pytest.mark.parametrize(("name", "value"), SPECIAL_VALUES.items())
def test_compose_encoder_round_trips_config_oracle(
    name: str, value: str, tmp_path: Path
) -> None:
    if not docker_compose_available(cwd=tmp_path):
        pytest.skip("requires Docker Compose 2.24 or newer")
    decoded = decode_with_supported_compose(
        encode_compose_environment({name: value}), tmp_path=tmp_path
    )

    assert decoded.substitution[name] == value
    assert decoded.rendered[name] == value.replace("$", "$$")


@pytest.mark.docker
def test_compose_service_probe_command_references_environment_without_baking_value(
    tmp_path: Path,
) -> None:
    if not docker_compose_available(cwd=tmp_path):
        pytest.skip("requires Docker Compose 2.24 or newer")
    value = "must-not-be-baked-into-shell-source"

    decoded = decode_with_supported_compose(
        encode_compose_environment({"PROBE_VALUE": value}), tmp_path=tmp_path
    )

    command = " ".join(decoded.commands["PROBE_VALUE"])
    assert value not in command
    assert "${PROBE_VALUE}" in command


@pytest.mark.docker
def test_compose_encoder_round_trips_real_container_without_second_interpolation(
    tmp_path: Path,
) -> None:
    if not docker_compose_available(cwd=tmp_path) or not docker_daemon_available(
        cwd=tmp_path
    ):
        pytest.skip("requires a Docker daemon")

    decoded = decode_with_supported_compose(
        encode_compose_environment(SPECIAL_VALUES),
        tmp_path=tmp_path,
        run_service=True,
    )

    assert dict(decoded.substitution) == SPECIAL_VALUES
    assert dict(decoded.runtime) == SPECIAL_VALUES


@pytest.mark.parametrize("value", ["nul\0value", "control\x01value"])
def test_compose_encoder_rejects_unrepresentable_values_without_echoing_them(
    value: str,
) -> None:
    with pytest.raises(DockerRuntimeError) as exc_info:
        encode_compose_environment({"PRIVATE_TOKEN": value})

    assert "PRIVATE_TOKEN" in str(exc_info.value)
    assert value not in str(exc_info.value)


def test_compose_invocation_is_explicit_control_only_and_value_redacted(
    tmp_path: Path,
) -> None:
    env_file = tmp_path / "agentseek-compose-private"
    compose_file = tmp_path / "compose.yaml"

    invocation = build_compose_invocation(
        compose_file=compose_file,
        env_file=env_file,
        docker_control={"DOCKER_HOST": "unix:///private/docker.sock"},
        application_payload={"TOKEN": "compose-secret"},
        selected_names=frozenset({"TOKEN"}),
        cwd=tmp_path,
        recreate=True,
    )

    assert invocation.argv == (
        "docker",
        "compose",
        "--env-file",
        str(env_file),
        "-f",
        str(compose_file),
        "up",
        "-d",
        "--force-recreate",
    )
    assert dict(invocation.environment) == {
        "DOCKER_HOST": "unix:///private/docker.sock"
    }
    assert "compose-secret" not in repr(invocation)
    assert "compose-secret" not in " ".join(invocation.argv)


@pytest.mark.parametrize(
    ("application_payload", "selected_names", "docker_control", "message"),
    [
        ({}, frozenset({"MISSING"}), {}, "not present"),
        (
            {"DOCKER_HOST": "application"},
            frozenset({"DOCKER_HOST"}),
            {"DOCKER_HOST": "control"},
            "collides",
        ),
    ],
)
def test_compose_invocation_rejects_missing_names_and_control_collisions(
    tmp_path: Path,
    application_payload: dict[str, str],
    selected_names: frozenset[str],
    docker_control: dict[str, str],
    message: str,
) -> None:
    with pytest.raises(ContainerPolicyError, match=message):
        build_compose_invocation(
            compose_file=tmp_path / "compose.yaml",
            env_file=tmp_path / "private.env",
            docker_control=docker_control,
            application_payload=application_payload,
            selected_names=selected_names,
            cwd=tmp_path,
        )


def test_require_supported_compose_uses_bounded_control_only_query(
    tmp_path: Path,
) -> None:
    calls: list[ProcessInvocation] = []

    def transport(invocation: ProcessInvocation) -> ProcessResult:
        calls.append(invocation)
        return ProcessResult(returncode=0, stdout=b"v2.24.0\n")

    assert MINIMUM_COMPOSE_VERSION == (2, 24, 0)
    assert require_supported_compose(
        transport=transport,
        docker_control={"PATH": "/usr/bin"},
        cwd=tmp_path,
    ) == (2, 24, 0)
    assert len(calls) == 1
    query = calls[0]
    assert isinstance(query, ControlQueryInvocation)
    assert query.argv == ("docker", "compose", "version", "--short")
    assert dict(query.environment) == {"PATH": "/usr/bin"}
    assert query.timeout_seconds > 0


def test_require_supported_compose_rejects_old_version_value_free(
    tmp_path: Path,
) -> None:
    def transport(_invocation: ProcessInvocation) -> ProcessResult:
        return ProcessResult(returncode=0, stdout=b"2.23.3\n")

    with pytest.raises(DockerRuntimeError) as exc_info:
        require_supported_compose(
            transport=transport,
            docker_control={},
            cwd=tmp_path,
        )

    assert "2.24.0" in str(exc_info.value)
    assert "2.23.3" not in str(exc_info.value)


def test_require_supported_compose_rejects_minimum_prerelease(tmp_path: Path) -> None:
    def transport(_invocation: ProcessInvocation) -> ProcessResult:
        return ProcessResult(returncode=0, stdout=b"2.24.0-rc.1\n")

    with pytest.raises(DockerRuntimeError, match="2.24.0 or newer"):
        require_supported_compose(
            transport=transport,
            docker_control={},
            cwd=tmp_path,
        )


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


def test_docker_run_preserves_physical_newline_only_in_carrier(
    tmp_path: Path,
) -> None:
    value = "first line\nsecond line"

    invocation = build_docker_run_invocation(
        base_argv=("docker", "run", "--rm"),
        image="agentseek:test",
        docker_control={},
        application_payload={"MULTILINE": value},
        container_argv=(),
        cwd=tmp_path,
    )

    assert invocation.environment["MULTILINE"] == value
    assert value not in " ".join(invocation.argv)
    assert invocation.argv == (
        "docker",
        "run",
        "--rm",
        "-e",
        "MULTILINE",
        "agentseek:test",
    )


def test_windows_docker_run_rejects_casefolded_cross_map_collision(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ContainerPolicyError,
        match="Application payload collides with Docker control keys",
    ):
        build_docker_run_invocation(
            base_argv=("docker", "run"),
            image="agentseek:test",
            docker_control={"Path": "control"},
            application_payload={"PATH": "application"},
            container_argv=(),
            cwd=tmp_path,
            platform="win32",
        )


@pytest.mark.parametrize(
    ("docker_control", "application_payload", "message"),
    [
        (
            {"Path": "first", "PATH": "second"},
            {},
            "Docker control environment contains duplicate Windows environment names.",
        ),
        (
            {},
            {"Token": "first", "TOKEN": "second"},
            "Application payload contains duplicate Windows environment names.",
        ),
    ],
)
def test_windows_docker_run_rejects_casefolded_duplicates_within_map(
    tmp_path: Path,
    docker_control: dict[str, str],
    application_payload: dict[str, str],
    message: str,
) -> None:
    with pytest.raises(ContainerPolicyError) as exc_info:
        build_docker_run_invocation(
            base_argv=("docker", "run"),
            image="agentseek:test",
            docker_control=docker_control,
            application_payload=application_payload,
            container_argv=(),
            cwd=tmp_path,
            platform="win32",
        )

    assert str(exc_info.value) == message


def test_linux_docker_run_keeps_case_sensitive_name_semantics(tmp_path: Path) -> None:
    invocation = build_docker_run_invocation(
        base_argv=("docker", "run"),
        image="agentseek:test",
        docker_control={"Path": "control"},
        application_payload={"PATH": "application"},
        container_argv=(),
        cwd=tmp_path,
        platform="linux",
    )

    assert invocation.environment == {"Path": "control", "PATH": "application"}


def test_nul_collision_is_rejected_before_value_free_collision_diagnostic(
    tmp_path: Path,
) -> None:
    name = "PRIVATE\0NAME"

    with pytest.raises(ContainerPolicyError) as exc_info:
        build_docker_run_invocation(
            base_argv=("docker", "run"),
            image="agentseek:test",
            docker_control={name: "control"},
            application_payload={name: "application"},
            container_argv=(),
            cwd=tmp_path,
        )

    message = str(exc_info.value)
    assert message == "Docker invocation environment name contains NUL."
    assert "\0" not in message
    assert "PRIVATE" not in message


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
