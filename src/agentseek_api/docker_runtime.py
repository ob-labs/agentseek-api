"""Value-redacted, shell-free process transport for Docker boundaries."""

from __future__ import annotations

import json
import math
import re
import subprocess
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Protocol

from agentseek_api.container_policy import select_compose_payload
from agentseek_api.environment import ContainerPolicyError

DEFAULT_CONTROL_QUERY_TIMEOUT_SECONDS = 10.0
MINIMUM_COMPOSE_VERSION = (2, 24, 0)
IMAGE_COMPATIBILITY_FORMAT = (
    "[{{json .Config.Labels}},{{json .Config.Entrypoint}},{{json .Config.Cmd}}]"
)

_SEMANTIC_VERSION = r"[0-9]+\.[0-9]+\.[0-9]+(?:[-+][0-9A-Za-z.-]+)?"
_COMPOSE_VERSION = re.compile(rf"^v?(?P<version>{_SEMANTIC_VERSION})$")
_BUILDX_VERSION = re.compile(
    rf"^github\.com/docker/buildx v?(?P<version>{_SEMANTIC_VERSION})(?:\s+\S.*)?$"
)
_ENVIRONMENT_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class DockerRuntimeError(RuntimeError):
    """A value-free Docker transport failure."""


@dataclass(frozen=True, kw_only=True)
class ProcessInvocation:
    argv: tuple[str, ...]
    environment: Mapping[str, str] = field(repr=False)
    cwd: Path
    stdin_bytes: bytes | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "argv", tuple(self.argv))
        object.__setattr__(
            self, "environment", MappingProxyType(dict(self.environment))
        )
        object.__setattr__(self, "cwd", Path(self.cwd))


@dataclass(frozen=True, kw_only=True)
class ControlQueryInvocation(ProcessInvocation):
    """A bounded query whose captured output never falls through to the terminal."""

    timeout_seconds: float = field(
        default=DEFAULT_CONTROL_QUERY_TIMEOUT_SECONDS, repr=False
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ContainerPolicyError(
                "Docker control query timeout must be a positive finite number."
            )


@dataclass(frozen=True, kw_only=True)
class DockerRunInvocation(ProcessInvocation):
    application_names: frozenset[str]

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(self, "application_names", frozenset(self.application_names))


@dataclass(frozen=True, kw_only=True)
class ProcessResult:
    returncode: int
    stdout: bytes = field(default=b"", repr=False)
    stderr: bytes = field(default=b"", repr=False)


@dataclass(frozen=True, kw_only=True)
class DockerImageConfig:
    labels: Mapping[str, str] = field(repr=False)
    entrypoint: tuple[str, ...] | str | None = field(repr=False)
    command: tuple[str, ...] | str | None = field(repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "labels", MappingProxyType(dict(self.labels)))
        if isinstance(self.entrypoint, list):
            object.__setattr__(self, "entrypoint", tuple(self.entrypoint))
        if isinstance(self.command, list):
            object.__setattr__(self, "command", tuple(self.command))


class ProcessTransport(Protocol):
    def __call__(self, invocation: ProcessInvocation) -> ProcessResult: ...


LegacyRunner = Callable[..., int]


def _validate_argv(argv: tuple[str, ...]) -> None:
    if not argv:
        raise ContainerPolicyError("Docker invocation argv must not be empty.")
    if any("\0" in item for item in argv):
        raise ContainerPolicyError("Docker invocation argv contains NUL.")


def _validated_environment(
    environment: Mapping[str, str],
    *,
    label: str = "Docker invocation environment",
    platform: str = sys.platform,
) -> dict[str, str]:
    validated: dict[str, str] = {}
    windows_names: set[str] = set()
    for name, value in environment.items():
        if "\0" in name:
            raise ContainerPolicyError(
                "Docker invocation environment name contains NUL."
            )
        if "\0" in value:
            raise ContainerPolicyError(
                f"Docker invocation environment value for '{name}' contains NUL."
            )
        if platform == "win32":
            logical_name = name.casefold()
            if logical_name in windows_names:
                raise ContainerPolicyError(
                    f"{label} contains duplicate Windows environment names."
                )
            windows_names.add(logical_name)
        validated[name] = value
    return validated


def build_docker_run_invocation(
    *,
    base_argv: tuple[str, ...],
    image: str,
    docker_control: Mapping[str, str],
    application_payload: Mapping[str, str],
    container_argv: tuple[str, ...],
    cwd: Path,
    platform: str = sys.platform,
) -> DockerRunInvocation:
    control = _validated_environment(
        docker_control,
        label="Docker control environment",
        platform=platform,
    )
    application = _validated_environment(
        application_payload,
        label="Application payload",
        platform=platform,
    )
    if platform == "win32":
        control_names = {name.casefold() for name in control}
        collisions = {name for name in application if name.casefold() in control_names}
    else:
        collisions = control.keys() & application.keys()
    if collisions:
        names = ", ".join(sorted(collisions))
        raise ContainerPolicyError(
            f"Application payload collides with Docker control keys: {names}"
        )
    argv = [*base_argv]
    for name in sorted(application):
        argv.extend(("-e", name))
    argv.extend((image, *container_argv))
    immutable_argv = tuple(argv)
    _validate_argv(immutable_argv)
    return DockerRunInvocation(
        argv=immutable_argv,
        environment=MappingProxyType({**control, **application}),
        cwd=cwd,
        stdin_bytes=None,
        application_names=frozenset(application),
    )


def build_docker_control_invocation(
    *,
    argv: tuple[str, ...],
    docker_control: Mapping[str, str],
    cwd: Path,
    stdin_bytes: bytes | None = None,
) -> ProcessInvocation:
    _validate_argv(argv)
    return ProcessInvocation(
        argv=argv,
        environment=_validated_environment(docker_control),
        cwd=cwd,
        stdin_bytes=stdin_bytes,
    )


def build_docker_query_invocation(
    *,
    argv: tuple[str, ...],
    docker_control: Mapping[str, str],
    cwd: Path,
    timeout_seconds: float = DEFAULT_CONTROL_QUERY_TIMEOUT_SECONDS,
) -> ControlQueryInvocation:
    _validate_argv(argv)
    return ControlQueryInvocation(
        argv=argv,
        environment=_validated_environment(docker_control),
        cwd=cwd,
        stdin_bytes=None,
        timeout_seconds=timeout_seconds,
    )


def validate_environment_name(name: str) -> None:
    if not _ENVIRONMENT_NAME.fullmatch(name):
        raise ContainerPolicyError("Compose environment name is invalid.")


def encode_compose_environment(values: Mapping[str, str]) -> str:
    """Encode values with the one supported literal Compose dotenv grammar."""

    lines: list[str] = []
    for name, value in sorted(values.items()):
        validate_environment_name(name)
        if "\x00" in value or any(
            ord(character) < 0x20 and character not in "\n\r\t" for character in value
        ):
            raise DockerRuntimeError(
                f"Compose value for {name} contains an unsupported control."
            )
        literal = (
            value.replace("\\", "\\\\")
            .replace('"', '\\"')
            .replace("$", "$$")
            .replace("\n", "\\n")
            .replace("\r", "\\r")
            .replace("\t", "\\t")
        )
        lines.append(f'{name}="{literal}"')
    return "\n".join(lines) + "\n"


def build_compose_invocation(
    *,
    compose_file: Path,
    env_file: Path,
    docker_control: Mapping[str, str],
    application_payload: Mapping[str, str],
    selected_names: frozenset[str],
    cwd: Path,
    recreate: bool = False,
    platform: str = sys.platform,
) -> ProcessInvocation:
    """Build an explicit-env-file Compose invocation with no carrier values."""

    select_compose_payload(
        application_payload=application_payload,
        selected_names=selected_names,
        docker_control=docker_control,
        platform=platform,
    )
    argv = [
        "docker",
        "compose",
        "--env-file",
        str(env_file),
        "-f",
        str(compose_file),
        "up",
        "-d",
    ]
    if recreate:
        argv.append("--force-recreate")
    return build_docker_control_invocation(
        argv=tuple(argv), docker_control=docker_control, cwd=cwd
    )


def require_supported_compose(
    *,
    transport: ProcessTransport,
    docker_control: Mapping[str, str],
    cwd: Path,
) -> tuple[int, int, int]:
    """Reject old or unavailable Compose before private artifacts are created."""

    query = build_docker_query_invocation(
        argv=("docker", "compose", "version", "--short"),
        docker_control=docker_control,
        cwd=cwd,
    )
    version_text = parse_compose_version_result(transport(query))
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)", version_text)
    if match is None:
        raise DockerRuntimeError(
            "Docker Compose version query returned an invalid result."
        )
    version = tuple(int(component) for component in match.groups())
    prerelease = "-" in version_text
    if version < MINIMUM_COMPOSE_VERSION or (
        version == MINIMUM_COMPOSE_VERSION and prerelease
    ):
        required = ".".join(str(component) for component in MINIMUM_COMPOSE_VERSION)
        raise DockerRuntimeError(f"Docker Compose {required} or newer is required.")
    return version


@dataclass(frozen=True)
class SubprocessTransport:
    def __call__(self, invocation: ProcessInvocation) -> ProcessResult:
        is_query = isinstance(invocation, ControlQueryInvocation)
        try:
            completed = subprocess.run(
                list(invocation.argv),
                env=dict(invocation.environment),
                cwd=invocation.cwd,
                input=invocation.stdin_bytes,
                stdout=subprocess.PIPE if is_query else None,
                stderr=subprocess.PIPE if is_query else None,
                shell=False,
                check=False,
                timeout=invocation.timeout_seconds if is_query else None,
            )
        except subprocess.TimeoutExpired as exc:
            raise DockerRuntimeError("Docker control query timed out.") from exc
        except OSError as exc:
            raise DockerRuntimeError("Docker process could not be started.") from exc
        return ProcessResult(
            returncode=completed.returncode,
            stdout=(completed.stdout or b"") if is_query else b"",
            stderr=(completed.stderr or b"") if is_query else b"",
        )


@dataclass(frozen=True)
class LegacyRunnerAdapter:
    runner: LegacyRunner = field(repr=False)

    def __call__(self, invocation: ProcessInvocation) -> ProcessResult:
        if isinstance(invocation, ControlQueryInvocation):
            raise DockerRuntimeError(
                "Legacy runners cannot represent Docker control queries safely."
            )
        if invocation.stdin_bytes is not None:
            raise DockerRuntimeError(
                "Legacy runners cannot represent Docker standard input safely."
            )
        return ProcessResult(
            returncode=self.runner(
                list(invocation.argv),
                env=dict(invocation.environment),
                cwd=str(invocation.cwd),
            )
        )


def _one_private_output_line(result: ProcessResult, *, error_message: str) -> str:
    if result.returncode != 0:
        raise DockerRuntimeError(error_message)
    try:
        text = result.stdout.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise DockerRuntimeError(error_message) from exc
    lines = text.splitlines()
    if len(lines) != 1 or not lines[0] or lines[0] != lines[0].strip():
        raise DockerRuntimeError(error_message)
    return lines[0]


def parse_compose_version_result(result: ProcessResult) -> str:
    error_message = "Docker Compose version query returned an invalid result."
    line = _one_private_output_line(result, error_message=error_message)
    match = _COMPOSE_VERSION.fullmatch(line)
    if match is None:
        raise DockerRuntimeError(error_message)
    return match.group("version")


def parse_buildx_version_result(result: ProcessResult) -> str:
    error_message = "Docker Buildx version query returned an invalid result."
    line = _one_private_output_line(result, error_message=error_message)
    match = _BUILDX_VERSION.fullmatch(line)
    if match is None:
        raise DockerRuntimeError(error_message)
    return match.group("version")


def require_buildx_available(result: ProcessResult) -> None:
    if result.returncode != 0:
        raise DockerRuntimeError("Docker Buildx builder is unavailable.")


def _parse_string_or_argv(value: object) -> tuple[str, ...] | str | None:
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return tuple(value)
    raise TypeError


def parse_image_compatibility_result(result: ProcessResult) -> DockerImageConfig:
    error_message = "Docker image compatibility query returned an invalid result."
    line = _one_private_output_line(result, error_message=error_message)
    try:
        payload = json.loads(line)
        if not isinstance(payload, list) or len(payload) != 3:
            raise TypeError
        raw_labels, raw_entrypoint, raw_command = payload
        if raw_labels is None:
            labels: dict[str, str] = {}
        elif isinstance(raw_labels, dict) and all(
            isinstance(name, str) and isinstance(value, str)
            for name, value in raw_labels.items()
        ):
            labels = dict(raw_labels)
        else:
            raise TypeError
        entrypoint = _parse_string_or_argv(raw_entrypoint)
        command = _parse_string_or_argv(raw_command)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise DockerRuntimeError(error_message) from exc
    return DockerImageConfig(
        labels=labels,
        entrypoint=entrypoint,
        command=command,
    )


__all__ = [
    "ControlQueryInvocation",
    "DEFAULT_CONTROL_QUERY_TIMEOUT_SECONDS",
    "DockerRunInvocation",
    "DockerImageConfig",
    "DockerRuntimeError",
    "LegacyRunnerAdapter",
    "IMAGE_COMPATIBILITY_FORMAT",
    "MINIMUM_COMPOSE_VERSION",
    "ProcessInvocation",
    "ProcessResult",
    "ProcessTransport",
    "SubprocessTransport",
    "build_docker_control_invocation",
    "build_compose_invocation",
    "build_docker_query_invocation",
    "build_docker_run_invocation",
    "encode_compose_environment",
    "parse_buildx_version_result",
    "parse_compose_version_result",
    "parse_image_compatibility_result",
    "require_buildx_available",
    "require_supported_compose",
    "validate_environment_name",
]
