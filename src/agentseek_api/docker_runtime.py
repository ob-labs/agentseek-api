"""Value-redacted, shell-free process transport for Docker boundaries."""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import stat
import subprocess
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Protocol

from agentseek_api.container_policy import select_compose_payload
from agentseek_api.container_build import (
    ContainerBuildBundle,
    ContainerBuildPlan,
    container_build_plan_fingerprint,
)
from agentseek_api.environment import ContainerPolicyError
from agentseek_api.constants import DEFAULT_API_PORT

DEFAULT_CONTROL_QUERY_TIMEOUT_SECONDS = 10.0
MINIMUM_COMPOSE_VERSION = (2, 24, 0)
MINIMUM_BUILDX_VERSION = (0, 12, 0)
IMAGE_COMPATIBILITY_FORMAT = (
    "[{{json .Config.Labels}},{{json .Config.Entrypoint}},{{json .Config.Cmd}}]"
)
PRELOADED_MANIFEST_PATH = "/opt/agentseek/manifest.v1.json"

_SEMANTIC_VERSION = r"[0-9]+\.[0-9]+\.[0-9]+(?:[-+][0-9A-Za-z.-]+)?"
_COMPOSE_VERSION = re.compile(rf"^v?(?P<version>{_SEMANTIC_VERSION})$")
_BUILDX_VERSION = re.compile(
    rf"^github\.com/docker/buildx v?(?P<version>{_SEMANTIC_VERSION})(?:\s+\S.*)?$"
)
_ENVIRONMENT_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class DockerRuntimeError(RuntimeError):
    """A value-free Docker transport failure."""


class ImageContractError(DockerRuntimeError):
    """A value-free incompatible custom-image failure."""


@dataclass(frozen=True, kw_only=True)
class ProcessInvocation:
    argv: tuple[str, ...]
    environment: Mapping[str, str] = field(repr=False)
    cwd: Path
    stdin_bytes: bytes | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "argv", tuple(self.argv))
        _validate_argv(self.argv)
        object.__setattr__(
            self, "environment", MappingProxyType(dict(self.environment))
        )
        object.__setattr__(self, "cwd", Path(self.cwd))


@dataclass(frozen=True, kw_only=True)
class BuildImageInvocation(ProcessInvocation):
    stdin_bytes: bytes = field(repr=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.stdin_bytes:
            raise ContainerPolicyError("Docker image build input must not be empty.")


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


@dataclass(frozen=True)
class PreloadedImageContract:
    manifest_path: str
    container_argv: tuple[str, ...]


class ProcessTransport(Protocol):
    def __call__(self, invocation: ProcessInvocation) -> ProcessResult: ...


LegacyRunner = Callable[..., int]


def _validate_argv(argv: tuple[str, ...]) -> None:
    if not argv:
        raise ContainerPolicyError("Docker invocation argv must not be empty.")
    if any("\0" in item for item in argv):
        raise ContainerPolicyError("Docker invocation argv contains NUL.")
    if not Path(argv[0]).is_absolute():
        raise ContainerPolicyError(
            "Docker invocation executable must be an absolute path."
        )


def resolve_docker_executable(
    docker_control: Mapping[str, str],
    *,
    platform: str = sys.platform,
) -> str:
    """Resolve Docker once from the selected control-plane search path."""

    if platform == "win32":
        selected = [
            value for name, value in docker_control.items() if name.casefold() == "path"
        ]
        path_value = selected[0] if len(selected) == 1 else None
    else:
        path_value = docker_control.get("PATH")
    if not path_value:
        raise DockerRuntimeError(
            "Docker executable could not be resolved from the selected PATH."
        )
    executable = shutil.which("docker", path=path_value)
    if executable is None:
        raise DockerRuntimeError(
            "Docker executable could not be resolved from the selected PATH."
        )
    if platform == "win32":
        from pathlib import PureWindowsPath

        is_absolute = PureWindowsPath(executable).is_absolute()
    else:
        is_absolute = Path(executable).is_absolute()
    if not is_absolute:
        raise DockerRuntimeError(
            "Docker executable could not be resolved from the selected PATH."
        )
    return executable


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


def _pip_config_identity(status: os.stat_result) -> tuple[int, int, int, int]:
    return (status.st_dev, status.st_ino, status.st_size, status.st_mtime_ns)


def validate_pip_config_identity(plan: ContainerBuildPlan) -> None:
    if plan.pip_config_file is None:
        return
    try:
        status = plan.pip_config_file.lstat()
    except OSError as exc:
        raise DockerRuntimeError(
            "The pip config identity changed before build."
        ) from exc
    if (
        stat.S_ISLNK(status.st_mode)
        or not stat.S_ISREG(status.st_mode)
        or plan.pip_config_identity is None
        or _pip_config_identity(status) != plan.pip_config_identity
    ):
        raise DockerRuntimeError("The pip config identity changed before build.")


def build_image_invocation(
    bundle: ContainerBuildBundle,
    *,
    plan: ContainerBuildPlan,
    docker_executable: str,
    docker_control: Mapping[str, str],
    tag: str | None = None,
    platform: str | None = None,
    pull: bool = False,
) -> BuildImageInvocation:
    """Freeze a Buildx invocation whose only build context is verified tar stdin."""

    if bundle.plan_fingerprint != container_build_plan_fingerprint(plan):
        raise DockerRuntimeError("The build bundle does not match the supplied plan.")
    archive = bundle.archive_bytes()
    validate_pip_config_identity(plan)
    argv = [docker_executable, "buildx", "build", "--load", "--file", "Dockerfile"]
    if platform:
        argv.extend(("--platform", platform))
    if pull:
        argv.append("--pull")
    if tag:
        argv.extend(("--tag", tag))
    if plan.pip_config_file is not None:
        source = str(plan.pip_config_file)
        argv.extend(("--secret", f"id=pip_config,src={source}"))
    argv.append("-")
    immutable_argv = tuple(argv)
    _validate_argv(immutable_argv)
    environment = {
        name: value
        for name, value in _validated_environment(docker_control).items()
        if name != "DOCKER_BUILDKIT"
    }
    return BuildImageInvocation(
        argv=immutable_argv,
        environment=environment,
        cwd=plan.invocation_cwd,
        stdin_bytes=archive,
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
    docker_executable: str,
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
        docker_executable,
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
    docker_executable: str,
    transport: ProcessTransport,
    docker_control: Mapping[str, str],
    cwd: Path,
) -> tuple[int, int, int]:
    """Reject old or unavailable Compose before private artifacts are created."""

    query = build_docker_query_invocation(
        argv=(docker_executable, "compose", "version", "--short"),
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


def require_supported_buildx(
    *,
    docker_executable: str,
    transport: ProcessTransport,
    docker_control: Mapping[str, str],
    cwd: Path,
    plan: ContainerBuildPlan | None = None,
) -> tuple[int, int, int]:
    """Require a usable Buildx plugin before any side-effecting build call."""

    version_query = build_docker_query_invocation(
        argv=(docker_executable, "buildx", "version"),
        docker_control=docker_control,
        cwd=cwd,
    )
    version_text = parse_buildx_version_result(transport(version_query))
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)", version_text)
    if match is None:  # pragma: no cover - parser already enforces this
        raise DockerRuntimeError(
            "Docker Buildx version query returned an invalid result."
        )
    version = tuple(int(component) for component in match.groups())
    prerelease = "-" in version_text
    if version < MINIMUM_BUILDX_VERSION or (
        version == MINIMUM_BUILDX_VERSION and prerelease
    ):
        required = ".".join(str(component) for component in MINIMUM_BUILDX_VERSION)
        raise DockerRuntimeError(
            f"Docker Buildx {required} or newer is required for BuildKit secrets."
        )
    inspect_query = build_docker_query_invocation(
        argv=(docker_executable, "buildx", "inspect"),
        docker_control=docker_control,
        cwd=cwd,
    )
    require_buildx_available(transport(inspect_query))
    if plan is not None:
        validate_pip_config_identity(plan)
    return version


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


_PRELOADED_V1_LABELS = {
    "org.agentseek.environment-contract": "preloaded-v1",
    "org.agentseek.runtime-manifest": PRELOADED_MANIFEST_PATH,
    "org.agentseek.runtime-distribution": "agentseek-api",
    "org.agentseek.runtime-version": "0.3.1",
}


def require_preloaded_v1(labels: Mapping[str, str]) -> str:
    """Require the complete immutable preloaded-v1 image label contract."""

    if any(labels.get(name) != value for name, value in _PRELOADED_V1_LABELS.items()):
        raise ImageContractError("The custom image must implement preloaded-v1.")
    return labels["org.agentseek.runtime-manifest"]


def inspect_image_contract(
    image: str,
    *,
    docker_executable: str,
    transport: ProcessTransport,
    docker_control: Mapping[str, str],
    cwd: Path,
) -> PreloadedImageContract:
    """Inspect only labels/entrypoint/cmd and require an explicit-mode carrier."""

    query = build_docker_query_invocation(
        argv=(
            docker_executable,
            "image",
            "inspect",
            "--format",
            IMAGE_COMPATIBILITY_FORMAT,
            image,
        ),
        docker_control=docker_control,
        cwd=cwd,
    )
    try:
        config = parse_image_compatibility_result(transport(query))
        manifest_path = require_preloaded_v1(config.labels)
    except DockerRuntimeError as exc:
        if isinstance(exc, ImageContractError):
            raise
        raise ImageContractError(
            "The custom image compatibility check failed."
        ) from exc
    serve = (
        "serve",
        "--environment-mode",
        "preloaded-v1",
        "--host",
        "0.0.0.0",
        "--port",
        str(DEFAULT_API_PORT),
    )
    if config.entrypoint in (None, ()):
        command = ("agentseek-api", *serve)
    elif config.entrypoint in (
        ("agentseek-api",),
        ("python", "-m", "agentseek_api.cli"),
    ):
        command = serve
    else:
        raise ImageContractError(
            "The custom image entrypoint cannot receive the preloaded-v1 command."
        )
    return PreloadedImageContract(
        manifest_path=manifest_path,
        container_argv=command,
    )


__all__ = [
    "BuildImageInvocation",
    "ControlQueryInvocation",
    "DEFAULT_CONTROL_QUERY_TIMEOUT_SECONDS",
    "DockerRunInvocation",
    "DockerImageConfig",
    "DockerRuntimeError",
    "ImageContractError",
    "LegacyRunnerAdapter",
    "IMAGE_COMPATIBILITY_FORMAT",
    "MINIMUM_BUILDX_VERSION",
    "MINIMUM_COMPOSE_VERSION",
    "PRELOADED_MANIFEST_PATH",
    "ProcessInvocation",
    "ProcessResult",
    "ProcessTransport",
    "PreloadedImageContract",
    "SubprocessTransport",
    "build_docker_control_invocation",
    "build_image_invocation",
    "build_compose_invocation",
    "build_docker_query_invocation",
    "build_docker_run_invocation",
    "encode_compose_environment",
    "parse_buildx_version_result",
    "parse_compose_version_result",
    "parse_image_compatibility_result",
    "inspect_image_contract",
    "require_preloaded_v1",
    "require_buildx_available",
    "require_supported_buildx",
    "require_supported_compose",
    "resolve_docker_executable",
    "validate_pip_config_identity",
    "validate_environment_name",
]
