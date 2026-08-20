"""Explicit, immutable selection policies for container process boundaries."""

from __future__ import annotations

import re
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from agentseek_api.environment import (
    ContainerPolicyError,
    EnvironmentPlan,
    EnvironmentTarget,
    NameScope,
    ResolutionPolicy,
    ResolvedEnvironment,
)


DOCKER_CONTROL_KEYS_BY_PLATFORM = MappingProxyType(
    {
        "common": frozenset(
            {
                "PATH",
                "TMP",
                "TEMP",
                "TMPDIR",
                "DOCKER_HOST",
                "DOCKER_CONTEXT",
                "DOCKER_CONFIG",
                "DOCKER_TLS_VERIFY",
                "DOCKER_CERT_PATH",
                "DOCKER_AUTH_CONFIG",
                "DOCKER_BUILDKIT",
                "DOCKER_DEFAULT_PLATFORM",
                "SSL_CERT_FILE",
                "SSL_CERT_DIR",
                "HTTP_PROXY",
                "HTTPS_PROXY",
                "ALL_PROXY",
                "NO_PROXY",
                "http_proxy",
                "https_proxy",
                "all_proxy",
                "no_proxy",
                "SSH_AUTH_SOCK",
            }
        ),
        "linux": frozenset({"HOME", "XDG_CONFIG_HOME", "XDG_RUNTIME_DIR"}),
        "darwin": frozenset({"HOME", "XDG_CONFIG_HOME"}),
        "win32": frozenset({"USERPROFILE", "SYSTEMROOT", "COMSPEC", "PATHEXT"}),
    }
)

APPLICATION_COMPATIBILITY_KEYS = frozenset(
    {
        "AGENTSEEK_API_BASE",
        "AGENTSEEK_API_KEY",
        "AGENTSEEK_GRAPHS",
        "AGENTSEEK_MODEL",
        "AGENTSEEK_MODEL_API_KEY",
        "AGENTSEEK_MODEL_PROVIDER",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_API_URL",
        "APP_NAME",
        "AUTH_MODULE_PATH",
        "BUB_API_BASE",
        "BUB_API_KEY",
        "BUB_MODEL",
        "BUB_OPENAI_API_BASE",
        "BUB_OPENAI_API_KEY",
        "DAYTONA_API_KEY",
        "DEEPAGENTS_MODEL",
        "EMBEDDING_API_KEY",
        "EMBEDDING_BASE_URL",
        "EXECUTOR_BACKEND",
        "GOOGLE_API_BASE",
        "GOOGLE_API_KEY",
        "LANGSMITH_API_KEY",
        "METADATA_DB_BACKEND",
        "METADATA_DB_URL",
        "OCEANBASE_DB_NAME",
        "OCEANBASE_HOST",
        "OCEANBASE_PASSWORD",
        "OCEANBASE_PORT",
        "OCEANBASE_USER",
        "OPENAI_API_BASE",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "OPENAI_MODEL",
        "PORT",
        "REDIS_RUN_PROCESSING_KEY",
        "REDIS_RUN_QUEUE_KEY",
        "REDIS_SCHEDULER_LOCK_KEY",
        "REDIS_SCHEDULER_LOCK_TTL_SECONDS",
        "REDIS_STREAM_MAXLEN",
        "REDIS_STREAM_TTL_SECONDS",
        "REDIS_URL",
        "REDIS_WORKER_LOCK_KEY",
        "REDIS_WORKER_LOCK_TTL_SECONDS",
        "REDIS_WORKER_POLL_TIMEOUT_SECONDS",
        "SCHEDULER_CLAIM_LIMIT",
        "SCHEDULER_POLL_INTERVAL_SECONDS",
        "SCHEDULER_STARTED_TICK_STALE_AFTER_SECONDS",
        "SEEKDB_EMBED",
        "SEEKDB_EMBED_DIR",
        "SEEKDB_URL",
        "SILICONFLOW_API_KEY",
        "STUDIO_AUTH_LOCAL_DEV",
        "TAVILY_API_KEY",
        "VLM_API_KEY",
        "VLM_BASE_URL",
        "WORKER_CONCURRENT_JOBS",
    }
)

HOST_RUNTIME_POLICY = ResolutionPolicy(
    target=EnvironmentTarget.HOST_RUNTIME,
    interpolation_scope=NameScope.ALL,
    assignment_scope=NameScope.ALL,
    export_scope=NameScope.ALL,
    malformed="error",
    unresolved="empty",
)
DOCKER_CONTROL_POLICY = ResolutionPolicy(
    target=EnvironmentTarget.DOCKER_CONTROL_PLANE,
    interpolation_scope=NameScope.DOCKER_CONTROL,
    assignment_scope=NameScope.DOCKER_CONTROL,
    export_scope=NameScope.DOCKER_CONTROL,
    malformed="error",
    unresolved="error",
)
APP_CONTAINER_POLICY = ResolutionPolicy(
    target=EnvironmentTarget.APP_CONTAINER,
    interpolation_scope=NameScope.CONTAINER_ELIGIBLE,
    assignment_scope=NameScope.CONTAINER_ELIGIBLE,
    export_scope=NameScope.CONTAINER_ELIGIBLE,
    malformed="error",
    unresolved="error",
)
COMPOSE_CONTROL_POLICY = ResolutionPolicy(
    target=EnvironmentTarget.COMPOSE_CONTROL_PLANE,
    interpolation_scope=NameScope.NONE,
    assignment_scope=NameScope.NONE,
    export_scope=NameScope.COMPOSE_SELECTED,
    malformed="error",
    unresolved="error",
)

_ENVIRONMENT_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_ALL_DOCKER_CONTROL_KEYS = frozenset().union(*DOCKER_CONTROL_KEYS_BY_PLATFORM.values())
_WINDOWS_SPAWNED_NAMES = {
    "path": "Path",
    "systemroot": "SystemRoot",
    "userprofile": "UserProfile",
    "comspec": "ComSpec",
    "pathext": "PATHEXT",
}


@dataclass(frozen=True)
class ContainerSelection:
    pass_env: frozenset[str]
    compose_env: frozenset[str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "pass_env", frozenset(self.pass_env))
        object.__setattr__(self, "compose_env", frozenset(self.compose_env))


def _validate_name(name: str) -> None:
    if "\0" in name or not _ENVIRONMENT_NAME.fullmatch(name):
        raise ContainerPolicyError(f"Invalid environment name '{name}'.")


def _validate_value(name: str, value: str) -> None:
    if "\0" in value:
        raise ContainerPolicyError(f"Environment value for '{name}' contains NUL.")


def _windows_index(
    mapping: Mapping[str, str], *, label: str, platform: str
) -> dict[str, tuple[str, str]]:
    if platform != "win32":
        return {name: (name, value) for name, value in mapping.items()}
    indexed: dict[str, tuple[str, str]] = {}
    for name, value in mapping.items():
        normalized = name.casefold()
        if normalized in indexed and indexed[normalized][0] != name:
            raise ContainerPolicyError(
                f"{label} contains duplicate Windows environment name '{indexed[normalized][0]}' and '{name}'."
            )
        indexed[normalized] = (name, value)
    return indexed


def _check_windows_selected_names(names: frozenset[str], *, platform: str) -> None:
    if platform != "win32":
        return
    seen: dict[str, str] = {}
    for name in names:
        normalized = name.casefold()
        previous = seen.get(normalized)
        if previous is not None and previous != name:
            raise ContainerPolicyError(
                f"Selected environment names contain duplicate Windows name '{previous}' and '{name}'."
            )
        seen[normalized] = name


def _canonical_name(name: str, *, platform: str) -> str:
    if platform == "win32":
        return _WINDOWS_SPAWNED_NAMES.get(name.casefold(), name.upper())
    return name


def _control_collision(name: str, control: Mapping[str, str], *, platform: str) -> bool:
    if platform == "win32":
        return name.casefold() in _windows_index(
            control, label="docker control environment", platform=platform
        )
    return name in control


def docker_control_environment(
    plan: EnvironmentPlan,
    *,
    platform: str = sys.platform,
) -> Mapping[str, str]:
    """Select only Docker-client controls directly from the launch snapshot."""

    selected = DOCKER_CONTROL_KEYS_BY_PLATFORM[
        "common"
    ] | DOCKER_CONTROL_KEYS_BY_PLATFORM.get(platform, frozenset())
    launch = _windows_index(
        plan.launch_environment, label="launch environment", platform=platform
    )
    payload: dict[str, str] = {}
    for name in selected:
        lookup = name.casefold() if platform == "win32" else name
        item = launch.get(lookup)
        if item is None:
            continue
        source_name, value = item
        _validate_name(source_name)
        _validate_value(source_name, value)
        payload[_canonical_name(source_name, platform=platform)] = value
    return MappingProxyType(dict(payload))


def select_application_payload(
    resolved: ResolvedEnvironment,
    selection: ContainerSelection,
    *,
    platform: str = sys.platform,
) -> Mapping[str, str]:
    """Produce the application-only payload from final resolved values."""

    values = _windows_index(
        resolved.values, label="application environment", platform=platform
    )
    selected = (
        set(resolved.declared_keys)
        | set(APPLICATION_COMPATIBILITY_KEYS)
        | set(selection.pass_env)
    )
    _check_windows_selected_names(selection.pass_env, platform=platform)
    for name in selected:
        _validate_name(name)
    for name in selection.pass_env:
        lookup = name.casefold() if platform == "win32" else name
        if lookup not in values:
            raise ContainerPolicyError(
                f"Explicitly selected environment name '{name}' is not present."
            )

    payload: dict[str, str] = {}
    for name in selected:
        lookup = name.casefold() if platform == "win32" else name
        item = values.get(lookup)
        if item is None:
            continue
        source_name, value = item
        if (source_name.casefold() if platform == "win32" else source_name) in {
            key.casefold() if platform == "win32" else key
            for key in _ALL_DOCKER_CONTROL_KEYS
        }:
            raise ContainerPolicyError(
                f"Application environment name '{source_name}' collides with Docker control plane."
            )
        if source_name in resolved.unresolved_references:
            raise ContainerPolicyError(
                f"Application environment key '{source_name}' has unresolved reference(s): "
                f"{', '.join(sorted(resolved.unresolved_references[source_name]))}."
            )
        _validate_name(source_name)
        _validate_value(source_name, value)
        payload[_canonical_name(source_name, platform=platform)] = value
    return MappingProxyType(dict(payload))


def select_compose_payload(
    *,
    application_payload: Mapping[str, str],
    selected_names: frozenset[str],
    docker_control: Mapping[str, str],
    platform: str = sys.platform,
) -> Mapping[str, str]:
    """Copy selected, already-final application values without re-resolution."""

    application = _windows_index(
        application_payload, label="application payload", platform=platform
    )
    _windows_index(
        docker_control, label="docker control environment", platform=platform
    )
    _check_windows_selected_names(selected_names, platform=platform)
    payload: dict[str, str] = {}
    for name in selected_names:
        _validate_name(name)
        lookup = name.casefold() if platform == "win32" else name
        item = application.get(lookup)
        if item is None:
            raise ContainerPolicyError(
                f"Compose-selected environment name '{name}' is not present in application payload."
            )
        source_name, value = item
        if _control_collision(source_name, docker_control, platform=platform):
            raise ContainerPolicyError(
                f"Compose-selected environment name '{source_name}' collides with Docker control plane."
            )
        _validate_value(source_name, value)
        payload[_canonical_name(source_name, platform=platform)] = value
    return MappingProxyType(dict(payload))


__all__ = [
    "APPLICATION_COMPATIBILITY_KEYS",
    "APP_CONTAINER_POLICY",
    "COMPOSE_CONTROL_POLICY",
    "ContainerPolicyError",
    "ContainerSelection",
    "DOCKER_CONTROL_KEYS_BY_PLATFORM",
    "DOCKER_CONTROL_POLICY",
    "HOST_RUNTIME_POLICY",
    "docker_control_environment",
    "select_application_payload",
    "select_compose_payload",
]
