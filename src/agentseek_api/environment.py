"""Typed, value-redacted environment resolution primitives."""

from __future__ import annotations

import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Literal

from dotenv.variables import Variable

from agentseek_api.dotenv_adapter import DotenvDocument, parse_dotenv_document


class EnvironmentTarget(StrEnum):
    HOST_RUNTIME = "host-runtime"
    DOCKER_CONTROL_PLANE = "docker-control-plane"
    APP_CONTAINER = "app-container"
    COMPOSE_CONTROL_PLANE = "compose-control-plane"


class NameScope(StrEnum):
    NONE = "none"
    ALL = "all"
    DOCKER_CONTROL = "docker-control"
    CONTAINER_ELIGIBLE = "container-eligible"
    COMPOSE_SELECTED = "compose-selected"


@dataclass(frozen=True)
class CommandDerivedAssignment:
    targets: frozenset[EnvironmentTarget]
    values: Mapping[str, str] = field(repr=False)
    reason: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "targets", frozenset(self.targets))
        object.__setattr__(self, "values", MappingProxyType(dict(self.values)))


@dataclass(frozen=True)
class EnvironmentPlan:
    config_path: Path | None
    config_dotenv: Path | None
    config_mapping: Mapping[str, str] = field(repr=False)
    auth_path: str | None = field(repr=False)
    cli_dotenv: Path | None = None
    launch_environment: Mapping[str, str] = field(default_factory=dict, repr=False)
    command_assignments: tuple[CommandDerivedAssignment, ...] = field(
        default=(), repr=False
    )
    explicit_names: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "config_mapping", MappingProxyType(dict(self.config_mapping))
        )
        object.__setattr__(
            self, "launch_environment", MappingProxyType(dict(self.launch_environment))
        )
        object.__setattr__(self, "command_assignments", tuple(self.command_assignments))
        object.__setattr__(self, "explicit_names", frozenset(self.explicit_names))


@dataclass(frozen=True)
class ResolutionPolicy:
    target: EnvironmentTarget
    interpolation_scope: NameScope
    assignment_scope: NameScope
    export_scope: NameScope
    malformed: Literal["error"]
    unresolved: Literal["empty", "error"]


@dataclass(frozen=True)
class EnvironmentOrigin:
    source_kind: Literal[
        "config-dotenv", "config-mapping", "auth", "cli-dotenv", "launch", "command"
    ]
    source_name: str
    line: int | None = None


@dataclass(frozen=True)
class EnvironmentDiagnostic:
    code: str
    key: str
    source_name: str
    line: int | None = None


@dataclass(frozen=True)
class ResolvedEnvironment:
    values: Mapping[str, str] = field(repr=False)
    origins: Mapping[str, EnvironmentOrigin]
    declared_keys: frozenset[str]
    unresolved_references: Mapping[str, frozenset[str]]
    diagnostics: tuple[EnvironmentDiagnostic, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", MappingProxyType(dict(self.values)))
        object.__setattr__(self, "origins", MappingProxyType(dict(self.origins)))
        object.__setattr__(self, "declared_keys", frozenset(self.declared_keys))
        object.__setattr__(
            self,
            "unresolved_references",
            MappingProxyType(
                {
                    key: frozenset(names)
                    for key, names in self.unresolved_references.items()
                }
            ),
        )
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))


class ContainerPolicyError(ValueError):
    pass


def _source_name(path: Path) -> str:
    return str(path)


def _document_declared_names(document: DotenvDocument) -> set[str]:
    return {
        binding.key
        for binding in document
        if binding.key is not None and binding.has_value
    }


def _resolve_document(
    document: DotenvDocument,
    *,
    ambient: Mapping[str, str],
    source_kind: Literal["config-dotenv", "cli-dotenv"],
) -> tuple[
    dict[str, str | None],
    dict[str, EnvironmentOrigin],
    set[str],
    dict[str, frozenset[str]],
    tuple[EnvironmentDiagnostic, ...],
]:
    context: dict[str, str | None] = dict(ambient)
    values: dict[str, str | None] = {}
    origins: dict[str, EnvironmentOrigin] = {}
    declared: set[str] = set()
    unresolved: dict[str, frozenset[str]] = {}
    diagnostics: list[EnvironmentDiagnostic] = []
    for binding in document:
        if binding.key is None:
            continue
        if not binding.has_value:
            values[binding.key] = None
            context[binding.key] = None
            continue
        missing = frozenset(
            atom.name
            for atom in binding.atoms
            if isinstance(atom, Variable)
            and atom.default is None
            and context.get(atom.name) is None
        )
        value = "".join(atom.resolve(context) for atom in binding.atoms)
        values[binding.key] = value
        context[binding.key] = value
        origins[binding.key] = EnvironmentOrigin(
            source_kind, _source_name(document.path), binding.line
        )
        declared.add(binding.key)
        if missing:
            unresolved[binding.key] = missing
            diagnostics.extend(
                EnvironmentDiagnostic(
                    "unresolved-reference",
                    binding.key,
                    _source_name(document.path),
                    binding.line,
                )
                for _ in missing
            )
    return values, origins, declared, unresolved, tuple(diagnostics)


def _check_windows_duplicates(
    mapping: Mapping[str, object], *, label: str, platform: str
) -> None:
    if platform != "win32":
        return
    seen: dict[str, str] = {}
    for name in mapping:
        normalized = name.casefold()
        previous = seen.get(normalized)
        if previous is not None and previous != name:
            raise ContainerPolicyError(
                f"{label} contains duplicate Windows environment name '{previous}' and '{name}'."
            )
        seen[normalized] = name


def _merge_values(
    destination: dict[str, str],
    origins: dict[str, EnvironmentOrigin],
    unresolved: dict[str, frozenset[str]],
    values: Mapping[str, str | None],
    value_origins: Mapping[str, EnvironmentOrigin],
    value_unresolved: Mapping[str, frozenset[str]],
) -> None:
    for key, value in values.items():
        if value is None:
            continue
        destination[key] = value
        origins[key] = value_origins[key]
        if key in value_unresolved:
            unresolved[key] = value_unresolved[key]
        else:
            unresolved.pop(key, None)


def _allowed_ambient(
    launch: Mapping[str, str],
    *,
    policy: ResolutionPolicy,
    eligible_names: set[str],
) -> dict[str, str]:
    if policy.interpolation_scope is NameScope.ALL:
        return dict(launch)
    if policy.interpolation_scope is NameScope.CONTAINER_ELIGIBLE:
        return {key: value for key, value in launch.items() if key in eligible_names}
    return {}


def resolve_environment(
    plan: EnvironmentPlan,
    policy: ResolutionPolicy,
    *,
    platform: str = sys.platform,
) -> ResolvedEnvironment:
    """Resolve one policy target without ever mutating the input plan."""

    if policy.target is EnvironmentTarget.COMPOSE_CONTROL_PLANE:
        raise ContainerPolicyError(
            "Compose control policy is export-only and cannot resolve source environments."
        )

    _check_windows_duplicates(
        plan.launch_environment, label="launch environment", platform=platform
    )
    _check_windows_duplicates(
        plan.config_mapping, label="config mapping", platform=platform
    )
    documents: list[tuple[DotenvDocument, Literal["config-dotenv", "cli-dotenv"]]] = []
    if plan.config_dotenv is not None:
        documents.append((parse_dotenv_document(plan.config_dotenv), "config-dotenv"))
    if plan.cli_dotenv is not None:
        documents.append((parse_dotenv_document(plan.cli_dotenv), "cli-dotenv"))

    eligible_names = set(plan.explicit_names)
    eligible_names.update(plan.config_mapping)
    if plan.auth_path is not None:
        eligible_names.add("AUTH_MODULE_PATH")
    for document, _ in documents:
        eligible_names.update(_document_declared_names(document))
    if policy.interpolation_scope is NameScope.CONTAINER_ELIGIBLE:
        from agentseek_api.container_policy import APPLICATION_COMPATIBILITY_KEYS

        eligible_names.update(APPLICATION_COMPATIBILITY_KEYS)

    ambient = _allowed_ambient(
        plan.launch_environment, policy=policy, eligible_names=eligible_names
    )
    final: dict[str, str] = {}
    origins: dict[str, EnvironmentOrigin] = {}
    unresolved: dict[str, frozenset[str]] = {}
    declared: set[str] = set()
    diagnostics: list[EnvironmentDiagnostic] = []

    for document, source_kind in documents:
        if source_kind != "config-dotenv":
            continue
        (
            values,
            document_origins,
            document_declared,
            document_unresolved,
            document_diagnostics,
        ) = _resolve_document(document, ambient=ambient, source_kind=source_kind)
        _merge_values(
            final, origins, unresolved, values, document_origins, document_unresolved
        )
        declared.update(document_declared)
        diagnostics.extend(document_diagnostics)

    mapping_values = {key: value for key, value in plan.config_mapping.items()}
    mapping_origins = {
        key: EnvironmentOrigin(
            "config-mapping",
            _source_name(plan.config_path) if plan.config_path else "config",
        )
        for key in mapping_values
    }
    _merge_values(final, origins, unresolved, mapping_values, mapping_origins, {})
    declared.update(mapping_values)

    if plan.auth_path is not None:
        auth_values = {"AUTH_MODULE_PATH": plan.auth_path}
        _merge_values(
            final,
            origins,
            unresolved,
            auth_values,
            {
                "AUTH_MODULE_PATH": EnvironmentOrigin(
                    "auth",
                    _source_name(plan.config_path) if plan.config_path else "config",
                )
            },
            {},
        )
        declared.add("AUTH_MODULE_PATH")

    for document, source_kind in documents:
        if source_kind != "cli-dotenv":
            continue
        (
            values,
            document_origins,
            document_declared,
            document_unresolved,
            document_diagnostics,
        ) = _resolve_document(document, ambient=ambient, source_kind=source_kind)
        _merge_values(
            final, origins, unresolved, values, document_origins, document_unresolved
        )
        declared.update(document_declared)
        diagnostics.extend(document_diagnostics)

    if policy.assignment_scope is NameScope.ALL:
        launch_values = dict(plan.launch_environment)
    elif policy.assignment_scope is NameScope.CONTAINER_ELIGIBLE:
        launch_values = {
            key: value
            for key, value in plan.launch_environment.items()
            if key in eligible_names
        }
    else:
        launch_values = {}
    _merge_values(
        final,
        origins,
        unresolved,
        launch_values,
        {key: EnvironmentOrigin("launch", "launch") for key in launch_values},
        {},
    )

    for assignment in plan.command_assignments:
        if policy.target not in assignment.targets:
            continue
        _check_windows_duplicates(
            assignment.values, label="command assignment", platform=platform
        )
        _merge_values(
            final,
            origins,
            unresolved,
            assignment.values,
            {
                key: EnvironmentOrigin("command", assignment.reason)
                for key in assignment.values
            },
            {},
        )

    if policy.unresolved == "error":
        failed = next(
            ((key, names) for key, names in unresolved.items() if key in final), None
        )
        if failed is not None:
            key, names = failed
            raise ContainerPolicyError(
                f"Container environment key '{key}' has unresolved reference(s): {', '.join(sorted(names))}."
            )

    return ResolvedEnvironment(
        values=final,
        origins=origins,
        declared_keys=frozenset(declared),
        unresolved_references=unresolved,
        diagnostics=tuple(diagnostics),
    )
