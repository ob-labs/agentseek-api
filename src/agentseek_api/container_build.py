"""Sanitized, deterministic build planning for the preloaded-v1 container contract."""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import shutil
import stat
import tarfile
import tomllib
import zipfile
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from enum import StrEnum
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Literal, TypeAlias
from urllib.parse import unquote, urlsplit

from packaging.requirements import InvalidRequirement, Requirement
from packaging.version import Version

from agentseek_api.constants import DEFAULT_API_PORT
from agentseek_api.dotenv_adapter import DotenvFileError, parse_dotenv_file
from agentseek_api.environment import EnvironmentOrigin
from agentseek_api.secure_temp import (
    SecureArtifactError,
    create_private_directory,
    verify_private_directory,
)

JsonScalar: TypeAlias = None | bool | int | float | str
JsonValue: TypeAlias = JsonScalar | tuple["JsonValue", ...] | Mapping[str, "JsonValue"]

_RUNTIME_VERSION = "0.3.0"
_CONTAINER_ROOT = PurePosixPath("/deps/agent")
_VCS_METADATA_NAMES = frozenset({".git", ".hg", ".svn", ".bzr"})


class ContainerBuildError(ValueError):
    """A value-free failure to produce a safe container build plan."""


def _freeze_json(value: object, *, location: str) -> JsonValue:
    if isinstance(value, float) and not math.isfinite(value):
        raise ContainerBuildError(f"{location} must contain finite numbers only.")
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item, location=location) for item in value)
    if isinstance(value, Mapping):
        frozen: dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ContainerBuildError(f"{location} keys must be strings.")
            frozen[key] = _freeze_json(item, location=f"{location}.{key}")
        return MappingProxyType(frozen)
    raise ContainerBuildError(f"{location} must contain JSON values only.")


def _json_value(value: JsonValue) -> object:
    if isinstance(value, Mapping):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    return value


@dataclass(frozen=True)
class StructuredGraphV1:
    graph: str
    prepare_input: str | None = None
    extract_output: str | None = None
    name: str | None = None
    description: str | None = None
    input_schema: Mapping[str, JsonValue] | None = field(default=None, repr=False)
    output_schema: Mapping[str, JsonValue] | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        for name in ("input_schema", "output_schema"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _freeze_json(dict(value), location=name))


@dataclass(frozen=True)
class StoreTtlManifestV1:
    refresh_on_read: bool | None = None
    default_ttl: float | None = None
    sweep_interval_minutes: int | None = None


@dataclass(frozen=True)
class StoreIndexManifestV1:
    embed: str | None = None
    dims: int | None = None
    fields: tuple[str, ...] | None = None


@dataclass(frozen=True)
class StoreManifestV1:
    ttl: StoreTtlManifestV1 | None = None
    index: StoreIndexManifestV1 | None = None


@dataclass(frozen=True)
class CorsManifestV1:
    allow_origins: tuple[str, ...] | None = None
    allow_origin_regex: str | None = None
    allow_methods: tuple[str, ...] | None = None
    allow_headers: tuple[str, ...] | None = None
    allow_credentials: bool | None = None
    expose_headers: tuple[str, ...] | None = None
    max_age: int | None = None


@dataclass(frozen=True)
class HttpManifestV1:
    app: str | None = None
    cors: CorsManifestV1 | None = None
    disable_mcp: bool | None = None
    disable_a2a: bool | None = None


@dataclass(frozen=True)
class AuthOpenApiManifestV1:
    security_schemes: Mapping[str, Mapping[str, JsonValue]] | None = field(
        default=None, repr=False
    )
    security: tuple[Mapping[str, tuple[str, ...]], ...] | None = field(
        default=None, repr=False
    )

    def __post_init__(self) -> None:
        if self.security_schemes is not None:
            schemes: dict[str, Mapping[str, JsonValue]] = {}
            for name, scheme in self.security_schemes.items():
                frozen = _freeze_json(
                    dict(scheme), location=f"auth.openapi.securitySchemes.{name}"
                )
                assert isinstance(frozen, Mapping)
                schemes[name] = frozen
            object.__setattr__(self, "security_schemes", MappingProxyType(schemes))
        if self.security is not None:
            object.__setattr__(
                self,
                "security",
                tuple(
                    MappingProxyType(
                        {name: tuple(scopes) for name, scopes in requirement.items()}
                    )
                    for requirement in self.security
                ),
            )


@dataclass(frozen=True)
class AuthPolicyManifestV1:
    openapi: AuthOpenApiManifestV1 | None = field(default=None, repr=False)
    disable_studio_auth: bool | None = None


class InstallActionKind(StrEnum):
    PROJECT = "project"
    REQUIREMENTS = "requirements"
    PEP508 = "pep508"
    SOURCE_ONLY = "source-only"


@dataclass(frozen=True)
class InstallAction:
    kind: InstallActionKind
    operand: str = field(repr=False)


class SourceReason(StrEnum):
    GRAPH = "graph"
    DEPENDENCY = "dependency"
    GRAPH_HOOK = "graph-hook"
    STORE_HOOK = "store-hook"
    HTTP_APP = "http-app"
    AUTH = "auth"
    BUILD_INCLUDE = "build-include"
    RUNTIME_ARTIFACT = "runtime-artifact"


@dataclass(frozen=True)
class SelectedSource:
    source_path: Path = field(repr=False)
    reasons: frozenset[SourceReason]
    source_identity: tuple[int, int, int, int] | None = field(
        default=None, repr=False, compare=False
    )
    source_sha256: str | None = field(default=None, repr=False, compare=False)
    ancestor_identities: tuple[tuple[Path, tuple[int, int]], ...] = field(
        default=(), repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if not self.reasons:
            raise ContainerBuildError("A selected build source must have a reason.")


@dataclass(frozen=True)
class FinalAuthSelection:
    value: str = field(repr=False)
    origin: EnvironmentOrigin


@dataclass(frozen=True)
class AuthPayloadPatch:
    value: str = field(repr=False)


class RuntimeArtifactSource(StrEnum):
    PUBLISHED_INDEX = "published-index"
    CANDIDATE_WHEEL = "candidate-wheel"


@dataclass(frozen=True)
class RuntimeArtifactV1:
    distribution: Literal["agentseek-api"]
    extra: Literal["embedded"]
    version: Literal["0.3.0"]
    source: RuntimeArtifactSource
    candidate_wheel: Path | None = field(default=None, repr=False)
    candidate_sha256: str | None = None
    candidate_identity: tuple[int, int, int, int] | None = field(
        default=None, repr=False
    )

    def __post_init__(self) -> None:
        if (
            self.distribution != "agentseek-api"
            or self.extra != "embedded"
            or self.version != _RUNTIME_VERSION
        ):
            raise ContainerBuildError("The runtime artifact identity is incompatible.")
        if self.source is RuntimeArtifactSource.PUBLISHED_INDEX:
            if (
                self.candidate_wheel is not None
                or self.candidate_sha256 is not None
                or self.candidate_identity is not None
            ):
                raise ContainerBuildError(
                    "A published runtime artifact cannot carry candidate state."
                )
        elif self.source is RuntimeArtifactSource.CANDIDATE_WHEEL:
            if (
                self.candidate_wheel is None
                or not re.fullmatch(r"[0-9a-f]{64}", self.candidate_sha256 or "")
                or self.candidate_identity is None
            ):
                raise ContainerBuildError(
                    "A candidate runtime artifact requires a wheel and SHA-256."
                )
        else:  # pragma: no cover - enum construction normally prevents this
            raise ContainerBuildError("The runtime artifact source is unsupported.")

    @property
    def requirement(self) -> str:
        return "agentseek-api[embedded]==0.3.0"


PUBLISHED_RUNTIME_ARTIFACT = RuntimeArtifactV1(
    distribution="agentseek-api",
    extra="embedded",
    version="0.3.0",
    source=RuntimeArtifactSource.PUBLISHED_INDEX,
)


@dataclass(frozen=True)
class RuntimeManifestV1:
    distribution: Literal["agentseek-api"]
    version: Literal["0.3.0"]
    contract: Literal["preloaded-v1"]


@dataclass(frozen=True)
class ContainerRuntimeManifestV1:
    schema_version: Literal[1]
    runtime: RuntimeManifestV1
    graphs: Mapping[str, str | StructuredGraphV1] = field(repr=False)
    dependencies: tuple[str, ...] = field(repr=False)
    store: StoreManifestV1 | None = field(repr=False)
    http: HttpManifestV1 | None = field(default=None, repr=False)
    auth: AuthPolicyManifestV1 | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.runtime != RuntimeManifestV1(
            distribution="agentseek-api", version="0.3.0", contract="preloaded-v1"
        ):
            raise ContainerBuildError("The runtime manifest identity is incompatible.")
        object.__setattr__(self, "graphs", MappingProxyType(dict(self.graphs)))
        object.__setattr__(self, "dependencies", tuple(self.dependencies))

    def to_json_object(self) -> dict[str, object]:
        document: dict[str, object] = {
            "schema_version": 1,
            "runtime": {
                "distribution": self.runtime.distribution,
                "version": self.runtime.version,
                "contract": self.runtime.contract,
            },
            "graphs": {name: _graph_json(graph) for name, graph in self.graphs.items()},
            "dependencies": list(self.dependencies),
        }
        if self.store is not None:
            document["store"] = _store_json(self.store)
        if self.http is not None:
            document["http"] = _http_json(self.http)
        if self.auth is not None:
            document["auth"] = _auth_json(self.auth)
        return document

    def to_json_bytes(self) -> bytes:
        return (
            json.dumps(
                self.to_json_object(),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
            + b"\n"
        )


@dataclass(frozen=True)
class ContainerBuildPlan:
    base_image: str
    python_version: str
    image_distro: str
    dockerfile_lines: tuple[str, ...] = field(repr=False)
    runtime_artifact: RuntimeArtifactV1 = field(repr=False)
    install_actions: tuple[InstallAction, ...] = field(repr=False)
    pip_config_file: Path | None = field(repr=False)
    manifest: ContainerRuntimeManifestV1 = field(repr=False)
    selected_sources: Mapping[str, SelectedSource] = field(repr=False)
    config_path: Path = field(repr=False)
    project_root: Path = field(repr=False)
    project_root_identity: tuple[int, int] = field(repr=False)
    invocation_cwd: Path = field(repr=False)
    excluded_paths: frozenset[Path] = field(default_factory=frozenset, repr=False)
    pip_config_identity: tuple[int, int, int, int] | None = field(
        default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "selected_sources", MappingProxyType(dict(self.selected_sources))
        )


@dataclass(frozen=True)
class BuildInventoryEntry:
    relative_path: str
    sha256: str
    size: int


@dataclass(frozen=True)
class ContainerBuildBundle:
    root: Path
    context: Path
    dockerfile: Path
    manifest: Path
    inventory: tuple[BuildInventoryEntry, ...]
    plan_fingerprint: str = field(repr=False)

    def archive_bytes(self) -> bytes:
        return create_deterministic_context_archive(
            context=self.context, expected_inventory=self.inventory
        )


def container_build_plan_fingerprint(plan: ContainerBuildPlan) -> str:
    """Return a value-free digest binding a bundle to every frozen plan field."""

    artifact = plan.runtime_artifact
    payload = {
        "base_image": plan.base_image,
        "python_version": plan.python_version,
        "image_distro": plan.image_distro,
        "dockerfile_lines": plan.dockerfile_lines,
        "runtime_artifact": {
            "distribution": artifact.distribution,
            "extra": artifact.extra,
            "version": artifact.version,
            "source": artifact.source.value,
            "candidate_wheel": (
                str(artifact.candidate_wheel)
                if artifact.candidate_wheel is not None
                else None
            ),
            "candidate_sha256": artifact.candidate_sha256,
            "candidate_identity": artifact.candidate_identity,
        },
        "install_actions": [
            {"kind": action.kind.value, "operand": action.operand}
            for action in plan.install_actions
        ],
        "pip_config_file": (
            str(plan.pip_config_file) if plan.pip_config_file is not None else None
        ),
        "pip_config_identity": plan.pip_config_identity,
        "manifest_sha256": hashlib.sha256(plan.manifest.to_json_bytes()).hexdigest(),
        "selected_sources": [
            {
                "destination": destination,
                "source_path": str(selected.source_path),
                "reasons": sorted(reason.value for reason in selected.reasons),
                "source_identity": selected.source_identity,
                "source_sha256": selected.source_sha256,
                "ancestor_identities": [
                    [str(path), identity]
                    for path, identity in selected.ancestor_identities
                ],
            }
            for destination, selected in sorted(plan.selected_sources.items())
        ],
        "config_path": str(plan.config_path),
        "project_root": str(plan.project_root),
        "project_root_identity": plan.project_root_identity,
        "invocation_cwd": str(plan.invocation_cwd),
        "excluded_paths": sorted(str(path) for path in plan.excluded_paths),
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


@dataclass(frozen=True)
class EffectiveRuntimePolicyV1:
    mcp_enabled: bool
    a2a_enabled: bool
    cors_middleware: Mapping[str, JsonValue] = field(repr=False)
    custom_app: str | None
    auth_openapi: Mapping[str, JsonValue] | None = field(default=None, repr=False)
    studio_auth_disabled: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "cors_middleware",
            _freeze_json(dict(self.cors_middleware), location="cors_middleware"),
        )
        if self.auth_openapi is not None:
            object.__setattr__(
                self,
                "auth_openapi",
                _freeze_json(dict(self.auth_openapi), location="auth_openapi"),
            )


def _omit_none(**items: object) -> dict[str, object]:
    return {name: value for name, value in items.items() if value is not None}


def _graph_json(graph: str | StructuredGraphV1) -> object:
    if isinstance(graph, str):
        return graph
    return _omit_none(
        graph=graph.graph,
        prepare_input=graph.prepare_input,
        extract_output=graph.extract_output,
        name=graph.name,
        description=graph.description,
        input_schema=None
        if graph.input_schema is None
        else _json_value(graph.input_schema),
        output_schema=None
        if graph.output_schema is None
        else _json_value(graph.output_schema),
    )


def _store_json(store: StoreManifestV1) -> dict[str, object]:
    document: dict[str, object] = {}
    if store.ttl is not None:
        document["ttl"] = _omit_none(
            refresh_on_read=store.ttl.refresh_on_read,
            default_ttl=store.ttl.default_ttl,
            sweep_interval_minutes=store.ttl.sweep_interval_minutes,
        )
    if store.index is not None:
        document["index"] = _omit_none(
            embed=store.index.embed,
            dims=store.index.dims,
            fields=None if store.index.fields is None else list(store.index.fields),
        )
    return document


def _http_json(http: HttpManifestV1) -> dict[str, object]:
    document = _omit_none(
        app=http.app,
        disable_mcp=http.disable_mcp,
        disable_a2a=http.disable_a2a,
    )
    if http.cors is not None:
        document["cors"] = _omit_none(
            allow_origins=None
            if http.cors.allow_origins is None
            else list(http.cors.allow_origins),
            allow_origin_regex=http.cors.allow_origin_regex,
            allow_methods=None
            if http.cors.allow_methods is None
            else list(http.cors.allow_methods),
            allow_headers=None
            if http.cors.allow_headers is None
            else list(http.cors.allow_headers),
            allow_credentials=http.cors.allow_credentials,
            expose_headers=None
            if http.cors.expose_headers is None
            else list(http.cors.expose_headers),
            max_age=http.cors.max_age,
        )
    return document


def _auth_json(auth: AuthPolicyManifestV1) -> dict[str, object]:
    document = _omit_none(disable_studio_auth=auth.disable_studio_auth)
    if auth.openapi is not None:
        openapi: dict[str, object] = {}
        if auth.openapi.security_schemes is not None:
            openapi["securitySchemes"] = {
                key: _json_value(value)
                for key, value in auth.openapi.security_schemes.items()
            }
        if auth.openapi.security is not None:
            openapi["security"] = [
                {name: list(scopes) for name, scopes in item.items()}
                for item in auth.openapi.security
            ]
        document["openapi"] = openapi
    return document


def _safe_regular(path: Path, *, project_root: Path, purpose: str) -> Path:
    try:
        supplied = path.absolute()
        raw = Path(os.path.normpath(supplied))
        resolved = supplied.resolve(strict=True)
        relative_resolved = resolved.relative_to(project_root)
        lexical_root = raw
        for _ in relative_resolved.parts:
            lexical_root = lexical_root.parent
        relative_raw = raw.relative_to(lexical_root)
        if (
            lexical_root.resolve(strict=True) != project_root
            or relative_raw.parts != relative_resolved.parts
        ):
            raise ContainerBuildError(
                f"The {purpose} source has an unsafe intermediate directory."
            )
        current = lexical_root
        for part in relative_raw.parts[:-1]:
            current = current / part
            parent_status = current.lstat()
            if stat.S_ISLNK(parent_status.st_mode) or not stat.S_ISDIR(
                parent_status.st_mode
            ):
                raise ContainerBuildError(
                    f"The {purpose} source has an unsafe intermediate directory."
                )
        status = raw.lstat()
    except ValueError as exc:
        raise ContainerBuildError(
            f"The {purpose} source must remain inside the project root."
        ) from exc
    except OSError as exc:
        raise ContainerBuildError(f"The {purpose} source is missing.") from exc
    if stat.S_ISLNK(status.st_mode) or not stat.S_ISREG(status.st_mode):
        raise ContainerBuildError(f"The {purpose} source must be a regular file.")
    if _VCS_METADATA_NAMES.intersection(resolved.relative_to(project_root).parts):
        raise ContainerBuildError(
            f"The {purpose} source selects excluded VCS metadata."
        )
    return resolved


def _safe_directory(path: Path, *, project_root: Path, purpose: str) -> Path:
    try:
        raw = path.absolute()
        relative_raw = raw.relative_to(project_root)
        current = project_root
        for part in relative_raw.parts:
            current = current / part
            component_status = current.lstat()
            if stat.S_ISLNK(component_status.st_mode) or not stat.S_ISDIR(
                component_status.st_mode
            ):
                raise ContainerBuildError(
                    f"The {purpose} source has an unsafe intermediate directory."
                )
        raw_status = raw.lstat()
    except ValueError as exc:
        raise ContainerBuildError(
            f"The {purpose} source must remain inside the project root."
        ) from exc
    except OSError as exc:
        raise ContainerBuildError(f"The {purpose} source is missing.") from exc
    if stat.S_ISLNK(raw_status.st_mode) or not stat.S_ISDIR(raw_status.st_mode):
        raise ContainerBuildError(f"The {purpose} source must be a directory.")
    resolved = path.resolve()
    try:
        resolved.relative_to(project_root)
    except ValueError as exc:
        raise ContainerBuildError(
            f"The {purpose} source must remain inside the project root."
        ) from exc
    if _VCS_METADATA_NAMES.intersection(resolved.relative_to(project_root).parts):
        raise ContainerBuildError(
            f"The {purpose} source selects excluded VCS metadata."
        )
    return resolved


def _pip_config_source(path: Path) -> tuple[Path, tuple[int, int, int, int]]:
    """Freeze an external secret carrier's identity without reading its contents."""

    try:
        raw = path.absolute()
        status = raw.lstat()
        if stat.S_ISLNK(status.st_mode) or not stat.S_ISREG(status.st_mode):
            raise ContainerBuildError("The pip config must be a readable regular file.")
        if os.name != "nt" and status.st_mode & 0o444 == 0:
            raise ContainerBuildError("The pip config must be a readable regular file.")
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(raw, flags)
    except ContainerBuildError:
        raise
    except OSError as exc:
        raise ContainerBuildError(
            "The pip config must be a readable regular file."
        ) from exc
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or _file_identity(opened) != _file_identity(
            status
        ):
            raise ContainerBuildError(
                "The pip config identity changed during planning."
            )
        identity = _file_identity(opened)
    finally:
        os.close(descriptor)
    return raw.resolve(strict=True), identity


def _container_path(relative: Path) -> str:
    return str(_CONTAINER_ROOT / PurePosixPath(relative.as_posix()))


def _module_file(reference: str, *, base: Path) -> tuple[Path, str] | None:
    module = reference.rsplit(":", 1)[0]
    if (
        module.endswith(".py")
        or module.startswith(".")
        or "/" in module
        or "\\" in module
    ):
        path = Path(module).expanduser()
        if not path.is_absolute():
            path = base / path
        return path, reference.rsplit(":", 1)[1] if ":" in reference else ""
    candidate = base / (module.replace(".", os.sep) + ".py")
    if candidate.exists():
        return candidate, reference.rsplit(":", 1)[1] if ":" in reference else ""
    return None


def _is_path_reference(reference: str) -> bool:
    module = reference.rsplit(":", 1)[0]
    return (
        module.endswith(".py")
        or module.startswith(".")
        or "/" in module
        or "\\" in module
        or Path(module).is_absolute()
    )


def _add_selected(
    selected: dict[str, SelectedSource],
    *,
    destination: str,
    source: Path,
    reason: SourceReason,
    project_root: Path | None = None,
) -> None:
    if destination.startswith("/") or ".." in PurePosixPath(destination).parts:
        raise ContainerBuildError("A selected destination escaped the build context.")
    selection_root = source.parent if project_root is None else project_root
    source_status = source.lstat()
    source_identity = _file_identity(source_status)
    source_sha256 = hashlib.sha256(
        _read_regular_source(source, expected_identity=source_identity)
    ).hexdigest()
    ancestors: list[tuple[Path, tuple[int, int]]] = []
    current = source.parent
    while True:
        ancestors.append((current, _directory_identity(current.lstat())))
        if current == selection_root:
            break
        if selection_root not in current.parents:
            raise ContainerBuildError("A selected source escaped the project root.")
        current = current.parent
    ancestor_identities = tuple(reversed(ancestors))
    existing = selected.get(destination)
    if existing is None:
        selected[destination] = SelectedSource(
            source,
            frozenset({reason}),
            source_identity=source_identity,
            source_sha256=source_sha256,
            ancestor_identities=ancestor_identities,
        )
        return
    if existing.source_path != source:
        raise ContainerBuildError(
            "Two different sources selected the same destination."
        )
    if (
        existing.source_identity != source_identity
        or existing.source_sha256 != source_sha256
        or existing.ancestor_identities != ancestor_identities
    ):
        raise ContainerBuildError("A selected source identity changed during planning.")
    selected[destination] = SelectedSource(
        source,
        existing.reasons | frozenset({reason}),
        source_identity=source_identity,
        source_sha256=source_sha256,
        ancestor_identities=ancestor_identities,
    )


def _select_file(
    selected: dict[str, SelectedSource],
    *,
    source: Path,
    project_root: Path,
    reason: SourceReason,
    excluded: frozenset[Path],
    destination_prefix: str = "app",
) -> Path:
    source = _safe_regular(source, project_root=project_root, purpose=reason.value)
    relative = source.relative_to(project_root)
    if (
        source in excluded
        or _VCS_METADATA_NAMES.intersection(relative.parts)
        or source.name
        in {
            ".gitignore",
            ".gitattributes",
        }
    ):
        raise ContainerBuildError(f"The selected {reason.value} source is excluded.")
    if relative.parts and relative.parts[0] in {"agentseek_api", "agentseek_api.py"}:
        raise ContainerBuildError(
            "A selected source could shadow the installed runtime."
        )
    destination = str(
        PurePosixPath(destination_prefix) / PurePosixPath(relative.as_posix())
    )
    _add_selected(
        selected,
        destination=destination,
        source=source,
        reason=reason,
        project_root=project_root,
    )
    return source


def _select_tree(
    selected: dict[str, SelectedSource],
    *,
    root: Path,
    project_root: Path,
    reason: SourceReason,
    excluded: frozenset[Path],
) -> None:
    root = _safe_directory(root, project_root=project_root, purpose=reason.value)
    for candidate in sorted(root.rglob("*")):
        relative = candidate.relative_to(project_root)
        if _VCS_METADATA_NAMES.intersection(relative.parts):
            continue
        status = candidate.lstat()
        if stat.S_ISLNK(status.st_mode):
            raise ContainerBuildError(
                f"The selected {reason.value} tree contains a symlink."
            )
        if stat.S_ISDIR(status.st_mode):
            continue
        if not stat.S_ISREG(status.st_mode):
            raise ContainerBuildError(
                f"The selected {reason.value} tree contains a non-regular file."
            )
        if candidate.resolve() in excluded:
            continue
        _select_file(
            selected,
            source=candidate,
            project_root=project_root,
            reason=reason,
            excluded=excluded,
        )


def _validate_allowed(
    raw: Mapping[str, object], allowed: set[str], location: str
) -> None:
    unknown = set(raw) - allowed
    if unknown:
        raise ContainerBuildError(f"{location} contains unsupported fields.")


def _string_tuple(value: object, *, location: str) -> tuple[str, ...] | None:
    if value is None:
        return None
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ContainerBuildError(f"{location} must be an array of strings.")
    return tuple(value)


def _optional_bool(value: object, *, location: str) -> bool | None:
    if value is None:
        return None
    if not isinstance(value, bool):
        raise ContainerBuildError(f"{location} must be a boolean.")
    return value


def _optional_number(
    value: object, *, location: str, integer: bool = False
) -> float | int | None:
    if value is None:
        return None
    expected = int if integer else (int, float)
    if isinstance(value, bool) or not isinstance(value, expected):
        raise ContainerBuildError(f"{location} must be numeric.")
    if isinstance(value, float) and not math.isfinite(value):
        raise ContainerBuildError(f"{location} must be finite.")
    return value


def _parse_graphs(
    raw: object,
    *,
    reference_base: Path,
    project_root: Path,
    selected: dict[str, SelectedSource],
    excluded: frozenset[Path],
) -> Mapping[str, str | StructuredGraphV1]:
    if not isinstance(raw, dict) or not raw:
        raise ContainerBuildError("graphs must be a non-empty object.")
    graphs: dict[str, str | StructuredGraphV1] = {}
    for name, value in raw.items():
        if not isinstance(name, str) or not name:
            raise ContainerBuildError("graph names must be non-empty strings.")
        references: list[tuple[str, SourceReason]] = []
        if isinstance(value, str):
            graph: str | StructuredGraphV1 = value
            references.append((value, SourceReason.GRAPH))
        elif isinstance(value, dict):
            _validate_allowed(
                value,
                {
                    "graph",
                    "prepare_input",
                    "extract_output",
                    "name",
                    "description",
                    "input_schema",
                    "output_schema",
                },
                f"graphs.{name}",
            )
            if not isinstance(value.get("graph"), str):
                raise ContainerBuildError(f"graphs.{name}.graph must be a string.")
            optional_strings: dict[str, str | None] = {}
            for key in ("prepare_input", "extract_output", "name", "description"):
                item = value.get(key)
                if item is not None and not isinstance(item, str):
                    raise ContainerBuildError(f"graphs.{name}.{key} must be a string.")
                optional_strings[key] = item
            input_schema = value.get("input_schema")
            output_schema = value.get("output_schema")
            if input_schema is not None and not isinstance(input_schema, dict):
                raise ContainerBuildError(
                    f"graphs.{name}.input_schema must be an object."
                )
            if output_schema is not None and not isinstance(output_schema, dict):
                raise ContainerBuildError(
                    f"graphs.{name}.output_schema must be an object."
                )
            graph = StructuredGraphV1(
                graph=value["graph"],
                prepare_input=optional_strings["prepare_input"],
                extract_output=optional_strings["extract_output"],
                name=optional_strings["name"],
                description=optional_strings["description"],
                input_schema=None
                if input_schema is None
                else _freeze_json(input_schema, location=f"graphs.{name}.input_schema"),  # type: ignore[arg-type]
                output_schema=None
                if output_schema is None
                else _freeze_json(
                    output_schema, location=f"graphs.{name}.output_schema"
                ),  # type: ignore[arg-type]
            )
            references.append((graph.graph, SourceReason.GRAPH))
            if graph.prepare_input is not None:
                references.append((graph.prepare_input, SourceReason.GRAPH_HOOK))
            if graph.extract_output is not None:
                references.append((graph.extract_output, SourceReason.GRAPH_HOOK))
        else:
            raise ContainerBuildError(f"graphs.{name} must be a string or object.")
        for reference, reason in references:
            located = _module_file(reference, base=reference_base)
            if located is None:
                continue
            module_file, symbol = located
            module_file = _safe_regular(
                module_file, project_root=project_root, purpose=reason.value
            )
            _select_file(
                selected,
                source=module_file,
                project_root=project_root,
                reason=reason,
                excluded=excluded,
            )
            package_root = module_file.parent
            if (package_root / "__init__.py").exists():
                _select_file(
                    selected,
                    source=package_root / "__init__.py",
                    project_root=project_root,
                    reason=reason,
                    excluded=excluded,
                )
            module_text = reference.rsplit(":", 1)[0]
            if (
                module_text.endswith(".py")
                or module_text.startswith(".")
                or "/" in module_text
                or "\\" in module_text
            ):
                normalized = (
                    f"{_container_path(module_file.relative_to(project_root))}:{symbol}"
                )
                if isinstance(graph, str):
                    graph = normalized
                elif reference == graph.graph:
                    graph = replace(graph, graph=normalized)
                elif reference == graph.prepare_input:
                    graph = replace(graph, prepare_input=normalized)
                elif reference == graph.extract_output:
                    graph = replace(graph, extract_output=normalized)
        graphs[name] = graph
    return MappingProxyType(graphs)


def _parse_store(
    raw: object,
    *,
    reference_base: Path,
    project_root: Path,
    selected: dict[str, SelectedSource],
    excluded: frozenset[Path],
) -> StoreManifestV1 | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ContainerBuildError("store must be an object.")
    _validate_allowed(raw, {"ttl", "index"}, "store")
    ttl: StoreTtlManifestV1 | None = None
    if (raw_ttl := raw.get("ttl")) is not None:
        if not isinstance(raw_ttl, dict):
            raise ContainerBuildError("store.ttl must be an object.")
        _validate_allowed(
            raw_ttl,
            {"refresh_on_read", "default_ttl", "sweep_interval_minutes"},
            "store.ttl",
        )
        ttl = StoreTtlManifestV1(
            refresh_on_read=_optional_bool(
                raw_ttl.get("refresh_on_read"), location="store.ttl.refresh_on_read"
            ),
            default_ttl=_optional_number(
                raw_ttl.get("default_ttl"), location="store.ttl.default_ttl"
            ),  # type: ignore[arg-type]
            sweep_interval_minutes=_optional_number(
                raw_ttl.get("sweep_interval_minutes"),
                location="store.ttl.sweep_interval_minutes",
                integer=True,
            ),  # type: ignore[arg-type]
        )
    index: StoreIndexManifestV1 | None = None
    if (raw_index := raw.get("index")) is not None:
        if not isinstance(raw_index, dict):
            raise ContainerBuildError("store.index must be an object.")
        _validate_allowed(raw_index, {"embed", "dims", "fields"}, "store.index")
        embed = raw_index.get("embed")
        if embed is not None and not isinstance(embed, str):
            raise ContainerBuildError("store.index.embed must be a string.")
        if isinstance(embed, str) and _is_path_reference(embed):
            located = _module_file(embed, base=reference_base)
            if located is not None:
                path, symbol = located
                path = _select_file(
                    selected,
                    source=path,
                    project_root=project_root,
                    reason=SourceReason.STORE_HOOK,
                    excluded=excluded,
                )
                embed = f"{_container_path(path.relative_to(project_root))}:{symbol}"
        index = StoreIndexManifestV1(
            embed=embed,
            dims=_optional_number(
                raw_index.get("dims"), location="store.index.dims", integer=True
            ),  # type: ignore[arg-type]
            fields=_string_tuple(
                raw_index.get("fields"), location="store.index.fields"
            ),
        )
    return StoreManifestV1(ttl=ttl, index=index)


def _parse_cors(raw: object) -> CorsManifestV1 | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ContainerBuildError("http.cors must be an object.")
    _validate_allowed(
        raw,
        {
            "allow_origins",
            "allow_origin_regex",
            "allow_methods",
            "allow_headers",
            "allow_credentials",
            "expose_headers",
            "max_age",
        },
        "http.cors",
    )
    regex = raw.get("allow_origin_regex")
    if regex is not None and not isinstance(regex, str):
        raise ContainerBuildError("http.cors.allow_origin_regex must be a string.")
    return CorsManifestV1(
        allow_origins=_string_tuple(
            raw.get("allow_origins"), location="http.cors.allow_origins"
        ),
        allow_origin_regex=regex,
        allow_methods=_string_tuple(
            raw.get("allow_methods"), location="http.cors.allow_methods"
        ),
        allow_headers=_string_tuple(
            raw.get("allow_headers"), location="http.cors.allow_headers"
        ),
        allow_credentials=_optional_bool(
            raw.get("allow_credentials"), location="http.cors.allow_credentials"
        ),
        expose_headers=_string_tuple(
            raw.get("expose_headers"), location="http.cors.expose_headers"
        ),
        max_age=_optional_number(
            raw.get("max_age"), location="http.cors.max_age", integer=True
        ),  # type: ignore[arg-type]
    )


def _parse_http(
    raw: object,
    *,
    reference_base: Path,
    project_root: Path,
    selected: dict[str, SelectedSource],
    excluded: frozenset[Path],
) -> HttpManifestV1 | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ContainerBuildError("http must be an object.")
    _validate_allowed(raw, {"app", "cors", "disable_mcp", "disable_a2a"}, "http")
    app = raw.get("app")
    if app is not None and not isinstance(app, str):
        raise ContainerBuildError("http.app must be a string.")
    if isinstance(app, str) and _is_path_reference(app):
        located = _module_file(app, base=reference_base)
        if located is not None:
            path, symbol = located
            path = _select_file(
                selected,
                source=path,
                project_root=project_root,
                reason=SourceReason.HTTP_APP,
                excluded=excluded,
            )
            app = f"{_container_path(path.relative_to(project_root))}:{symbol}"
    return HttpManifestV1(
        app=app,
        cors=_parse_cors(raw.get("cors")),
        disable_mcp=_optional_bool(raw.get("disable_mcp"), location="http.disable_mcp"),
        disable_a2a=_optional_bool(raw.get("disable_a2a"), location="http.disable_a2a"),
    )


def _validate_security_scheme(name: str, raw: object) -> Mapping[str, JsonValue]:
    if not isinstance(raw, dict):
        raise ContainerBuildError(
            f"auth.openapi.securitySchemes.{name} must be an object."
        )
    location = f"auth.openapi.securitySchemes.{name}"
    scheme_type = raw.get("type")
    if not isinstance(scheme_type, str):
        raise ContainerBuildError(f"{location}.type is required and must be a string.")
    allowed_by_type = {
        "apiKey": {"type", "description", "name", "in"},
        "http": {"type", "description", "scheme", "bearerFormat"},
        "oauth2": {"type", "description", "flows"},
        "openIdConnect": {"type", "description", "openIdConnectUrl"},
    }
    allowed = allowed_by_type.get(scheme_type)
    if allowed is None:
        raise ContainerBuildError(f"{location}.type is unsupported.")
    _validate_allowed(raw, allowed, location)
    required_by_type = {
        "apiKey": {"name", "in"},
        "http": {"scheme"},
        "oauth2": {"flows"},
        "openIdConnect": {"openIdConnectUrl"},
    }
    missing = required_by_type[scheme_type] - raw.keys()
    if missing:
        raise ContainerBuildError(f"{location} is missing required fields.")
    if scheme_type == "apiKey" and raw.get("in") not in {"query", "header", "cookie"}:
        raise ContainerBuildError(f"{location}.in has an unsupported value.")
    for key, value in raw.items():
        if key.startswith("x-"):
            raise ContainerBuildError(
                "OpenAPI security metadata cannot contain extensions."
            )
        if key == "flows":
            _validate_oauth_flows(value, location=f"{location}.flows")
            continue
        if not isinstance(value, str):
            raise ContainerBuildError(f"{location}.{key} must be a string.")
        if key.endswith("Url"):
            _reject_credential_url(value)
    return _freeze_json(raw, location=location)  # type: ignore[return-value]


def _reject_credential_url(value: str) -> None:
    parsed = urlsplit(value)
    if parsed.username is not None or parsed.password is not None:
        raise ContainerBuildError(
            "OpenAPI security metadata has a credential-bearing URL."
        )
    if parsed.query:
        raise ContainerBuildError(
            "OpenAPI security metadata URL query parameters are not permitted."
        )
    if parsed.fragment and not re.fullmatch(r"sha256=[0-9a-fA-F]{64}", parsed.fragment):
        raise ContainerBuildError(
            "OpenAPI security metadata URL fragment is not permitted."
        )


def _validate_oauth_flows(value: object, *, location: str) -> None:
    if not isinstance(value, dict) or not value:
        raise ContainerBuildError(f"{location} must be an object.")
    _validate_allowed(
        value,
        {"implicit", "password", "clientCredentials", "authorizationCode"},
        location,
    )
    for flow_name, flow in value.items():
        if not isinstance(flow, dict):
            raise ContainerBuildError(f"{location}.{flow_name} must be an object.")
        allowed = {"tokenUrl", "refreshUrl", "scopes"}
        if flow_name in {"implicit", "authorizationCode"}:
            allowed.add("authorizationUrl")
        _validate_allowed(flow, allowed, f"{location}.{flow_name}")
        required = {
            "implicit": {"authorizationUrl", "scopes"},
            "password": {"tokenUrl", "scopes"},
            "clientCredentials": {"tokenUrl", "scopes"},
            "authorizationCode": {"authorizationUrl", "tokenUrl", "scopes"},
        }[flow_name]
        if required - flow.keys():
            raise ContainerBuildError(
                f"{location}.{flow_name} is missing required flow fields."
            )
        scopes = flow.get("scopes")
        if not isinstance(scopes, dict) or not all(
            isinstance(key, str) and isinstance(item, str)
            for key, item in scopes.items()
        ):
            raise ContainerBuildError(
                f"{location}.{flow_name}.scopes must be a string map."
            )
        for key in ("authorizationUrl", "tokenUrl", "refreshUrl"):
            url = flow.get(key)
            if url is not None:
                if not isinstance(url, str):
                    raise ContainerBuildError(
                        f"{location}.{flow_name}.{key} must be a string."
                    )
                _reject_credential_url(url)


def _parse_auth(raw: object) -> tuple[AuthPolicyManifestV1 | None, str | None]:
    if raw is None:
        return None, None
    if not isinstance(raw, dict):
        raise ContainerBuildError("auth must be an object.")
    _validate_allowed(raw, {"path", "openapi", "disable_studio_auth"}, "auth")
    auth_path = raw.get("path")
    if auth_path is not None and not isinstance(auth_path, str):
        raise ContainerBuildError("auth.path must be a string.")
    openapi: AuthOpenApiManifestV1 | None = None
    if (raw_openapi := raw.get("openapi")) is not None:
        if not isinstance(raw_openapi, dict):
            raise ContainerBuildError("auth.openapi must be an object.")
        _validate_allowed(raw_openapi, {"securitySchemes", "security"}, "auth.openapi")
        schemes: Mapping[str, Mapping[str, JsonValue]] | None = None
        if (raw_schemes := raw_openapi.get("securitySchemes")) is not None:
            if not isinstance(raw_schemes, dict):
                raise ContainerBuildError(
                    "auth.openapi.securitySchemes must be an object."
                )
            schemes = MappingProxyType(
                {
                    str(name): _validate_security_scheme(str(name), scheme)
                    for name, scheme in raw_schemes.items()
                }
            )
        security: tuple[Mapping[str, tuple[str, ...]], ...] | None = None
        if (raw_security := raw_openapi.get("security")) is not None:
            if not isinstance(raw_security, list):
                raise ContainerBuildError("auth.openapi.security must be an array.")
            parsed: list[Mapping[str, tuple[str, ...]]] = []
            for requirement in raw_security:
                if not isinstance(requirement, dict):
                    raise ContainerBuildError(
                        "auth.openapi.security requirements must be objects."
                    )
                item: dict[str, tuple[str, ...]] = {}
                for name, scopes in requirement.items():
                    if name not in (raw_schemes or {}):
                        raise ContainerBuildError(
                            "auth.openapi.security references an unknown scheme."
                        )
                    parsed_scopes = _string_tuple(
                        scopes, location="auth.openapi.security scopes"
                    )
                    assert parsed_scopes is not None
                    item[name] = parsed_scopes
                parsed.append(MappingProxyType(item))
            security = tuple(parsed)
        openapi = AuthOpenApiManifestV1(security_schemes=schemes, security=security)
    policy = AuthPolicyManifestV1(
        openapi=openapi,
        disable_studio_auth=_optional_bool(
            raw.get("disable_studio_auth"), location="auth.disable_studio_auth"
        ),
    )
    return policy, auth_path


def _dependency_is_local(value: str) -> bool:
    if re.match(r"^[A-Za-z]:[\\/]", value):
        return True
    try:
        Requirement(value)
    except InvalidRequirement:
        pass
    else:
        return False
    if re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", value) or re.match(
        r"^[^/@\s]+@[^/:\s]+:", value
    ):
        return False
    return (
        value in {".", ".."}
        or value.startswith(("./", "../", "/", "~", ".\\", "..\\"))
        or "/" in value
        or "\\" in value
    )


def _dependency_error(message: str) -> ContainerBuildError:
    return ContainerBuildError(
        f"{message}; use pip_config_file for private dependencies."
    )


def _validate_dependency_fragment(fragment: str) -> None:
    if not fragment:
        return
    seen: set[str] = set()
    for component in fragment.split("&"):
        raw_name, separator, raw_value = component.partition("=")
        if not separator:
            raise _dependency_error("Dependency URL fragment components are invalid")
        name = unquote(raw_name)
        value = unquote(raw_value)
        if name in seen or any(
            ord(character) < 0x20 or ord(character) == 0x7F
            for character in name + value
        ):
            raise _dependency_error("Dependency URL fragment components are invalid")
        if re.search(
            r"(?i)(?:^|[?&#;/])(?:token|auth|password|passwd|secret|credential|api[_-]?key)=",
            value,
        ):
            raise _dependency_error(
                "Dependency URL fragment contains credential-like data"
            )
        seen.add(name)
        if name == "sha256":
            if not re.fullmatch(r"[0-9a-fA-F]{64}", value):
                raise _dependency_error("Dependency URL sha256 fragment is invalid")
            continue
        if name == "subdirectory":
            normalized = value.replace("\\", "/")
            path = PurePosixPath(normalized)
            if (
                not normalized
                or normalized.startswith("/")
                or any(part in {"", ".", ".."} for part in normalized.split("/"))
                or path.as_posix() != normalized
            ):
                raise _dependency_error(
                    "Dependency URL subdirectory fragment must be normalized and relative"
                )
            continue
        raise _dependency_error("Dependency URL fragment component is not supported")


def validate_dependency_specification(value: str) -> None:
    """Validate the V1 dependency grammar without disclosing the operand."""

    if _dependency_is_local(value):
        return
    try:
        requirement = Requirement(value)
    except InvalidRequirement:
        requirement = None
    candidate = requirement.url if requirement is not None else None
    if candidate is None and re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", value):
        candidate = value
    if candidate is None:
        if requirement is not None:
            return
        raise ContainerBuildError("A dependency is not valid PEP 508 or HTTPS.")

    decoded_candidate = unquote(candidate)
    if any(
        ord(character) < 0x20 or ord(character) == 0x7F
        for character in decoded_candidate
    ):
        raise _dependency_error("Dependency URL contains a control character")
    try:
        parsed = urlsplit(candidate)
        decoded = urlsplit(decoded_candidate)
        credentials_present = any(
            part is not None
            for part in (
                parsed.username,
                parsed.password,
                decoded.username,
                decoded.password,
            )
        )
    except ValueError as exc:
        raise _dependency_error("Dependency URL is invalid") from exc
    if credentials_present:
        raise _dependency_error("Dependency URLs cannot contain credentials")
    if parsed.query or decoded.query:
        raise _dependency_error("Dependency URL query parameters are not permitted")
    if parsed.scheme.lower() != "https" or not parsed.netloc:
        raise _dependency_error("Dependency URLs must use direct HTTPS artifacts")
    _validate_dependency_fragment(parsed.fragment)
    if decoded.fragment != parsed.fragment:
        _validate_dependency_fragment(decoded.fragment)


def _validate_requirement_url(value: str) -> None:
    validate_dependency_specification(value)


def _check_runtime_requirement(text: str, *, location: str) -> None:
    try:
        requirement = Requirement(text)
    except InvalidRequirement:
        return
    if requirement.name.lower().replace("_", "-") != "agentseek-api":
        return
    if requirement.specifier and not requirement.specifier.contains(
        Version(_RUNTIME_VERSION), prereleases=True
    ):
        raise ContainerBuildError(
            f"{location} excludes agentseek-api 0.3.0; migrate the project runtime pin to 0.3.0."
        )


def _check_static_metadata(root: Path) -> None:
    requirements = root / "requirements.txt"
    if requirements.is_file():
        for line in requirements.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped.startswith(("-r ", "--requirement ", "-r=", "--requirement=")):
                raise ContainerBuildError(
                    "Nested requirements files are ambiguous; consolidate requirements.txt."
                )
            if stripped.startswith(
                (
                    "--index-url",
                    "--extra-index-url",
                    "--find-links",
                    "-e ",
                    "--editable ",
                )
            ):
                _, _, option_value = stripped.replace("=", " ", 1).partition(" ")
                _validate_requirement_url(option_value.strip())
            elif not stripped.startswith("-"):
                _validate_requirement_url(stripped)
                _check_runtime_requirement(stripped, location="requirements.txt")
    pyproject = root / "pyproject.toml"
    if pyproject.is_file():
        try:
            payload = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, tomllib.TOMLDecodeError) as exc:
            raise ContainerBuildError(
                "The selected pyproject.toml is invalid."
            ) from exc
        project = payload.get("project")
        if isinstance(project, dict) and "dependencies" not in set(
            project.get("dynamic", [])
        ):
            dependencies = project.get("dependencies", [])
            if isinstance(dependencies, list):
                for dependency in dependencies:
                    if isinstance(dependency, str):
                        _validate_requirement_url(dependency)
                        _check_runtime_requirement(
                            dependency, location="pyproject.toml"
                        )


def _classify_local_dependency(
    path: Path, *, project_root: Path
) -> tuple[InstallAction, bool]:
    _safe_directory(path, project_root=project_root, purpose="dependency")
    _check_static_metadata(path)
    container = _container_path(path.relative_to(project_root))
    if (path / "pyproject.toml").is_file() or (path / "setup.py").is_file():
        return InstallAction(InstallActionKind.PROJECT, container), False
    requirements = sorted(path.glob("requirements*.txt"))
    if len(requirements) > 1 and not (path / "requirements.txt").is_file():
        raise ContainerBuildError("A dependency has ambiguous requirements files.")
    if (path / "requirements.txt").is_file():
        return (
            InstallAction(
                InstallActionKind.REQUIREMENTS,
                _container_path((path / "requirements.txt").relative_to(project_root)),
            ),
            False,
        )
    return InstallAction(InstallActionKind.SOURCE_ONLY, container), True


def _wheel_metadata_bytes(data: bytes) -> tuple[str, str]:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            matches = [
                name
                for name in archive.namelist()
                if name.endswith(".dist-info/METADATA")
            ]
            if len(matches) != 1:
                raise ContainerBuildError("The candidate wheel metadata is ambiguous.")
            text = archive.read(matches[0]).decode("utf-8")
    except (OSError, UnicodeError, zipfile.BadZipFile, KeyError) as exc:
        raise ContainerBuildError(
            "The candidate wheel metadata could not be read."
        ) from exc
    fields: dict[str, str] = {}
    for line in text.splitlines():
        if ":" in line:
            name, _, value = line.partition(":")
            fields.setdefault(name.lower(), value.strip())
    return fields.get("name", ""), fields.get("version", "")


def candidate_runtime_artifact(
    wheel_path: Path, expected_sha256: str
) -> RuntimeArtifactV1:
    wheel = Path(wheel_path).absolute()
    try:
        status = wheel.lstat()
    except OSError as exc:
        raise ContainerBuildError("The candidate wheel is missing.") from exc
    if stat.S_ISLNK(status.st_mode) or not stat.S_ISREG(status.st_mode):
        raise ContainerBuildError("The candidate wheel must be a regular file.")
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ContainerBuildError("The candidate wheel SHA-256 is invalid.")
    data = _read_regular_source(wheel, expected_identity=_file_identity(status))
    if hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ContainerBuildError("The candidate wheel SHA-256 does not match.")
    name, version = _wheel_metadata_bytes(data)
    if name.lower().replace("_", "-") != "agentseek-api" or version != _RUNTIME_VERSION:
        raise ContainerBuildError("The candidate wheel identity is incompatible.")
    return RuntimeArtifactV1(
        distribution="agentseek-api",
        extra="embedded",
        version="0.3.0",
        source=RuntimeArtifactSource.CANDIDATE_WHEEL,
        candidate_wheel=wheel,
        candidate_sha256=expected_sha256,
        candidate_identity=_file_identity(status),
    )


def _file_identity(status: os.stat_result) -> tuple[int, int, int, int]:
    return (status.st_dev, status.st_ino, status.st_size, status.st_mtime_ns)


def _load_config(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContainerBuildError(
            "The container config is missing or invalid JSON."
        ) from exc
    if not isinstance(value, dict):
        raise ContainerBuildError("The container config must be an object.")
    return value


def _discover_project_root(config: Path) -> Path:
    """Use the nearest installable/VCS ancestor, otherwise the config directory."""

    start = config.parent.resolve()
    for candidate in (start, *start.parents):
        if any(
            (candidate / marker).exists()
            for marker in ("pyproject.toml", "setup.py", ".git")
        ):
            return candidate
    return start


def _logical_runtime_reference(
    reference: object,
    *,
    config_path: Path | None,
) -> str | None:
    if not isinstance(reference, str):
        return None
    module, separator, symbol = reference.rpartition(":")
    if not separator:
        return reference
    if module.startswith("/deps/agent/"):
        return f"{module.removeprefix('/deps/agent/')}:{symbol}"
    if config_path is not None and _is_path_reference(reference):
        path = Path(module).expanduser()
        if not path.is_absolute():
            path = config_path.parent / path
        resolved = path.resolve()
        project_root = _discover_project_root(config_path)
        try:
            relative = resolved.relative_to(project_root)
        except ValueError:
            return reference
        return f"{relative.as_posix()}:{symbol}"
    return reference


def _effective_runtime_policy(
    document: Mapping[str, object],
    *,
    config_path: Path | None,
) -> EffectiveRuntimePolicyV1:
    raw_http = document.get("http")
    http = raw_http if isinstance(raw_http, Mapping) else {}
    raw_cors = http.get("cors")
    cors = raw_cors if isinstance(raw_cors, Mapping) else {}
    origins = cors.get("allow_origins", ["*"])
    allow_credentials = cors.get("allow_credentials", origins not in (["*"], "*"))
    cors_middleware: dict[str, JsonValue] = {
        "allow_origins": _freeze_json(origins, location="http.cors.allow_origins"),
        "allow_credentials": bool(allow_credentials),
        "allow_methods": _freeze_json(
            cors.get("allow_methods", ["*"]), location="http.cors.allow_methods"
        ),
        "allow_headers": _freeze_json(
            cors.get("allow_headers", ["*"]), location="http.cors.allow_headers"
        ),
        "allow_origin_regex": _freeze_json(
            cors.get("allow_origin_regex"), location="http.cors.allow_origin_regex"
        ),
        "expose_headers": _freeze_json(
            cors.get("expose_headers", ["Content-Location", "Location"]),
            location="http.cors.expose_headers",
        ),
        "max_age": _freeze_json(cors.get("max_age", 600), location="http.cors.max_age"),
    }
    raw_auth = document.get("auth")
    auth = raw_auth if isinstance(raw_auth, Mapping) else {}
    raw_openapi = auth.get("openapi")
    openapi = (
        _freeze_json(dict(raw_openapi), location="auth.openapi")
        if isinstance(raw_openapi, Mapping)
        else None
    )
    return EffectiveRuntimePolicyV1(
        mcp_enabled=http.get("disable_mcp") is not True,
        a2a_enabled=http.get("disable_a2a") is not True,
        cors_middleware=cors_middleware,
        custom_app=_logical_runtime_reference(http.get("app"), config_path=config_path),
        auth_openapi=openapi,  # type: ignore[arg-type]
        studio_auth_disabled=auth.get("disable_studio_auth") is True,
    )


def interpret_host_runtime_policy(
    payload: Mapping[str, object], *, config_path: Path
) -> EffectiveRuntimePolicyV1:
    """Interpret released host config through the V1 policy comparison seam."""

    return _effective_runtime_policy(payload, config_path=config_path)


def interpret_manifest_runtime_policy(
    manifest: ContainerRuntimeManifestV1,
) -> EffectiveRuntimePolicyV1:
    """Interpret the sanitized manifest through the same effective-policy seam."""

    return _effective_runtime_policy(manifest.to_json_object(), config_path=None)


def plan_container_image(
    *,
    config_path: Path,
    dotenv_paths: Sequence[Path] = (),
    build_include: Sequence[str] | None = None,
    base_image_override: str | None = None,
    runtime_artifact: RuntimeArtifactV1 = PUBLISHED_RUNTIME_ARTIFACT,
    invocation_cwd: Path | None = None,
) -> ContainerBuildPlan:
    config = Path(config_path).absolute()
    if invocation_cwd is None:
        project_root = _discover_project_root(config)
        resolved_invocation_cwd = project_root
    else:
        resolved_invocation_cwd = Path(invocation_cwd).resolve()
        try:
            invocation_status = resolved_invocation_cwd.lstat()
        except OSError as exc:
            raise ContainerBuildError("The invocation cwd is missing.") from exc
        if stat.S_ISLNK(invocation_status.st_mode) or not stat.S_ISDIR(
            invocation_status.st_mode
        ):
            raise ContainerBuildError("The invocation cwd must be a directory.")
        project_root = resolved_invocation_cwd
    reference_base = config.parent.resolve()
    config = _safe_regular(config, project_root=project_root, purpose="config")
    payload = _load_config(config)
    raw_env = payload.get("env")
    configured_dotenv: list[Path] = []
    if isinstance(raw_env, str):
        path = Path(raw_env).expanduser()
        if not path.is_absolute():
            path = reference_base / path
        configured_dotenv.append(path)
    all_dotenv = [*configured_dotenv, *(Path(item) for item in dotenv_paths)]
    resolved_dotenv: list[Path] = []
    for dotenv in all_dotenv:
        path = dotenv if dotenv.is_absolute() else project_root / dotenv
        try:
            parse_dotenv_file(path, ambient={})
        except DotenvFileError as exc:
            raise ContainerBuildError(str(exc)) from exc
        resolved_dotenv.append(path.resolve())
    raw_pip = payload.get("pip_config_file")
    pip_path: Path | None = None
    pip_identity: tuple[int, int, int, int] | None = None
    if raw_pip is not None:
        if not isinstance(raw_pip, str) or not raw_pip.strip():
            raise ContainerBuildError("pip_config_file must be a non-empty string.")
        pip_candidate = Path(raw_pip).expanduser()
        if not pip_candidate.is_absolute():
            pip_candidate = reference_base / pip_candidate
        pip_path, pip_identity = _pip_config_source(pip_candidate)
    candidate_source: Path | None = None
    if runtime_artifact.source is RuntimeArtifactSource.CANDIDATE_WHEEL:
        assert runtime_artifact.candidate_wheel is not None
        assert runtime_artifact.candidate_sha256 is not None
        candidate_source = _safe_regular(
            runtime_artifact.candidate_wheel,
            project_root=project_root,
            purpose="candidate runtime artifact",
        )
        rechecked_artifact = candidate_runtime_artifact(
            candidate_source, runtime_artifact.candidate_sha256
        )
        if rechecked_artifact.candidate_identity != runtime_artifact.candidate_identity:
            raise ContainerBuildError(
                "The candidate wheel identity changed before planning."
            )
    excluded = frozenset(
        {
            config,
            *resolved_dotenv,
            *(() if pip_path is None else (pip_path,)),
            *(() if candidate_source is None else (candidate_source,)),
        }
    )

    selected: dict[str, SelectedSource] = {}
    graphs = _parse_graphs(
        payload.get("graphs"),
        reference_base=reference_base,
        project_root=project_root,
        selected=selected,
        excluded=excluded,
    )
    store = _parse_store(
        payload.get("store"),
        reference_base=reference_base,
        project_root=project_root,
        selected=selected,
        excluded=excluded,
    )
    http = _parse_http(
        payload.get("http"),
        reference_base=reference_base,
        project_root=project_root,
        selected=selected,
        excluded=excluded,
    )
    auth, dedicated_auth = _parse_auth(payload.get("auth"))

    static_auth: str | None = dedicated_auth
    static_auth_base = reference_base
    if static_auth is None and isinstance(raw_env, dict):
        value = raw_env.get("AUTH_MODULE_PATH")
        if isinstance(value, str):
            static_auth = value
            static_auth_base = resolved_invocation_cwd
    if static_auth and _is_path_reference(static_auth):
        located = _module_file(static_auth, base=static_auth_base)
        if located is not None:
            auth_file, _ = located
            _select_file(
                selected,
                source=auth_file,
                project_root=project_root,
                reason=SourceReason.AUTH,
                excluded=excluded,
            )

    includes = (
        tuple(payload.get("build_include", ()))
        if build_include is None
        else tuple(build_include)
    )
    if not all(isinstance(item, str) for item in includes):
        raise ContainerBuildError("build_include must contain strings.")
    for item in includes:
        candidate = Path(item).expanduser()
        if not candidate.is_absolute():
            candidate = reference_base / candidate
        try:
            status = candidate.absolute().lstat()
        except OSError as exc:
            raise ContainerBuildError(
                "A build_include source is missing from the project."
            ) from exc
        if stat.S_ISDIR(status.st_mode):
            _select_tree(
                selected,
                root=candidate,
                project_root=project_root,
                reason=SourceReason.BUILD_INCLUDE,
                excluded=excluded,
            )
        else:
            _select_file(
                selected,
                source=candidate,
                project_root=project_root,
                reason=SourceReason.BUILD_INCLUDE,
                excluded=excluded,
            )

    raw_dependencies = payload.get("dependencies", [])
    if not isinstance(raw_dependencies, list) or not all(
        isinstance(item, str) and item for item in raw_dependencies
    ):
        raise ContainerBuildError("dependencies must be an array of non-empty strings.")
    actions: list[InstallAction] = []
    runtime_roots: list[str] = []
    for dependency in raw_dependencies:
        assert isinstance(dependency, str)
        if _dependency_is_local(dependency):
            local = Path(dependency).expanduser()
            if not local.is_absolute():
                local = reference_base / local
            local = _safe_directory(
                local, project_root=project_root, purpose="dependency"
            )
            action, needs_runtime_path = _classify_local_dependency(
                local, project_root=project_root
            )
            actions.append(action)
            _select_tree(
                selected,
                root=local,
                project_root=project_root,
                reason=SourceReason.DEPENDENCY,
                excluded=excluded,
            )
            if needs_runtime_path and action.operand not in runtime_roots:
                runtime_roots.append(action.operand)
        else:
            _validate_requirement_url(dependency)
            _check_runtime_requirement(dependency, location="dependencies")
            try:
                Requirement(dependency)
            except InvalidRequirement:
                if not dependency.startswith("https://"):
                    raise ContainerBuildError(
                        "A dependency is not valid PEP 508 or HTTPS."
                    )
            actions.append(InstallAction(InstallActionKind.PEP508, dependency))

    if runtime_artifact.source is RuntimeArtifactSource.CANDIDATE_WHEEL:
        assert candidate_source is not None
        destination = f"runtime/{candidate_source.name}"
        _add_selected(
            selected,
            destination=destination,
            source=candidate_source,
            reason=SourceReason.RUNTIME_ARTIFACT,
            project_root=project_root,
        )

    raw_base = payload.get("base_image")
    raw_python = payload.get("python_version", "3.12")
    raw_distro = payload.get("image_distro", "debian")
    raw_lines = payload.get("dockerfile_lines", [])
    if raw_base is not None and not isinstance(raw_base, str):
        raise ContainerBuildError("base_image must be a string.")
    if not isinstance(raw_python, str) or not isinstance(raw_distro, str):
        raise ContainerBuildError(
            "Container Python and distro settings must be strings."
        )
    if not isinstance(raw_lines, list) or not all(
        isinstance(line, str) for line in raw_lines
    ):
        raise ContainerBuildError("dockerfile_lines must be an array of strings.")
    if base_image_override or raw_base:
        base = base_image_override or raw_base
        assert base is not None
    elif raw_distro in {"", "debian"}:
        base = f"python:{raw_python}-slim"
    elif raw_distro in {"bookworm", "bullseye"}:
        base = f"python:{raw_python}-slim-{raw_distro}"
    else:
        raise ContainerBuildError(
            "image_distro is not supported without an explicit base_image."
        )
    manifest = ContainerRuntimeManifestV1(
        schema_version=1,
        runtime=RuntimeManifestV1(
            distribution="agentseek-api", version="0.3.0", contract="preloaded-v1"
        ),
        graphs=graphs,
        dependencies=tuple(runtime_roots),
        store=store,
        http=http,
        auth=auth,
    )
    return ContainerBuildPlan(
        base_image=base,
        python_version=raw_python,
        image_distro=raw_distro,
        dockerfile_lines=tuple(raw_lines),
        runtime_artifact=runtime_artifact,
        install_actions=tuple(actions),
        pip_config_file=pip_path,
        manifest=manifest,
        selected_sources=selected,
        config_path=config,
        project_root=project_root,
        project_root_identity=_directory_identity(project_root.lstat()),
        invocation_cwd=resolved_invocation_cwd,
        excluded_paths=excluded,
        pip_config_identity=pip_identity,
    )


def _without_auth_reasons(plan: ContainerBuildPlan) -> dict[str, SelectedSource]:
    selected: dict[str, SelectedSource] = {}
    for destination, source in plan.selected_sources.items():
        reasons = source.reasons - frozenset({SourceReason.AUTH})
        if reasons:
            selected[destination] = SelectedSource(
                source.source_path,
                reasons,
                source_identity=source.source_identity,
                source_sha256=source.source_sha256,
                ancestor_identities=source.ancestor_identities,
            )
    return selected


def plan_generated_up_auth(
    plan: ContainerBuildPlan, selection: FinalAuthSelection | None
) -> tuple[ContainerBuildPlan, AuthPayloadPatch | None]:
    selected = _without_auth_reasons(plan)
    if selection is None:
        return replace(plan, selected_sources=selected), None
    if selection.value == "":
        return replace(plan, selected_sources=selected), AuthPayloadPatch("")
    if not _is_path_reference(selection.value):
        return replace(plan, selected_sources=selected), AuthPayloadPatch(
            selection.value
        )
    auth_base = (
        plan.config_path.parent
        if selection.origin.source_kind == "auth"
        else plan.invocation_cwd
    )
    located = _module_file(selection.value, base=auth_base)
    if located is None:
        return replace(plan, selected_sources=selected), AuthPayloadPatch(
            selection.value
        )
    source, symbol = located
    source = _select_file(
        selected,
        source=source,
        project_root=plan.project_root,
        reason=SourceReason.AUTH,
        excluded=plan.excluded_paths,
    )
    rewritten = f"{_container_path(source.relative_to(plan.project_root))}:{symbol}"
    return replace(plan, selected_sources=selected), AuthPayloadPatch(rewritten)


def _docker_exec_run(argv: Sequence[str], *, pip_secret: bool = False) -> str:
    mount = (
        "--mount=type=secret,id=pip_config,target=/etc/pip.conf " if pip_secret else ""
    )
    return f"RUN {mount}{json.dumps(list(argv), ensure_ascii=False)}"


def _pip_install_argv(action: InstallAction) -> tuple[str, ...] | None:
    base = (
        "python",
        "-m",
        "pip",
        "install",
        "--no-cache-dir",
        "--constraint",
        "/opt/agentseek/runtime-constraints.txt",
    )
    if action.kind is InstallActionKind.PROJECT:
        return (*base, action.operand)
    if action.kind is InstallActionKind.REQUIREMENTS:
        return (*base, "--requirement", action.operand)
    if action.kind is InstallActionKind.PEP508:
        return (*base, action.operand)
    if action.kind is InstallActionKind.SOURCE_ONLY:
        return None
    raise ContainerBuildError("The build plan contains an unsupported install action.")


def _candidate_destination(plan: ContainerBuildPlan) -> str | None:
    if plan.runtime_artifact.source is not RuntimeArtifactSource.CANDIDATE_WHEEL:
        return None
    candidates = [
        destination
        for destination, source in plan.selected_sources.items()
        if SourceReason.RUNTIME_ARTIFACT in source.reasons
    ]
    if len(candidates) != 1 or not candidates[0].startswith("runtime/"):
        raise ContainerBuildError(
            "The candidate runtime artifact selection is invalid."
        )
    return candidates[0]


def render_build_dockerfile(plan: ContainerBuildPlan) -> bytes:
    """Render final Dockerfile bytes using only the immutable build plan."""

    if (
        not plan.base_image
        or any(character.isspace() for character in plan.base_image)
        or "\0" in plan.base_image
    ):
        raise ContainerBuildError("The build plan base image is invalid.")
    has_app = any(
        destination == "app" or destination.startswith("app/")
        for destination in plan.selected_sources
    )
    pip_secret = plan.pip_config_file is not None
    candidate_source = _candidate_destination(plan)
    artifact = plan.runtime_artifact
    manifest_sha256 = hashlib.sha256(plan.manifest.to_json_bytes()).hexdigest()

    lines = [
        "# syntax=docker/dockerfile:1.7",
        f"FROM {plan.base_image}",
        f"# agentseek-python-version={json.dumps(plan.python_version)}",
        f"# agentseek-image-distro={json.dumps(plan.image_distro)}",
        "ENV PYTHONDONTWRITEBYTECODE=1",
        "ENV PYTHONUNBUFFERED=1",
        "WORKDIR /deps/agent",
    ]
    if has_app:
        lines.append("COPY app /deps/agent")
    lines.append("COPY runtime-constraints.txt /opt/agentseek/runtime-constraints.txt")
    if candidate_source is not None:
        lines.append(
            "COPY "
            + json.dumps(
                [candidate_source, "/opt/agentseek/runtime/agentseek-api-0.3.0.whl"],
                ensure_ascii=False,
            )
        )

    for action in plan.install_actions:
        argv = _pip_install_argv(action)
        if argv is not None:
            lines.append(_docker_exec_run(argv, pip_secret=pip_secret))
    lines.extend(plan.dockerfile_lines)

    if candidate_source is not None:
        if artifact.candidate_sha256 is None:
            raise ContainerBuildError(
                "The candidate runtime artifact selection is invalid."
            )
        candidate_check = (
            "import hashlib,pathlib;"
            "p=pathlib.Path('/opt/agentseek/runtime/agentseek-api-0.3.0.whl');"
            "raise SystemExit('candidate runtime hash mismatch') if "
            f"hashlib.sha256(p.read_bytes()).hexdigest()!='{artifact.candidate_sha256}' "
            "else None"
        )
        lines.append(_docker_exec_run(("python", "-c", candidate_check)))
        runtime_operand = "/opt/agentseek/runtime/agentseek-api-0.3.0.whl[embedded]"
    else:
        runtime_operand = artifact.requirement
    lines.append(
        _docker_exec_run(
            (
                "python",
                "-m",
                "pip",
                "install",
                "--no-cache-dir",
                "--force-reinstall",
                "--constraint",
                "/opt/agentseek/runtime-constraints.txt",
                runtime_operand,
            ),
            pip_secret=pip_secret,
        )
    )
    lines.append("COPY manifest.v1.json /opt/agentseek/manifest.v1.json")
    manifest_check = (
        "import hashlib,json,pathlib;"
        "p=pathlib.Path('/opt/agentseek/manifest.v1.json');raw=p.read_bytes();"
        "doc=json.loads(raw);canonical=(json.dumps(doc,ensure_ascii=False,sort_keys=True,"
        "separators=(',',':'),allow_nan=False)+'\\n').encode();"
        "raise SystemExit('runtime manifest integrity mismatch') if "
        f"raw!=canonical or hashlib.sha256(raw).hexdigest()!='{manifest_sha256}' "
        "else None"
    )
    lines.append(_docker_exec_run(("python", "-c", manifest_check)))
    lines.append(_docker_exec_run(("python", "-m", "pip", "check")))
    runtime_check = (
        "import importlib.metadata,pathlib,sys,sysconfig,agentseek_api.cli;"
        "distribution=importlib.metadata.distribution('agentseek-api');"
        "raise SystemExit('runtime distribution version mismatch') if "
        f"distribution.version!='{artifact.version}' else None;"
        "module=pathlib.Path(agentseek_api.cli.__file__).resolve();"
        "files=distribution.files;"
        "raise SystemExit('runtime distribution file inventory missing') if "
        "files is None else None;"
        "owned={pathlib.Path(distribution.locate_file(item)).resolve() for item in files};"
        "raise SystemExit('runtime module is not owned by distribution') if "
        "module not in owned else None;"
        "roots={pathlib.Path(value).resolve() for key,value in sysconfig.get_paths().items() "
        "if key in {'purelib','platlib'}};"
        "raise SystemExit('runtime module is outside site packages') if "
        "not roots or not any(module.is_relative_to(root) for root in roots) else None;"
        "raise SystemExit('Python 3.12 or newer is required') if "
        "sys.version_info[:2]<(3,12) else None"
    )
    lines.append(_docker_exec_run(("python", "-c", runtime_check)))
    lines.extend(
        (
            "LABEL org.agentseek.environment-contract=preloaded-v1",
            "LABEL org.agentseek.runtime-manifest=/opt/agentseek/manifest.v1.json",
            f"LABEL org.agentseek.runtime-distribution={artifact.distribution}",
            f"LABEL org.agentseek.runtime-version={artifact.version}",
            "ENTRYPOINT []",
            "CMD "
            + json.dumps(
                [
                    "python",
                    "-m",
                    "agentseek_api.cli",
                    "serve",
                    "--environment-mode",
                    "preloaded-v1",
                    "--host",
                    "0.0.0.0",
                    "--port",
                    str(DEFAULT_API_PORT),
                ]
            ),
        )
    )
    return ("\n".join(lines) + "\n").encode("utf-8")


def _write_file(path: Path, data: bytes) -> None:
    target = path.absolute()
    if os.name == "nt":  # pragma: no cover - native Windows only
        current = Path(target.anchor)
        for part in target.parts[1:-1]:
            current /= part
            try:
                status = current.lstat()
            except FileNotFoundError:
                current.mkdir(mode=0o700)
                status = current.lstat()
            expected = _directory_identity(status)
            if not _directory_identity_matches(current, expected):
                raise ContainerBuildError(
                    "An output ancestor changed or became unsafe."
                )
        _write_file_descriptor(os.open(target, _output_file_flags(), 0o600), data)
        return

    with _opened_output_parent(target, anchor=Path(target.anchor), create=False) as (
        directory_fd,
        _,
    ):
        file_fd = os.open(
            target.name,
            _output_file_flags(),
            0o600,
            dir_fd=directory_fd,
        )
        _write_file_descriptor(file_fd, data)


def _prepare_output_ancestors(
    path: Path, *, anchor: Path
) -> dict[Path, tuple[int, int]]:
    target = path.absolute()
    root = anchor.absolute()
    try:
        relative_parent = target.parent.relative_to(root)
    except ValueError as exc:
        raise ContainerBuildError("A build output escaped its verified root.") from exc

    identities: dict[Path, tuple[int, int]] = {}
    if os.name == "nt":  # pragma: no cover - native Windows only
        current = root
        for part in ("", *relative_parent.parts):
            if part:
                current /= part
            try:
                status = current.lstat()
            except FileNotFoundError:
                current.mkdir(mode=0o700)
                status = current.lstat()
            identity = _directory_identity(status)
            if not _directory_identity_matches(current, identity):
                raise ContainerBuildError(
                    "A build output ancestor changed or became unsafe."
                )
            identities[current] = identity
        return identities

    with _opened_output_parent(target, anchor=root, create=True) as (
        _,
        identities,
    ):
        return identities


@contextmanager
def _opened_output_parent(
    target: Path, *, anchor: Path, create: bool
) -> Iterator[tuple[int, dict[Path, tuple[int, int]]]]:
    try:
        relative_parent = target.parent.relative_to(anchor)
    except ValueError as exc:
        raise ContainerBuildError("A build output escaped its verified root.") from exc

    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    descriptors: list[int] = []
    identities: dict[Path, tuple[int, int]] = {}
    try:
        directory_fd = os.open(anchor, directory_flags)
        descriptors.append(directory_fd)
        root_status = os.fstat(directory_fd)
        identities[anchor] = _directory_identity(root_status)
        current = anchor
        for part in relative_parent.parts:
            try:
                child_fd = os.open(part, directory_flags, dir_fd=directory_fd)
            except FileNotFoundError:
                if not create:
                    raise
                os.mkdir(part, 0o700, dir_fd=directory_fd)
                child_fd = os.open(part, directory_flags, dir_fd=directory_fd)
            child_status = os.fstat(child_fd)
            if not stat.S_ISDIR(child_status.st_mode):
                os.close(child_fd)
                raise ContainerBuildError(
                    "A build output ancestor changed or became unsafe."
                )
            current /= part
            identities[current] = _directory_identity(child_status)
            descriptors.append(child_fd)
            directory_fd = child_fd
        yield directory_fd, identities
    except ContainerBuildError:
        raise
    except OSError as exc:
        raise ContainerBuildError(
            "A build output ancestor changed or became unsafe."
        ) from exc
    finally:
        for descriptor in reversed(descriptors):
            try:
                os.close(descriptor)
            except OSError:
                pass


def _output_file_flags() -> int:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    return flags


def _write_file_descriptor(fd: int, data: bytes) -> None:
    try:
        remaining = memoryview(data)
        while remaining:
            written = os.write(fd, remaining)
            if written <= 0:
                raise ContainerBuildError("Could not write the build bundle.")
            remaining = remaining[written:]
        os.fsync(fd)
    finally:
        os.close(fd)


def _read_regular_source(
    path: Path,
    *,
    expected_identity: tuple[int, int, int, int] | None = None,
    identity_error: str = "A selected build source identity changed before materialization.",
) -> bytes:
    try:
        before = path.lstat()
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
            raise ContainerBuildError("A selected build source changed before copy.")
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(path, flags)
    except ContainerBuildError:
        raise
    except OSError as exc:
        raise ContainerBuildError(
            "A selected build source changed before copy."
        ) from exc
    try:
        opened = os.fstat(fd)
        opened_identity = _file_identity(opened)
        if opened_identity != _file_identity(before):
            raise ContainerBuildError(
                "A selected build source identity changed before copy."
            )
        if expected_identity is not None and opened_identity != expected_identity:
            raise ContainerBuildError(identity_error)
        chunks: list[bytes] = []
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(fd)


def materialize_build_bundle(
    plan: ContainerBuildPlan,
    *,
    dockerfile_bytes: bytes,
    output_root: Path,
) -> ContainerBuildBundle:
    try:
        if _directory_identity(plan.project_root.lstat()) != plan.project_root_identity:
            raise ContainerBuildError(
                "The project root identity changed before materialization."
            )
    except OSError as exc:
        raise ContainerBuildError(
            "The project root changed before materialization."
        ) from exc
    requested_root = Path(output_root).absolute()
    try:
        root_status = requested_root.lstat()
    except FileNotFoundError:
        root_status = None
    except OSError as exc:
        raise ContainerBuildError(
            "The build output root could not be verified."
        ) from exc
    root_was_created = root_status is None
    if root_status is not None:
        if (
            stat.S_ISLNK(root_status.st_mode)
            or not stat.S_ISDIR(root_status.st_mode)
            or any(requested_root.iterdir())
        ):
            raise ContainerBuildError(
                "The build output root must be a new empty directory."
            )
        root = requested_root.resolve(strict=True)
    else:
        root = requested_root.parent.resolve() / requested_root.name
        try:
            root.parent.mkdir(parents=True, exist_ok=True)
            create_private_directory(root)
        except (OSError, SecureArtifactError) as exc:
            raise ContainerBuildError(
                "The build output root could not be created."
            ) from exc
    _verify_private_output_root(root)
    root_identity = _directory_identity(root.lstat())
    context = root / "context"
    context.mkdir(mode=0o700)
    context_identity = _directory_identity(context.lstat())
    created: list[Path] = []
    frozen_outputs: dict[Path, bytes] = {}
    output_directory_identities = {
        root: root_identity,
        context: context_identity,
    }

    def verify_output_directories() -> None:
        for directory, expected in output_directory_identities.items():
            if directory == root:
                message = "The build output root identity changed."
            elif directory == context:
                message = "The build output context identity changed."
            else:
                message = "A build output ancestor identity changed."
            _verify_directory_identity(
                directory,
                expected=expected,
                message=message,
            )

    def freeze_output_ancestors(path: Path) -> None:
        try:
            relative_parent = path.parent.relative_to(root)
        except ValueError as exc:
            raise ContainerBuildError(
                "A build output escaped its verified root."
            ) from exc
        current = root
        for part in relative_parent.parts:
            current /= part
            try:
                status = current.lstat()
            except OSError as exc:
                raise ContainerBuildError("A build output ancestor changed.") from exc
            identity = _directory_identity(status)
            if not _directory_identity_matches(current, identity):
                raise ContainerBuildError(
                    "A build output ancestor changed or became unsafe."
                )
            expected = output_directory_identities.setdefault(current, identity)
            if identity != expected:
                raise ContainerBuildError("A build output ancestor identity changed.")

    def write_output(path: Path, data: bytes) -> None:
        verify_output_directories()
        for directory, identity in _prepare_output_ancestors(path, anchor=root).items():
            expected = output_directory_identities.setdefault(directory, identity)
            if identity != expected:
                raise ContainerBuildError("A build output ancestor identity changed.")
        verify_output_directories()
        _write_file(path, data)
        verify_output_directories()
        freeze_output_ancestors(path)
        verify_output_directories()
        frozen_outputs[path] = data

    try:
        for destination, selected in sorted(plan.selected_sources.items()):
            source = selected.source_path
            try:
                resolved_source = source.resolve(strict=True)
                resolved_source.relative_to(plan.project_root)
            except (OSError, ValueError) as exc:
                raise ContainerBuildError(
                    "A selected source escaped the project root before materialization."
                ) from exc
            if resolved_source != source:
                raise ContainerBuildError(
                    "A selected source gained an unsafe symlink before materialization."
                )
            for ancestor, expected_identity in selected.ancestor_identities:
                try:
                    current_identity = _directory_identity(ancestor.lstat())
                except OSError as exc:
                    raise ContainerBuildError(
                        "A selected source ancestor changed before materialization."
                    ) from exc
                if current_identity != expected_identity:
                    raise ContainerBuildError(
                        "A selected source ancestor identity changed before materialization."
                    )
            if selected.source_identity is None:
                raise ContainerBuildError("A selected source has no frozen identity.")
            if selected.source_sha256 is None:
                raise ContainerBuildError("A selected source has no frozen hash.")
            data = _read_regular_source(
                source,
                expected_identity=selected.source_identity,
                identity_error=(
                    "The candidate wheel identity changed before materialization."
                    if selected.reasons == frozenset({SourceReason.RUNTIME_ARTIFACT})
                    else "A selected build source identity changed before materialization."
                ),
            )
            if hashlib.sha256(data).hexdigest() != selected.source_sha256:
                raise ContainerBuildError(
                    "A selected build source hash changed before materialization."
                )
            if selected.reasons == frozenset({SourceReason.RUNTIME_ARTIFACT}):
                expected = plan.runtime_artifact.candidate_sha256
                if hashlib.sha256(data).hexdigest() != expected:
                    raise ContainerBuildError(
                        "The candidate wheel changed before materialization."
                    )
                name, version = _wheel_metadata_bytes(data)
                if (
                    name.lower().replace("_", "-") != "agentseek-api"
                    or version != _RUNTIME_VERSION
                ):
                    raise ContainerBuildError("The candidate wheel identity changed.")
            target = context / PurePosixPath(destination)
            write_output(target, data)
            created.append(target)
        manifest = context / "manifest.v1.json"
        write_output(manifest, plan.manifest.to_json_bytes())
        constraints = context / "runtime-constraints.txt"
        write_output(constraints, b"agentseek-api==0.3.0\n")
        dockerfile = context / "Dockerfile"
        write_output(dockerfile, bytes(dockerfile_bytes))
        inventory = tuple(
            BuildInventoryEntry(
                relative_path=path.relative_to(context).as_posix(),
                sha256=hashlib.sha256(frozen_outputs[path]).hexdigest(),
                size=len(frozen_outputs[path]),
            )
            for path in sorted(created + [manifest, constraints, dockerfile])
        )
        inventory_path = root / "inventory.json"
        write_output(
            inventory_path,
            (
                json.dumps(
                    [
                        {
                            "relative_path": entry.relative_path,
                            "sha256": entry.sha256,
                            "size": entry.size,
                        }
                        for entry in inventory
                    ],
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
                + b"\n"
            ),
        )
        return ContainerBuildBundle(
            root=root,
            context=context,
            dockerfile=dockerfile,
            manifest=manifest,
            inventory=inventory,
            plan_fingerprint=container_build_plan_fingerprint(plan),
        )
    except Exception:
        root_is_original = _directory_identity_matches(root, root_identity)
        context_is_original = root_is_original and _directory_identity_matches(
            context, context_identity
        )
        if context_is_original:
            shutil.rmtree(context, ignore_errors=True)
        inventory_path = root / "inventory.json"
        if (
            root_is_original
            and inventory_path.exists()
            and not inventory_path.is_symlink()
        ):
            try:
                inventory_path.unlink()
            except OSError:
                pass
        if root_was_created and root_is_original:
            try:
                root.rmdir()
            except OSError:
                pass
        raise


def _verify_private_output_root(root: Path) -> None:
    try:
        verify_private_directory(root)
    except SecureArtifactError as exc:
        raise ContainerBuildError(
            "The build output root could not be verified private."
        ) from exc


def _directory_identity_matches(path: Path, expected: tuple[int, int]) -> bool:
    try:
        status = path.lstat()
    except OSError:
        return False
    is_junction = getattr(path, "is_junction", None)
    return (
        stat.S_ISDIR(status.st_mode)
        and not stat.S_ISLNK(status.st_mode)
        and not (is_junction is not None and is_junction())
        and _directory_identity(status) == expected
    )


def _verify_directory_identity(
    path: Path,
    *,
    expected: tuple[int, int],
    message: str,
) -> None:
    if not _directory_identity_matches(path, expected):
        raise ContainerBuildError(message)


def _directory_identity(status: os.stat_result) -> tuple[int, int]:
    return (status.st_dev, status.st_ino)


def create_deterministic_context_archive(
    *, context: Path, expected_inventory: Sequence[BuildInventoryEntry]
) -> bytes:
    expected = {entry.relative_path: entry for entry in expected_inventory}
    actual_files: dict[str, Path] = {}
    for path in context.rglob("*"):
        status = path.lstat()
        if stat.S_ISLNK(status.st_mode) or (
            not stat.S_ISDIR(status.st_mode) and not stat.S_ISREG(status.st_mode)
        ):
            raise ContainerBuildError("The build context contains an unsafe entry.")
        if stat.S_ISREG(status.st_mode):
            actual_files[path.relative_to(context).as_posix()] = path
    if set(actual_files) != set(expected):
        raise ContainerBuildError("The build context inventory changed.")
    frozen_data: dict[str, bytes] = {}
    for name, entry in expected.items():
        path = actual_files[name]
        data = _read_regular_source(path)
        digest = hashlib.sha256(data).hexdigest()
        size = len(data)
        if digest != entry.sha256 or size != entry.size:
            raise ContainerBuildError("A build context file changed after inventory.")
        frozen_data[name] = data
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for name in sorted(expected):
            data = frozen_data[name]
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = 0o644
            info.uid = info.gid = info.mtime = 0
            info.uname = info.gname = ""
            archive.addfile(info, io.BytesIO(data))
    return output.getvalue()


__all__ = [
    "AuthOpenApiManifestV1",
    "AuthPayloadPatch",
    "AuthPolicyManifestV1",
    "BuildInventoryEntry",
    "ContainerBuildBundle",
    "ContainerBuildError",
    "ContainerBuildPlan",
    "ContainerRuntimeManifestV1",
    "CorsManifestV1",
    "EffectiveRuntimePolicyV1",
    "FinalAuthSelection",
    "HttpManifestV1",
    "InstallAction",
    "InstallActionKind",
    "PUBLISHED_RUNTIME_ARTIFACT",
    "RuntimeArtifactSource",
    "RuntimeArtifactV1",
    "RuntimeManifestV1",
    "SelectedSource",
    "SourceReason",
    "StoreIndexManifestV1",
    "StoreManifestV1",
    "StoreTtlManifestV1",
    "StructuredGraphV1",
    "candidate_runtime_artifact",
    "container_build_plan_fingerprint",
    "create_deterministic_context_archive",
    "interpret_host_runtime_policy",
    "interpret_manifest_runtime_policy",
    "materialize_build_bundle",
    "plan_container_image",
    "plan_generated_up_auth",
    "render_build_dockerfile",
    "validate_dependency_specification",
]
