from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import json
import os
import runpy
import sys
from collections.abc import Sequence
from pathlib import Path
from urllib.parse import unquote, urlparse

from pydantic import ValidationError


TARGET_MODULES = {
    "uvicorn": "uvicorn.__main__",
    "worker": "agentseek_api.worker",
    "scheduler": "agentseek_api.scheduler",
}

TARGET_DISTRIBUTIONS = {
    "uvicorn": ("uvicorn", "uvicorn", None),
    "worker": ("agentseek-api", "agentseek_api", "0.3.1"),
    "scheduler": ("agentseek-api", "agentseek_api", "0.3.1"),
}


class RuntimeBootstrapError(RuntimeError):
    pass


def _owned_distribution_locations(
    distribution_name: str,
    package_name: str,
    *,
    expected_version: str | None,
) -> tuple[frozenset[Path], tuple[Path, ...]]:
    try:
        distribution = importlib.metadata.distribution(distribution_name)
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeBootstrapError from exc
    if expected_version is not None and distribution.version != expected_version:
        raise RuntimeBootstrapError

    owned_files = frozenset(
        distribution.locate_file(item).resolve()
        for item in (distribution.files or ())
        if item.parts[:1] == (package_name,)
    )
    editable_roots: tuple[Path, ...] = ()
    try:
        direct_url = json.loads(distribution.read_text("direct_url.json") or "{}")
        url = direct_url.get("url")
        editable = direct_url.get("dir_info", {}).get("editable") is True
        if editable and isinstance(url, str):
            parsed = urlparse(url)
            if parsed.scheme == "file":
                checkout = Path(unquote(parsed.path)).resolve()
                editable_roots = tuple(
                    candidate.resolve()
                    for candidate in (
                        checkout / "src" / package_name,
                        checkout / package_name,
                    )
                    if candidate.is_dir()
                )
    except (json.JSONDecodeError, OSError, TypeError):
        editable_roots = ()
    if not owned_files and not editable_roots:
        raise RuntimeBootstrapError
    return owned_files, editable_roots


def _owned_runtime_locations() -> tuple[frozenset[Path], tuple[Path, ...]]:
    return _owned_distribution_locations(
        "agentseek-api", "agentseek_api", expected_version="0.3.1"
    )


def _is_owned_path(
    path: Path, *, owned_files: frozenset[Path], editable_roots: tuple[Path, ...]
) -> bool:
    return path in owned_files or any(
        path == root or root in path.parents for root in editable_roots
    )


def _require_distribution_owned_runtime() -> None:
    owned_files, editable_roots = _owned_runtime_locations()
    for module_name, module in tuple(sys.modules.items()):
        if module_name != "agentseek_api" and not module_name.startswith(
            "agentseek_api."
        ):
            continue
        module_file = getattr(module, "__file__", None)
        if module_file is None:
            continue
        path = Path(module_file).resolve()
        if _is_owned_path(path, owned_files=owned_files, editable_roots=editable_roots):
            continue
        raise RuntimeBootstrapError


def _require_trusted_target_module(target_name: str) -> None:
    distribution_name, package_name, expected_version = TARGET_DISTRIBUTIONS[
        target_name
    ]
    target_module = TARGET_MODULES[target_name]
    owned_files, editable_roots = _owned_distribution_locations(
        distribution_name,
        package_name,
        expected_version=expected_version,
    )
    try:
        spec = importlib.util.find_spec(target_module)
    except (ImportError, ModuleNotFoundError, ValueError) as exc:
        raise RuntimeBootstrapError from exc
    if spec is None or spec.origin is None:
        raise RuntimeBootstrapError
    if not _is_owned_path(
        Path(spec.origin).resolve(),
        owned_files=owned_files,
        editable_roots=editable_roots,
    ):
        raise RuntimeBootstrapError


def _activate_preloaded_runtime() -> None:
    from agentseek_api.container_build import (
        ContainerBuildError,
        load_container_runtime_manifest_v1,
    )

    manifest_value = os.environ.get("AGENTSEEK_GRAPHS")
    if not manifest_value:
        raise RuntimeBootstrapError
    try:
        manifest = load_container_runtime_manifest_v1(Path(manifest_value))
    except ContainerBuildError as exc:
        raise RuntimeBootstrapError from exc
    for dependency in manifest.dependencies:
        if dependency not in sys.path:
            sys.path.append(dependency)


def _format_settings_validation_error(exc: ValidationError) -> str:
    fields = sorted(
        {
            ".".join(str(part) for part in error["loc"]) + f" ({error['type']})"
            for error in exc.errors(include_input=False, include_url=False)
        }
    )
    return f"Invalid runtime setting(s): {', '.join(fields)}."


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    preloaded = arguments[:1] == ["--preloaded-v1"]
    if preloaded:
        arguments = arguments[1:]
    if not arguments or arguments[0] not in TARGET_MODULES:
        sys.stderr.write("Invalid internal runtime target.\n")
        return 2
    target_name, *target_argv = arguments
    if target_argv[:1] == ["--"]:
        target_argv = target_argv[1:]
    target_module = TARGET_MODULES[target_name]
    previous_argv = sys.argv
    sys.argv = [target_module, *target_argv]
    try:
        try:
            if preloaded:
                _require_distribution_owned_runtime()
                _require_trusted_target_module(target_name)
                _activate_preloaded_runtime()
            importlib.import_module("agentseek_api.settings")
            if preloaded:
                _require_distribution_owned_runtime()
        except ValidationError as exc:
            sys.stderr.write(_format_settings_validation_error(exc) + "\n")
            return 2
        except RuntimeBootstrapError:
            sys.stderr.write("The preloaded runtime identity is incompatible.\n")
            return 2
        try:
            runpy.run_module(target_module, run_name="__main__")
        except SystemExit as exc:
            return (
                exc.code
                if isinstance(exc.code, int)
                else (0 if exc.code is None else 1)
            )
        return 0
    finally:
        sys.argv = previous_argv


if __name__ == "__main__":
    raise SystemExit(main())
