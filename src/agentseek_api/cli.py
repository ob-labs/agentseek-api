from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from urllib import error as urllib_error
from urllib import request as urllib_request
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TextIO

from agentseek_api import __version__
from agentseek_api.container_policy import (
    APP_CONTAINER_POLICY,
    HOST_RUNTIME_POLICY,
    ContainerSelection,
    docker_control_environment,
    select_application_payload,
    select_compose_payload,
)
from agentseek_api.container_build import (
    PUBLISHED_RUNTIME_ARTIFACT,
    ContainerBuildError,
    FinalAuthSelection,
    RuntimeArtifactV1,
    load_container_runtime_manifest_v1,
    materialize_build_bundle,
    plan_container_image,
    plan_generated_up_auth,
    render_build_dockerfile,
)
from agentseek_api.constants import DEFAULT_API_PORT
from agentseek_api.docker_runtime import (
    DockerRuntimeError,
    LegacyRunnerAdapter,
    ProcessTransport,
    SubprocessTransport,
    build_compose_invocation,
    build_docker_control_invocation,
    build_image_invocation,
    build_docker_query_invocation,
    build_docker_run_invocation,
    encode_compose_environment,
    inspect_image_contract,
    require_supported_compose,
    require_supported_buildx,
)
from agentseek_api.dotenv_adapter import DotenvFileError, parse_dotenv_file
from agentseek_api.environment import (
    CommandDerivedAssignment,
    ContainerPolicyError,
    EnvironmentPlan,
    EnvironmentMode,
    EnvironmentTarget,
    PRELOADED_V1_POLICY,
    ResolvedEnvironment,
    resolve_environment,
)
from agentseek_api.process_supervisor import (
    ForegroundChildSupervisor,
    ForwardingSignalGuard,
    ProcessSupervisionError,
    _ForwardedSignal,
)
from agentseek_api.secure_temp import (
    SecureArtifactError,
    private_artifact,
    private_directory,
    sweep_expired_artifacts,
)

DEFAULT_CLI_NAME = "agentseek-api"

AGENTSEEK_ONBOARD_BANNER = (
    "\n"
    "        Welcome to\n"
    "\n"
    "╔═╗┌─┐┌─┐┌┐┌┌┬┐╔═╗┌─┐┌─┐┬┌─\n"
    "╠═╣│ ┬├┤ │││ │ ╚═╗├┤ ├┤ ├┴┐\n"
    "╩ ╩└─┘└─┘┘└┘ ┴ ╚═╝└─┘└─┘┴ ┴\n"
    "\n"
    "     AgentSeek v{version}\n"
)

AGENTSEEK_ONBOARD_BANNER_ASCII = (
    "\n"
    "        Welcome to\n"
    "\n"
    "========================\n"
    "     AgentSeek v{version}\n"
    "========================\n"
)

__all__ = [
    "CliError",
    "build_container_env",
    "build_runtime_env",
    "build_uvicorn_command",
    "create_parser",
    "main",
    "resolve_runtime_for_mode",
    "register_subcommands",
    "run_namespace",
]

_CONTAINER_ENV_PREFIXES = (
    "AGENTSEEK_",
    "ANTHROPIC_",
    "AUTH_",
    "LANGCHAIN_",
    "LANGSMITH_",
    "LIVE_",
    "METADATA_",
    "OCEANBASE_",
    "OPENAI_",
    "SEEKDB_",
)


class CliError(RuntimeError):
    pass


@dataclass
class CliConfig:
    graphs: dict[str, object]
    dependencies: list[str] = field(default_factory=list)
    env_mapping: dict[str, str] = field(default_factory=dict)
    env_file: Path | None = None
    auth_path: str | None = None
    base_image: str | None = None
    python_version: str | None = None
    image_distro: str | None = None
    pip_config_file: Path | None = None
    dockerfile_lines: list[str] = field(default_factory=list)
    compose_env: tuple[str, ...] = ()
    build_include: tuple[str, ...] = ()


@dataclass(frozen=True)
class DevServerUrls:
    api_url: str
    docs_url: str
    studio_url: str


def _write_banner(
    stdout: TextIO,
    *,
    unicode_text: str,
    ascii_text: str,
) -> None:
    text = unicode_text
    encoding = getattr(stdout, "encoding", None)
    if isinstance(encoding, str) and encoding:
        try:
            unicode_text.encode(encoding, errors="strict")
        except (UnicodeEncodeError, LookupError):
            text = ascii_text.encode("ascii", errors="replace").decode("ascii")
    stdout.write(text)
    stdout.flush()


def _write_onboard_banner(stdout: TextIO) -> None:
    _write_banner(
        stdout,
        unicode_text=AGENTSEEK_ONBOARD_BANNER.format(version=__version__) + "\n",
        ascii_text=AGENTSEEK_ONBOARD_BANNER_ASCII.format(version=__version__) + "\n",
    )


def _resolve_path(path_text: str, *, cwd: Path) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = cwd / path
    return path.resolve()


def _flag_name(option_name: str) -> str:
    return f"--{option_name.replace('_', '-')}"


def _cli_name(args: argparse.Namespace) -> str:
    return getattr(args, "cli_name", DEFAULT_CLI_NAME)


def _infer_cli_name() -> str:
    candidate = Path(sys.argv[0]).name
    if candidate == DEFAULT_CLI_NAME:
        return candidate
    return DEFAULT_CLI_NAME


def _reject_unsupported_options(
    args: argparse.Namespace,
    *,
    command_name: str,
    option_names: Sequence[str],
    hint: str | None = None,
) -> None:
    unsupported = []
    for option_name in option_names:
        value = getattr(args, option_name)
        if isinstance(value, bool):
            if value:
                unsupported.append(_flag_name(option_name))
            continue
        if value is not None:
            unsupported.append(_flag_name(option_name))
    if unsupported:
        message = f"Unsupported option(s) for '{_cli_name(args)} {command_name}': {', '.join(unsupported)}"
        if hint:
            message = f"{message} {hint}"
        raise CliError(message)


def discover_config_path(*, explicit_path: str | None, cwd: Path) -> Path | None:
    if explicit_path:
        resolved = _resolve_path(explicit_path, cwd=cwd)
        if not resolved.exists():
            raise CliError(f"Config file '{resolved}' does not exist.")
        return resolved

    for candidate in ("agentseek.json", "langgraph.json"):
        resolved = (cwd / candidate).resolve()
        if resolved.exists():
            return resolved

    env_manifest = os.environ.get("AGENTSEEK_GRAPHS")
    if env_manifest:
        resolved = Path(env_manifest).expanduser().resolve()
        if resolved.exists():
            return resolved
    return None


def _apply_env_layer(
    env: dict[str, str],
    layer: dict[str, str | None],
) -> None:
    for key, value in layer.items():
        if value is not None:
            env[key] = value


def _read_env_layer(
    path: Path,
    *,
    inherited: dict[str, str],
) -> dict[str, str | None]:
    try:
        return parse_dotenv_file(path, ambient=inherited)
    except DotenvFileError as exc:
        raise CliError(str(exc)) from exc


def _resolve_path_from_config(path_text: str, *, config_path: Path) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = config_path.parent / path
    return path.resolve()


def _split_symbol_reference(reference: str) -> tuple[str, str] | None:
    if ":" not in reference:
        return None
    module_name, symbol_name = reference.rsplit(":", maxsplit=1)
    if not module_name or not symbol_name:
        return None
    return module_name, symbol_name


def _normalize_symbol_reference(reference: str, *, config_path: Path) -> str:
    parts = _split_symbol_reference(reference)
    if parts is None:
        return reference
    module_name, symbol_name = parts
    if (
        module_name.endswith(".py")
        or module_name.startswith(".")
        or "/" in module_name
        or "\\" in module_name
    ):
        resolved_module = _resolve_path_from_config(
            module_name, config_path=config_path
        )
        return f"{resolved_module}:{symbol_name}"
    return reference


def _normalize_env_mapping(
    raw_env: object, *, config_path: Path
) -> tuple[dict[str, str], Path | None]:
    if raw_env is None:
        return {}, None
    if isinstance(raw_env, str):
        resolved_env = _resolve_path_from_config(raw_env, config_path=config_path)
        if not resolved_env.exists():
            raise CliError(f"Env file '{resolved_env}' does not exist.")
        return {}, resolved_env
    if isinstance(raw_env, dict):
        env_mapping: dict[str, str] = {}
        for key, value in raw_env.items():
            if not isinstance(key, str):
                raise CliError(
                    f"Config file '{config_path}' env mapping keys must be strings."
                )
            if isinstance(value, (str, int, float, bool)) or value is None:
                env_mapping[key] = "" if value is None else str(value)
            else:
                raise CliError(
                    f"Config file '{config_path}' env mapping values must be scalar."
                )
        return env_mapping, None
    raise CliError(
        f"Config file '{config_path}' must set 'env' to a path string or key/value object."
    )


def _normalize_name_list(
    raw_value: object, *, config_path: Path, field_name: str
) -> tuple[str, ...]:
    if raw_value is None:
        return ()
    if not isinstance(raw_value, list) or not all(
        isinstance(name, str) and name.strip() for name in raw_value
    ):
        raise CliError(
            f"Config file '{config_path}' field '{field_name}' must be an array of non-empty strings."
        )
    return tuple(name.strip() for name in raw_value)


def _load_cli_config(config_path: Path) -> CliConfig:
    payload = _load_config_payload(config_path)
    env_mapping, env_file = _normalize_env_mapping(
        payload.get("env"), config_path=config_path
    )
    raw_dependencies = payload.get("dependencies", [])
    if raw_dependencies is None:
        raw_dependencies = []
    if not isinstance(raw_dependencies, list) or not all(
        isinstance(item, str) and item.strip() for item in raw_dependencies
    ):
        raise CliError(
            f"Config file '{config_path}' field 'dependencies' must be an array of non-empty strings."
        )

    auth_path: str | None = None
    raw_auth = payload.get("auth")
    if raw_auth is not None:
        if not isinstance(raw_auth, dict):
            raise CliError(
                f"Config file '{config_path}' field 'auth' must be an object."
            )
        raw_auth_path = raw_auth.get("path")
        if raw_auth_path is not None:
            if not isinstance(raw_auth_path, str) or not raw_auth_path.strip():
                raise CliError(
                    f"Config file '{config_path}' field 'auth.path' must be a non-empty string."
                )
            auth_path = _normalize_symbol_reference(
                raw_auth_path.strip(), config_path=config_path
            )

    raw_pip_config = payload.get("pip_config_file")
    pip_config_file: Path | None = None
    if raw_pip_config is not None:
        if not isinstance(raw_pip_config, str) or not raw_pip_config.strip():
            raise CliError(
                f"Config file '{config_path}' field 'pip_config_file' must be a non-empty string."
            )
        pip_config_file = _resolve_path_from_config(
            raw_pip_config, config_path=config_path
        )
        if not pip_config_file.exists():
            raise CliError(f"Pip config file '{pip_config_file}' does not exist.")

    raw_base_image = payload.get("base_image")
    if raw_base_image is not None and (
        not isinstance(raw_base_image, str) or not raw_base_image.strip()
    ):
        raise CliError(
            f"Config file '{config_path}' field 'base_image' must be a non-empty string."
        )

    raw_python_version = payload.get("python_version")
    if raw_python_version is not None and (
        not isinstance(raw_python_version, str) or not raw_python_version.strip()
    ):
        raise CliError(
            f"Config file '{config_path}' field 'python_version' must be a non-empty string."
        )

    raw_image_distro = payload.get("image_distro")
    if raw_image_distro is not None and (
        not isinstance(raw_image_distro, str) or not raw_image_distro.strip()
    ):
        raise CliError(
            f"Config file '{config_path}' field 'image_distro' must be a non-empty string."
        )

    raw_dockerfile_lines = payload.get("dockerfile_lines", [])
    if raw_dockerfile_lines is None:
        raw_dockerfile_lines = []
    if not isinstance(raw_dockerfile_lines, list) or not all(
        isinstance(item, str) for item in raw_dockerfile_lines
    ):
        raise CliError(
            f"Config file '{config_path}' field 'dockerfile_lines' must be an array of strings."
        )

    compose_env = _normalize_name_list(
        payload.get("compose_env"), config_path=config_path, field_name="compose_env"
    )
    build_include = _normalize_name_list(
        payload.get("build_include"),
        config_path=config_path,
        field_name="build_include",
    )

    return CliConfig(
        dependencies=[item.strip() for item in raw_dependencies],
        graphs=payload["graphs"],  # validated by _load_config_payload
        env_mapping=env_mapping,
        env_file=env_file,
        auth_path=auth_path,
        base_image=raw_base_image.strip() if isinstance(raw_base_image, str) else None,
        python_version=raw_python_version.strip()
        if isinstance(raw_python_version, str)
        else None,
        image_distro=raw_image_distro.strip()
        if isinstance(raw_image_distro, str)
        else None,
        pip_config_file=pip_config_file,
        dockerfile_lines=list(raw_dockerfile_lines),
        compose_env=compose_env,
        build_include=build_include,
    )


def build_runtime_env(
    *,
    config_path: Path | None,
    env_file: str | None,
    cwd: Path,
    base_env: dict[str, str] | None = None,
) -> dict[str, str]:
    plan = build_host_environment_plan(
        config_path=config_path,
        env_file=env_file,
        cwd=cwd,
        base_env=base_env,
        role=None,
    )
    return _resolve_host_environment_plan(plan)


def resolve_runtime_for_mode(
    *,
    mode: EnvironmentMode,
    config_path: str | None,
    env_file: str | None,
    inherited: dict[str, str],
    cwd: Path,
    role: str | None = None,
) -> ResolvedEnvironment:
    """Resolve ordinary sources or one exact preloaded manifest, never both."""

    if mode is EnvironmentMode.RESOLVE:
        discovered = discover_config_path(explicit_path=config_path, cwd=cwd)
        plan = build_host_environment_plan(
            config_path=discovered,
            env_file=env_file,
            cwd=cwd,
            base_env=inherited,
            role=role,
        )
        try:
            return resolve_environment(plan, HOST_RUNTIME_POLICY)
        except DotenvFileError as exc:
            raise CliError(str(exc)) from exc

    manifest_value = inherited.get("AGENTSEEK_GRAPHS")
    if not manifest_value:
        raise CliError("preloaded-v1 requires inherited AGENTSEEK_GRAPHS.")
    manifest_path = Path(manifest_value)
    if not manifest_path.is_absolute():
        raise CliError("preloaded-v1 AGENTSEEK_GRAPHS must be an absolute path.")
    if env_file is not None:
        raise CliError("--env-file is not supported in preloaded-v1 mode.")
    if (
        config_path is not None
        and str(_resolve_path(config_path, cwd=cwd)) != manifest_value
    ):
        raise CliError(
            "--config must resolve to the inherited preloaded manifest path."
        )
    try:
        manifest = load_container_runtime_manifest_v1(manifest_path)
    except ContainerBuildError as exc:
        raise CliError(str(exc)) from exc
    try:
        installed_version = importlib.metadata.version(manifest.runtime.distribution)
    except importlib.metadata.PackageNotFoundError as exc:
        raise CliError(
            "The installed runtime distribution is incompatible with the manifest."
        ) from exc
    if installed_version != manifest.runtime.version:
        raise CliError(
            "The installed runtime distribution is incompatible with the manifest."
        )
    assignments: tuple[CommandDerivedAssignment, ...] = ()
    if role == "dev":
        assignments = (
            CommandDerivedAssignment(
                targets=frozenset({EnvironmentTarget.HOST_RUNTIME}),
                values={"STUDIO_AUTH_LOCAL_DEV": "true"},
                reason="development role safety",
            ),
        )
    plan = EnvironmentPlan(
        config_path=manifest_path,
        config_dotenv=None,
        config_mapping={},
        auth_path=None,
        cli_dotenv=None,
        launch_environment=inherited,
        command_assignments=assignments,
    )
    return resolve_environment(plan, PRELOADED_V1_POLICY)


def build_host_environment_plan(
    *,
    config_path: Path | None,
    env_file: str | None,
    cwd: Path,
    base_env: dict[str, str] | None = None,
    role: str | None,
) -> EnvironmentPlan:
    inherited = dict(os.environ if base_env is None else base_env)
    config = _load_cli_config(config_path) if config_path is not None else None
    cli_dotenv = _resolve_path(env_file, cwd=cwd) if env_file else None
    assignments: list[CommandDerivedAssignment] = []
    inherited.pop("AGENTSEEK_GRAPHS", None)
    if config_path is not None:
        assignments.append(
            CommandDerivedAssignment(
                targets=frozenset({EnvironmentTarget.HOST_RUNTIME}),
                values={"AGENTSEEK_GRAPHS": str(config_path)},
                reason="selected host config path",
            )
        )
    if role == "dev":
        assignments.append(
            CommandDerivedAssignment(
                targets=frozenset({EnvironmentTarget.HOST_RUNTIME}),
                values={"STUDIO_AUTH_LOCAL_DEV": "true"},
                reason="development role safety",
            )
        )
    return EnvironmentPlan(
        config_path=config_path,
        config_dotenv=config.env_file if config is not None else None,
        config_mapping=config.env_mapping if config is not None else {},
        auth_path=config.auth_path if config is not None else None,
        cli_dotenv=cli_dotenv,
        launch_environment=inherited,
        command_assignments=tuple(assignments),
        explicit_names=frozenset(),
    )


def _resolve_host_environment_plan(plan: EnvironmentPlan) -> dict[str, str]:
    try:
        resolved = resolve_environment(plan, HOST_RUNTIME_POLICY)
    except DotenvFileError as exc:
        raise CliError(str(exc)) from exc
    return dict(resolved.values)


def build_container_command_assignments(
    *,
    config_path: Path,
    cwd: Path,
    postgres_uri: str | None,
    role: str | None = None,
    environment_mode: str | None = None,
) -> tuple[CommandDerivedAssignment, ...]:
    assignments = [
        CommandDerivedAssignment(
            targets=frozenset({EnvironmentTarget.APP_CONTAINER}),
            values={
                "AGENTSEEK_GRAPHS": _container_config_path(
                    config_path=config_path, cwd=cwd
                )
            },
            reason="container manifest path",
        )
    ]
    if postgres_uri is not None:
        assignments.append(
            CommandDerivedAssignment(
                targets=frozenset({EnvironmentTarget.APP_CONTAINER}),
                values={
                    "METADATA_DB_URL": postgres_uri,
                    "METADATA_DB_BACKEND": "postgresql",
                },
                reason="postgres uri override",
            )
        )
    if role == "dev" and environment_mode == "preloaded-v1":
        assignments.append(
            CommandDerivedAssignment(
                targets=frozenset({EnvironmentTarget.APP_CONTAINER}),
                values={"STUDIO_AUTH_LOCAL_DEV": "true"},
                reason="development role safety in preloaded runtime",
            )
        )
    return tuple(assignments)


def build_container_selection(
    *,
    config_path: Path,
    pass_env: Sequence[str],
    compose_pass_env: Sequence[str],
) -> ContainerSelection:
    """Parse container selectors without coupling them to Docker execution."""

    config = _load_cli_config(config_path)

    def normalized(names: Sequence[str]) -> frozenset[str]:
        normalized_names = frozenset(name.strip() for name in names)
        if "" in normalized_names:
            raise CliError("Container environment selector names must be non-empty.")
        return normalized_names

    return ContainerSelection(
        pass_env=normalized(pass_env),
        compose_env=normalized((*config.compose_env, *compose_pass_env)),
    )


def _build_container_environment_plan(
    *,
    config_path: Path,
    env_file: str | None,
    cwd: Path,
    selection: ContainerSelection,
    postgres_uri: str | None,
) -> EnvironmentPlan:
    host_plan = build_host_environment_plan(
        config_path=config_path,
        env_file=env_file,
        cwd=cwd,
        role=None,
    )
    return EnvironmentPlan(
        config_path=host_plan.config_path,
        config_dotenv=host_plan.config_dotenv,
        config_mapping=host_plan.config_mapping,
        auth_path=host_plan.auth_path,
        cli_dotenv=host_plan.cli_dotenv,
        launch_environment=host_plan.launch_environment,
        command_assignments=build_container_command_assignments(
            config_path=config_path,
            cwd=cwd,
            postgres_uri=postgres_uri,
        ),
        explicit_names=selection.pass_env,
    )


def _resolve_application_container_payload(
    plan: EnvironmentPlan,
    *,
    selection: ContainerSelection,
) -> tuple[dict[str, str], FinalAuthSelection | None]:
    try:
        resolved = resolve_environment(plan, APP_CONTAINER_POLICY)
        payload = dict(select_application_payload(resolved, selection))
    except (ContainerPolicyError, DotenvFileError) as exc:
        raise CliError(str(exc)) from exc
    auth_selection: FinalAuthSelection | None = None
    if "AUTH_MODULE_PATH" in payload:
        auth_selection = FinalAuthSelection(
            payload["AUTH_MODULE_PATH"], resolved.origins["AUTH_MODULE_PATH"]
        )
    return payload, auth_selection


def _is_valid_custom_image_auth(reference: str) -> bool:
    if reference == "":
        return True
    if (
        reference.startswith(".")
        or reference.partition(":")[0].endswith(".py")
        or "/" in reference
        or "\\" in reference
        or re.match(r"^[A-Za-z]:", reference)
    ):
        return False
    parts = _split_symbol_reference(reference)
    if parts is None or reference.count(":") != 1:
        return False
    module_name, symbol_name = parts
    identifier = r"[A-Za-z_][A-Za-z0-9_]*"
    return bool(
        re.fullmatch(rf"{identifier}(?:\.{identifier})*", module_name)
        and re.fullmatch(identifier, symbol_name)
    )


def _planner_dotenv_paths(env_file: str | None, *, cwd: Path) -> tuple[Path, ...]:
    return () if env_file is None else (_resolve_path(env_file, cwd=cwd),)


def _runtime_entrypoint_prefix(*, isolated: bool) -> list[str]:
    return [
        sys.executable,
        *(["-I"] if isolated else []),
        "-m",
        "agentseek_api.runtime_entrypoint",
        *(["--preloaded-v1"] if isolated else []),
    ]


def build_uvicorn_command(
    *, host: str, port: int, reload_enabled: bool, isolated: bool = False
) -> list[str]:
    command = [
        *_runtime_entrypoint_prefix(isolated=isolated),
        "uvicorn",
        "--",
        "agentseek_api.main:app",
        "--host",
        host,
        "--port",
        str(port),
    ]
    if reload_enabled:
        command.append("--reload")
    return command


def build_worker_command(*, isolated: bool = False) -> list[str]:
    return [
        *_runtime_entrypoint_prefix(isolated=isolated),
        "worker",
    ]


def build_scheduler_command(*, isolated: bool = False) -> list[str]:
    return [
        *_runtime_entrypoint_prefix(isolated=isolated),
        "scheduler",
    ]


def _default_runner(
    command: list[str], *, env: dict[str, str], cwd: str | None = None
) -> int:
    try:
        with ForwardingSignalGuard() as signals:
            child = ForegroundChildSupervisor.start(command, env=env, cwd=cwd)
            try:
                signals.attach(child)
                exit_code = child.wait()
                child.close_remaining_tree(timeout=5.0)
                return exit_code
            except KeyboardInterrupt:
                signals.begin_cleanup()
                child.forward_and_reap(signal.SIGINT, timeout=5.0)
                return 130
            except _ForwardedSignal as exc:
                signals.begin_cleanup()
                child.forward_and_reap(exc.signum, timeout=5.0)
                return 128 + exc.signum
            except BaseException:
                signals.begin_cleanup()
                child.terminate_and_reap(timeout=5.0)
                raise
            finally:
                signals.begin_cleanup()
                try:
                    child.ensure_closed(timeout=5.0)
                finally:
                    child.close()
    except ProcessSupervisionError as exc:
        raise CliError("Could not supervise the runtime child safely.") from exc


def _format_http_host(host: str) -> str:
    if ":" in host and not host.startswith("["):
        return f"[{host}]"
    return host


def _is_loopbackish_host(host: str) -> bool:
    return host in {"127.0.0.1", "localhost", "0.0.0.0", "::", "[::]"}


def _resolve_dev_urls(*, host: str, port: int, studio_url: str | None) -> DevServerUrls:
    display_host = "localhost" if _is_loopbackish_host(host) else host
    studio_host = "127.0.0.1" if _is_loopbackish_host(host) else host
    studio_origin = (studio_url or "https://smith.langchain.com").rstrip("/")
    api_url = f"http://{_format_http_host(display_host)}:{port}"
    studio_base_url = f"http://{_format_http_host(studio_host)}:{port}"
    return DevServerUrls(
        api_url=api_url,
        docs_url=f"{api_url}/docs",
        studio_url=f"{studio_origin}/studio/?baseUrl={studio_base_url}",
    )


def _render_dev_ready_banner(urls: DevServerUrls) -> str:
    return (
        f"- \U0001f680 API: {urls.api_url}\n"
        f"- \U0001f4da Docs: {urls.docs_url}\n"
        f"- \U0001f3a8 Studio UI: {urls.studio_url}\n"
        "\n\n"
    )


def _render_ascii_dev_ready_banner(urls: DevServerUrls) -> str:
    return (
        f"- API: {urls.api_url}\n"
        f"- Docs: {urls.docs_url}\n"
        f"- Studio UI: {urls.studio_url}\n"
        "\n\n"
    )


def _wait_for_dev_server_ready(
    api_url: str,
    *,
    process,
    timeout_seconds: float = 30.0,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None
    ready_urls = [f"{api_url}/ok", f"{api_url}/health"]
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise CliError(
                f"Development server exited before becoming ready (exit code {process.returncode})."
            )
        for ready_url in ready_urls:
            try:
                with urllib_request.urlopen(ready_url, timeout=2.0) as response:
                    if 200 <= response.status < 300:
                        return
            except (urllib_error.URLError, TimeoutError, OSError) as exc:
                last_error = exc
        sleep(0.2)
    raise CliError(f"Timed out waiting for '{api_url}' to become ready: {last_error}")


def _default_process_factory(command: list[str], *, env: dict[str, str], cwd: str):
    return subprocess.Popen(command, env=env, cwd=cwd)


def _run_managed_dev_server(
    *,
    command: list[str],
    env: dict[str, str],
    cwd: Path,
    urls: DevServerUrls,
    stdout: TextIO,
    open_browser: bool = True,
    process_factory: Callable[..., object] = _default_process_factory,
    wait_for_ready: Callable[..., None] = _wait_for_dev_server_ready,
    browser_opener: Callable[[str], object] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    process = process_factory(command, env=env, cwd=str(cwd))
    previous_handlers: dict[int, object] = {}

    def _terminate_child(_signum, _frame) -> None:
        if process.poll() is None:
            process.terminate()

    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            previous_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, _terminate_child)
        except ValueError:
            continue

    try:
        _write_banner(
            stdout,
            unicode_text=_render_dev_ready_banner(urls),
            ascii_text=_render_ascii_dev_ready_banner(urls),
        )
        wait_for_ready(urls.api_url, process=process, sleep=sleep)
        if open_browser:
            if browser_opener is None:
                import webbrowser

                browser_opener = webbrowser.open
            browser_opener(urls.studio_url)
        return process.wait()
    except CliError:
        if process.poll() is not None:
            return process.returncode
        raise
    except KeyboardInterrupt:
        if process.poll() is None:
            process.terminate()
        return process.wait()
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait()
        for signum, handler in previous_handlers.items():
            try:
                signal.signal(signum, handler)
            except ValueError:
                continue


def _execute_runtime_command(
    args: argparse.Namespace, *, runner: Callable[..., int], cwd: Path
) -> int:
    env = dict(
        resolve_runtime_for_mode(
            mode=args.environment_mode,
            config_path=args.config,
            env_file=args.env_file,
            inherited=dict(os.environ),
            cwd=cwd,
        ).values
    )
    command = build_uvicorn_command(
        host=args.host,
        port=args.port,
        reload_enabled=getattr(args, "reload", False),
        isolated=args.environment_mode is EnvironmentMode.PRELOADED_V1,
    )
    return runner(command, env=env, cwd=str(cwd))


def _execute_dev_command(
    args: argparse.Namespace,
    *,
    runner: Callable[..., int],
    cwd: Path,
    stdout: TextIO,
) -> int:
    _write_onboard_banner(stdout)
    args.reload = not args.no_reload
    env = dict(
        resolve_runtime_for_mode(
            mode=args.environment_mode,
            config_path=args.config,
            env_file=args.env_file,
            inherited=dict(os.environ),
            cwd=cwd,
            role="dev",
        ).values
    )
    command = build_uvicorn_command(
        host=args.host,
        port=args.port,
        reload_enabled=args.reload,
        isolated=args.environment_mode is EnvironmentMode.PRELOADED_V1,
    )
    if runner is not _default_runner:
        return runner(command, env=env, cwd=str(cwd))
    urls = _resolve_dev_urls(host=args.host, port=args.port, studio_url=args.studio_url)
    return _run_managed_dev_server(
        command=command,
        env=env,
        cwd=cwd,
        urls=urls,
        stdout=stdout,
        open_browser=not args.no_browser,
    )


def _execute_worker_command(
    args: argparse.Namespace, *, runner: Callable[..., int], cwd: Path
) -> int:
    env = dict(
        resolve_runtime_for_mode(
            mode=args.environment_mode,
            config_path=args.config,
            env_file=args.env_file,
            inherited=dict(os.environ),
            cwd=cwd,
        ).values
    )
    command = (
        build_worker_command(isolated=True)
        if args.environment_mode is EnvironmentMode.PRELOADED_V1
        else build_worker_command()
    )
    return runner(command, env=env, cwd=str(cwd))


def _execute_scheduler_command(
    args: argparse.Namespace, *, runner: Callable[..., int], cwd: Path
) -> int:
    env = dict(
        resolve_runtime_for_mode(
            mode=args.environment_mode,
            config_path=args.config,
            env_file=args.env_file,
            inherited=dict(os.environ),
            cwd=cwd,
        ).values
    )
    return runner(
        build_scheduler_command(
            isolated=args.environment_mode is EnvironmentMode.PRELOADED_V1
        ),
        env=env,
        cwd=str(cwd),
    )


def _load_config_payload(config_path: Path) -> dict[str, object]:
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise CliError(
            f"Config file '{config_path}' could not be parsed as JSON: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise CliError(
            f"Config file '{config_path}' must contain a top-level JSON object."
        )
    graphs = payload.get("graphs")
    if not isinstance(graphs, dict) or not graphs:
        raise CliError(
            f"Config file '{config_path}' must contain a non-empty 'graphs' object."
        )
    return payload


def _container_config_path(*, config_path: Path, cwd: Path) -> str:
    try:
        relative_path = config_path.relative_to(cwd)
    except ValueError as exc:
        raise CliError(
            f"Config file '{config_path}' must live under the project root '{cwd}' for Docker builds."
        ) from exc
    return f"/deps/agent/{relative_path.as_posix()}"


def _containerize_symbol_reference(reference: str, *, cwd: Path) -> str:
    parts = _split_symbol_reference(reference)
    if parts is None:
        return reference
    module_name, symbol_name = parts
    if (
        module_name.endswith(".py")
        or module_name.startswith(".")
        or "/" in module_name
        or "\\" in module_name
    ):
        resolved_module = _resolve_path(module_name, cwd=cwd)
        return f"{_container_config_path(config_path=resolved_module, cwd=cwd)}:{symbol_name}"
    return reference


def _ambient_container_env() -> dict[str, str]:
    return {
        key: value
        for key, value in os.environ.items()
        if key.startswith(_CONTAINER_ENV_PREFIXES)
    }


def build_container_env(
    *,
    config_path: Path,
    env_file: str | None,
    cwd: Path,
) -> dict[str, str]:
    env = build_runtime_env(
        config_path=config_path,
        env_file=env_file,
        cwd=cwd,
        base_env=_ambient_container_env(),
    )
    env["AGENTSEEK_GRAPHS"] = _container_config_path(config_path=config_path, cwd=cwd)
    auth_module_path = env.get("AUTH_MODULE_PATH")
    if auth_module_path:
        env["AUTH_MODULE_PATH"] = _containerize_symbol_reference(
            auth_module_path, cwd=cwd
        )
    return env


def _execute_dockerfile_command(
    args: argparse.Namespace,
    *,
    stdout: TextIO,
    cwd: Path,
    runtime_artifact: RuntimeArtifactV1,
) -> int:
    config_path = discover_config_path(explicit_path=args.config, cwd=cwd)
    if config_path is None:
        raise CliError(
            f"No config file found in '{cwd}'. Expected agentseek.json or langgraph.json."
        )
    _load_cli_config(config_path)
    save_path = _resolve_path(args.save_path, cwd=cwd)
    plan = plan_container_image(
        config_path=config_path,
        dotenv_paths=_planner_dotenv_paths(args.env_file, cwd=cwd),
        runtime_artifact=runtime_artifact,
        invocation_cwd=cwd,
    )
    dockerfile_bytes = render_build_dockerfile(plan)
    bundle = materialize_build_bundle(
        plan,
        dockerfile_bytes=dockerfile_bytes,
        output_root=save_path,
    )
    stdout.write(f"{bundle.dockerfile}\n")
    return 0


def _execute_build_command(
    args: argparse.Namespace,
    *,
    process_transport: ProcessTransport,
    cwd: Path,
    runtime_artifact: RuntimeArtifactV1,
) -> int:
    config_path = discover_config_path(explicit_path=args.config, cwd=cwd)
    if config_path is None:
        raise CliError(
            f"No config file found in '{cwd}'. Expected agentseek.json or langgraph.json."
        )
    _load_cli_config(config_path)
    build_plan = plan_container_image(
        config_path=config_path,
        dotenv_paths=_planner_dotenv_paths(args.env_file, cwd=cwd),
        runtime_artifact=runtime_artifact,
        invocation_cwd=cwd,
    )
    dockerfile_bytes = render_build_dockerfile(build_plan)
    environment_plan = build_host_environment_plan(
        config_path=config_path,
        env_file=args.env_file,
        cwd=cwd,
        role=None,
    )
    with private_directory(prefix="agentseek-build-") as output_root:
        bundle = materialize_build_bundle(
            build_plan,
            dockerfile_bytes=dockerfile_bytes,
            output_root=output_root,
        )
        invocation = build_image_invocation(
            bundle,
            plan=build_plan,
            docker_control=docker_control_environment(environment_plan),
            tag=args.tag,
            platform=args.platform,
            pull=args.pull,
        )
        try:
            require_supported_buildx(
                transport=process_transport,
                docker_control=docker_control_environment(environment_plan),
                cwd=cwd,
                plan=build_plan,
            )
        except DockerRuntimeError as exc:
            raise CliError(str(exc)) from exc
        return process_transport(invocation).returncode


def _container_name_for_port(port: int) -> str:
    return f"agentseek-up-{port}"


def _wait_for_http_ready(url: str, *, timeout_seconds: float) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urllib_request.urlopen(url, timeout=2.0) as response:
                if 200 <= response.status < 300:
                    return
        except (urllib_error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
        time.sleep(1.0)
    raise CliError(f"Timed out waiting for '{url}' to become ready: {last_error}")


def _container_exists(
    name: str,
    *,
    process_transport: ProcessTransport,
    docker_control: dict[str, str],
    cwd: Path,
) -> bool:
    invocation = build_docker_query_invocation(
        argv=("docker", "container", "inspect", name),
        docker_control=docker_control,
        cwd=cwd,
    )
    return process_transport(invocation).returncode == 0


def _execute_up_command(
    args: argparse.Namespace,
    *,
    process_transport: ProcessTransport,
    cwd: Path,
    runtime_artifact: RuntimeArtifactV1,
) -> int:
    config_path = discover_config_path(explicit_path=args.config, cwd=cwd)
    if config_path is None:
        raise CliError(
            f"No config file found in '{cwd}'. Expected agentseek.json or langgraph.json."
        )

    image = args.image
    selection = build_container_selection(
        config_path=config_path,
        pass_env=args.pass_env,
        compose_pass_env=args.compose_pass_env,
    )
    environment_plan = _build_container_environment_plan(
        config_path=config_path,
        env_file=args.env_file,
        cwd=cwd,
        selection=selection,
        postgres_uri=args.postgres_uri,
    )
    docker_control = dict(docker_control_environment(environment_plan))
    application_payload, final_auth = _resolve_application_container_payload(
        environment_plan, selection=selection
    )

    compose_path: Path | None = None
    if args.docker_compose:
        compose_path = _resolve_path(args.docker_compose, cwd=cwd)
        if not compose_path.exists():
            raise CliError(f"Docker compose file '{compose_path}' does not exist.")

    generated_plan = None
    generated_dockerfile_bytes: bytes | None = None
    custom_image_contract = None
    if image:
        if final_auth is not None and not _is_valid_custom_image_auth(final_auth.value):
            raise CliError(
                "Custom-image auth cannot reference a host file; bake the module into the image and use an importable package reference."
            )
        custom_image_contract = inspect_image_contract(
            image,
            transport=process_transport,
            docker_control=docker_control,
            cwd=cwd,
        )
        application_payload = dict(application_payload)
        application_payload["AGENTSEEK_GRAPHS"] = custom_image_contract.manifest_path
    else:
        _load_cli_config(config_path)
        generated_plan = plan_container_image(
            config_path=config_path,
            dotenv_paths=_planner_dotenv_paths(args.env_file, cwd=cwd),
            base_image_override=args.base_image,
            runtime_artifact=runtime_artifact,
            invocation_cwd=cwd,
        )
        generated_plan, auth_patch = plan_generated_up_auth(generated_plan, final_auth)
        application_payload = dict(application_payload)
        if auth_patch is None:
            application_payload.pop("AUTH_MODULE_PATH", None)
        else:
            application_payload["AUTH_MODULE_PATH"] = auth_patch.value
        generated_dockerfile_bytes = render_build_dockerfile(generated_plan)

    compose_payload: dict[str, str] = {}
    encoded_compose: bytes | None = None
    if compose_path is not None:
        compose_payload = dict(
            select_compose_payload(
                application_payload=application_payload,
                selected_names=selection.compose_env,
                docker_control=docker_control,
            )
        )
        require_supported_compose(
            transport=process_transport,
            docker_control=docker_control,
            cwd=cwd,
        )
        sweep_expired_artifacts(
            prefix="agentseek-compose-",
            older_than_seconds=24 * 60 * 60,
        )
        encoded_compose = encode_compose_environment(compose_payload).encode("utf-8")

    if not image:
        assert generated_plan is not None
        assert generated_dockerfile_bytes is not None
        image = f"agentseek-up:{args.port}"
        with private_directory(prefix="agentseek-build-") as output_root:
            bundle = materialize_build_bundle(
                generated_plan,
                dockerfile_bytes=generated_dockerfile_bytes,
                output_root=output_root,
            )
            build_invocation = build_image_invocation(
                bundle,
                plan=generated_plan,
                docker_control=docker_control,
                tag=image,
                pull=args.pull,
            )
            try:
                require_supported_buildx(
                    transport=process_transport,
                    docker_control=docker_control,
                    cwd=cwd,
                    plan=generated_plan,
                )
            except DockerRuntimeError as exc:
                raise CliError(str(exc)) from exc
            build_exit_code = process_transport(build_invocation).returncode
        if build_exit_code != 0:
            return build_exit_code

    container_name = _container_name_for_port(args.port)
    if args.recreate:
        remove_invocation = build_docker_control_invocation(
            argv=("docker", "rm", "-f", container_name),
            docker_control=docker_control,
            cwd=cwd,
        )
        process_transport(remove_invocation)
    elif _container_exists(
        container_name,
        process_transport=process_transport,
        docker_control=docker_control,
        cwd=cwd,
    ):
        raise CliError(
            f"Container '{container_name}' already exists. Re-run with '--recreate' or remove it manually."
        )

    if compose_path is not None:
        assert encoded_compose is not None
        with private_artifact(
            prefix="agentseek-compose-",
            contents=encoded_compose,
        ) as env_path:
            compose_invocation = build_compose_invocation(
                compose_file=compose_path,
                env_file=env_path,
                docker_control=docker_control,
                application_payload=application_payload,
                selected_names=selection.compose_env,
                cwd=cwd,
                recreate=args.recreate,
            )
            compose_exit_code = process_transport(compose_invocation).returncode
        if compose_exit_code != 0:
            return compose_exit_code

    base_argv = (
        "docker",
        "run",
        "--detach",
        "--name",
        container_name,
        "--add-host",
        "host.docker.internal:host-gateway",
        "-p",
        f"{args.port}:{DEFAULT_API_PORT}",
    )
    run_invocation = build_docker_run_invocation(
        base_argv=base_argv,
        image=image,
        docker_control=docker_control,
        application_payload=application_payload,
        container_argv=(
            ()
            if custom_image_contract is None
            else custom_image_contract.container_argv
        ),
        cwd=cwd,
    )
    run_exit_code = process_transport(run_invocation).returncode
    if run_exit_code != 0:
        return run_exit_code
    if args.wait:
        _wait_for_http_ready(
            f"http://127.0.0.1:{args.port}/health", timeout_seconds=30.0
        )
    return 0


def _print_version(*, stdout: TextIO) -> int:
    stdout.write(f"agentseek-api {__version__}\n")
    return 0


def _add_command_parsers(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    *,
    runtime_parent: argparse.ArgumentParser,
) -> None:
    dev_parser = subparsers.add_parser("dev", parents=[runtime_parent])
    dev_parser.add_argument("--host", default="127.0.0.1")
    dev_parser.add_argument("--port", default=DEFAULT_API_PORT, type=int)
    dev_parser.add_argument("--no-reload", action="store_true")
    dev_parser.add_argument("--n-jobs-per-worker", type=int)
    dev_parser.add_argument("--debug-port", type=int)
    dev_parser.add_argument("--wait-for-client", action="store_true")
    dev_parser.add_argument("--no-browser", action="store_true")
    dev_parser.add_argument("--studio-url")
    dev_parser.add_argument("--allow-blocking", action="store_true")
    dev_parser.add_argument("--tunnel", action="store_true")
    dev_parser.add_argument(
        "--environment-mode",
        type=EnvironmentMode,
        choices=tuple(EnvironmentMode),
        default=EnvironmentMode.RESOLVE,
    )

    serve_parser = subparsers.add_parser("serve", parents=[runtime_parent])
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", default=DEFAULT_API_PORT, type=int)
    serve_parser.add_argument(
        "--environment-mode",
        type=EnvironmentMode,
        choices=tuple(EnvironmentMode),
        default=EnvironmentMode.RESOLVE,
    )

    worker_parser = subparsers.add_parser("worker", parents=[runtime_parent])
    worker_parser.add_argument(
        "--environment-mode",
        type=EnvironmentMode,
        choices=tuple(EnvironmentMode),
        default=EnvironmentMode.RESOLVE,
    )
    scheduler_parser = subparsers.add_parser("scheduler", parents=[runtime_parent])
    scheduler_parser.add_argument(
        "--environment-mode",
        type=EnvironmentMode,
        choices=tuple(EnvironmentMode),
        default=EnvironmentMode.RESOLVE,
    )

    subparsers.add_parser("version")

    build_parser = subparsers.add_parser("build", parents=[runtime_parent])
    build_parser.add_argument("--platform")
    build_parser.add_argument("-t", "--tag", required=True)
    build_parser.add_argument("--pull", dest="pull", action="store_true", default=True)
    build_parser.add_argument("--no-pull", dest="pull", action="store_false")

    up_parser = subparsers.add_parser("up", parents=[runtime_parent])
    up_parser.add_argument("--wait", action="store_true")
    up_parser.add_argument("--base-image")
    up_parser.add_argument("--image")
    up_parser.add_argument("--postgres-uri")
    up_parser.add_argument("--pass-env", action="append", default=[])
    up_parser.add_argument("--compose-pass-env", action="append", default=[])
    up_parser.add_argument("--watch", action="store_true")
    up_parser.add_argument("--debugger-base-url")
    up_parser.add_argument("--debugger-port", type=int)
    up_parser.add_argument("--verbose", action="store_true")
    up_parser.add_argument("-d", "--docker-compose")
    up_parser.add_argument("-p", "--port", default=8123, type=int)
    up_parser.add_argument("--pull", dest="pull", action="store_true", default=True)
    up_parser.add_argument("--no-pull", dest="pull", action="store_false")
    up_parser.add_argument("--recreate", dest="recreate", action="store_true")
    up_parser.add_argument("--no-recreate", dest="recreate", action="store_false")
    up_parser.set_defaults(recreate=False)

    dockerfile_parser = subparsers.add_parser("dockerfile", parents=[runtime_parent])
    dockerfile_parser.add_argument("save_path")


def register_subcommands(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    *,
    command_name: str = DEFAULT_CLI_NAME,
) -> argparse.ArgumentParser:
    runtime_parent = argparse.ArgumentParser(add_help=False)
    runtime_parent.add_argument("-c", "--config")
    runtime_parent.add_argument("--env-file")

    command_parser = subparsers.add_parser(command_name)
    command_parser.set_defaults(cli_name=command_name)
    command_subparsers = command_parser.add_subparsers(dest="command", required=True)
    _add_command_parsers(command_subparsers, runtime_parent=runtime_parent)
    return command_parser


def create_parser(*, prog: str = DEFAULT_CLI_NAME) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog)
    parser.set_defaults(cli_name=prog)
    runtime_parent = argparse.ArgumentParser(add_help=False)
    runtime_parent.add_argument("-c", "--config")
    runtime_parent.add_argument("--env-file")
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_command_parsers(subparsers, runtime_parent=runtime_parent)
    return parser


def run_namespace(
    args: argparse.Namespace,
    *,
    runner: Callable[..., int] | None = None,
    process_transport: ProcessTransport | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
    cwd: str | Path | None = None,
    runtime_artifact: RuntimeArtifactV1 = PUBLISHED_RUNTIME_ARTIFACT,
) -> int:
    command = args.command
    workdir = Path(cwd or Path.cwd()).resolve()
    run = runner or _default_runner
    docker_transport = process_transport or (
        LegacyRunnerAdapter(run) if runner is not None else SubprocessTransport()
    )
    out = stdout or sys.stdout
    err = stderr or sys.stderr

    try:
        if command == "version":
            return _print_version(stdout=out)
        if command == "dev":
            _reject_unsupported_options(
                args,
                command_name="dev",
                option_names=(
                    "n_jobs_per_worker",
                    "debug_port",
                    "wait_for_client",
                    "allow_blocking",
                    "tunnel",
                ),
                hint="Use 'langgraph dev' for mocked or tunneled local workflows.",
            )
            return _execute_dev_command(args, runner=run, cwd=workdir, stdout=out)
        if command == "serve":
            _write_onboard_banner(out)
            args.reload = False
            return _execute_runtime_command(args, runner=run, cwd=workdir)
        if command == "worker":
            return _execute_worker_command(args, runner=run, cwd=workdir)
        if command == "scheduler":
            return _execute_scheduler_command(args, runner=run, cwd=workdir)
        if command == "dockerfile":
            return _execute_dockerfile_command(
                args,
                stdout=out,
                cwd=workdir,
                runtime_artifact=runtime_artifact,
            )
        if command == "build":
            return _execute_build_command(
                args,
                process_transport=docker_transport,
                cwd=workdir,
                runtime_artifact=runtime_artifact,
            )
        if command == "up":
            _reject_unsupported_options(
                args,
                command_name="up",
                option_names=(
                    "watch",
                    "debugger_base_url",
                    "debugger_port",
                    "verbose",
                ),
            )
            return _execute_up_command(
                args,
                process_transport=docker_transport,
                cwd=workdir,
                runtime_artifact=runtime_artifact,
            )
        raise CliError(f"Unsupported command '{command}'.")
    except (
        CliError,
        ContainerPolicyError,
        DockerRuntimeError,
        SecureArtifactError,
        ContainerBuildError,
    ) as exc:
        err.write(f"{exc}\n")
        return 2


def main(
    argv: Sequence[str] | None = None,
    *,
    prog: str | None = None,
    runner: Callable[..., int] | None = None,
    process_transport: ProcessTransport | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
    cwd: str | Path | None = None,
    runtime_artifact: RuntimeArtifactV1 = PUBLISHED_RUNTIME_ARTIFACT,
) -> int:
    if prog is None:
        prog = _infer_cli_name() if argv is None else DEFAULT_CLI_NAME
    parser = create_parser(prog=prog)
    args = parser.parse_args(list(argv) if argv is not None else None)
    return run_namespace(
        args,
        runner=runner,
        process_transport=process_transport,
        stdout=stdout,
        stderr=stderr,
        cwd=cwd,
        runtime_artifact=runtime_artifact,
    )


if __name__ == "__main__":
    raise SystemExit(main())
