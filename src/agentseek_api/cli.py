from __future__ import annotations

import argparse
import json
import os
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
from agentseek_api.settings import DEFAULT_API_PORT

DEFAULT_CLI_NAME = "agentseek-api"

__all__ = [
    "CliError",
    "build_container_env",
    "build_runtime_env",
    "build_uvicorn_command",
    "create_parser",
    "main",
    "register_subcommands",
    "run_namespace",
    "write_dockerfile",
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


@dataclass(frozen=True)
class DevServerUrls:
    api_url: str
    docs_url: str
    scalar_docs_url: str
    studio_url: str


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


def _parse_env_file(env_file: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(env_file.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            raise CliError(f"Env file '{env_file}' has an invalid line {line_number}: '{raw_line}'.")
        key, value = line.split("=", maxsplit=1)
        values[key.strip()] = value.strip().strip("\"'")
    return values


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
    if module_name.endswith(".py") or module_name.startswith(".") or "/" in module_name or "\\" in module_name:
        resolved_module = _resolve_path_from_config(module_name, config_path=config_path)
        return f"{resolved_module}:{symbol_name}"
    return reference


def _normalize_env_mapping(raw_env: object, *, config_path: Path) -> tuple[dict[str, str], Path | None]:
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
                raise CliError(f"Config file '{config_path}' env mapping keys must be strings.")
            if isinstance(value, (str, int, float, bool)) or value is None:
                env_mapping[key] = "" if value is None else str(value)
            else:
                raise CliError(f"Config file '{config_path}' env mapping values must be scalar.")
        return env_mapping, None
    raise CliError(f"Config file '{config_path}' must set 'env' to a path string or key/value object.")


def _load_cli_config(config_path: Path) -> CliConfig:
    payload = _load_config_payload(config_path)
    env_mapping, env_file = _normalize_env_mapping(payload.get("env"), config_path=config_path)
    raw_dependencies = payload.get("dependencies", [])
    if raw_dependencies is None:
        raw_dependencies = []
    if not isinstance(raw_dependencies, list) or not all(isinstance(item, str) and item.strip() for item in raw_dependencies):
        raise CliError(f"Config file '{config_path}' field 'dependencies' must be an array of non-empty strings.")

    auth_path: str | None = None
    raw_auth = payload.get("auth")
    if raw_auth is not None:
        if not isinstance(raw_auth, dict):
            raise CliError(f"Config file '{config_path}' field 'auth' must be an object.")
        raw_auth_path = raw_auth.get("path")
        if raw_auth_path is not None:
            if not isinstance(raw_auth_path, str) or not raw_auth_path.strip():
                raise CliError(f"Config file '{config_path}' field 'auth.path' must be a non-empty string.")
            auth_path = _normalize_symbol_reference(raw_auth_path.strip(), config_path=config_path)

    raw_pip_config = payload.get("pip_config_file")
    pip_config_file: Path | None = None
    if raw_pip_config is not None:
        if not isinstance(raw_pip_config, str) or not raw_pip_config.strip():
            raise CliError(f"Config file '{config_path}' field 'pip_config_file' must be a non-empty string.")
        pip_config_file = _resolve_path_from_config(raw_pip_config, config_path=config_path)
        if not pip_config_file.exists():
            raise CliError(f"Pip config file '{pip_config_file}' does not exist.")

    raw_base_image = payload.get("base_image")
    if raw_base_image is not None and (not isinstance(raw_base_image, str) or not raw_base_image.strip()):
        raise CliError(f"Config file '{config_path}' field 'base_image' must be a non-empty string.")

    raw_python_version = payload.get("python_version")
    if raw_python_version is not None and (not isinstance(raw_python_version, str) or not raw_python_version.strip()):
        raise CliError(f"Config file '{config_path}' field 'python_version' must be a non-empty string.")

    raw_image_distro = payload.get("image_distro")
    if raw_image_distro is not None and (not isinstance(raw_image_distro, str) or not raw_image_distro.strip()):
        raise CliError(f"Config file '{config_path}' field 'image_distro' must be a non-empty string.")

    raw_dockerfile_lines = payload.get("dockerfile_lines", [])
    if raw_dockerfile_lines is None:
        raw_dockerfile_lines = []
    if not isinstance(raw_dockerfile_lines, list) or not all(isinstance(item, str) for item in raw_dockerfile_lines):
        raise CliError(f"Config file '{config_path}' field 'dockerfile_lines' must be an array of strings.")

    return CliConfig(
        dependencies=[item.strip() for item in raw_dependencies],
        graphs=payload["graphs"],  # validated by _load_config_payload
        env_mapping=env_mapping,
        env_file=env_file,
        auth_path=auth_path,
        base_image=raw_base_image.strip() if isinstance(raw_base_image, str) else None,
        python_version=raw_python_version.strip() if isinstance(raw_python_version, str) else None,
        image_distro=raw_image_distro.strip() if isinstance(raw_image_distro, str) else None,
        pip_config_file=pip_config_file,
        dockerfile_lines=list(raw_dockerfile_lines),
    )


def build_runtime_env(
    *,
    config_path: Path | None,
    env_file: str | None,
    cwd: Path,
    base_env: dict[str, str] | None = None,
) -> dict[str, str]:
    env = dict(os.environ if base_env is None else base_env)
    config: CliConfig | None = _load_cli_config(config_path) if config_path is not None else None
    if config is not None:
        if config.env_file is not None:
            env.update(_parse_env_file(config.env_file))
        env.update(config.env_mapping)
        if config.auth_path:
            env["AUTH_MODULE_PATH"] = config.auth_path
    if env_file:
        resolved_env_file = _resolve_path(env_file, cwd=cwd)
        if not resolved_env_file.exists():
            raise CliError(f"Env file '{resolved_env_file}' does not exist.")
        env.update(_parse_env_file(resolved_env_file))
    if config_path is not None:
        env["AGENTSEEK_GRAPHS"] = str(config_path)
    return env


def build_uvicorn_command(*, host: str, port: int, reload_enabled: bool) -> list[str]:
    command = ["uvicorn", "agentseek_api.main:app", "--host", host, "--port", str(port)]
    if reload_enabled:
        command.append("--reload")
    return command


def build_worker_command() -> list[str]:
    return [sys.executable, "-m", "agentseek_api.worker"]


def build_scheduler_command() -> list[str]:
    return [sys.executable, "-m", "agentseek_api.scheduler"]


def _default_runner(command: list[str], *, env: dict[str, str], cwd: str | None = None) -> int:
    completed = subprocess.run(command, env=env, cwd=cwd, check=False)
    return completed.returncode


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
        scalar_docs_url=f"{api_url}/scalar-docs",
        studio_url=f"{studio_origin}/studio/?baseUrl={studio_base_url}",
    )


def _render_dev_ready_banner(urls: DevServerUrls) -> str:
    return (
        "> Ready!\n"
        ">\n"
        f"> - API: {urls.api_url}\n"
        ">\n"
        f"> - Docs (Swagger): {urls.docs_url}\n"
        ">\n"
        f"> - Docs (Scalar): {urls.scalar_docs_url}\n"
        ">\n"
        f"> - LangSmith Studio Web UI: {urls.studio_url}\n"
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
            raise CliError(f"Development server exited before becoming ready (exit code {process.returncode}).")
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
        wait_for_ready(urls.api_url, process=process, sleep=sleep)
        stdout.write(_render_dev_ready_banner(urls))
        stdout.flush()
        if open_browser:
            if browser_opener is None:
                import webbrowser

                browser_opener = webbrowser.open
            browser_opener(urls.studio_url)
        return process.wait()
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


def _execute_runtime_command(args: argparse.Namespace, *, runner: Callable[..., int], cwd: Path) -> int:
    config_path = discover_config_path(explicit_path=args.config, cwd=cwd)
    env = build_runtime_env(config_path=config_path, env_file=args.env_file, cwd=cwd)
    command = build_uvicorn_command(
        host=args.host,
        port=args.port,
        reload_enabled=getattr(args, "reload", False),
    )
    return runner(command, env=env, cwd=str(cwd))


def _execute_dev_command(
    args: argparse.Namespace,
    *,
    runner: Callable[..., int],
    cwd: Path,
    stdout: TextIO,
) -> int:
    args.reload = not args.no_reload
    config_path = discover_config_path(explicit_path=args.config, cwd=cwd)
    env = build_runtime_env(config_path=config_path, env_file=args.env_file, cwd=cwd)
    env["STUDIO_AUTH_LOCAL_DEV"] = "true"
    command = build_uvicorn_command(
        host=args.host,
        port=args.port,
        reload_enabled=args.reload,
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


def _execute_worker_command(args: argparse.Namespace, *, runner: Callable[..., int], cwd: Path) -> int:
    config_path = discover_config_path(explicit_path=args.config, cwd=cwd)
    env = build_runtime_env(config_path=config_path, env_file=args.env_file, cwd=cwd)
    if runner is _default_runner:
        from agentseek_api import worker as worker_module

        previous_env = os.environ.copy()
        previous_cwd = Path.cwd()
        try:
            os.environ.clear()
            os.environ.update(env)
            os.chdir(cwd)
            return worker_module.main()
        finally:
            os.chdir(previous_cwd)
            os.environ.clear()
            os.environ.update(previous_env)
    return runner(build_worker_command(), env=env, cwd=str(cwd))


def _execute_scheduler_command(args: argparse.Namespace, *, runner: Callable[..., int], cwd: Path) -> int:
    config_path = discover_config_path(explicit_path=args.config, cwd=cwd)
    env = build_runtime_env(config_path=config_path, env_file=args.env_file, cwd=cwd)
    if runner is _default_runner:
        from agentseek_api import scheduler as scheduler_module

        previous_env = os.environ.copy()
        previous_cwd = Path.cwd()
        try:
            os.environ.clear()
            os.environ.update(env)
            os.chdir(cwd)
            return scheduler_module.main()
        finally:
            os.chdir(previous_cwd)
            os.environ.clear()
            os.environ.update(previous_env)
    return runner(build_scheduler_command(), env=env, cwd=str(cwd))


def _load_config_payload(config_path: Path) -> dict[str, object]:
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise CliError(f"Config file '{config_path}' could not be parsed as JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise CliError(f"Config file '{config_path}' must contain a top-level JSON object.")
    graphs = payload.get("graphs")
    if not isinstance(graphs, dict) or not graphs:
        raise CliError(f"Config file '{config_path}' must contain a non-empty 'graphs' object.")
    return payload


def _container_config_path(*, config_path: Path, cwd: Path) -> str:
    try:
        relative_path = config_path.relative_to(cwd)
    except ValueError as exc:
        raise CliError(f"Config file '{config_path}' must live under the project root '{cwd}' for Docker builds.") from exc
    return f"/deps/agent/{relative_path.as_posix()}"


def _containerize_symbol_reference(reference: str, *, cwd: Path) -> str:
    parts = _split_symbol_reference(reference)
    if parts is None:
        return reference
    module_name, symbol_name = parts
    if module_name.endswith(".py") or module_name.startswith(".") or "/" in module_name or "\\" in module_name:
        resolved_module = _resolve_path(module_name, cwd=cwd)
        return f"{_container_config_path(config_path=resolved_module, cwd=cwd)}:{symbol_name}"
    return reference


def _resolve_dependency_path(dependency: str, *, config_path: Path) -> Path:
    dependency_path = Path(dependency).expanduser()
    if dependency_path.is_absolute():
        return dependency_path.resolve()
    return (config_path.parent / dependency_path).resolve()


def _is_local_dependency(dependency: str) -> bool:
    return dependency == "." or dependency.startswith(".") or "/" in dependency or "\\" in dependency


def _dependency_install_command(*, dependency_path: Path, cwd: Path) -> str | None:
    container_path = _container_config_path(config_path=dependency_path, cwd=cwd)
    if (dependency_path / "pyproject.toml").exists() or (dependency_path / "setup.py").exists():
        return f"pip install --no-cache-dir {container_path}"
    if (dependency_path / "requirements.txt").exists():
        return f"pip install --no-cache-dir -r {container_path}/requirements.txt"
    return None


def _find_installable_project_root(*, start: Path, cwd: Path) -> Path:
    start_resolved = start.resolve()
    cwd_resolved = cwd.resolve()
    for candidate in [start_resolved, *start_resolved.parents]:
        if candidate == cwd_resolved or cwd_resolved in candidate.parents:
            if (
                (candidate / "pyproject.toml").exists()
                or (candidate / "setup.py").exists()
                or (candidate / "requirements.txt").exists()
            ):
                return candidate
        if candidate == cwd_resolved:
            break
    return start_resolved


def _root_install_command(*, project_root: Path, cwd: Path) -> str | None:
    container_path = _container_config_path(config_path=project_root, cwd=cwd)
    if (project_root / "pyproject.toml").exists() or (project_root / "setup.py").exists():
        return f"pip install --no-cache-dir {container_path}"
    if (project_root / "requirements.txt").exists():
        return f"pip install --no-cache-dir -r {container_path}/requirements.txt"
    return None


def _docker_dependency_plan(*, config: CliConfig, config_path: Path, cwd: Path) -> tuple[list[str], list[str]]:
    pythonpath_entries = ["/deps/agent"]
    install_commands: list[str] = []
    seen_pythonpath: set[str] = set()
    seen_install_commands: set[str] = set()

    for dependency in config.dependencies:
        if _is_local_dependency(dependency):
            dependency_path = _resolve_dependency_path(dependency, config_path=config_path)
            container_path = _container_config_path(config_path=dependency_path, cwd=cwd)
            if container_path not in seen_pythonpath:
                pythonpath_entries.append(container_path)
                seen_pythonpath.add(container_path)
            install_command = _dependency_install_command(dependency_path=dependency_path, cwd=cwd)
            if install_command is not None and install_command not in seen_install_commands:
                install_commands.append(install_command)
                seen_install_commands.add(install_command)
            continue

        install_command = f"pip install --no-cache-dir {dependency}"
        if install_command not in seen_install_commands:
            install_commands.append(install_command)
            seen_install_commands.add(install_command)

    return pythonpath_entries, install_commands


def _ambient_container_env() -> dict[str, str]:
    return {
        key: value
        for key, value in os.environ.items()
        if key.startswith(_CONTAINER_ENV_PREFIXES)
    }


def build_container_env(*, config_path: Path, env_file: str | None, cwd: Path) -> dict[str, str]:
    env = build_runtime_env(
        config_path=config_path,
        env_file=env_file,
        cwd=cwd,
        base_env=_ambient_container_env(),
    )
    env["AGENTSEEK_GRAPHS"] = _container_config_path(config_path=config_path, cwd=cwd)
    auth_module_path = env.get("AUTH_MODULE_PATH")
    if auth_module_path:
        env["AUTH_MODULE_PATH"] = _containerize_symbol_reference(auth_module_path, cwd=cwd)
    return env


def _default_base_image(*, python_version: str | None, image_distro: str | None) -> str:
    version = (python_version or "3.12").strip()
    distro = (image_distro or "debian").strip().lower()
    if distro in {"", "debian"}:
        return f"python:{version}-slim"
    if distro in {"bookworm", "bullseye"}:
        return f"python:{version}-slim-{distro}"
    if distro == "wolfi":
        raise CliError("image_distro 'wolfi' is not supported without an explicit base_image.")
    raise CliError(f"Unsupported image_distro '{image_distro}'.")


def _supports_apt_get_base_image(base_image: str) -> bool:
    normalized = base_image.strip().lower()
    if normalized.startswith(("python:", "debian:", "ubuntu:", "langchain/langgraph")):
        return "alpine" not in normalized and "wolfi" not in normalized
    return any(marker in normalized for marker in ("debian", "ubuntu", "bookworm", "bullseye"))


def _validate_base_image(base_image: str) -> None:
    if _supports_apt_get_base_image(base_image):
        return
    raise CliError(
        f"Base image '{base_image}' is not supported because generated Dockerfiles require apt-get. "
        "Use a Debian/Ubuntu-compatible image such as 'python:3.12-slim' or 'langchain/langgraph-api'."
    )


def render_dockerfile(*, config_path: Path, cwd: Path, base_image_override: str | None = None) -> str:
    config = _load_cli_config(config_path)
    project_root = _find_installable_project_root(start=config_path.parent, cwd=cwd)
    container_config = _container_config_path(config_path=config_path, cwd=cwd)
    pythonpath_entries, dependency_install_commands = _docker_dependency_plan(
        config=config,
        config_path=config_path,
        cwd=cwd,
    )
    root_install_command = _root_install_command(project_root=project_root, cwd=cwd)
    if root_install_command is not None and root_install_command not in dependency_install_commands:
        dependency_install_commands.append(root_install_command)
    base_image = base_image_override or config.base_image or _default_base_image(
        python_version=config.python_version,
        image_distro=config.image_distro,
    )
    _validate_base_image(base_image)
    pip_install_prefix = ""
    if config.pip_config_file is not None:
        pip_config_path = _container_config_path(config_path=config.pip_config_file, cwd=cwd)
        pip_install_prefix = f"PIP_CONFIG_FILE={pip_config_path} "
    return "\n".join(
        [
            f"FROM {base_image}",
            "",
            "ENV PYTHONDONTWRITEBYTECODE=1",
            "ENV PYTHONUNBUFFERED=1",
            f"ENV PYTHONPATH={':'.join(pythonpath_entries)}",
            "",
            "RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*",
            "",
            "WORKDIR /deps/agent",
            "COPY . /deps/agent",
            *[f"RUN {pip_install_prefix}{command}" for command in dependency_install_commands],
            *config.dockerfile_lines,
            f"ENV AGENTSEEK_GRAPHS={container_config}",
            f"EXPOSE {DEFAULT_API_PORT}",
            f'CMD ["python", "-m", "agentseek_api.cli", "serve", "--host", "0.0.0.0", "--port", "{DEFAULT_API_PORT}"]',
            "",
        ]
    )


def write_dockerfile(*, config_path: Path, save_path: Path, cwd: Path, base_image_override: str | None = None) -> Path:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(
        render_dockerfile(config_path=config_path, cwd=cwd, base_image_override=base_image_override),
        encoding="utf-8",
    )
    return save_path


def _execute_dockerfile_command(args: argparse.Namespace, *, stdout: TextIO, cwd: Path) -> int:
    config_path = discover_config_path(explicit_path=args.config, cwd=cwd)
    if config_path is None:
        raise CliError(f"No config file found in '{cwd}'. Expected agentseek.json or langgraph.json.")
    save_path = _resolve_path(args.save_path, cwd=cwd)
    write_dockerfile(config_path=config_path, save_path=save_path, cwd=cwd)
    stdout.write(f"{save_path}\n")
    return 0


def _execute_build_command(args: argparse.Namespace, *, runner: Callable[..., int], cwd: Path) -> int:
    config_path = discover_config_path(explicit_path=args.config, cwd=cwd)
    if config_path is None:
        raise CliError(f"No config file found in '{cwd}'. Expected agentseek.json or langgraph.json.")
    generated_dockerfile = write_dockerfile(
        config_path=config_path,
        save_path=(cwd / ".agentseek" / "Dockerfile").resolve(),
        cwd=cwd,
    )
    command = ["docker", "build"]
    if args.platform:
        command.extend(["--platform", args.platform])
    if args.pull:
        command.append("--pull")
    command.extend(["-t", args.tag, "-f", str(generated_dockerfile), "."])
    env = build_runtime_env(config_path=config_path, env_file=args.env_file, cwd=cwd)
    return runner(command, env=env, cwd=str(cwd))


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
    runner: Callable[..., int],
    env: dict[str, str],
    cwd: Path,
) -> bool:
    inspect_command = ["docker", "container", "inspect", name]
    if runner is _default_runner:
        completed = subprocess.run(
            inspect_command,
            env=env,
            cwd=str(cwd),
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return completed.returncode == 0
    return runner(inspect_command, env=env, cwd=str(cwd)) == 0


def _execute_up_command(args: argparse.Namespace, *, runner: Callable[..., int], cwd: Path) -> int:
    config_path = discover_config_path(explicit_path=args.config, cwd=cwd)
    if config_path is None:
        raise CliError(f"No config file found in '{cwd}'. Expected agentseek.json or langgraph.json.")

    image = args.image
    env = build_runtime_env(config_path=config_path, env_file=args.env_file, cwd=cwd)
    container_env = build_container_env(config_path=config_path, env_file=args.env_file, cwd=cwd)
    if not image:
        image = f"agentseek-up:{args.port}"
        generated_dockerfile = write_dockerfile(
            config_path=config_path,
            save_path=(cwd / ".agentseek" / "Dockerfile").resolve(),
            cwd=cwd,
            base_image_override=args.base_image,
        )
        build_command = ["docker", "build"]
        if args.pull:
            build_command.append("--pull")
        build_command.extend(["-t", image, "-f", str(generated_dockerfile), "."])
        build_exit_code = runner(build_command, env=env, cwd=str(cwd))
        if build_exit_code != 0:
            return build_exit_code

    compose_path: Path | None = None
    if args.docker_compose:
        compose_path = _resolve_path(args.docker_compose, cwd=cwd)
        if not compose_path.exists():
            raise CliError(f"Docker compose file '{compose_path}' does not exist.")

    container_name = _container_name_for_port(args.port)
    if args.recreate:
        runner(["docker", "rm", "-f", container_name], env=env, cwd=str(cwd))
    elif _container_exists(container_name, runner=runner, env=env, cwd=cwd):
        raise CliError(
            f"Container '{container_name}' already exists. Re-run with '--recreate' or remove it manually."
        )

    if compose_path is not None:
        compose_command = ["docker", "compose", "-f", str(compose_path), "up", "-d"]
        if args.recreate:
            compose_command.append("--force-recreate")
        compose_exit_code = runner(compose_command, env=env, cwd=str(cwd))
        if compose_exit_code != 0:
            return compose_exit_code

    command = [
        "docker",
        "run",
        "--detach",
        "--name",
        container_name,
        "--add-host",
        "host.docker.internal:host-gateway",
        "-p",
        f"{args.port}:{DEFAULT_API_PORT}",
    ]
    for key, value in sorted(container_env.items()):
        command.extend(["-e", f"{key}={value}"])
    if args.postgres_uri:
        command.extend(
            [
                "-e",
                f"METADATA_DB_URL={args.postgres_uri}",
                "-e",
                "METADATA_DB_BACKEND=postgresql",
            ]
        )
    command.append(image)
    run_exit_code = runner(command, env=env, cwd=str(cwd))
    if run_exit_code != 0:
        return run_exit_code
    if args.wait:
        _wait_for_http_ready(f"http://127.0.0.1:{args.port}/health", timeout_seconds=30.0)
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

    serve_parser = subparsers.add_parser("serve", parents=[runtime_parent])
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", default=DEFAULT_API_PORT, type=int)

    subparsers.add_parser("worker", parents=[runtime_parent])
    subparsers.add_parser("scheduler", parents=[runtime_parent])

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
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
    cwd: str | Path | None = None,
) -> int:
    command = args.command
    workdir = Path(cwd or Path.cwd()).resolve()
    run = runner or _default_runner
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
            args.reload = False
            return _execute_runtime_command(args, runner=run, cwd=workdir)
        if command == "worker":
            return _execute_worker_command(args, runner=run, cwd=workdir)
        if command == "scheduler":
            return _execute_scheduler_command(args, runner=run, cwd=workdir)
        if command == "dockerfile":
            return _execute_dockerfile_command(args, stdout=out, cwd=workdir)
        if command == "build":
            return _execute_build_command(args, runner=run, cwd=workdir)
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
            return _execute_up_command(args, runner=run, cwd=workdir)
        raise CliError(f"Unsupported command '{command}'.")
    except CliError as exc:
        err.write(f"{exc}\n")
        return 2


def main(
    argv: Sequence[str] | None = None,
    *,
    prog: str | None = None,
    runner: Callable[..., int] | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
    cwd: str | Path | None = None,
) -> int:
    if prog is None:
        prog = _infer_cli_name() if argv is None else DEFAULT_CLI_NAME
    parser = create_parser(prog=prog)
    args = parser.parse_args(list(argv) if argv is not None else None)
    return run_namespace(args, runner=runner, stdout=stdout, stderr=stderr, cwd=cwd)


if __name__ == "__main__":
    raise SystemExit(main())
