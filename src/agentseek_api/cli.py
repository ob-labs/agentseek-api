from __future__ import annotations

import argparse
import json
import os
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


class CliError(RuntimeError):
    pass


@dataclass
class CliConfig:
    graphs: dict[str, object]
    env_mapping: dict[str, str] = field(default_factory=dict)
    env_file: Path | None = None
    auth_path: str | None = None
    base_image: str | None = None
    python_version: str | None = None
    image_distro: str | None = None
    pip_config_file: Path | None = None
    dockerfile_lines: list[str] = field(default_factory=list)


def _resolve_path(path_text: str, *, cwd: Path) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = cwd / path
    return path.resolve()


def _flag_name(option_name: str) -> str:
    return f"--{option_name.replace('_', '-')}"


def _reject_unsupported_options(args: argparse.Namespace, *, command_name: str, option_names: Sequence[str]) -> None:
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
        raise CliError(f"Unsupported option(s) for 'agentseek {command_name}': {', '.join(unsupported)}")


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
            env["AUTH_TYPE"] = "custom"
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


def _default_runner(command: list[str], *, env: dict[str, str], cwd: str | None = None) -> int:
    completed = subprocess.run(command, env=env, cwd=cwd, check=False)
    return completed.returncode


def _execute_runtime_command(args: argparse.Namespace, *, runner: Callable[..., int], cwd: Path) -> int:
    config_path = discover_config_path(explicit_path=args.config, cwd=cwd)
    env = build_runtime_env(config_path=config_path, env_file=args.env_file, cwd=cwd)
    command = build_uvicorn_command(
        host=args.host,
        port=args.port,
        reload_enabled=getattr(args, "reload", False),
    )
    return runner(command, env=env, cwd=str(cwd))


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


def build_container_env(*, config_path: Path, env_file: str | None, cwd: Path) -> dict[str, str]:
    env = build_runtime_env(config_path=config_path, env_file=env_file, cwd=cwd, base_env={})
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


def render_dockerfile(*, config_path: Path, cwd: Path, base_image_override: str | None = None) -> str:
    config = _load_cli_config(config_path)
    container_config = _container_config_path(config_path=config_path, cwd=cwd)
    base_image = base_image_override or config.base_image or _default_base_image(
        python_version=config.python_version,
        image_distro=config.image_distro,
    )
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
            "ENV PYTHONPATH=/deps/agent",
            "",
            "RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*",
            "",
            "WORKDIR /deps/agent",
            "COPY . /deps/agent",
            *config.dockerfile_lines,
            f"RUN {pip_install_prefix}pip install --no-cache-dir .",
            f"ENV AGENTSEEK_GRAPHS={container_config}",
            "EXPOSE 2026",
            'CMD ["agentseek", "serve", "--host", "0.0.0.0", "--port", "2026"]',
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

    container_name = _container_name_for_port(args.port)
    if args.recreate:
        runner(["docker", "rm", "-f", container_name], env=env, cwd=str(cwd))
    if args.docker_compose:
        compose_path = _resolve_path(args.docker_compose, cwd=cwd)
        if not compose_path.exists():
            raise CliError(f"Docker compose file '{compose_path}' does not exist.")
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
        "--rm",
        "--name",
        container_name,
        "--add-host",
        "host.docker.internal:host-gateway",
        "-p",
        f"{args.port}:2026",
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
    stdout.write(f"agentseek {__version__}\n")
    stdout.write(f"agentseek-api {__version__}\n")
    return 0


def _unimplemented(command_name: str) -> int:
    raise CliError(f"'agentseek {command_name}' is not implemented yet in this milestone slice.")


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agentseek")
    subparsers = parser.add_subparsers(dest="command", required=True)

    runtime_parent = argparse.ArgumentParser(add_help=False)
    runtime_parent.add_argument("-c", "--config")
    runtime_parent.add_argument("--env-file")

    dev_parser = subparsers.add_parser("dev", parents=[runtime_parent])
    dev_parser.add_argument("--host", default="127.0.0.1")
    dev_parser.add_argument("--port", default=2026, type=int)
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
    serve_parser.add_argument("--port", default=2026, type=int)

    subparsers.add_parser("version")

    build_parser = subparsers.add_parser("build", parents=[runtime_parent])
    build_parser.add_argument("--platform")
    build_parser.add_argument("-t", "--tag", required=True)
    build_parser.add_argument("--pull", dest="pull", action="store_true", default=True)
    build_parser.add_argument("--no-pull", dest="pull", action="store_false")

    deploy_parser = subparsers.add_parser("deploy", parents=[runtime_parent])
    deploy_parser.add_argument("--api-key")
    deploy_parser.add_argument("--name")
    deploy_parser.add_argument("--deployment-id")
    deploy_parser.add_argument("--deployment-type", default="dev")
    deploy_parser.add_argument("--no-wait", action="store_true")
    deploy_parser.add_argument("--verbose", action="store_true")
    deploy_subparsers = deploy_parser.add_subparsers(dest="deploy_command")
    deploy_subparsers.add_parser("list")
    revisions_parser = deploy_subparsers.add_parser("revisions")
    revisions_subparsers = revisions_parser.add_subparsers(dest="revisions_command")
    revisions_list_parser = revisions_subparsers.add_parser("list")
    revisions_list_parser.add_argument("deployment_id")
    revisions_list_parser.add_argument("--limit", default=10, type=int)
    delete_parser = deploy_subparsers.add_parser("delete")
    delete_parser.add_argument("deployment_id")
    delete_parser.add_argument("--force", action="store_true")
    logs_parser = deploy_subparsers.add_parser("logs")
    logs_parser.add_argument("-f", "--follow", action="store_true")
    logs_parser.add_argument("--end-time")
    logs_parser.add_argument("--start-time")
    logs_parser.add_argument("-q", "--query")
    logs_parser.add_argument("--limit", default=100, type=int)
    logs_parser.add_argument("--level")
    logs_parser.add_argument("--revision-id")
    logs_parser.add_argument("--type", default="deploy")
    logs_parser.add_argument("--deployment-id")
    logs_parser.add_argument("--name")

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

    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    runner: Callable[..., int] | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
    cwd: str | Path | None = None,
) -> int:
    parser = create_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
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
                    "no_browser",
                    "studio_url",
                    "allow_blocking",
                    "tunnel",
                ),
            )
            args.reload = not args.no_reload
            return _execute_runtime_command(args, runner=run, cwd=workdir)
        if command == "serve":
            args.reload = False
            return _execute_runtime_command(args, runner=run, cwd=workdir)
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
        return _unimplemented(command)
    except CliError as exc:
        err.write(f"{exc}\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
