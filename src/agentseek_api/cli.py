from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from urllib import error as urllib_error
from urllib import request as urllib_request
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TextIO

from agentseek_api import __version__


class CliError(RuntimeError):
    pass


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


def build_runtime_env(
    *,
    config_path: Path | None,
    env_file: str | None,
    cwd: Path,
    base_env: dict[str, str] | None = None,
) -> dict[str, str]:
    env = dict(base_env or os.environ)
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


def render_dockerfile(*, config_path: Path, cwd: Path) -> str:
    _load_config_payload(config_path)
    container_config = _container_config_path(config_path=config_path, cwd=cwd)
    return "\n".join(
        [
            "FROM python:3.12-slim",
            "",
            "ENV PYTHONDONTWRITEBYTECODE=1",
            "ENV PYTHONUNBUFFERED=1",
            "ENV PYTHONPATH=/deps/agent",
            "",
            "RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*",
            "",
            "WORKDIR /deps/agent",
            "COPY . /deps/agent",
            "RUN pip install --no-cache-dir .",
            f"ENV AGENTSEEK_GRAPHS={container_config}",
            "EXPOSE 2026",
            'CMD ["agentseek", "serve", "--host", "0.0.0.0", "--port", "2026"]',
            "",
        ]
    )


def write_dockerfile(*, config_path: Path, save_path: Path, cwd: Path) -> Path:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(render_dockerfile(config_path=config_path, cwd=cwd), encoding="utf-8")
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
    if not image:
        image = f"agentseek-up:{args.port}"
        generated_dockerfile = write_dockerfile(
            config_path=config_path,
            save_path=(cwd / ".agentseek" / "Dockerfile").resolve(),
            cwd=cwd,
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
    if args.env_file:
        resolved_env_file = _resolve_path(args.env_file, cwd=cwd)
        if not resolved_env_file.exists():
            raise CliError(f"Env file '{resolved_env_file}' does not exist.")
        command.extend(["--env-file", str(resolved_env_file)])
    command.extend(["-e", f"AGENTSEEK_GRAPHS={_container_config_path(config_path=config_path, cwd=cwd)}"])
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
                    "base_image",
                    "watch",
                    "debugger_base_url",
                    "debugger_port",
                    "verbose",
                    "docker_compose",
                ),
            )
            return _execute_up_command(args, runner=run, cwd=workdir)
        return _unimplemented(command)
    except CliError as exc:
        err.write(f"{exc}\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
