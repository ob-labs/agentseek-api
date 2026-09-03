#!/usr/bin/env python3
"""Real Docker/Compose acceptance proof for the container environment boundary."""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import stat
import subprocess
import sys
import traceback
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Protocol

from agentseek_api.cli import main as cli_main
from agentseek_api.container_build import candidate_runtime_artifact
from agentseek_api.docker_runtime import (
    BuildImageInvocation,
    ControlQueryInvocation,
    DockerRunInvocation,
    ProcessInvocation,
    ProcessResult,
)
from agentseek_api.secure_temp import private_artifact, private_directory

try:
    from container_image_archive import scan_image_archive
except ModuleNotFoundError:  # Loaded as a module by the unit acceptance tests.
    from scripts.container_image_archive import scan_image_archive


_ROOT = Path(__file__).resolve().parents[1]
_PORT = "48123"
_SUCCESS = "container boundary verification passed"
_ENVIRONMENT_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_URL_USERINFO = re.compile(r"(?P<scheme>https?://)[^/@\s]+@", re.IGNORECASE)


class BoundaryFailure(RuntimeError):
    """A value-free acceptance-boundary failure."""


@dataclass(frozen=True)
class CapturedInvocation:
    kind: str
    argv: tuple[str, ...] = field(repr=False)
    environment: Mapping[str, str] = field(repr=False)
    stdin_sha256: str | None = field(repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "argv", tuple(self.argv))
        object.__setattr__(
            self, "environment", MappingProxyType(dict(self.environment))
        )


@dataclass(frozen=True)
class BoundaryEvidence:
    disallowed_value: str = field(repr=False)
    allowed_value: str = field(repr=False)
    build_environment: Mapping[str, str] = field(repr=False)
    compose_environment: Mapping[str, str] = field(repr=False)
    build_context_archive: bytes = field(repr=False)
    image_archive_and_history: bytes = field(repr=False)
    application_environment: Mapping[str, str] = field(repr=False)
    invocations: tuple[CapturedInvocation, ...] = field(repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "build_environment", MappingProxyType(dict(self.build_environment))
        )
        object.__setattr__(
            self,
            "compose_environment",
            MappingProxyType(dict(self.compose_environment)),
        )
        object.__setattr__(
            self,
            "application_environment",
            MappingProxyType(dict(self.application_environment)),
        )
        object.__setattr__(self, "invocations", tuple(self.invocations))


class EvidenceCollector(Protocol):
    def __call__(
        self,
        *,
        disallowed_name: str,
        disallowed_value: str,
        allowed_name: str,
        allowed_value: str,
    ) -> BoundaryEvidence: ...


def _safe_process(
    argv: tuple[str, ...],
    *,
    cwd: Path,
    environment: Mapping[str, str],
    stdin: bytes | None = None,
    timeout: float | None = None,
) -> ProcessResult:
    try:
        completed = subprocess.run(
            list(argv),
            cwd=cwd,
            env=dict(environment),
            input=stdin,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            shell=False,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise BoundaryFailure("container control query timed out") from exc
    except OSError as exc:
        raise BoundaryFailure("container process could not be started") from exc
    return ProcessResult(
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def _classify(invocation: ProcessInvocation) -> str:
    argv = invocation.argv[1:]
    if isinstance(invocation, BuildImageInvocation):
        return "build"
    if isinstance(invocation, DockerRunInvocation):
        return "docker-run"
    if argv[:2] == ("rm", "-f"):
        return "remove"
    if argv[:1] == ("compose",):
        return "compose"
    return "inspect"


def _value_free_process_tail(
    result: ProcessResult, *, redactions: tuple[bytes, ...]
) -> str:
    output = result.stdout + b"\n" + result.stderr
    for value in redactions:
        if value:
            output = output.replace(value, b"<redacted>")
    text = output.decode("utf-8", errors="replace")
    text = _URL_USERINFO.sub(r"\g<scheme><redacted>@", text)
    normalized = " | ".join(line.strip() for line in text.splitlines() if line.strip())
    return normalized[-800:]


def _invocation_stage(invocation: ProcessInvocation) -> str:
    if isinstance(invocation, BuildImageInvocation):
        return "candidate image build"
    if isinstance(invocation, DockerRunInvocation):
        return "direct carrier probe"
    argv = invocation.argv[1:]
    if argv[:2] == ("compose", "version"):
        return "Compose capability query"
    if argv[:2] == ("buildx", "version"):
        return "Buildx version query"
    if argv[:2] == ("buildx", "inspect"):
        return "Buildx availability query"
    if argv[:2] == ("image", "inspect"):
        return "image contract query"
    if argv[:2] == ("rm", "-f"):
        return "container cleanup"
    if argv[:1] == ("compose",):
        return "Compose render"
    return "Docker control query"


def _probe_script(names: tuple[str, ...], result_name: str) -> str:
    if Path(result_name).name != result_name or not result_name:
        raise BoundaryFailure("synthetic probe result name boundary failed")
    encoded_names = json.dumps(names, ensure_ascii=True, separators=(",", ":"))
    encoded_result_path = json.dumps(f"/result/{result_name}", ensure_ascii=True)
    return (
        "import json,os,pathlib;"
        f"p=pathlib.Path({encoded_result_path});"
        f"names={encoded_names};"
        "p.write_text(json.dumps({n:os.environ.get(n) for n in names},"
        "ensure_ascii=False,sort_keys=True,separators=(',',':')),encoding='utf-8');"
        "p.chmod(0o600)"
    )


def _run_probe(
    *,
    docker_executable: str,
    image: str,
    application: Mapping[str, str],
    docker_environment: Mapping[str, str],
    cwd: Path,
    result_path: Path,
) -> Mapping[str, str]:
    names = tuple(sorted(application))
    container_name = f"agentseek-boundary-probe-{secrets.token_hex(6)}"
    argv = [
        docker_executable,
        "run",
        "--rm",
        "--name",
        container_name,
    ]
    if os.name != "nt":
        argv.extend(("--user", f"{os.getuid()}:{os.getgid()}"))
    argv.extend(
        (
            "--mount",
            f"type=bind,src={result_path.parent},dst=/result",
        )
    )
    for name in names:
        argv.extend(("-e", name))
    argv.extend(
        (
            "--entrypoint",
            "python",
            image,
            "-I",
            "-c",
            _probe_script(names, result_path.name),
        )
    )
    result = _safe_process(
        tuple(argv),
        cwd=cwd,
        environment={**docker_environment, **application},
    )
    if result.returncode != 0:
        raise BoundaryFailure("synthetic Docker carrier probe failed")
    remaining = _safe_process(
        (docker_executable, "container", "inspect", container_name),
        cwd=cwd,
        environment=docker_environment,
        timeout=30,
    )
    if remaining.returncode == 0:
        raise BoundaryFailure("synthetic probe cleanup boundary failed")
    try:
        status = result_path.stat()
        if stat.S_IMODE(status.st_mode) != 0o600:
            raise BoundaryFailure("synthetic probe result mode boundary failed")
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BoundaryFailure("synthetic probe result boundary failed") from exc
    if not isinstance(payload, dict) or not all(
        isinstance(name, str) and (value is None or isinstance(value, str))
        for name, value in payload.items()
    ):
        raise BoundaryFailure("synthetic probe result shape boundary failed")
    return MappingProxyType(dict(payload))


@dataclass
class _EvidenceTransport:
    result_path: Path
    compose_dotenv_canary: str = field(repr=False)
    forbidden_values: tuple[bytes, ...] = field(repr=False)
    invocations: list[CapturedInvocation] = field(default_factory=list, repr=False)
    build_environment: Mapping[str, str] = field(
        default_factory=lambda: MappingProxyType({}), repr=False
    )
    compose_environment: Mapping[str, str] = field(
        default_factory=lambda: MappingProxyType({}), repr=False
    )
    build_context_archive: bytes = field(default=b"", repr=False)
    image_archive_and_history: bytes = field(default=b"", repr=False)
    application_environment: Mapping[str, str] = field(
        default_factory=lambda: MappingProxyType({}), repr=False
    )
    image: str | None = None
    docker_executable: str | None = None
    container_name: str | None = None
    last_stage: str = "planning"
    last_failure_detail: str = field(default="", repr=False)

    def require_docker_executable(self) -> str:
        if self.docker_executable is None:
            raise BoundaryFailure("Docker executable evidence was missing")
        return self.docker_executable

    def _record(self, invocation: ProcessInvocation) -> None:
        if self.docker_executable is None:
            self.docker_executable = invocation.argv[0]
        elif self.docker_executable != invocation.argv[0]:
            raise BoundaryFailure("Docker executable identity changed")
        digest = (
            None
            if invocation.stdin_bytes is None
            else hashlib.sha256(invocation.stdin_bytes).hexdigest()
        )
        self.invocations.append(
            CapturedInvocation(
                kind=_classify(invocation),
                argv=invocation.argv,
                environment=invocation.environment,
                stdin_sha256=digest,
            )
        )

    def _render_compose(self, invocation: ProcessInvocation) -> ProcessResult:
        try:
            command_index = invocation.argv.index("up")
        except ValueError as exc:
            raise BoundaryFailure("Compose invocation classification failed") from exc
        prefix = invocation.argv[:command_index]
        rendered_result = _safe_process(
            (*prefix, "config", "--format", "json"),
            cwd=invocation.cwd,
            environment=invocation.environment,
            timeout=30,
        )
        if rendered_result.returncode != 0:
            raise BoundaryFailure("Compose document render failed")
        try:
            rendered = json.loads(rendered_result.stdout)
            service_environment = rendered["services"]["probe"]["environment"]
            if not isinstance(service_environment, dict) or not all(
                isinstance(name, str) and isinstance(value, str)
                for name, value in service_environment.items()
            ):
                raise TypeError
            observed = service_environment["PROJECT_DOTENV_CANARY"]
        except (
            KeyError,
            TypeError,
            UnicodeDecodeError,
            json.JSONDecodeError,
        ) as exc:
            raise BoundaryFailure("Compose document boundary was malformed") from exc
        self.compose_environment = MappingProxyType(dict(service_environment))
        if (
            observed != "unset"
            or self.compose_dotenv_canary in service_environment.values()
        ):
            raise BoundaryFailure("explicit Compose env-file isolation failed")
        return ProcessResult(returncode=0)

    def _capture_image(self, invocation: BuildImageInvocation) -> None:
        try:
            tag_index = invocation.argv.index("--tag")
            self.image = invocation.argv[tag_index + 1]
        except (ValueError, IndexError) as exc:
            raise BoundaryFailure("built image tag boundary was missing") from exc
        with private_artifact(
            prefix="agentseek-boundary-image-",
            contents=b"",
            tmp_root=invocation.cwd.parent,
        ) as image_archive:
            save = _safe_process(
                (
                    invocation.argv[0],
                    "image",
                    "save",
                    "--output",
                    str(image_archive),
                    self.image,
                ),
                cwd=invocation.cwd,
                environment=invocation.environment,
            )
            if save.returncode != 0:
                raise BoundaryFailure("built image export failed")
            archive_bytes = image_archive.read_bytes()
        history = _safe_process(
            (
                invocation.argv[0],
                "history",
                "--no-trunc",
                "--format",
                "{{json .}}",
                self.image,
            ),
            cwd=invocation.cwd,
            environment=invocation.environment,
            timeout=30,
        )
        if history.returncode != 0:
            raise BoundaryFailure("built image history query failed")
        for forbidden in self.forbidden_values:
            scan_image_archive(
                archive_bytes,
                forbidden=forbidden,
                history=history.stdout,
            )
        self.image_archive_and_history = archive_bytes + b"\n" + history.stdout

    def __call__(self, invocation: ProcessInvocation) -> ProcessResult:
        self._record(invocation)
        self.last_stage = _invocation_stage(invocation)
        if isinstance(invocation, BuildImageInvocation):
            self.build_environment = MappingProxyType(dict(invocation.environment))
            self.build_context_archive = invocation.stdin_bytes
            result = _safe_process(
                invocation.argv,
                cwd=invocation.cwd,
                environment=invocation.environment,
                stdin=invocation.stdin_bytes,
            )
            if result.returncode == 0:
                self._capture_image(invocation)
            else:
                self.last_failure_detail = _value_free_process_tail(
                    result,
                    redactions=(
                        *self.forbidden_values,
                        *(
                            value.encode()
                            for value in invocation.environment.values()
                            if value
                        ),
                    ),
                )
            return result
        if isinstance(invocation, DockerRunInvocation):
            if self.image is None:
                raise BoundaryFailure("direct-run image boundary was missing")
            try:
                name_index = invocation.argv.index("--name")
                self.container_name = invocation.argv[name_index + 1]
            except (ValueError, IndexError) as exc:
                raise BoundaryFailure(
                    "generated container ownership boundary was missing"
                ) from exc
            application = {
                name: invocation.environment[name]
                for name in invocation.application_names
            }
            docker_environment = {
                name: value
                for name, value in invocation.environment.items()
                if name not in invocation.application_names
            }
            self.application_environment = _run_probe(
                docker_executable=invocation.argv[0],
                image=self.image,
                application=application,
                docker_environment=docker_environment,
                cwd=invocation.cwd,
                result_path=self.result_path,
            )
            return _safe_process(
                invocation.argv,
                cwd=invocation.cwd,
                environment=invocation.environment,
                stdin=invocation.stdin_bytes,
            )
        if invocation.argv[1:2] == ("compose",) and "up" in invocation.argv:
            return self._render_compose(invocation)
        timeout = (
            invocation.timeout_seconds
            if isinstance(invocation, ControlQueryInvocation)
            else None
        )
        result = _safe_process(
            invocation.argv,
            cwd=invocation.cwd,
            environment=invocation.environment,
            stdin=invocation.stdin_bytes,
            timeout=timeout,
        )
        if result.returncode != 0:
            self.last_failure_detail = _value_free_process_tail(
                result,
                redactions=(
                    *self.forbidden_values,
                    *(
                        value.encode()
                        for value in invocation.environment.values()
                        if value
                    ),
                ),
            )
        return result


def _build_candidate(project: Path):
    output = project / "candidate-dist"
    output.mkdir(mode=0o700)
    environment = {
        name: value
        for name, value in os.environ.items()
        if name
        in {
            "PATH",
            "HOME",
            "UV_CACHE_DIR",
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
        }
    }
    result = _safe_process(
        ("uv", "build", "--wheel", "--out-dir", str(output), str(_ROOT)),
        cwd=_ROOT,
        environment=environment,
    )
    wheels = tuple(output.glob("agentseek_api-0.3.1-*.whl"))
    if result.returncode != 0 or len(wheels) != 1:
        raise BoundaryFailure("candidate runtime wheel build failed")
    wheel = wheels[0]
    digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
    return candidate_runtime_artifact(wheel, digest)


def _write_project(
    root: Path,
    *,
    allowed_name: str,
    allowed_value: str,
    compose_dotenv_canary: str,
) -> tuple[Path, Path]:
    package = root / "chat"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "graph.py").write_text(
        "from langgraph.graph import END, START, StateGraph\n"
        "\n"
        "def respond(state):\n"
        "    return {'message': state.get('message', '')}\n"
        "\n"
        "builder = StateGraph(dict)\n"
        "builder.add_node('respond', respond)\n"
        "builder.add_edge(START, 'respond')\n"
        "builder.add_edge('respond', END)\n"
        "graph = builder.compile(name='Boundary Graph')\n",
        encoding="utf-8",
    )
    application_dotenv = root / "application.env"
    application_dotenv.write_text(
        f"{allowed_name}={allowed_value}\n"
        "METADATA_DB_URL=sqlite+aiosqlite:////tmp/agentseek-boundary.db\n",
        encoding="utf-8",
    )
    application_dotenv.chmod(0o600)
    config = root / "agentseek.json"
    config.write_text(
        json.dumps(
            {
                "dependencies": [],
                "graphs": {"chat": "chat/graph.py:graph"},
                "env": "application.env",
                "compose_env": [],
            },
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    (root / ".env").write_text(
        f"PROJECT_DOTENV_CANARY={compose_dotenv_canary}\n", encoding="utf-8"
    )
    compose = root / "compose.yaml"
    compose.write_text(
        "services:\n"
        "  probe:\n"
        "    image: busybox:1.36\n"
        "    environment:\n"
        "      PROJECT_DOTENV_CANARY: ${PROJECT_DOTENV_CANARY-unset}\n",
        encoding="utf-8",
    )
    return config, compose


def _prove_value_domain_carrier(
    *,
    docker_executable: str,
    image: str,
    docker_environment: Mapping[str, str],
    cwd: Path,
    result_path: Path,
) -> None:
    values = MappingProxyType(
        {
            "VALUE_EMPTY": "",
            "VALUE_NEWLINE": "physical\nnewline",
            "VALUE_UNICODE": "海洋数据库",
            "VALUE_DOLLAR": "$cash",
            "VALUE_EXPANSION": "${NOT_EXPANDED}",
            "VALUE_HASH": "value#fragment",
            "VALUE_SPACES": "  leading and trailing  ",
            "VALUE_EQUALS": "left=right",
            "VALUE_SINGLE_QUOTE": "it's literal",
            "VALUE_DOUBLE_QUOTE": 'say "hello"',
            "VALUE_BACKSLASH": r"C:\boundary\path",
        }
    )
    result_path.write_bytes(b"")
    result_path.chmod(0o600)
    observed = _run_probe(
        docker_executable=docker_executable,
        image=image,
        application=values,
        docker_environment=docker_environment,
        cwd=cwd,
        result_path=result_path,
    )
    if dict(observed) != dict(values):
        raise BoundaryFailure("direct Docker carrier value-domain boundary failed")


def _remove_owned_image(
    *,
    docker_executable: str,
    image: str,
    cwd: Path,
    environment: Mapping[str, str],
) -> None:
    removed = _safe_process(
        (docker_executable, "image", "rm", "--force", image),
        cwd=cwd,
        environment=environment,
        timeout=30,
    )
    remaining = _safe_process(
        (docker_executable, "image", "inspect", image),
        cwd=cwd,
        environment=environment,
        timeout=30,
    )
    if removed.returncode != 0 or remaining.returncode == 0:
        raise BoundaryFailure("owned image cleanup boundary failed")


def _remove_owned_container(
    *,
    docker_executable: str,
    container_name: str,
    cwd: Path,
    environment: Mapping[str, str],
) -> None:
    _safe_process(
        (docker_executable, "rm", "-f", container_name),
        cwd=cwd,
        environment=environment,
        timeout=30,
    )
    remaining = _safe_process(
        (docker_executable, "container", "inspect", container_name),
        cwd=cwd,
        environment=environment,
        timeout=30,
    )
    if remaining.returncode == 0:
        raise BoundaryFailure("owned container cleanup boundary failed")


def collect_boundary_evidence(
    *,
    disallowed_name: str,
    disallowed_value: str,
    allowed_name: str,
    allowed_value: str,
) -> BoundaryEvidence:
    """Collect value-redacted evidence from real Docker and Compose processes."""

    if not all(
        _ENVIRONMENT_NAME.fullmatch(name) for name in (disallowed_name, allowed_name)
    ):
        raise BoundaryFailure("boundary sentinel name was invalid")
    if not all(
        value.isascii() and value.isprintable()
        for value in (disallowed_value, allowed_value)
    ):
        raise BoundaryFailure("boundary sentinel value domain was invalid")

    compose_dotenv_canary = secrets.token_urlsafe(24)
    previous = os.environ.get(disallowed_name)
    had_previous = disallowed_name in os.environ
    with private_directory(prefix="agentseek-boundary-") as workspace:
        project = workspace / "project"
        project.mkdir(mode=0o700)
        config, compose = _write_project(
            project,
            allowed_name=allowed_name,
            allowed_value=allowed_value,
            compose_dotenv_canary=compose_dotenv_canary,
        )
        artifact = _build_candidate(project)
        with private_directory(
            prefix="agentseek-boundary-results-", tmp_root=workspace
        ) as result_root:
            with private_artifact(
                prefix="environment-", contents=b"", tmp_root=result_root
            ) as result_path:
                transport = _EvidenceTransport(
                    result_path=result_path,
                    compose_dotenv_canary=compose_dotenv_canary,
                    forbidden_values=(
                        disallowed_value.encode(),
                        allowed_value.encode(),
                    ),
                )
                try:
                    os.environ[disallowed_name] = disallowed_value
                    exit_code = cli_main(
                        (
                            "up",
                            "--config",
                            str(config),
                            "--docker-compose",
                            str(compose),
                            "--port",
                            _PORT,
                            "--recreate",
                            "--wait",
                        ),
                        process_transport=transport,
                        cwd=project,
                        runtime_artifact=artifact,
                    )
                    if exit_code != 0:
                        detail = (
                            f": {transport.last_failure_detail}"
                            if transport.last_failure_detail
                            else ""
                        )
                        raise BoundaryFailure(
                            "agentseek-api up failed during "
                            f"{transport.last_stage}{detail}"
                        )
                    if transport.image is None:
                        raise BoundaryFailure(
                            "built image boundary evidence was missing"
                        )
                    controls = dict(transport.build_environment)
                    _prove_value_domain_carrier(
                        docker_executable=transport.require_docker_executable(),
                        image=transport.image,
                        docker_environment=controls,
                        cwd=project,
                        result_path=result_path,
                    )
                    return BoundaryEvidence(
                        disallowed_value=disallowed_value,
                        allowed_value=allowed_value,
                        build_environment=transport.build_environment,
                        compose_environment=transport.compose_environment,
                        build_context_archive=transport.build_context_archive,
                        image_archive_and_history=transport.image_archive_and_history,
                        application_environment=transport.application_environment,
                        invocations=tuple(transport.invocations),
                    )
                finally:
                    active_failure = sys.exception()
                    if had_previous:
                        assert previous is not None
                        os.environ[disallowed_name] = previous
                    else:
                        os.environ.pop(disallowed_name, None)
                    first = transport.invocations[0] if transport.invocations else None
                    environment = {} if first is None else first.environment
                    cleanup_failure: BoundaryFailure | None = None
                    if transport.container_name is not None:
                        try:
                            _remove_owned_container(
                                docker_executable=transport.require_docker_executable(),
                                container_name=transport.container_name,
                                cwd=project,
                                environment=environment,
                            )
                        except BoundaryFailure as exc:
                            cleanup_failure = exc
                    if transport.image is not None:
                        try:
                            _remove_owned_image(
                                docker_executable=transport.require_docker_executable(),
                                image=transport.image,
                                cwd=project,
                                environment=environment,
                            )
                        except BoundaryFailure as exc:
                            cleanup_failure = cleanup_failure or exc
                    if cleanup_failure is not None:
                        if active_failure is None:
                            raise cleanup_failure
                        active_failure.add_note(
                            "Owned container cleanup could not be verified."
                        )


def _require(condition: bool, boundary: str) -> None:
    if not condition:
        raise BoundaryFailure(f"{boundary} boundary failed")


def _report_github_failure(message: str) -> None:
    if os.environ.get("GITHUB_ACTIONS") != "true":
        return
    escaped = message.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
    print(
        f"::error title=AgentSeek container boundary::{escaped}",
        file=sys.stderr,
    )


def _unexpected_failure_fingerprint(exc: BaseException) -> str:
    location = ""
    for frame in reversed(traceback.extract_tb(exc.__traceback__)):
        path = Path(frame.filename).resolve()
        try:
            relative = path.relative_to(_ROOT)
        except ValueError:
            continue
        location = f" at {relative.as_posix()}:{frame.lineno}"
        break
    return f"real-runtime boundary failed ({type(exc).__name__}{location})"


def _verify_evidence(evidence: BoundaryEvidence) -> None:
    _require(
        evidence.disallowed_value not in evidence.build_environment.values(),
        "build environment isolation",
    )
    _require(
        evidence.disallowed_value not in evidence.compose_environment.values(),
        "Compose environment isolation",
    )
    _require(
        evidence.disallowed_value.encode() not in evidence.build_context_archive,
        "build context isolation",
    )
    _require(
        evidence.disallowed_value.encode() not in evidence.image_archive_and_history,
        "image layer and history isolation",
    )
    _require(
        evidence.disallowed_value not in evidence.application_environment.values(),
        "application environment isolation",
    )
    _require(
        evidence.allowed_value not in evidence.build_environment.values(),
        "selected application build isolation",
    )
    _require(
        evidence.allowed_value not in evidence.compose_environment.values(),
        "selected application Compose isolation",
    )
    _require(
        evidence.allowed_value.encode() not in evidence.build_context_archive,
        "selected application context isolation",
    )
    _require(
        evidence.allowed_value.encode() not in evidence.image_archive_and_history,
        "selected application image isolation",
    )
    _require(
        evidence.application_environment.get("ALLOWED_SENTINEL")
        == evidence.allowed_value,
        "selected application carrier",
    )
    for invocation in evidence.invocations:
        _require(
            all(evidence.disallowed_value not in arg for arg in invocation.argv),
            f"{invocation.kind} argv disallowed-value isolation",
        )
        _require(
            evidence.disallowed_value not in invocation.environment.values(),
            f"{invocation.kind} environment disallowed-value isolation",
        )
        if invocation.kind == "docker-run":
            _require(
                invocation.environment.get("ALLOWED_SENTINEL")
                == evidence.allowed_value,
                "direct-run selected application carrier",
            )
            _require(
                all(evidence.allowed_value not in arg for arg in invocation.argv),
                "direct-run argv selected-value isolation",
            )
        else:
            _require(
                all(evidence.allowed_value not in arg for arg in invocation.argv),
                f"{invocation.kind} argv selected-value isolation",
            )
            _require(
                evidence.allowed_value not in invocation.environment.values(),
                f"{invocation.kind} environment selected-value isolation",
            )


def main(*, evidence_collector: EvidenceCollector = collect_boundary_evidence) -> int:
    try:
        evidence = evidence_collector(
            disallowed_name="DISALLOWED_CANARY",
            disallowed_value=secrets.token_urlsafe(24),
            allowed_name="ALLOWED_SENTINEL",
            allowed_value=secrets.token_hex(32),
        )
        _verify_evidence(evidence)
    except BoundaryFailure as exc:
        _report_github_failure(str(exc))
        print(f"container boundary verification failed: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        failure = _unexpected_failure_fingerprint(exc)
        _report_github_failure(failure)
        print(f"container boundary verification failed: {failure}", file=sys.stderr)
        return 1
    print(_SUCCESS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
