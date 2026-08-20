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
    argv = invocation.argv
    if isinstance(invocation, BuildImageInvocation):
        return "build"
    if isinstance(invocation, DockerRunInvocation):
        return "docker-run"
    if argv[:3] == ("docker", "rm", "-f"):
        return "remove"
    if argv[:2] == ("docker", "compose"):
        return "compose"
    return "inspect"


def _parse_environment(output: bytes) -> Mapping[str, str]:
    try:
        text = output.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise BoundaryFailure("Compose environment output was not UTF-8") from exc
    parsed: dict[str, str] = {}
    for line in text.splitlines():
        if not line:
            continue
        name, separator, value = line.partition("=")
        if not separator or not _ENVIRONMENT_NAME.fullmatch(name):
            raise BoundaryFailure("Compose environment output was malformed")
        parsed[name] = value
    return MappingProxyType(parsed)


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
    image: str,
    application: Mapping[str, str],
    docker_environment: Mapping[str, str],
    cwd: Path,
    result_path: Path,
) -> Mapping[str, str]:
    names = tuple(sorted(application))
    container_name = f"agentseek-boundary-probe-{secrets.token_hex(6)}"
    argv = [
        "docker",
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
    argv.extend((image, "python", "-c", _probe_script(names, result_path.name)))
    result = _safe_process(
        tuple(argv),
        cwd=cwd,
        environment={**docker_environment, **application},
    )
    if result.returncode != 0:
        raise BoundaryFailure("synthetic Docker carrier probe failed")
    remaining = _safe_process(
        ("docker", "container", "inspect", container_name),
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

    def _record(self, invocation: ProcessInvocation) -> None:
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
        environment_result = _safe_process(
            (*prefix, "config", "--environment"),
            cwd=invocation.cwd,
            environment=invocation.environment,
            timeout=30,
        )
        if environment_result.returncode != 0:
            raise BoundaryFailure("Compose environment render failed")
        self.compose_environment = _parse_environment(environment_result.stdout)
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
            observed = rendered["services"]["probe"]["environment"][
                "PROJECT_DOTENV_CANARY"
            ]
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise BoundaryFailure("Compose document boundary was malformed") from exc
        if observed != "unset" or self.compose_dotenv_canary in (
            self.compose_environment.values()
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
                ("docker", "image", "save", "--output", str(image_archive), self.image),
                cwd=invocation.cwd,
                environment=invocation.environment,
            )
            if save.returncode != 0:
                raise BoundaryFailure("built image export failed")
            archive_bytes = image_archive.read_bytes()
        history = _safe_process(
            ("docker", "history", "--no-trunc", "--format", "{{json .}}", self.image),
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
            return result
        if isinstance(invocation, DockerRunInvocation):
            if self.image is None:
                raise BoundaryFailure("direct-run image boundary was missing")
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
                image=self.image,
                application=application,
                docker_environment=docker_environment,
                cwd=invocation.cwd,
                result_path=self.result_path,
            )
            return ProcessResult(returncode=0)
        if invocation.argv[:2] == ("docker", "compose") and "up" in invocation.argv:
            return self._render_compose(invocation)
        timeout = (
            invocation.timeout_seconds
            if isinstance(invocation, ControlQueryInvocation)
            else None
        )
        return _safe_process(
            invocation.argv,
            cwd=invocation.cwd,
            environment=invocation.environment,
            stdin=invocation.stdin_bytes,
            timeout=timeout,
        )


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
    wheels = tuple(output.glob("agentseek_api-0.3.0-*.whl"))
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
    (package / "graph.py").write_text("graph = object()\n", encoding="utf-8")
    application_dotenv = root / "application.env"
    application_dotenv.write_text(f"{allowed_name}={allowed_value}\n", encoding="utf-8")
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
    *, image: str, docker_environment: Mapping[str, str], cwd: Path, result_path: Path
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
        image=image,
        application=values,
        docker_environment=docker_environment,
        cwd=cwd,
        result_path=result_path,
    )
    if dict(observed) != dict(values):
        raise BoundaryFailure("direct Docker carrier value-domain boundary failed")


def _remove_owned_image(
    *, image: str, cwd: Path, environment: Mapping[str, str]
) -> None:
    removed = _safe_process(
        ("docker", "image", "rm", "--force", image),
        cwd=cwd,
        environment=environment,
        timeout=30,
    )
    remaining = _safe_process(
        ("docker", "image", "inspect", image),
        cwd=cwd,
        environment=environment,
        timeout=30,
    )
    if removed.returncode != 0 or remaining.returncode == 0:
        raise BoundaryFailure("owned image cleanup boundary failed")


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
                        ),
                        process_transport=transport,
                        cwd=project,
                        runtime_artifact=artifact,
                    )
                    if exit_code != 0:
                        raise BoundaryFailure("agentseek-api up boundary failed")
                    if transport.image is None:
                        raise BoundaryFailure(
                            "built image boundary evidence was missing"
                        )
                    first_run = next(
                        item
                        for item in transport.invocations
                        if item.kind == "docker-run"
                    )
                    controls = {
                        name: value
                        for name, value in first_run.environment.items()
                        if name not in transport.application_environment
                    }
                    _prove_value_domain_carrier(
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
                    if had_previous:
                        assert previous is not None
                        os.environ[disallowed_name] = previous
                    else:
                        os.environ.pop(disallowed_name, None)
                    if transport.image is not None:
                        first = (
                            transport.invocations[0] if transport.invocations else None
                        )
                        environment = {} if first is None else first.environment
                        _remove_owned_image(
                            image=transport.image,
                            cwd=project,
                            environment=environment,
                        )


def _require(condition: bool, boundary: str) -> None:
    if not condition:
        raise BoundaryFailure(f"{boundary} boundary failed")


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
        print(f"container boundary verification failed: {exc}", file=sys.stderr)
        return 1
    except Exception:
        print(
            "container boundary verification failed: real-runtime boundary failed",
            file=sys.stderr,
        )
        return 1
    print(_SUCCESS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
