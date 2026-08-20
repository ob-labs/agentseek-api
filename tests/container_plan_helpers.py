from __future__ import annotations

import json
import os
import re
import subprocess
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import AbstractSet

from agentseek_api.environment import ResolvedEnvironment


@dataclass(frozen=True)
class ComposeDecodedEnvironment(Mapping[str, str]):
    substitution: Mapping[str, str] = field(repr=False)
    rendered: Mapping[str, str] = field(repr=False)
    commands: Mapping[str, tuple[str, ...]] = field(
        default_factory=lambda: MappingProxyType({}), repr=False
    )
    runtime: Mapping[str, str] = field(
        default_factory=lambda: MappingProxyType({}), repr=False
    )

    def __getitem__(self, key: str) -> str:
        return self.rendered[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.rendered)

    def __len__(self) -> int:
        return len(self.rendered)


def _synthetic_docker_environment() -> dict[str, str]:
    allowed = {
        "PATH",
        "HOME",
        "USERPROFILE",
        "SYSTEMROOT",
        "DOCKER_HOST",
        "DOCKER_CONTEXT",
        "DOCKER_CONFIG",
        "DOCKER_TLS_VERIFY",
        "DOCKER_CERT_PATH",
    }
    return {name: value for name, value in os.environ.items() if name in allowed}


def _run_private(command: list[str], *, cwd: Path) -> bytes:
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=_synthetic_docker_environment(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        shell=False,
        timeout=30,
    )
    if completed.returncode != 0:
        raise RuntimeError("Compose conformance query failed.")
    return completed.stdout


def docker_daemon_available(*, cwd: Path) -> bool:
    try:
        completed = subprocess.run(
            ["docker", "info", "--format", "{{json .ServerVersion}}"],
            cwd=cwd,
            env=_synthetic_docker_environment(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            shell=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return completed.returncode == 0


def docker_compose_available(*, cwd: Path) -> bool:
    try:
        completed = subprocess.run(
            ["docker", "compose", "version", "--short"],
            cwd=cwd,
            env=_synthetic_docker_environment(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            shell=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    if completed.returncode != 0:
        return False
    match = re.fullmatch(
        rb"v?(\d+)\.(\d+)\.(\d+)(?:[-+][0-9A-Za-z.-]+)?\r?\n?", completed.stdout
    )
    return match is not None and tuple(int(part) for part in match.groups()) >= (
        2,
        24,
        0,
    )


def _parse_compose_environment(output: bytes) -> dict[str, str]:
    text = output.decode("utf-8", errors="strict")
    matches = list(
        re.finditer(
            r"(?ms)^([A-Za-z_][A-Za-z0-9_]*)=(.*?)(?=^[A-Za-z_][A-Za-z0-9_]*=|\Z)",
            text,
        )
    )
    return {match.group(1): match.group(2).removesuffix("\n") for match in matches}


def decode_with_supported_compose(
    encoded: str,
    *,
    tmp_path: Path,
    run_service: bool = False,
) -> ComposeDecodedEnvironment:
    """Inspect Compose substitution, rendered JSON, and optional runtime bytes."""

    names = tuple(
        line.partition("=")[0] for line in encoded.splitlines() if line.strip()
    )
    if any(not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) for name in names):
        raise ValueError("Encoded Compose input has an invalid name.")
    env_path = tmp_path / "explicit-compose.env"
    compose_path = tmp_path / "compose-conformance.json"
    result_path = tmp_path / "compose-results"
    result_path.mkdir(mode=0o700, exist_ok=True)
    env_path.write_text(encoded, encoding="utf-8")
    (tmp_path / ".env").write_text(
        "PROJECT_DOTENV_CANARY=must-not-load\n", encoding="utf-8"
    )
    services = {
        f"probe-{name.lower().replace('_', '-')}": {
            "image": "busybox:1.36",
            "environment": {
                name: f"${{{name}}}",
                "PROJECT_DOTENV_CANARY": "${PROJECT_DOTENV_CANARY-unset}",
            },
            "command": [
                "sh",
                "-c",
                f"umask 077; printf '%s' \"$${{{name}}}\" > /result/{name}",
            ],
            "volumes": [f"{result_path}:/result"],
        }
        for name in names
    }
    compose_path.write_text(
        json.dumps({"name": "agentseek-compose-conformance", "services": services}),
        encoding="utf-8",
    )
    base = [
        "docker",
        "compose",
        "--env-file",
        str(env_path),
        "-f",
        str(compose_path),
    ]
    substitution = _parse_compose_environment(
        _run_private([*base, "config", "--environment"], cwd=tmp_path)
    )
    rendered_document = json.loads(
        _run_private([*base, "config", "--format", "json"], cwd=tmp_path)
    )
    rendered: dict[str, str] = {}
    commands: dict[str, tuple[str, ...]] = {}
    for name in names:
        service = rendered_document["services"][
            f"probe-{name.lower().replace('_', '-')}"
        ]
        if service["environment"]["PROJECT_DOTENV_CANARY"] != "unset":
            raise RuntimeError("Compose loaded the project dotenv unexpectedly.")
        rendered[name] = service["environment"][name]
        commands[name] = tuple(service["command"])

    runtime: dict[str, str] = {}
    if run_service:
        try:
            _run_private(
                [*base, "up", "--abort-on-container-exit", "--remove-orphans"],
                cwd=tmp_path,
            )
            runtime = {
                name: (result_path / name).read_bytes().decode("utf-8")
                for name in names
            }
        finally:
            subprocess.run(
                [*base, "down", "--remove-orphans"],
                cwd=tmp_path,
                env=_synthetic_docker_environment(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                shell=False,
                timeout=30,
            )
    return ComposeDecodedEnvironment(
        substitution=MappingProxyType(substitution),
        rendered=MappingProxyType(rendered),
        commands=MappingProxyType(commands),
        runtime=MappingProxyType(runtime),
    )


def resolved_fixture(
    *,
    values: Mapping[str, str],
    declared_keys: AbstractSet[str],
    unresolved_references: Mapping[str, AbstractSet[str]] | None = None,
) -> ResolvedEnvironment:
    return ResolvedEnvironment(
        values=MappingProxyType(dict(values)),
        origins=MappingProxyType({}),
        declared_keys=frozenset(declared_keys),
        unresolved_references=MappingProxyType(
            {
                key: frozenset(names)
                for key, names in (unresolved_references or {}).items()
            }
        ),
        diagnostics=(),
    )
