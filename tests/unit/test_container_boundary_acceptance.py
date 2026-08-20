from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from agentseek_api.docker_runtime import ProcessResult


SCRIPT = Path("scripts/test_container_env_boundary.py")
AUTODISCOVERY_SCRIPT = Path("scripts/test_cli_config_autodiscovery.py")


def _load_script():
    if not SCRIPT.is_file():
        pytest.fail("the executable container-boundary acceptance module is missing")
    spec = importlib.util.spec_from_file_location(
        "container_boundary_acceptance", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_autodiscovery_script():
    spec = importlib.util.spec_from_file_location(
        "cli_config_autodiscovery", AUTODISCOVERY_SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_evidence_objects_hide_all_values_from_repr() -> None:
    module = _load_script()
    value = "private-boundary-value"

    invocation = module.CapturedInvocation(
        kind="docker-run",
        argv=("docker", value),
        environment={"VALUE": value},
        stdin_sha256=value,
    )
    evidence = module.BoundaryEvidence(
        disallowed_value=value,
        allowed_value=value,
        build_environment={"VALUE": value},
        compose_environment={"VALUE": value},
        build_context_archive=value.encode(),
        image_archive_and_history=value.encode(),
        application_environment={"VALUE": value},
        invocations=(invocation,),
    )

    assert value not in repr(invocation)
    assert value not in repr(evidence)


def test_success_output_is_one_value_free_line() -> None:
    module = _load_script()
    evidence = module.BoundaryEvidence(
        disallowed_value="disallowed-private-value",
        allowed_value="allowed-private-value",
        build_environment={},
        compose_environment={},
        build_context_archive=b"",
        image_archive_and_history=b"",
        application_environment={"ALLOWED_SENTINEL": "allowed-private-value"},
        invocations=(
            module.CapturedInvocation(
                kind="docker-run",
                argv=("docker", "run", "-e", "ALLOWED_SENTINEL"),
                environment={"ALLOWED_SENTINEL": "allowed-private-value"},
                stdin_sha256=None,
            ),
        ),
    )
    output = io.StringIO()

    with contextlib.redirect_stdout(output):
        result = module.main(evidence_collector=lambda **_: evidence)

    assert result == 0
    assert output.getvalue() == "container boundary verification passed\n"


def test_boundary_failure_emits_value_free_github_annotation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()
    stderr = io.StringIO()
    monkeypatch.setenv("GITHUB_ACTIONS", "true")

    def fail(**_kwargs):
        raise module.BoundaryFailure("synthetic Docker carrier probe failed")

    with contextlib.redirect_stderr(stderr):
        result = module.main(evidence_collector=fail)

    assert result == 1
    lines = stderr.getvalue().splitlines()
    assert lines == [
        "::error title=AgentSeek container boundary::synthetic Docker carrier probe failed",
        "container boundary verification failed: synthetic Docker carrier probe failed",
    ]


def test_generated_compose_flow_uses_build_controls_for_direct_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()
    observed_controls: list[dict[str, str]] = []

    monkeypatch.setattr(module, "_build_candidate", lambda _project: object())
    monkeypatch.setattr(module, "_remove_owned_image", lambda **_kwargs: None)

    def fake_cli(_argv, *, process_transport, **_kwargs):
        process_transport.image = "agentseek:test"
        process_transport.build_environment = {"PATH": "docker-control-only"}
        process_transport.build_context_archive = b"context"
        process_transport.compose_environment = {}
        process_transport.application_environment = {"ALLOWED_SENTINEL": "allowed"}
        process_transport.image_archive_and_history = b"image"
        process_transport.invocations.append(
            module.CapturedInvocation(
                kind="compose",
                argv=("docker", "compose", "up"),
                environment={"PATH": "docker-control-only"},
                stdin_sha256=None,
            )
        )
        return 0

    def fake_probe(*, docker_environment, **_kwargs):
        observed_controls.append(dict(docker_environment))

    monkeypatch.setattr(module, "cli_main", fake_cli)
    monkeypatch.setattr(module, "_prove_value_domain_carrier", fake_probe)

    evidence = module.collect_boundary_evidence(
        disallowed_name="DISALLOWED_CANARY",
        disallowed_value="disallowed",
        allowed_name="ALLOWED_SENTINEL",
        allowed_value="allowed",
    )

    assert evidence.invocations[0].kind == "compose"
    assert observed_controls == [{"PATH": "docker-control-only"}]


def test_process_failure_tail_is_bounded_and_value_free() -> None:
    module = _load_script()
    forbidden = "abc"
    credential = "tiny-password"
    result = ProcessResult(
        returncode=1,
        stdout=("x" * 2_000).encode(),
        stderr=(
            f"prefix{forbidden}suffix "
            f"https://alice:{credential}@registry.invalid real build error"
        ).encode(),
    )

    diagnostic = module._value_free_process_tail(
        result,
        redactions=(forbidden.encode(), credential.encode()),
    )

    assert len(diagnostic) <= 800
    assert "real build error" in diagnostic
    assert forbidden not in diagnostic
    assert credential not in diagnostic
    assert "alice" not in diagnostic


@pytest.mark.parametrize(
    "application",
    [
        {"ALLOWED_SENTINEL": "selected-value"},
        {
            "VALUE_EMPTY": "",
            "VALUE_NEWLINE": "physical\nnewline",
            "VALUE_UNICODE": "海洋数据库",
        },
    ],
)
def test_probe_writes_and_reads_the_exact_private_result_basename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    application: dict[str, str],
) -> None:
    module = _load_script()
    result_path = tmp_path / f"environment-{os.urandom(8).hex()}"
    result_path.write_bytes(b"")
    result_path.chmod(0o600)
    calls: list[tuple[str, ...]] = []

    def fake_process(argv, *, cwd, environment, stdin=None, timeout=None):
        del cwd, stdin, timeout
        calls.append(argv)
        if argv[:3] == ("docker", "container", "inspect"):
            return ProcessResult(returncode=1)
        script = argv[-1].replace("/result/", f"{result_path.parent}/")
        completed = subprocess.run(
            [sys.executable, "-c", script],
            env=dict(environment),
            capture_output=True,
            check=False,
        )
        return ProcessResult(
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )

    monkeypatch.setattr(module, "_safe_process", fake_process)

    observed = module._run_probe(
        image="synthetic:test",
        application=application,
        docker_environment={"PATH": os.environ["PATH"]},
        cwd=tmp_path,
        result_path=result_path,
    )

    assert dict(observed) == application
    assert json.loads(result_path.read_text(encoding="utf-8")) == application
    assert tuple(path.name for path in tmp_path.iterdir()) == (result_path.name,)
    run_argv = calls[0]
    entrypoint_index = run_argv.index("--entrypoint")
    assert run_argv[entrypoint_index : entrypoint_index + 3] == (
        "--entrypoint",
        "python",
        "synthetic:test",
    )
    assert run_argv[-3:-1] == ("-I", "-c")
    assert calls[1][:3] == ("docker", "container", "inspect")


def test_probe_fails_if_auto_remove_leaves_the_owned_container(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_script()
    result_path = tmp_path / "environment-random"
    result_path.write_text("{}", encoding="utf-8")
    result_path.chmod(0o600)

    monkeypatch.setattr(
        module,
        "_safe_process",
        lambda *args, **kwargs: ProcessResult(returncode=0),
    )

    with pytest.raises(module.BoundaryFailure, match="cleanup"):
        module._run_probe(
            image="synthetic:test",
            application={},
            docker_environment={"PATH": os.environ["PATH"]},
            cwd=tmp_path,
            result_path=result_path,
        )


def test_owned_image_cleanup_requires_removal_and_absence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_script()
    results = iter((ProcessResult(returncode=0), ProcessResult(returncode=0)))
    monkeypatch.setattr(module, "_safe_process", lambda *args, **kwargs: next(results))

    with pytest.raises(module.BoundaryFailure, match="cleanup"):
        module._remove_owned_image(
            image="synthetic:test",
            cwd=tmp_path,
            environment={"PATH": os.environ["PATH"]},
        )


@pytest.mark.parametrize("name", ["README.md", "README.zh-CN.md", "CHANGELOG.md"])
def test_container_migration_docs_cover_the_public_boundary(name: str) -> None:
    text = Path(name).read_text(encoding="utf-8")
    required_literals = {
        "build_include",
        "compose_env",
        "--pass-env",
        "--compose-pass-env",
        "preloaded-v1",
        "org.agentseek.environment-contract",
        "org.agentseek.runtime-manifest",
        "org.agentseek.runtime-distribution",
        "org.agentseek.runtime-version",
    }

    assert required_literals <= set(re.findall(r"[\w.-]+|--[\w-]+", text))


def test_release_docs_name_template_catalog_and_agentseek_as_separate_releases() -> (
    None
):
    for name in ("README.md", "README.zh-CN.md", "CHANGELOG.md"):
        text = Path(name).read_text(encoding="utf-8")
        assert re.search(r"template/catalog.{0,80}0\.1\.4", text, re.DOTALL)
        assert re.search(r"AgentSeek.{0,80}0\.1\.4", text, re.DOTALL)


def test_cli_config_autodiscovery_executes_the_bundle_contract() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/test_cli_config_autodiscovery.py"],
        cwd=Path.cwd(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_cli_config_autodiscovery_emits_value_free_ci_failure_annotation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_autodiscovery_script()
    credential = "tiny-password"
    stderr = io.StringIO()
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=2,
            stdout="",
            stderr=(
                "The build output root could not be verified private at "
                f"https://alice:{credential}@registry.invalid\n"
            ),
        ),
    )

    with contextlib.redirect_stderr(stderr), pytest.raises(SystemExit):
        module.main([])

    diagnostic = stderr.getvalue()
    assert diagnostic.startswith("::error title=AgentSeek dockerfile smoke::")
    assert "build output root could not be verified private" in diagnostic
    assert credential not in diagnostic
    assert "alice" not in diagnostic


def test_cli_config_autodiscovery_reports_post_generation_contract_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_autodiscovery_script()
    stderr = io.StringIO()
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="", stderr=""),
    )

    with contextlib.redirect_stderr(stderr), pytest.raises(SystemExit):
        module.main([])

    assert "dockerfile bundle contract was incomplete" in stderr.getvalue()
