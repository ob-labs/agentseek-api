from __future__ import annotations

import contextlib
import importlib.util
import io
import re
import sys
from pathlib import Path

import pytest


SCRIPT = Path("scripts/test_container_env_boundary.py")


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
