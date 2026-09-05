import re
from pathlib import Path

import yaml


def test_cli_compatibility_step_runs_sqlite_runtime_regressions_on_every_os() -> None:
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    job = re.search(
        r"(?ms)^  cli-compatibility:\n(?P<body>.*?)(?=^  \S|\Z)",
        workflow,
    )
    assert job is not None

    step = re.search(
        r"(?ms)^      - name: CLI config, host environment, and process tests\n"
        r"(?P<body>.*?)(?=^      - name:|\Z)",
        job.group("body"),
    )
    assert step is not None

    command_tokens = set(step.group("body").split())
    required_tests = {
        "tests/unit/test_sqlite_checkpointer.py",
        "tests/integration/test_metadata_db_config.py",
    }
    assert required_tests <= command_tokens


def _job(workflow: str, name: str) -> str:
    match = re.search(
        rf"(?ms)^  {re.escape(name)}:\n(?P<body>.*?)(?=^  \S|\Z)",
        workflow,
    )
    assert match is not None
    return match.group("body")


def test_cli_docker_runtime_has_independent_current_and_floor_compose_legs() -> None:
    workflow_path = Path(".github/workflows/ci.yml")
    workflow = workflow_path.read_text(encoding="utf-8")
    parsed = yaml.safe_load(workflow)
    job = parsed["jobs"]["cli-docker-runtime"]

    assert job["strategy"]["fail-fast"] is False
    assert job["strategy"]["matrix"]["compose-version"] == [
        "runner-current",
        "v2.24.0",
    ]
    assert "${{ matrix.compose-version }}" in job["name"]

    steps = job["steps"]
    assert any(
        step.get("run") == "uv run python scripts/test_container_env_boundary.py"
        for step in steps
    )
    assert any(step.get("run") == "make test-cli-docker" for step in steps)


def test_docker_marked_tests_run_only_in_the_dedicated_docker_matrix() -> None:
    parsed = yaml.safe_load(
        Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    )

    fast_steps = parsed["jobs"]["fast-tests"]["steps"]
    fast_commands = tuple(step.get("run", "") for step in fast_steps)
    assert any("-m 'not docker'" in command for command in fast_commands)

    docker_steps = parsed["jobs"]["cli-docker-runtime"]["steps"]
    assert any(
        step.get("run")
        == "uv run pytest tests/unit/test_docker_runtime.py -m docker -q"
        for step in docker_steps
    )


def test_compose_floor_action_is_commit_pinned_and_floor_only() -> None:
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    body = _job(workflow, "cli-docker-runtime")
    references = re.findall(r"docker/setup-compose-action@([^\s#]+)", body)

    assert len(references) == 1
    assert re.fullmatch(r"[0-9a-f]{40}", references[0])
    assert "if: matrix.compose-version == 'v2.24.0'" in body
    assert re.search(r"(?m)^\s+version: v2\.24\.0$", body)


def test_cli_compatibility_runs_platform_container_selection_and_artifact_tests() -> (
    None
):
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    body = _job(workflow, "cli-compatibility")

    assert "tests/unit/test_container_policy.py" in body
    assert "tests/unit/test_secure_temp.py" in body


def test_cli_compatibility_uses_the_executable_bundle_smoke_for_dockerfile() -> None:
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    body = _job(workflow, "cli-compatibility")
    step = re.search(
        r"(?ms)^      - name: Dockerfile command renders a runnable config\n"
        r"(?P<body>.*?)(?=^      - name:|\Z)",
        body,
    )
    assert step is not None

    assert step.group("body").strip() == (
        "run: uv run python scripts/test_cli_config_autodiscovery.py "
        "--config examples/external_graph/manifest.json"
    )
