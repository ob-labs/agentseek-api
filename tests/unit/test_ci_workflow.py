import re
from pathlib import Path


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
