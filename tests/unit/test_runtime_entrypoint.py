from __future__ import annotations

import runpy
import sys

import pytest
from pydantic import ValidationError


@pytest.mark.parametrize("arguments", [[], ["unknown-target"]])
def test_runtime_entrypoint_rejects_unknown_internal_targets(
    arguments: list[str],
    capsys: pytest.CaptureFixture[str],
) -> None:
    from agentseek_api.runtime_entrypoint import main

    assert main(arguments) == 2
    assert capsys.readouterr().err == "Invalid internal runtime target.\n"


def test_runtime_entrypoint_dispatches_target_with_isolated_argv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agentseek_api import runtime_entrypoint

    original_argv = ["parent", "parent-canary"]
    observed: list[tuple[str, str, list[str]]] = []
    monkeypatch.setattr(sys, "argv", original_argv)
    monkeypatch.setattr(
        runtime_entrypoint.importlib,
        "import_module",
        lambda name: observed.append(("import", name, list(sys.argv))),
    )
    monkeypatch.setattr(
        runtime_entrypoint.runpy,
        "run_module",
        lambda name, *, run_name: observed.append((name, run_name, list(sys.argv))),
    )

    assert runtime_entrypoint.main(["uvicorn", "--", "app:api", "--port", "8080"]) == 0

    assert observed == [
        (
            "import",
            "agentseek_api.settings",
            ["uvicorn.__main__", "app:api", "--port", "8080"],
        ),
        (
            "uvicorn.__main__",
            "__main__",
            ["uvicorn.__main__", "app:api", "--port", "8080"],
        ),
    ]
    assert sys.argv is original_argv


def test_runtime_entrypoint_uses_process_argv_when_arguments_are_omitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agentseek_api import runtime_entrypoint

    observed: list[tuple[str, list[str]]] = []
    monkeypatch.setattr(sys, "argv", ["entrypoint", "scheduler", "--once"])
    monkeypatch.setattr(
        runtime_entrypoint.importlib, "import_module", lambda _name: None
    )
    monkeypatch.setattr(
        runtime_entrypoint.runpy,
        "run_module",
        lambda name, *, run_name: observed.append((name, list(sys.argv))),
    )

    assert runtime_entrypoint.main() == 0
    assert observed == [
        ("agentseek_api.scheduler", ["agentseek_api.scheduler", "--once"])
    ]
    assert sys.argv == ["entrypoint", "scheduler", "--once"]


def test_runtime_entrypoint_redacts_settings_validation_input(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from agentseek_api import runtime_entrypoint
    from agentseek_api.settings import Settings

    with pytest.raises(ValidationError) as captured:
        Settings.model_validate({"PORT": "settings-input-canary"})

    run_called = False

    def fail_settings_import(_name: str) -> None:
        raise captured.value

    def record_run(*_args, **_kwargs) -> None:
        nonlocal run_called
        run_called = True

    parent_argv = ["parent"]
    monkeypatch.setattr(sys, "argv", parent_argv)
    monkeypatch.setattr(
        runtime_entrypoint.importlib,
        "import_module",
        fail_settings_import,
    )
    monkeypatch.setattr(runtime_entrypoint.runpy, "run_module", record_run)

    assert runtime_entrypoint.main(["worker"]) == 2

    stderr = capsys.readouterr().err
    assert stderr == "Invalid runtime setting(s): PORT (int_parsing).\n"
    assert "settings-input-canary" not in stderr
    assert run_called is False
    assert sys.argv is parent_argv


@pytest.mark.parametrize(
    ("system_exit_code", "expected"),
    [(37, 37), (None, 0), ("non-integer-canary", 1)],
)
def test_runtime_entrypoint_normalizes_target_system_exit(
    system_exit_code: object,
    expected: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agentseek_api import runtime_entrypoint

    monkeypatch.setattr(
        runtime_entrypoint.importlib, "import_module", lambda _name: None
    )
    monkeypatch.setattr(
        runtime_entrypoint.runpy,
        "run_module",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(SystemExit(system_exit_code)),
    )

    assert runtime_entrypoint.main(["worker"]) == expected


def test_runtime_entrypoint_restores_argv_when_target_crashes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agentseek_api import runtime_entrypoint

    parent_argv = ["parent", "argv-canary"]
    monkeypatch.setattr(sys, "argv", parent_argv)
    monkeypatch.setattr(
        runtime_entrypoint.importlib, "import_module", lambda _name: None
    )
    monkeypatch.setattr(
        runtime_entrypoint.runpy,
        "run_module",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("target-canary")),
    )

    with pytest.raises(RuntimeError, match="target-canary"):
        runtime_entrypoint.main(["scheduler"])

    assert sys.argv is parent_argv


def test_runtime_entrypoint_module_execution_uses_cli_exit_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["runtime-entrypoint", "invalid-target"])
    monkeypatch.delitem(sys.modules, "agentseek_api.runtime_entrypoint", raising=False)

    with pytest.raises(SystemExit) as captured:
        runpy.run_module("agentseek_api.runtime_entrypoint", run_name="__main__")

    assert captured.value.code == 2
