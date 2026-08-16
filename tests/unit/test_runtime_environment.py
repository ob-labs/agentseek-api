from __future__ import annotations

import json
from pathlib import Path

import pytest


def _write_config(
    root: Path,
    *,
    env: str | dict[str, object] | None,
    auth_path: str | None = None,
) -> Path:
    payload: dict[str, object] = {
        "graphs": {"chat": "chat.graph:graph"},
    }
    if env is not None:
        payload["env"] = env
    if auth_path is not None:
        payload["auth"] = {"path": auth_path}
    config_path = root / "langgraph.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    return config_path


@pytest.mark.parametrize(
    ("cli_binding", "inherited", "expected"),
    [
        ("TOKEN=from-cli\n", {}, "from-cli"),
        ("TOKEN\n", {}, "from-config"),
        ("TOKEN=\n", {}, ""),
        ("TOKEN=from-cli\n", {"TOKEN": ""}, ""),
        ("TOKEN=from-cli\n", {"TOKEN": "from-shell"}, "from-shell"),
    ],
    ids=[
        "cli-over-config",
        "valueless-does-not-assign",
        "explicit-empty-assigns",
        "inherited-empty-is-final",
        "inherited-nonempty-is-final",
    ],
)
def test_host_runtime_assignment_matrix(
    tmp_path: Path,
    cli_binding: str,
    inherited: dict[str, str],
    expected: str,
) -> None:
    from agentseek_api.cli import build_runtime_env

    config_env = tmp_path / "config.env"
    config_env.write_text("TOKEN=from-config\n", encoding="utf-8")
    config_path = _write_config(tmp_path, env="./config.env")
    cli_env = tmp_path / "cli.env"
    cli_env.write_text(cli_binding, encoding="utf-8")

    actual = build_runtime_env(
        config_path=config_path,
        env_file=str(cli_env),
        cwd=tmp_path,
        base_env=inherited,
    )

    assert actual["TOKEN"] == expected


def test_config_mapping_and_auth_are_below_cli_and_inherited(
    tmp_path: Path,
) -> None:
    from agentseek_api.cli import build_runtime_env

    config_path = _write_config(
        tmp_path,
        env={"TOKEN": "from-mapping", "AUTH_MODULE_PATH": "from-env-mapping"},
        auth_path="auth.module:backend",
    )
    cli_env = tmp_path / "cli.env"
    cli_env.write_text(
        "TOKEN=from-cli\nAUTH_MODULE_PATH=from-cli-auth\n",
        encoding="utf-8",
    )

    actual = build_runtime_env(
        config_path=config_path,
        env_file=str(cli_env),
        cwd=tmp_path,
        base_env={
            "TOKEN": "from-shell",
            "AUTH_MODULE_PATH": "",
        },
    )

    assert actual["TOKEN"] == "from-shell"
    assert actual["AUTH_MODULE_PATH"] == ""


def test_config_dotenv_valueless_is_absent_and_empty_is_present(
    tmp_path: Path,
) -> None:
    from agentseek_api.cli import build_runtime_env

    config_env = tmp_path / "config.env"
    config_env.write_text("VALUELESS\nEMPTY=\n", encoding="utf-8")
    config_path = _write_config(tmp_path, env="./config.env")

    actual = build_runtime_env(
        config_path=config_path,
        env_file=None,
        cwd=tmp_path,
        base_env={},
    )

    assert "VALUELESS" not in actual
    assert actual["EMPTY"] == ""


@pytest.mark.parametrize(
    "case",
    [
        {
            "id": "config-bare",
            "config_env": ("dotenv", "KEY\n"),
            "auth_path": None,
            "cli_dotenv": None,
            "inherited": {},
            "key": "KEY",
            "present": False,
            "value": None,
        },
        {
            "id": "config-empty",
            "config_env": ("dotenv", "KEY=\n"),
            "auth_path": None,
            "cli_dotenv": None,
            "inherited": {},
            "key": "KEY",
            "present": True,
            "value": "",
        },
        {
            "id": "config-value",
            "config_env": ("dotenv", "KEY=config\n"),
            "auth_path": None,
            "cli_dotenv": None,
            "inherited": {},
            "key": "KEY",
            "present": True,
            "value": "config",
        },
        {
            "id": "mapping-empty",
            "config_env": {"KEY": ""},
            "auth_path": None,
            "cli_dotenv": None,
            "inherited": {},
            "key": "KEY",
            "present": True,
            "value": "",
        },
        {
            "id": "mapping-value",
            "config_env": {"KEY": "mapping"},
            "auth_path": None,
            "cli_dotenv": None,
            "inherited": {},
            "key": "KEY",
            "present": True,
            "value": "mapping",
        },
        {
            "id": "auth-over-dotenv",
            "config_env": ("dotenv", "AUTH_MODULE_PATH=dotenv\n"),
            "auth_path": "auth.module:backend",
            "cli_dotenv": None,
            "inherited": {},
            "key": "AUTH_MODULE_PATH",
            "present": True,
            "value": "auth.module:backend",
        },
        {
            "id": "cli-bare",
            "config_env": {"KEY": "mapping"},
            "auth_path": None,
            "cli_dotenv": "KEY\n",
            "inherited": {},
            "key": "KEY",
            "present": True,
            "value": "mapping",
        },
        {
            "id": "cli-empty",
            "config_env": {"KEY": "mapping"},
            "auth_path": None,
            "cli_dotenv": "KEY=\n",
            "inherited": {},
            "key": "KEY",
            "present": True,
            "value": "",
        },
        {
            "id": "cli-value",
            "config_env": {"KEY": "mapping"},
            "auth_path": None,
            "cli_dotenv": "KEY=cli\n",
            "inherited": {},
            "key": "KEY",
            "present": True,
            "value": "cli",
        },
        {
            "id": "inherited-empty",
            "config_env": {"KEY": "mapping"},
            "auth_path": None,
            "cli_dotenv": "KEY=cli\n",
            "inherited": {"KEY": ""},
            "key": "KEY",
            "present": True,
            "value": "",
        },
        {
            "id": "inherited-value",
            "config_env": {"KEY": "mapping"},
            "auth_path": None,
            "cli_dotenv": "KEY=cli\n",
            "inherited": {"KEY": "shell"},
            "key": "KEY",
            "present": True,
            "value": "shell",
        },
        {
            "id": "inherited-empty-auth",
            "config_env": ("dotenv", "AUTH_MODULE_PATH=dotenv\n"),
            "auth_path": "auth.module:backend",
            "cli_dotenv": "AUTH_MODULE_PATH=cli\n",
            "inherited": {"AUTH_MODULE_PATH": ""},
            "key": "AUTH_MODULE_PATH",
            "present": True,
            "value": "",
        },
        {
            "id": "inherited-value-auth",
            "config_env": ("dotenv", "AUTH_MODULE_PATH=dotenv\n"),
            "auth_path": "auth.module:backend",
            "cli_dotenv": "AUTH_MODULE_PATH=cli\n",
            "inherited": {"AUTH_MODULE_PATH": "shell"},
            "key": "AUTH_MODULE_PATH",
            "present": True,
            "value": "shell",
        },
    ],
    ids=lambda case: case["id"],
)
def test_complete_host_assignment_collision_matrix(
    tmp_path: Path,
    case: dict[str, object],
) -> None:
    from agentseek_api.cli import build_runtime_env

    config_source = case["config_env"]
    if isinstance(config_source, tuple):
        _, contents = config_source
        assert isinstance(contents, str)
        (tmp_path / "config.env").write_text(contents, encoding="utf-8")
        config_env: str | dict[str, object] = "./config.env"
    else:
        assert isinstance(config_source, dict)
        config_env = config_source
    auth_path = case["auth_path"]
    assert auth_path is None or isinstance(auth_path, str)
    config_path = _write_config(
        tmp_path,
        env=config_env,
        auth_path=auth_path,
    )
    cli_dotenv = case["cli_dotenv"]
    cli_env: Path | None = None
    if cli_dotenv is not None:
        assert isinstance(cli_dotenv, str)
        cli_env = tmp_path / "cli.env"
        cli_env.write_text(cli_dotenv, encoding="utf-8")
    inherited = case["inherited"]
    assert isinstance(inherited, dict)

    actual = build_runtime_env(
        config_path=config_path,
        env_file=str(cli_env) if cli_env is not None else None,
        cwd=tmp_path,
        base_env=inherited,
    )

    key = case["key"]
    assert isinstance(key, str)
    assert (key in actual) is case["present"]
    if case["present"]:
        assert actual[key] == case["value"]


def test_each_dotenv_file_uses_an_independent_interpolation_context(
    tmp_path: Path,
) -> None:
    from agentseek_api.cli import build_runtime_env

    config_env = tmp_path / "config.env"
    config_env.write_text(
        "ORIGIN=https://config.example\n"
        "CONFIG_RESULT=${ORIGIN}/v1\n",
        encoding="utf-8",
    )
    config_path = _write_config(tmp_path, env="./config.env")
    cli_env = tmp_path / "cli.env"
    cli_env.write_text("CLI_RESULT=${ORIGIN}/v2\n", encoding="utf-8")

    actual = build_runtime_env(
        config_path=config_path,
        env_file=str(cli_env),
        cwd=tmp_path,
        base_env={"ORIGIN": "https://shell.example"},
    )

    assert actual == {
        "CONFIG_RESULT": "https://config.example/v1",
        "CLI_RESULT": "https://shell.example/v2",
        "ORIGIN": "https://shell.example",
        "AGENTSEEK_GRAPHS": str(config_path),
    }


def test_cli_dotenv_does_not_interpolate_literal_config_mapping(
    tmp_path: Path,
) -> None:
    from agentseek_api.cli import build_runtime_env

    config_path = _write_config(
        tmp_path,
        env={"ORIGIN": "https://mapping.example"},
    )
    cli_env = tmp_path / "cli.env"
    cli_env.write_text(
        "RESULT=${ORIGIN:-https://fallback.example}/v1\n",
        encoding="utf-8",
    )

    actual = build_runtime_env(
        config_path=config_path,
        env_file=str(cli_env),
        cwd=tmp_path,
        base_env={},
    )

    assert actual["ORIGIN"] == "https://mapping.example"
    assert actual["RESULT"] == "https://fallback.example/v1"


def test_inherited_override_does_not_recompute_earlier_file_value(
    tmp_path: Path,
) -> None:
    from agentseek_api.cli import build_runtime_env

    config_env = tmp_path / "config.env"
    config_env.write_text(
        "ORIGIN=https://config.example\n"
        "BASE_URL=${ORIGIN}/v1\n",
        encoding="utf-8",
    )
    config_path = _write_config(tmp_path, env="./config.env")

    actual = build_runtime_env(
        config_path=config_path,
        env_file=None,
        cwd=tmp_path,
        base_env={"ORIGIN": "https://shell.example"},
    )

    assert actual["ORIGIN"] == "https://shell.example"
    assert actual["BASE_URL"] == "https://config.example/v1"


def test_malformed_dotenv_returns_exit_2_without_starting_child(
    tmp_path: Path,
) -> None:
    from agentseek_api.cli import main

    config_path = _write_config(tmp_path, env=None)
    env_file = tmp_path / "broken.env"
    env_file.write_text('SECRET=must-not-leak\nBROKEN "value"\n', encoding="utf-8")
    calls: list[list[str]] = []

    def runner(
        command: list[str],
        *,
        env: dict[str, str],
        cwd: str | None = None,
    ) -> int:
        calls.append(command)
        return 0

    import io

    stderr = io.StringIO()
    exit_code = main(
        [
            "serve",
            "--config",
            str(config_path),
            "--env-file",
            str(env_file),
        ],
        runner=runner,
        cwd=tmp_path,
        stderr=stderr,
    )

    assert exit_code == 2
    assert calls == []
    assert "line 2" in stderr.getvalue()
    assert "must-not-leak" not in stderr.getvalue()


@pytest.mark.parametrize(
    "failure",
    ["missing", "decode", "read"],
)
def test_unreadable_dotenv_returns_exit_2_without_starting_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    import io

    from agentseek_api.cli import main

    config_path = _write_config(tmp_path, env=None)
    env_file = tmp_path / f"{failure}.env"
    if failure == "decode":
        env_file.write_bytes(b"TOKEN=\xff\n")
    elif failure == "read":
        env_file.write_text("TOKEN=hidden\n", encoding="utf-8")
        original_open = Path.open

        def fail_selected_open(path: Path, *args: object, **kwargs: object):
            if path == env_file:
                raise PermissionError(13, "Permission denied", str(path))
            return original_open(path, *args, **kwargs)

        monkeypatch.setattr(Path, "open", fail_selected_open)
    calls: list[list[str]] = []

    def runner(
        command: list[str],
        *,
        env: dict[str, str],
        cwd: str | None = None,
    ) -> int:
        calls.append(command)
        return 0

    stderr = io.StringIO()
    exit_code = main(
        [
            "serve",
            "--config",
            str(config_path),
            "--env-file",
            str(env_file),
        ],
        runner=runner,
        cwd=tmp_path,
        stderr=stderr,
    )

    assert exit_code == 2
    assert calls == []
    if failure == "missing":
        assert "does not exist" in stderr.getvalue()
    elif failure == "decode":
        assert "not valid UTF-8" in stderr.getvalue()
    else:
        assert "could not be read" in stderr.getvalue()
    assert "hidden" not in stderr.getvalue()


def test_command_owned_graph_path_overrides_inherited_value(
    tmp_path: Path,
) -> None:
    from agentseek_api.cli import build_runtime_env

    config_path = _write_config(tmp_path, env=None)

    actual = build_runtime_env(
        config_path=config_path,
        env_file=None,
        cwd=tmp_path,
        base_env={"AGENTSEEK_GRAPHS": "/stale/manifest.json"},
    )

    assert actual["AGENTSEEK_GRAPHS"] == str(config_path)


def test_command_owned_graph_path_is_absent_without_selected_config(
    tmp_path: Path,
) -> None:
    from agentseek_api.cli import build_runtime_env

    actual = build_runtime_env(
        config_path=None,
        env_file=None,
        cwd=tmp_path,
        base_env={"AGENTSEEK_GRAPHS": "/stale/manifest.json"},
    )

    assert "AGENTSEEK_GRAPHS" not in actual


def test_shared_lifecycle_dotenv_mutation_cannot_replace_inherited_present_values(
    tmp_path: Path,
) -> None:
    from agentseek_api.cli import build_runtime_env

    shared_env = tmp_path / ".env"
    shared_env.write_text(
        "PRESENT=initial\n"
        "EMPTY=\n"
        "CHILD_ONLY=initial\n",
        encoding="utf-8",
    )
    snapshot = {"PRESENT": "initial", "EMPTY": ""}
    shared_env.write_text(
        "PRESENT=mutated\n"
        "EMPTY=mutated\n"
        "CHILD_ONLY=added-later\n",
        encoding="utf-8",
    )
    config_path = _write_config(tmp_path, env="./.env")

    actual = build_runtime_env(
        config_path=config_path,
        env_file=None,
        cwd=tmp_path,
        base_env=snapshot,
    )

    assert actual["PRESENT"] == "initial"
    assert "EMPTY" in actual
    assert actual["EMPTY"] == ""
    assert actual["CHILD_ONLY"] == "added-later"


@pytest.mark.parametrize("role", ["dev", "serve", "worker", "scheduler"])
@pytest.mark.parametrize(
    ("source", "source_value"),
    [
        ("config-dotenv", "false"),
        ("config-mapping", "false"),
        ("cli-dotenv", "false"),
        ("inherited-empty", ""),
        ("inherited-nonempty", "false"),
    ],
)
def test_studio_auth_local_dev_is_command_owned_only_for_dev(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    role: str,
    source: str,
    source_value: str,
) -> None:
    from agentseek_api.cli import main

    monkeypatch.delenv("STUDIO_AUTH_LOCAL_DEV", raising=False)
    config_env: str | dict[str, object] | None = None
    env_file: Path | None = None
    if source == "config-dotenv":
        (tmp_path / "config.env").write_text(
            "STUDIO_AUTH_LOCAL_DEV=false\n",
            encoding="utf-8",
        )
        config_env = "./config.env"
    elif source == "config-mapping":
        config_env = {"STUDIO_AUTH_LOCAL_DEV": "false"}
    elif source == "cli-dotenv":
        env_file = tmp_path / "cli.env"
        env_file.write_text("STUDIO_AUTH_LOCAL_DEV=false\n", encoding="utf-8")
    else:
        monkeypatch.setenv("STUDIO_AUTH_LOCAL_DEV", source_value)
    config_path = _write_config(tmp_path, env=config_env)
    captured_env: dict[str, str] | None = None

    def runner(
        command: list[str],
        *,
        env: dict[str, str],
        cwd: str | None = None,
    ) -> int:
        nonlocal captured_env
        captured_env = env
        return 0

    argv = [role, "--config", str(config_path)]
    if role == "dev":
        argv.append("--no-reload")
    if env_file is not None:
        argv.extend(["--env-file", str(env_file)])

    exit_code = main(argv, runner=runner, cwd=tmp_path)

    assert exit_code == 0
    assert captured_env is not None
    expected = "true" if role == "dev" else source_value
    assert captured_env["STUDIO_AUTH_LOCAL_DEV"] == expected
