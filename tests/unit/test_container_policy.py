from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from types import MappingProxyType

import pytest

from agentseek_api.container_policy import (
    APPLICATION_COMPATIBILITY_KEYS,
    APP_CONTAINER_POLICY,
    ContainerPolicyError,
    ContainerSelection,
    HOST_RUNTIME_POLICY,
    docker_control_environment,
    select_application_payload,
    select_compose_payload,
)
from agentseek_api.environment import (
    CommandDerivedAssignment,
    EnvironmentPlan,
    EnvironmentTarget,
    resolve_environment,
)
from agentseek_api.settings import Settings
from tests.container_plan_helpers import resolved_fixture


def _plan(
    tmp_path: Path,
    *,
    config_dotenv: str | None = None,
    config_mapping: dict[str, str] | None = None,
    launch: dict[str, str] | None = None,
    cli_dotenv: str | None = None,
    explicit_names: frozenset[str] = frozenset(),
    assignments: tuple[CommandDerivedAssignment, ...] = (),
) -> EnvironmentPlan:
    config_path = tmp_path / "langgraph.json"
    config_path.write_text('{"graphs":{"chat":"chat.graph:graph"}}', encoding="utf-8")
    config_env_path = None
    if config_dotenv is not None:
        config_env_path = tmp_path / "config.env"
        config_env_path.write_text(config_dotenv, encoding="utf-8")
    cli_env_path = None
    if cli_dotenv is not None:
        cli_env_path = tmp_path / "cli.env"
        cli_env_path.write_text(cli_dotenv, encoding="utf-8")
    return EnvironmentPlan(
        config_path=config_path,
        config_dotenv=config_env_path,
        config_mapping=config_mapping or {},
        auth_path=None,
        cli_dotenv=cli_env_path,
        launch_environment=launch or {},
        command_assignments=assignments,
        explicit_names=explicit_names,
    )


def test_application_payload_requires_declaration_but_preserves_empty() -> None:
    resolved = resolved_fixture(
        values={"DECLARED_EMPTY": "", "AMBIENT_ONLY": "secret"},
        declared_keys={"DECLARED_EMPTY"},
    )

    payload = select_application_payload(
        resolved,
        ContainerSelection(pass_env=frozenset(), compose_env=frozenset()),
    )

    assert dict(payload) == {"DECLARED_EMPTY": ""}


def test_compose_selection_rejects_control_plane_collision() -> None:
    with pytest.raises(ContainerPolicyError, match="DOCKER_HOST"):
        select_compose_payload(
            application_payload={"DOCKER_HOST": "application-value"},
            selected_names=frozenset({"DOCKER_HOST"}),
            docker_control={"DOCKER_HOST": "unix:///var/run/docker.sock"},
        )


def test_compose_selection_does_not_parse_or_resolve_dotenv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agentseek_api.dotenv_adapter as dotenv_adapter

    def fail(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("Compose selection must not resolve dotenv sources")

    monkeypatch.setattr(dotenv_adapter, "parse_dotenv_document", fail)
    monkeypatch.setattr(dotenv_adapter, "parse_dotenv_file", fail)

    payload = select_compose_payload(
        application_payload=MappingProxyType({"TOKEN": "already-final"}),
        selected_names=frozenset({"TOKEN"}),
        docker_control=MappingProxyType({"PATH": "/safe/bin"}),
    )

    assert dict(payload) == {"TOKEN": "already-final"}


def test_typed_host_policy_matches_released_build_runtime_env_matrix(
    tmp_path: Path,
) -> None:
    from agentseek_api.cli import build_runtime_env

    plan = _plan(
        tmp_path,
        config_dotenv="TOKEN=config\nRESULT=${TOKEN}/dotenv\n",
        launch={"TOKEN": "shell", "INHERITED_EMPTY": ""},
        cli_dotenv="CLI_RESULT=${TOKEN}/cli\nVALUELESS\nEMPTY=\n",
    )
    plan.config_path.write_text(
        '{"graphs":{"chat":"chat.graph:graph"},"env":"./config.env"}', encoding="utf-8"
    )
    plan = EnvironmentPlan(
        config_path=plan.config_path,
        config_dotenv=plan.config_dotenv,
        config_mapping={},
        auth_path=None,
        cli_dotenv=plan.cli_dotenv,
        launch_environment=plan.launch_environment,
        command_assignments=(
            CommandDerivedAssignment(
                targets=frozenset({EnvironmentTarget.HOST_RUNTIME}),
                values={"AGENTSEEK_GRAPHS": str(plan.config_path)},
                reason="selected host config path",
            ),
        ),
        explicit_names=frozenset(),
    )
    expected = build_runtime_env(
        config_path=plan.config_path,
        env_file=str(plan.cli_dotenv),
        cwd=tmp_path,
        base_env=dict(plan.launch_environment),
    )

    resolved = resolve_environment(plan, HOST_RUNTIME_POLICY)

    assert dict(resolved.values) == expected
    assert "VALUELESS" not in resolved.declared_keys
    assert {"TOKEN", "RESULT", "CLI_RESULT", "EMPTY"} <= resolved.declared_keys


def test_container_document_collects_explicit_empty_but_not_bare_binding(
    tmp_path: Path,
) -> None:
    resolved = resolve_environment(
        _plan(tmp_path, config_dotenv="EMPTY=\nBARE\n"),
        APP_CONTAINER_POLICY,
    )

    assert "EMPTY" in resolved.declared_keys
    assert "BARE" not in resolved.declared_keys
    assert dict(
        select_application_payload(
            resolved, ContainerSelection(frozenset(), frozenset())
        )
    ) == {"EMPTY": ""}


def test_declared_key_uses_final_inherited_value_for_export(tmp_path: Path) -> None:
    resolved = resolve_environment(
        _plan(tmp_path, config_dotenv="TOKEN=dotenv\n", launch={"TOKEN": "shell"}),
        APP_CONTAINER_POLICY,
    )

    assert dict(
        select_application_payload(
            resolved, ContainerSelection(frozenset(), frozenset())
        )
    ) == {"TOKEN": "shell"}


def test_unselected_ambient_reference_fails_before_payload_selection(
    tmp_path: Path,
) -> None:
    plan = _plan(
        tmp_path,
        config_dotenv="RESULT=${AMBIENT_ONLY}\n",
        launch={"AMBIENT_ONLY": "secret"},
    )

    with pytest.raises(ContainerPolicyError, match="AMBIENT_ONLY"):
        resolve_environment(plan, APP_CONTAINER_POLICY)


def test_explicit_pass_env_allows_ambient_reference_and_export(tmp_path: Path) -> None:
    resolved = resolve_environment(
        _plan(
            tmp_path,
            config_dotenv="RESULT=${AMBIENT_ONLY}\n",
            launch={"AMBIENT_ONLY": "secret"},
            explicit_names=frozenset({"AMBIENT_ONLY"}),
        ),
        APP_CONTAINER_POLICY,
    )

    payload = select_application_payload(
        resolved,
        ContainerSelection(
            pass_env=frozenset({"AMBIENT_ONLY"}), compose_env=frozenset()
        ),
    )
    assert dict(payload) == {"AMBIENT_ONLY": "secret", "RESULT": "secret"}


def test_later_resolved_dotenv_binding_clears_earlier_unresolved_reference(
    tmp_path: Path,
) -> None:
    resolved = resolve_environment(
        _plan(tmp_path, config_dotenv="VALUE=${MISSING}\nVALUE=ok\n"),
        APP_CONTAINER_POLICY,
    )

    assert resolved.values["VALUE"] == "ok"
    assert "VALUE" not in resolved.unresolved_references


def test_windows_container_interpolation_matches_explicit_name_case_insensitively(
    tmp_path: Path,
) -> None:
    resolved = resolve_environment(
        _plan(
            tmp_path,
            config_dotenv="RESULT=${openai_api_key}\n",
            launch={"OpenAI_Api_Key": "value"},
        ),
        APP_CONTAINER_POLICY,
        platform="win32",
    )

    assert resolved.values["RESULT"] == "value"
    assert dict(
        select_application_payload(
            resolved,
            ContainerSelection(pass_env=frozenset(), compose_env=frozenset()),
            platform="win32",
        )
    ) == {"OPENAI_API_KEY": "value", "RESULT": "value"}


def test_windows_rejects_two_spellings_of_a_launch_environment_name(
    tmp_path: Path,
) -> None:
    with pytest.raises(ContainerPolicyError, match="Path.*PATH"):
        resolve_environment(
            _plan(tmp_path, launch={"Path": "one", "PATH": "two"}),
            APP_CONTAINER_POLICY,
            platform="win32",
        )


def test_windows_launch_assignment_replaces_lower_source_case_insensitively(
    tmp_path: Path,
) -> None:
    resolved = resolve_environment(
        _plan(
            tmp_path,
            config_dotenv="OPENAI_API_KEY=dotenv\n",
            launch={"OpenAI_Api_Key": "launch"},
        ),
        APP_CONTAINER_POLICY,
        platform="win32",
    )

    assert dict(
        select_application_payload(
            resolved,
            ContainerSelection(pass_env=frozenset(), compose_env=frozenset()),
            platform="win32",
        )
    ) == {"OPENAI_API_KEY": "launch"}


def test_windows_rejects_application_control_collision_across_spellings() -> None:
    with pytest.raises(ContainerPolicyError, match="Path"):
        select_compose_payload(
            application_payload={"Path": "application-value"},
            selected_names=frozenset({"PATH"}),
            docker_control={"PATH": "control-value"},
            platform="win32",
        )


def test_cli_builders_construct_target_scoped_command_assignments(
    tmp_path: Path,
) -> None:
    from agentseek_api.cli import (
        build_container_command_assignments,
        build_host_environment_plan,
    )

    config_path = tmp_path / "langgraph.json"
    config_path.write_text('{"graphs":{"chat":"chat.graph:graph"}}', encoding="utf-8")
    host_plan = build_host_environment_plan(
        config_path=config_path,
        env_file=None,
        cwd=tmp_path,
        base_env={},
        role="dev",
    )
    host = resolve_environment(host_plan, HOST_RUNTIME_POLICY)
    omitted = build_container_command_assignments(
        config_path=config_path, cwd=tmp_path, postgres_uri=None
    )
    explicit = build_container_command_assignments(
        config_path=config_path, cwd=tmp_path, postgres_uri="postgresql://db"
    )
    dev_preloaded = build_container_command_assignments(
        config_path=config_path,
        cwd=tmp_path,
        postgres_uri=None,
        role="dev",
        environment_mode="preloaded-v1",
    )
    dev_without_mode = build_container_command_assignments(
        config_path=config_path,
        cwd=tmp_path,
        postgres_uri=None,
        role="dev",
        environment_mode=None,
    )
    non_dev_preloaded = build_container_command_assignments(
        config_path=config_path,
        cwd=tmp_path,
        postgres_uri=None,
        role="serve",
        environment_mode="preloaded-v1",
    )

    assert host.values["AGENTSEEK_GRAPHS"] == str(config_path)
    assert host.values["STUDIO_AUTH_LOCAL_DEV"] == "true"
    assert [dict(item.values) for item in omitted] == [
        {"AGENTSEEK_GRAPHS": "/deps/agent/langgraph.json"}
    ]
    assert [dict(item.values) for item in explicit] == [
        {"AGENTSEEK_GRAPHS": "/deps/agent/langgraph.json"},
        {
            "METADATA_DB_URL": "postgresql://db",
            "METADATA_DB_BACKEND": "postgresql",
        },
    ]
    assert [dict(item.values) for item in dev_preloaded] == [
        {"AGENTSEEK_GRAPHS": "/deps/agent/langgraph.json"},
        {"STUDIO_AUTH_LOCAL_DEV": "true"},
    ]
    assert [dict(item.values) for item in dev_without_mode] == [
        {"AGENTSEEK_GRAPHS": "/deps/agent/langgraph.json"}
    ]
    assert [dict(item.values) for item in non_dev_preloaded] == [
        {"AGENTSEEK_GRAPHS": "/deps/agent/langgraph.json"}
    ]


def test_missing_explicit_pass_env_is_an_error() -> None:
    with pytest.raises(ContainerPolicyError, match="MISSING"):
        select_application_payload(
            resolved_fixture(values={}, declared_keys=set()),
            ContainerSelection(
                pass_env=frozenset({"MISSING"}), compose_env=frozenset()
            ),
        )


def test_application_registry_contains_every_settings_field() -> None:
    assert set(Settings.model_fields) <= APPLICATION_COMPATIBILITY_KEYS


def test_provider_registry_matches_documented_exact_snapshot() -> None:
    assert APPLICATION_COMPATIBILITY_KEYS == {
        "AGENTSEEK_API_BASE",
        "AGENTSEEK_API_KEY",
        "AGENTSEEK_GRAPHS",
        "AGENTSEEK_MODEL",
        "AGENTSEEK_MODEL_API_KEY",
        "AGENTSEEK_MODEL_PROVIDER",
        "APP_NAME",
        "AUTH_MODULE_PATH",
        "OPENAI_API_BASE",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "OPENAI_MODEL",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_API_URL",
        "GOOGLE_API_BASE",
        "GOOGLE_API_KEY",
        "TAVILY_API_KEY",
        "DAYTONA_API_KEY",
        "LANGSMITH_API_KEY",
        "BUB_API_BASE",
        "BUB_API_KEY",
        "BUB_MODEL",
        "BUB_OPENAI_API_BASE",
        "BUB_OPENAI_API_KEY",
        "DEEPAGENTS_MODEL",
        "EMBEDDING_API_KEY",
        "EMBEDDING_BASE_URL",
        "VLM_API_KEY",
        "VLM_BASE_URL",
        "SILICONFLOW_API_KEY",
        "EXECUTOR_BACKEND",
        "METADATA_DB_BACKEND",
        "METADATA_DB_URL",
        "OCEANBASE_DB_NAME",
        "OCEANBASE_HOST",
        "OCEANBASE_PASSWORD",
        "OCEANBASE_PORT",
        "OCEANBASE_USER",
        "PORT",
        "REDIS_RUN_PROCESSING_KEY",
        "REDIS_RUN_QUEUE_KEY",
        "REDIS_SCHEDULER_LOCK_KEY",
        "REDIS_SCHEDULER_LOCK_TTL_SECONDS",
        "REDIS_STREAM_MAXLEN",
        "REDIS_STREAM_TTL_SECONDS",
        "REDIS_URL",
        "REDIS_WORKER_LOCK_KEY",
        "REDIS_WORKER_LOCK_TTL_SECONDS",
        "REDIS_WORKER_POLL_TIMEOUT_SECONDS",
        "SCHEDULER_CLAIM_LIMIT",
        "SCHEDULER_POLL_INTERVAL_SECONDS",
        "SCHEDULER_STARTED_TICK_STALE_AFTER_SECONDS",
        "SEEKDB_EMBED",
        "SEEKDB_EMBED_DIR",
        "SEEKDB_URL",
        "STUDIO_AUTH_LOCAL_DEV",
        "WORKER_CONCURRENT_JOBS",
    }


def test_docker_control_environment_is_exact_for_linux_darwin_and_win32(
    tmp_path: Path,
) -> None:
    plan = _plan(
        tmp_path,
        launch={
            "PATH": "/safe/bin",
            "HOME": "/safe/home",
            "XDG_CONFIG_HOME": "/cfg",
            "XDG_RUNTIME_DIR": "/run",
            "USERPROFILE": "C:/Users/safe",
            "SYSTEMROOT": "C:/Windows",
            "UNRELATED": "must-not-pass",
        },
    )

    assert dict(docker_control_environment(plan, platform="linux")) == {
        "PATH": "/safe/bin",
        "HOME": "/safe/home",
        "XDG_CONFIG_HOME": "/cfg",
        "XDG_RUNTIME_DIR": "/run",
    }
    assert dict(docker_control_environment(plan, platform="darwin")) == {
        "PATH": "/safe/bin",
        "HOME": "/safe/home",
        "XDG_CONFIG_HOME": "/cfg",
    }
    assert dict(docker_control_environment(plan, platform="win32")) == {
        "Path": "/safe/bin",
        "UserProfile": "C:/Users/safe",
        "SystemRoot": "C:/Windows",
    }


def test_macos_docker_plugin_discovery_uses_sanitized_home_and_path(
    tmp_path: Path,
) -> None:
    plan = _plan(
        tmp_path, launch={"PATH": "/docker/bin", "HOME": "/safe/home", "SECRET": "no"}
    )
    assert dict(docker_control_environment(plan, platform="darwin")) == {
        "PATH": "/docker/bin",
        "HOME": "/safe/home",
    }


def test_windows_docker_native_invocation_uses_sanitized_systemroot_userprofile_and_path(
    tmp_path: Path,
) -> None:
    plan = _plan(
        tmp_path,
        launch={
            "Path": "C:/docker",
            "SystemRoot": "C:/Windows",
            "UserProfile": "C:/Users/safe",
        },
    )
    assert dict(docker_control_environment(plan, platform="win32")) == {
        "Path": "C:/docker",
        "SystemRoot": "C:/Windows",
        "UserProfile": "C:/Users/safe",
    }


def test_command_assignments_apply_only_to_their_declared_target(
    tmp_path: Path,
) -> None:
    assignment = CommandDerivedAssignment(
        targets=frozenset({EnvironmentTarget.HOST_RUNTIME}),
        values={"AGENTSEEK_GRAPHS": "host"},
        reason="host config",
    )
    plan = _plan(tmp_path, assignments=(assignment,))
    assert (
        resolve_environment(plan, HOST_RUNTIME_POLICY).values["AGENTSEEK_GRAPHS"]
        == "host"
    )
    assert (
        "AGENTSEEK_GRAPHS" not in resolve_environment(plan, APP_CONTAINER_POLICY).values
    )


def test_postgres_uri_overrides_app_metadata_without_reaching_host_or_docker_control(
    tmp_path: Path,
) -> None:
    assignment = CommandDerivedAssignment(
        targets=frozenset({EnvironmentTarget.APP_CONTAINER}),
        values={
            "METADATA_DB_URL": "postgresql://db",
            "METADATA_DB_BACKEND": "postgresql",
        },
        reason="postgres override",
    )
    plan = _plan(tmp_path, assignments=(assignment,))
    app = resolve_environment(plan, APP_CONTAINER_POLICY)
    assert app.values["METADATA_DB_URL"] == "postgresql://db"
    assert (
        "METADATA_DB_URL" not in resolve_environment(plan, HOST_RUNTIME_POLICY).values
    )
    assert "METADATA_DB_URL" not in docker_control_environment(plan, platform="linux")


def test_container_manifest_path_replaces_host_graph_path_only_in_app_payload(
    tmp_path: Path,
) -> None:
    assignments = (
        CommandDerivedAssignment(
            targets=frozenset({EnvironmentTarget.HOST_RUNTIME}),
            values={"AGENTSEEK_GRAPHS": "/host/config.json"},
            reason="host config",
        ),
        CommandDerivedAssignment(
            targets=frozenset({EnvironmentTarget.APP_CONTAINER}),
            values={"AGENTSEEK_GRAPHS": "/deps/agent/config.json"},
            reason="container config",
        ),
    )
    plan = _plan(tmp_path, assignments=assignments)
    assert (
        resolve_environment(plan, HOST_RUNTIME_POLICY).values["AGENTSEEK_GRAPHS"]
        == "/host/config.json"
    )
    assert (
        resolve_environment(plan, APP_CONTAINER_POLICY).values["AGENTSEEK_GRAPHS"]
        == "/deps/agent/config.json"
    )


def test_public_boundary_repr_redacts_values(tmp_path: Path) -> None:
    from agentseek_api.dotenv_adapter import parse_dotenv_document

    sentinel = "hostile-secret-sentinel"
    plan = _plan(
        tmp_path, config_mapping={"TOKEN": sentinel}, launch={"LAUNCH": sentinel}
    )
    assignment = CommandDerivedAssignment(
        targets=frozenset({EnvironmentTarget.HOST_RUNTIME}),
        values={"COMMAND": sentinel},
        reason="test",
    )
    resolved = resolved_fixture(values={"TOKEN": sentinel}, declared_keys={"TOKEN"})
    dotenv_path = tmp_path / "redacted.env"
    dotenv_path.write_text(f"TOKEN={sentinel}\n", encoding="utf-8")
    document = parse_dotenv_document(dotenv_path)

    rendered = [
        repr(plan),
        repr(assignment),
        repr(resolved),
        repr(document),
        repr(document[0]),
    ]
    assert all(sentinel not in item for item in rendered)
    assert {field.name for field in fields(plan)} >= {
        "config_mapping",
        "launch_environment",
    }
