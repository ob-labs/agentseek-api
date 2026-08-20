from __future__ import annotations

import hashlib
import io
import json
import os
import tarfile
import zipfile
from pathlib import Path

import pytest

import agentseek_api.container_build as container_build
from agentseek_api.container_build import (
    PUBLISHED_RUNTIME_ARTIFACT,
    AuthPayloadPatch,
    FinalAuthSelection,
    InstallActionKind,
    RuntimeArtifactSource,
    RuntimeArtifactV1,
    SourceReason,
    candidate_runtime_artifact,
    interpret_host_runtime_policy,
    interpret_manifest_runtime_policy,
    materialize_build_bundle,
    plan_container_image,
    plan_generated_up_auth,
)
from agentseek_api.environment import EnvironmentOrigin
from tests.container_plan_helpers import make_graph_project


def test_bundle_excludes_env_and_records_only_selected_regular_files(
    tmp_path: Path,
) -> None:
    project = make_graph_project(tmp_path)
    (project / ".env").write_text("TOKEN=build-canary", encoding="utf-8")
    (project / "asset.txt").write_text("allowed", encoding="utf-8")

    plan = plan_container_image(
        config_path=project / "agentseek.json",
        dotenv_paths=(project / ".env",),
        build_include=("asset.txt",),
    )
    bundle = materialize_build_bundle(
        plan,
        dockerfile_bytes=b"FROM scratch\n",
        output_root=tmp_path / "bundle",
    )

    assert not (bundle.context / ".env").exists()
    assert b"build-canary" not in bundle.archive_bytes()
    assert {entry.relative_path for entry in bundle.inventory} == {
        "app/asset.txt",
        "app/chat/__init__.py",
        "app/chat/graph.py",
        "manifest.v1.json",
        "runtime-constraints.txt",
        "Dockerfile",
    }


def test_published_runtime_artifact_is_exact_and_path_free() -> None:
    artifact = PUBLISHED_RUNTIME_ARTIFACT
    assert artifact.source is RuntimeArtifactSource.PUBLISHED_INDEX
    assert artifact.requirement == "agentseek-api[embedded]==0.3.0"
    assert artifact.candidate_wheel is None
    assert artifact.candidate_sha256 is None


def test_install_actions_are_separate_from_runtime_import_roots(tmp_path: Path) -> None:
    project = make_graph_project(tmp_path)
    source_only = project / "extras"
    source_only.mkdir()
    (source_only / "helper.py").write_text("VALUE = 1\n", encoding="utf-8")
    config = project / "agentseek.json"
    payload = json.loads(config.read_text(encoding="utf-8"))
    payload["dependencies"] = [".", "./extras", "httpx>=0.27"]
    config.write_text(json.dumps(payload), encoding="utf-8")

    plan = plan_container_image(config_path=config)

    assert [action.kind for action in plan.install_actions] == [
        InstallActionKind.SOURCE_ONLY,
        InstallActionKind.SOURCE_ONLY,
        InstallActionKind.PEP508,
    ]
    assert plan.manifest.dependencies == ("/deps/agent", "/deps/agent/extras")


def test_manifest_serialization_omits_absent_and_preserves_explicit_values(
    tmp_path: Path,
) -> None:
    project = make_graph_project(tmp_path)
    config = project / "agentseek.json"
    config.write_text(
        json.dumps(
            {
                "graphs": {
                    "chat": {
                        "graph": "chat.graph:graph",
                        "name": "",
                        "input_schema": {"nullable": None},
                    }
                },
                "store": {"ttl": {"refresh_on_read": False, "default_ttl": 0}},
                "http": {
                    "disable_mcp": False,
                    "disable_a2a": True,
                    "cors": {"allow_origins": [], "max_age": 0},
                },
                "auth": {"disable_studio_auth": False},
            }
        ),
        encoding="utf-8",
    )

    document = json.loads(
        plan_container_image(config_path=config).manifest.to_json_bytes()
    )

    graph = document["graphs"]["chat"]
    assert "prepare_input" not in graph
    assert "extract_output" not in graph
    assert graph["name"] == ""
    assert graph["input_schema"]["nullable"] is None
    assert document["store"]["ttl"] == {
        "refresh_on_read": False,
        "default_ttl": 0,
    }
    assert document["http"]["cors"] == {"allow_origins": [], "max_age": 0}
    assert document["auth"]["disable_studio_auth"] is False


@pytest.mark.parametrize(
    "payload,path",
    [
        ({"store": {"index": {"api_key": "manifest-canary"}}}, "store.index"),
        ({"http": {"unknown": True}}, "http"),
        ({"auth": {"unknown": "manifest-canary"}}, "auth"),
    ],
)
def test_unknown_or_launch_only_manifest_fields_fail_value_free(
    tmp_path: Path, payload: dict[str, object], path: str
) -> None:
    project = make_graph_project(tmp_path)
    config = project / "agentseek.json"
    document = {"graphs": {"chat": "chat.graph:graph"}, **payload}
    config.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(Exception, match=path) as caught:
        plan_container_image(config_path=config)

    assert "manifest-canary" not in str(caught.value)
    assert "manifest-canary" not in repr(caught.value)


def test_selected_source_reasons_merge_for_same_source(tmp_path: Path) -> None:
    project = make_graph_project(tmp_path)
    plan = plan_container_image(
        config_path=project / "agentseek.json",
        build_include=("chat/graph.py",),
    )
    assert plan.selected_sources["app/chat/graph.py"].reasons == frozenset(
        {SourceReason.GRAPH, SourceReason.BUILD_INCLUDE, SourceReason.DEPENDENCY}
    )


def test_deterministic_archive_rejects_changed_context(tmp_path: Path) -> None:
    project = make_graph_project(tmp_path)
    bundle = materialize_build_bundle(
        plan_container_image(config_path=project / "agentseek.json"),
        dockerfile_bytes=b"FROM scratch\n",
        output_root=tmp_path / "bundle",
    )
    first = bundle.archive_bytes()
    assert first == bundle.archive_bytes()
    with tarfile.open(fileobj=io.BytesIO(first), mode="r:") as archive:
        assert archive.getnames() == sorted(
            entry.relative_path for entry in bundle.inventory
        )
        assert all(member.uid == member.gid == member.mtime == 0 for member in archive)

    (bundle.context / "app/chat/graph.py").write_text("changed\n", encoding="utf-8")
    with pytest.raises(Exception, match="changed"):
        bundle.archive_bytes()


def test_static_incompatible_runtime_pin_fails_with_migration_guidance(
    tmp_path: Path,
) -> None:
    project = make_graph_project(tmp_path)
    (project / "requirements.txt").write_text(
        "agentseek-api==0.2.2\n", encoding="utf-8"
    )
    with pytest.raises(Exception, match=r"0\.3\.0.*migrat"):
        plan_container_image(config_path=project / "agentseek.json")


def test_direct_https_is_install_action_not_manifest_path(tmp_path: Path) -> None:
    project = make_graph_project(tmp_path)
    config = project / "agentseek.json"
    payload = json.loads(config.read_text(encoding="utf-8"))
    url = "https://packages.example.invalid/fixture.whl"
    payload["dependencies"] = [url]
    config.write_text(json.dumps(payload), encoding="utf-8")
    plan = plan_container_image(config_path=config)
    assert [(item.kind, item.operand) for item in plan.install_actions] == [
        (InstallActionKind.PEP508, url)
    ]
    assert plan.manifest.dependencies == ()
    assert url not in plan.manifest.to_json_bytes().decode()


def test_generated_up_auth_preserves_empty_and_rewrites_local_file(
    tmp_path: Path,
) -> None:
    project = make_graph_project(tmp_path)
    auth = project / "auth.py"
    auth.write_text("auth = object()\n", encoding="utf-8")
    plan = plan_container_image(config_path=project / "agentseek.json")
    origin = EnvironmentOrigin("launch", "AUTH_MODULE_PATH")

    empty_plan, empty_patch = plan_generated_up_auth(
        plan, FinalAuthSelection(value="", origin=origin)
    )
    assert empty_patch == AuthPayloadPatch("")
    assert empty_plan.selected_sources == plan.selected_sources

    local_plan, local_patch = plan_generated_up_auth(
        plan, FinalAuthSelection(value="auth.py:auth", origin=origin)
    )
    assert local_patch == AuthPayloadPatch("/deps/agent/auth.py:auth")
    assert local_plan.selected_sources["app/auth.py"].reasons == frozenset(
        {SourceReason.AUTH, SourceReason.DEPENDENCY}
    )


def test_candidate_wheel_hash_and_metadata_are_verified(tmp_path: Path) -> None:
    wheel = tmp_path / "agentseek_api-0.3.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(
            "agentseek_api-0.3.0.dist-info/METADATA",
            "Metadata-Version: 2.1\nName: agentseek-api\nVersion: 0.3.0\n",
        )
    digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
    artifact = candidate_runtime_artifact(wheel, digest)
    assert artifact.source is RuntimeArtifactSource.CANDIDATE_WHEEL
    assert artifact.candidate_sha256 == digest


@pytest.mark.parametrize("value", ["../escape", "/outside"])
def test_build_include_rejects_escape(tmp_path: Path, value: str) -> None:
    project = make_graph_project(tmp_path)
    with pytest.raises(Exception, match="project"):
        plan_container_image(
            config_path=project / "agentseek.json", build_include=(value,)
        )


@pytest.mark.parametrize(
    ("marker", "expected_kind", "expected_operand_suffix", "runtime_root"),
    [
        ("pyproject.toml", InstallActionKind.PROJECT, "/dep", None),
        ("setup.py", InstallActionKind.PROJECT, "/dep", None),
        (
            "requirements.txt",
            InstallActionKind.REQUIREMENTS,
            "/dep/requirements.txt",
            None,
        ),
        (None, InstallActionKind.SOURCE_ONLY, "/dep", "/deps/agent/dep"),
    ],
)
def test_local_dependency_classification_is_exact(
    tmp_path: Path,
    marker: str | None,
    expected_kind: InstallActionKind,
    expected_operand_suffix: str,
    runtime_root: str | None,
) -> None:
    project = make_graph_project(tmp_path)
    dependency = project / "dep"
    dependency.mkdir()
    (dependency / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    if marker == "pyproject.toml":
        (dependency / marker).write_text(
            '[project]\nname="fixture"\nversion="1.0.0"\n', encoding="utf-8"
        )
    elif marker == "setup.py":
        (dependency / marker).write_text(
            "raise RuntimeError('must never execute on host')\n", encoding="utf-8"
        )
    elif marker == "requirements.txt":
        (dependency / marker).write_text("httpx>=0.27\n", encoding="utf-8")
    config = project / "agentseek.json"
    payload = json.loads(config.read_text(encoding="utf-8"))
    payload["dependencies"] = ["./dep"]
    config.write_text(json.dumps(payload), encoding="utf-8")

    plan = plan_container_image(config_path=config)

    assert len(plan.install_actions) == 1
    assert plan.install_actions[0].kind is expected_kind
    assert plan.install_actions[0].operand.endswith(expected_operand_suffix)
    assert plan.manifest.dependencies == (
        () if runtime_root is None else (runtime_root,)
    )


def test_structured_hooks_http_store_and_auth_are_selected_and_normalized(
    tmp_path: Path,
) -> None:
    project = make_graph_project(tmp_path)
    for filename in ("prepare.py", "extract.py", "embed.py", "web.py", "auth.py"):
        (project / filename).write_text("value = object()\n", encoding="utf-8")
    config = project / "agentseek.json"
    config.write_text(
        json.dumps(
            {
                "graphs": {
                    "chat": {
                        "graph": "chat.graph:graph",
                        "prepare_input": "./prepare.py:value",
                        "extract_output": "./extract.py:value",
                    }
                },
                "store": {"index": {"embed": "./embed.py:value", "dims": 0}},
                "http": {"app": "./web.py:value"},
                "auth": {"path": "./auth.py:value"},
            }
        ),
        encoding="utf-8",
    )

    plan = plan_container_image(config_path=config)
    document = plan.manifest.to_json_object()

    assert plan.selected_sources["app/prepare.py"].reasons == frozenset(
        {SourceReason.GRAPH_HOOK}
    )
    assert plan.selected_sources["app/extract.py"].reasons == frozenset(
        {SourceReason.GRAPH_HOOK}
    )
    assert plan.selected_sources["app/embed.py"].reasons == frozenset(
        {SourceReason.STORE_HOOK}
    )
    assert plan.selected_sources["app/web.py"].reasons == frozenset(
        {SourceReason.HTTP_APP}
    )
    assert plan.selected_sources["app/auth.py"].reasons == frozenset(
        {SourceReason.AUTH}
    )
    assert document["graphs"]["chat"]["prepare_input"] == "/deps/agent/prepare.py:value"
    assert document["store"]["index"]["embed"] == "/deps/agent/embed.py:value"
    assert document["http"]["app"] == "/deps/agent/web.py:value"
    assert "path" not in document["auth"]


def test_package_only_project_materializes_without_app_sources(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    config = project / "agentseek.json"
    config.write_text(
        json.dumps(
            {
                "graphs": {"chat": "installed.graph:graph"},
                "dependencies": ["installed-package>=1"],
            }
        ),
        encoding="utf-8",
    )
    plan = plan_container_image(config_path=config)
    bundle = materialize_build_bundle(
        plan,
        dockerfile_bytes=b"FROM scratch\n",
        output_root=tmp_path / "bundle",
    )
    assert not any(entry.relative_path.startswith("app/") for entry in bundle.inventory)
    assert bundle.dockerfile.read_bytes() == b"FROM scratch\n"


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFO is POSIX-only")
def test_build_include_rejects_special_file(tmp_path: Path) -> None:
    project = make_graph_project(tmp_path)
    fifo = project / "pipe"
    os.mkfifo(fifo)
    with pytest.raises(Exception, match="regular file"):
        plan_container_image(
            config_path=project / "agentseek.json", build_include=("pipe",)
        )


def test_selected_tree_rejects_escaping_symlink(tmp_path: Path) -> None:
    project = make_graph_project(tmp_path)
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    (project / "escape").symlink_to(outside)
    with pytest.raises(Exception, match="symlink"):
        plan_container_image(config_path=project / "agentseek.json")


def test_explicit_config_dotenv_and_vcs_includes_are_rejected(tmp_path: Path) -> None:
    project = make_graph_project(tmp_path)
    dotenv = project / ".env"
    dotenv.write_text("TOKEN=canary\n", encoding="utf-8")
    vcs = project / ".git"
    vcs.mkdir()
    (vcs / "config").write_text("canary", encoding="utf-8")
    for include, match in [
        ("agentseek.json", "excluded"),
        (".env", "excluded"),
        (".git", "VCS"),
    ]:
        with pytest.raises(Exception, match=match):
            plan_container_image(
                config_path=project / "agentseek.json",
                dotenv_paths=(dotenv,),
                build_include=(include,),
            )


def test_invalid_dotenv_fails_before_bundle_creation(tmp_path: Path) -> None:
    project = make_graph_project(tmp_path)
    dotenv = project / ".env"
    dotenv.write_bytes(b"GOOD=1\n\xff")
    output = tmp_path / "bundle"
    with pytest.raises(Exception, match="UTF-8"):
        plan = plan_container_image(
            config_path=project / "agentseek.json", dotenv_paths=(dotenv,)
        )
        materialize_build_bundle(
            plan, dockerfile_bytes=b"FROM scratch\n", output_root=output
        )
    assert not output.exists()


@pytest.mark.parametrize("contents", [None, b"VALID=1\nthis is not an assignment\n"])
def test_missing_or_malformed_dotenv_fails_closed(
    tmp_path: Path, contents: bytes | None
) -> None:
    project = make_graph_project(tmp_path)
    dotenv = project / "selected.env"
    if contents is not None:
        dotenv.write_bytes(contents)
    with pytest.raises(Exception, match="Env file|dotenv"):
        plan_container_image(
            config_path=project / "agentseek.json", dotenv_paths=(dotenv,)
        )


def test_duplicate_content_is_retained_at_distinct_destinations(tmp_path: Path) -> None:
    project = make_graph_project(tmp_path)
    (project / "first.txt").write_text("same", encoding="utf-8")
    (project / "second.txt").write_text("same", encoding="utf-8")
    plan = plan_container_image(
        config_path=project / "agentseek.json",
        build_include=("first.txt", "second.txt"),
    )
    assert "app/first.txt" in plan.selected_sources
    assert "app/second.txt" in plan.selected_sources


def test_credential_dependency_url_fails_without_value_disclosure(
    tmp_path: Path,
) -> None:
    project = make_graph_project(tmp_path)
    config = project / "agentseek.json"
    payload = json.loads(config.read_text(encoding="utf-8"))
    payload["dependencies"] = [
        "fixture @ https://user:dependency-canary@packages.example/fixture.whl"
    ]
    config.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(Exception, match="credentials") as caught:
        plan_container_image(config_path=config)
    assert "dependency-canary" not in str(caught.value)


def test_compatible_static_runtime_pin_is_accepted_and_constrained(
    tmp_path: Path,
) -> None:
    project = make_graph_project(tmp_path)
    (project / "requirements.txt").write_text(
        "agentseek-api>=0.3,<0.4\n", encoding="utf-8"
    )
    bundle = materialize_build_bundle(
        plan_container_image(config_path=project / "agentseek.json"),
        dockerfile_bytes=b"FROM scratch\n",
        output_root=tmp_path / "bundle",
    )
    assert (bundle.context / "runtime-constraints.txt").read_text() == (
        "agentseek-api==0.3.0\n"
    )


def test_provider_payload_is_structurally_absent_from_plan_and_bundle(
    tmp_path: Path,
) -> None:
    project = make_graph_project(tmp_path)
    config = project / "agentseek.json"
    payload = json.loads(config.read_text(encoding="utf-8"))
    payload["env"] = {"OPENAI_API_KEY": "provider-structural-canary"}
    config.write_text(json.dumps(payload), encoding="utf-8")
    plan = plan_container_image(config_path=config)
    bundle = materialize_build_bundle(
        plan,
        dockerfile_bytes=b"FROM scratch\n",
        output_root=tmp_path / "bundle",
    )
    assert "provider-structural-canary" not in repr(plan)
    assert b"provider-structural-canary" not in bundle.archive_bytes()


def test_pip_config_is_retained_only_as_secret_source_not_copied(
    tmp_path: Path,
) -> None:
    project = make_graph_project(tmp_path)
    pip_config = project / "pip.conf"
    pip_config.write_text("password=pip-canary\n", encoding="utf-8")
    config = project / "agentseek.json"
    payload = json.loads(config.read_text(encoding="utf-8"))
    payload["pip_config_file"] = "./pip.conf"
    config.write_text(json.dumps(payload), encoding="utf-8")
    plan = plan_container_image(config_path=config)
    bundle = materialize_build_bundle(
        plan,
        dockerfile_bytes=b"FROM scratch\n",
        output_root=tmp_path / "bundle",
    )
    assert plan.pip_config_file == pip_config
    assert b"pip-canary" not in bundle.archive_bytes()


def _write_candidate_wheel(path: Path, *, version: str = "0.3.0") -> str:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            f"agentseek_api-{version}.dist-info/METADATA",
            f"Metadata-Version: 2.1\nName: agentseek-api\nVersion: {version}\n",
        )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_candidate_wrong_metadata_and_symlink_are_rejected(tmp_path: Path) -> None:
    wrong = tmp_path / "wrong.whl"
    wrong_digest = _write_candidate_wheel(wrong, version="0.2.2")
    with pytest.raises(Exception, match="identity"):
        candidate_runtime_artifact(wrong, wrong_digest)
    link = tmp_path / "link.whl"
    link.symlink_to(wrong)
    with pytest.raises(Exception, match="regular"):
        candidate_runtime_artifact(link, wrong_digest)


def test_candidate_inconsistent_states_and_swaps_fail_closed(tmp_path: Path) -> None:
    project = make_graph_project(tmp_path)
    wheel = project / "candidate.whl"
    digest = _write_candidate_wheel(wheel)
    with pytest.raises(Exception, match="published"):
        RuntimeArtifactV1(
            distribution="agentseek-api",
            extra="embedded",
            version="0.3.0",
            source=RuntimeArtifactSource.PUBLISHED_INDEX,
            candidate_wheel=wheel,
            candidate_sha256=digest,
        )
    with pytest.raises(Exception, match="SHA-256"):
        candidate_runtime_artifact(wheel, "0" * 64)

    artifact = candidate_runtime_artifact(wheel, digest)
    plan = plan_container_image(
        config_path=project / "agentseek.json", runtime_artifact=artifact
    )
    assert [
        destination
        for destination, selected in plan.selected_sources.items()
        if selected.source_path == wheel
    ] == ["runtime/candidate.whl"]
    _write_candidate_wheel(wheel, version="0.2.2")
    output = tmp_path / "candidate-bundle"
    with pytest.raises(Exception, match="candidate wheel .*changed"):
        materialize_build_bundle(
            plan, dockerfile_bytes=b"FROM scratch\n", output_root=output
        )
    assert not output.exists()


def test_candidate_same_bytes_inode_replacement_is_rejected(tmp_path: Path) -> None:
    project = make_graph_project(tmp_path)
    wheel = project / "candidate.whl"
    digest = _write_candidate_wheel(wheel)
    plan = plan_container_image(
        config_path=project / "agentseek.json",
        runtime_artifact=candidate_runtime_artifact(wheel, digest),
    )
    replacement = tmp_path / "replacement.whl"
    replacement.write_bytes(wheel.read_bytes())
    replacement.replace(wheel)
    output = tmp_path / "same-bytes-bundle"

    with pytest.raises(Exception, match="identity"):
        materialize_build_bundle(
            plan, dockerfile_bytes=b"FROM scratch\n", output_root=output
        )

    assert not output.exists()


def test_candidate_wheel_must_be_confined_to_project(tmp_path: Path) -> None:
    project = make_graph_project(tmp_path)
    wheel = tmp_path / "outside.whl"
    digest = _write_candidate_wheel(wheel)
    with pytest.raises(Exception, match="project root"):
        plan_container_image(
            config_path=project / "agentseek.json",
            runtime_artifact=candidate_runtime_artifact(wheel, digest),
        )


def test_auth_precedence_and_package_override_remove_only_auth_reason(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    nested = project / "config"
    nested.mkdir(parents=True)
    (project / "pyproject.toml").write_text(
        '[project]\nname="fixture"\nversion="1.0"\n', encoding="utf-8"
    )
    local_a = project / "mapping.py"
    local_b = nested / "dedicated.py"
    local_a.write_text("auth = object()\n", encoding="utf-8")
    local_b.write_text("auth = object()\n", encoding="utf-8")
    config = nested / "agentseek.json"
    config.write_text(
        json.dumps(
            {
                "graphs": {"chat": "installed.graph:graph"},
                "env": {"AUTH_MODULE_PATH": "mapping.py:auth"},
                "auth": {"path": "./dedicated.py:auth"},
            }
        ),
        encoding="utf-8",
    )
    plan = plan_container_image(config_path=config)
    assert "app/config/dedicated.py" in plan.selected_sources
    assert "app/mapping.py" not in plan.selected_sources
    origin = EnvironmentOrigin("launch", "AUTH_MODULE_PATH")
    overridden, patch = plan_generated_up_auth(
        plan, FinalAuthSelection("package.auth:auth", origin)
    )
    assert patch == AuthPayloadPatch("package.auth:auth")
    assert "app/config/dedicated.py" not in overridden.selected_sources


def test_openapi_security_metadata_round_trips_and_rejects_credentials(
    tmp_path: Path,
) -> None:
    project = make_graph_project(tmp_path)
    config = project / "agentseek.json"
    base = {
        "graphs": {"chat": "chat.graph:graph"},
        "auth": {
            "openapi": {
                "securitySchemes": {
                    "oauth": {
                        "type": "oauth2",
                        "flows": {
                            "authorizationCode": {
                                "authorizationUrl": "https://id.example/authorize",
                                "tokenUrl": "https://id.example/token",
                                "scopes": {"read": "Read"},
                            }
                        },
                    }
                },
                "security": [{"oauth": ["read"]}],
            }
        },
    }
    config.write_text(json.dumps(base), encoding="utf-8")
    document = plan_container_image(config_path=config).manifest.to_json_object()
    assert document["auth"]["openapi"] == base["auth"]["openapi"]

    base["auth"]["openapi"]["securitySchemes"]["oauth"]["flows"]["authorizationCode"][
        "tokenUrl"
    ] = "https://user:manifest-canary@id.example/token"
    config.write_text(json.dumps(base), encoding="utf-8")
    with pytest.raises(Exception, match="credential-bearing") as caught:
        plan_container_image(config_path=config)
    assert "manifest-canary" not in str(caught.value)


@pytest.mark.parametrize("app_ref", ["installed.web:app", "./web.py:app"])
def test_http_and_auth_policy_effect_matches_host_config(
    tmp_path: Path, app_ref: str
) -> None:
    project = make_graph_project(tmp_path)
    (project / "web.py").write_text("app = object()\n", encoding="utf-8")
    config = project / "agentseek.json"
    http = {
        "disable_mcp": True,
        "disable_a2a": False,
        "app": app_ref,
        "cors": {
            "allow_origins": ["https://example.test"],
            "allow_origin_regex": "https://.*",
            "allow_methods": ["GET"],
            "allow_headers": ["x-test"],
            "allow_credentials": False,
            "expose_headers": [],
            "max_age": 0,
        },
    }
    auth = {
        "openapi": {
            "securitySchemes": {
                "apiKey": {"type": "apiKey", "name": "x-api-key", "in": "header"}
            },
            "security": [{"apiKey": []}],
        },
        "disable_studio_auth": True,
    }
    host_payload = {
        "graphs": {"chat": "chat.graph:graph"},
        "http": http,
        "auth": auth,
    }
    config.write_text(json.dumps(host_payload), encoding="utf-8")

    manifest = plan_container_image(config_path=config).manifest

    assert interpret_host_runtime_policy(
        host_payload, config_path=config
    ) == interpret_manifest_runtime_policy(manifest)


def test_selected_source_collision_rules_are_fail_closed(tmp_path: Path) -> None:
    first = tmp_path / "first.py"
    second = tmp_path / "second.py"
    first.write_text("FIRST = 1\n", encoding="utf-8")
    second.write_text("SECOND = 1\n", encoding="utf-8")
    selected: dict[str, object] = {}
    container_build._add_selected(  # noqa: SLF001 - invariant-level regression
        selected,
        destination="app/first.py",
        source=first,
        reason=SourceReason.GRAPH,
    )
    container_build._add_selected(  # noqa: SLF001 - explicit second destination
        selected,
        destination="copy/first.py",
        source=first,
        reason=SourceReason.BUILD_INCLUDE,
    )
    assert len(selected) == 2
    with pytest.raises(Exception, match="same destination"):
        container_build._add_selected(  # noqa: SLF001 - collision regression
            selected,
            destination="app/first.py",
            source=second,
            reason=SourceReason.AUTH,
        )


def test_shared_auth_source_loses_only_auth_reason_on_package_override(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    shared = project / "shared.py"
    shared.write_text("graph = auth = object()\n", encoding="utf-8")
    config = project / "agentseek.json"
    config.write_text(
        json.dumps(
            {
                "graphs": {"chat": "./shared.py:graph"},
                "auth": {"path": "./shared.py:auth"},
                "build_include": ["shared.py"],
            }
        ),
        encoding="utf-8",
    )
    plan = plan_container_image(config_path=config)
    origin = EnvironmentOrigin("launch", "AUTH_MODULE_PATH")
    overridden, _ = plan_generated_up_auth(
        plan, FinalAuthSelection("installed.auth:auth", origin)
    )
    assert overridden.selected_sources["app/shared.py"].reasons == frozenset(
        {SourceReason.GRAPH, SourceReason.BUILD_INCLUDE}
    )


def test_generated_up_rejects_outside_local_auth_override(tmp_path: Path) -> None:
    project = make_graph_project(tmp_path)
    outside = tmp_path / "outside.py"
    outside.write_text("auth = object()\n", encoding="utf-8")
    plan = plan_container_image(config_path=project / "agentseek.json")
    origin = EnvironmentOrigin("launch", "AUTH_MODULE_PATH")
    with pytest.raises(Exception, match="project root"):
        plan_generated_up_auth(plan, FinalAuthSelection(f"{outside}:auth", origin))


def test_top_level_runtime_shadow_source_is_rejected(tmp_path: Path) -> None:
    project = make_graph_project(tmp_path)
    shadow = project / "agentseek_api.py"
    shadow.write_text("SHADOW = True\n", encoding="utf-8")
    with pytest.raises(Exception, match="shadow"):
        plan_container_image(
            config_path=project / "agentseek.json",
            build_include=("agentseek_api.py",),
        )


@pytest.mark.skipif(os.name == "nt", reason="POSIX mode assertion")
def test_existing_wrong_mode_output_root_is_rejected(tmp_path: Path) -> None:
    project = make_graph_project(tmp_path)
    output = tmp_path / "bundle"
    output.mkdir(mode=0o755)
    with pytest.raises(Exception, match="private"):
        materialize_build_bundle(
            plan_container_image(config_path=project / "agentseek.json"),
            dockerfile_bytes=b"FROM scratch\n",
            output_root=output,
        )


def test_archive_rejects_added_symlink(tmp_path: Path) -> None:
    project = make_graph_project(tmp_path)
    bundle = materialize_build_bundle(
        plan_container_image(config_path=project / "agentseek.json"),
        dockerfile_bytes=b"FROM scratch\n",
        output_root=tmp_path / "bundle",
    )
    (bundle.context / "escape").symlink_to(project / "agentseek.json")
    with pytest.raises(Exception, match="unsafe"):
        bundle.archive_bytes()
