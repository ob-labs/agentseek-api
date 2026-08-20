from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from dataclasses import replace
from pathlib import Path

import pytest

import agentseek_api.container_build as container_build
from agentseek_api.container_build import (
    PUBLISHED_RUNTIME_ARTIFACT,
    AuthPayloadPatch,
    ContainerBuildError,
    FinalAuthSelection,
    InstallActionKind,
    RuntimeArtifactSource,
    RuntimeArtifactV1,
    SourceReason,
    candidate_runtime_artifact,
    interpret_host_runtime_policy,
    interpret_manifest_runtime_policy,
    load_container_runtime_manifest_v1,
    materialize_build_bundle,
    plan_container_image,
    plan_generated_up_auth,
    render_build_dockerfile,
    validate_dependency_specification,
)
from agentseek_api.environment import EnvironmentOrigin
from tests.container_plan_helpers import (
    build_plan_fixture,
    make_graph_project,
    package_only_build_plan_fixture,
    read_archive_member,
    write_sanitized_manifest,
)


def test_canonical_manifest_loader_preserves_explicit_values_and_container_roots(
    tmp_path: Path,
) -> None:
    manifest_path = write_sanitized_manifest(tmp_path)
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    document.update(
        dependencies=["/deps/agent", "/deps/agent/source"],
        store={"ttl": {"refresh_on_read": False, "default_ttl": 0}},
        http={
            "app": "/deps/agent/web.py:app",
            "disable_mcp": False,
            "disable_a2a": True,
            "cors": {"allow_origins": [], "allow_credentials": False, "max_age": 0},
        },
        auth={
            "openapi": {"securitySchemes": {}, "security": []},
            "disable_studio_auth": False,
        },
    )
    manifest_path.write_text(json.dumps(document), encoding="utf-8")

    manifest = load_container_runtime_manifest_v1(manifest_path)

    assert manifest.to_json_object() == document


@pytest.mark.parametrize(
    "root",
    [
        ".",
        "source",
        "../escape",
        "/deps/agent/../escape",
        "/other/root",
        "/deps/agent/",
    ],
)
def test_canonical_manifest_loader_rejects_noncanonical_container_roots(
    tmp_path: Path, root: str
) -> None:
    manifest_path = write_sanitized_manifest(tmp_path)
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    document["dependencies"] = [root]
    manifest_path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ContainerBuildError, match="container"):
        load_container_runtime_manifest_v1(manifest_path)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (("runtime", "distribution", "other-runtime"), "identity"),
        (("runtime", "version", "9.9.9"), "identity"),
        (("runtime", "contract", "resolve"), "identity"),
        (("http", "unknown", True), "http"),
        (("http", "cors", {"unknown": True}), "cors"),
        (("auth", "path", "secret.py:auth"), "auth"),
        (("unknown", "field", True), "manifest"),
        (("unknown", "env", {"TOKEN": "secret.py"}), "manifest"),
        (("unknown", "pip_config_file", "secret.py"), "manifest"),
        (("unknown", "dockerfile_lines", ["secret.py"]), "manifest"),
    ],
)
def test_canonical_manifest_loader_rejects_unknown_or_forbidden_fields_value_free(
    tmp_path: Path, mutation: tuple[str, str, object], match: str
) -> None:
    manifest_path = write_sanitized_manifest(tmp_path)
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    parent, field, value = mutation
    if parent == "unknown":
        document[field] = value
    else:
        nested = document.setdefault(parent, {})
        assert isinstance(nested, dict)
        nested[field] = value
    manifest_path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ContainerBuildError, match=match) as caught:
        load_container_runtime_manifest_v1(manifest_path)
    assert "secret.py" not in str(caught.value)


def test_canonical_manifest_loader_rejects_boolean_schema_version(
    tmp_path: Path,
) -> None:
    manifest_path = write_sanitized_manifest(tmp_path)
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    document["schema_version"] = True
    manifest_path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ContainerBuildError, match="identity"):
        load_container_runtime_manifest_v1(manifest_path)


@pytest.mark.parametrize(
    ("section", "reference"),
    [
        ("graphs", "/deps/agent/../escape.py:graph"),
        ("store", "/deps/agent/source/../../escape.py:embed"),
        ("http", "/deps/agent/../escape.py:app"),
    ],
)
def test_canonical_manifest_loader_rejects_absolute_copied_module_escapes(
    tmp_path: Path, section: str, reference: str
) -> None:
    manifest_path = write_sanitized_manifest(tmp_path)
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    if section == "graphs":
        document["graphs"] = {"chat": reference}
    elif section == "store":
        document["store"] = {"index": {"embed": reference}}
    else:
        document["http"] = {"app": reference}
    manifest_path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ContainerBuildError, match="container"):
        load_container_runtime_manifest_v1(manifest_path)


@pytest.mark.parametrize("reference", ["missing-symbol", ":graph", "bad-module!:graph"])
def test_canonical_manifest_loader_rejects_nonimportable_package_references(
    tmp_path: Path, reference: str
) -> None:
    manifest_path = write_sanitized_manifest(tmp_path)
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    document["graphs"] = {"chat": reference}
    manifest_path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ContainerBuildError, match="importable package"):
        load_container_runtime_manifest_v1(manifest_path)


def test_canonical_manifest_loader_rejects_present_null_auth_path(
    tmp_path: Path,
) -> None:
    manifest_path = write_sanitized_manifest(tmp_path)
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    document["auth"] = {"path": None}
    manifest_path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ContainerBuildError, match="auth.path"):
        load_container_runtime_manifest_v1(manifest_path)


@pytest.mark.parametrize(
    "reference",
    ["", "agentseek_api.services.sample_graphs:_build_echo_graph.extra"],
)
def test_canonical_loader_and_runtime_consumer_reject_same_graph_reference(
    tmp_path: Path, reference: str
) -> None:
    from agentseek_api.services.langgraph_service import (
        GraphManifestError,
        _load_module_symbol,
    )

    manifest_path = write_sanitized_manifest(tmp_path)
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    document["graphs"] = {"chat": reference}
    manifest_path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ContainerBuildError):
        load_container_runtime_manifest_v1(manifest_path)
    with pytest.raises(GraphManifestError):
        _load_module_symbol(
            dotted_path=reference,
            graph_id="chat",
            field_name="graph",
            manifest_path=manifest_path,
        )


@pytest.mark.parametrize(
    "http",
    [
        {"disable_mcp": "bad"},
        {"cors": {"max_age": "bad"}},
        {"app": "../host.py:app"},
    ],
)
def test_build_planning_and_preloaded_loading_share_http_rejection(
    tmp_path: Path, http: dict[str, object]
) -> None:
    project = make_graph_project(tmp_path)
    config = project / "agentseek.json"
    host_document = json.loads(config.read_text(encoding="utf-8"))
    host_document["http"] = http
    config.write_text(json.dumps(host_document), encoding="utf-8")

    with pytest.raises(ContainerBuildError):
        plan_container_image(config_path=config)

    manifest_path = write_sanitized_manifest(tmp_path)
    manifest_document = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_document["http"] = http
    manifest_path.write_text(json.dumps(manifest_document), encoding="utf-8")
    with pytest.raises(ContainerBuildError):
        load_container_runtime_manifest_v1(manifest_path)


@pytest.mark.parametrize(
    "patch",
    [
        {"graphs": {"chat": {"graph": "chat.graph:graph", "unknown": True}}},
        {"store": {"index": {"unknown": True}}},
        {"auth": {"openapi": {"unknown": True}}},
        {
            "auth": {
                "openapi": {
                    "securitySchemes": {
                        "oidc": {
                            "type": "openIdConnect",
                            "openIdConnectUrl": "https://user:manifest-canary@id.example/config",
                        }
                    },
                    "security": [{"oidc": []}],
                }
            }
        },
    ],
)
def test_preloaded_loader_rejects_unknown_nested_or_credential_metadata_value_free(
    tmp_path: Path, patch: dict[str, object]
) -> None:
    manifest_path = write_sanitized_manifest(tmp_path)
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    document.update(patch)
    manifest_path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ContainerBuildError) as caught:
        load_container_runtime_manifest_v1(manifest_path)
    assert "manifest-canary" not in str(caught.value)


@pytest.mark.parametrize("flag", [False, True])
def test_loaded_manifest_runtime_policy_matches_host_for_boolean_matrix(
    tmp_path: Path, flag: bool
) -> None:
    project = make_graph_project(tmp_path)
    config = project / "agentseek.json"
    host = {
        "graphs": {"chat": "chat.graph:graph"},
        "http": {
            "app": "installed.web:app",
            "disable_mcp": flag,
            "disable_a2a": not flag,
            "cors": {
                "allow_origins": [],
                "allow_methods": [],
                "allow_headers": [],
                "allow_credentials": False,
                "expose_headers": [],
                "max_age": 0,
            },
        },
        "auth": {
            "openapi": {
                "securitySchemes": {
                    "key": {"type": "apiKey", "name": "x-key", "in": "header"}
                },
                "security": [{"key": []}],
            },
            "disable_studio_auth": flag,
        },
    }
    config.write_text(json.dumps(host), encoding="utf-8")
    planned = plan_container_image(config_path=config).manifest
    manifest_path = project / "manifest.v1.json"
    manifest_path.write_bytes(planned.to_json_bytes())
    loaded = load_container_runtime_manifest_v1(manifest_path)

    assert interpret_host_runtime_policy(
        host, config_path=config
    ) == interpret_manifest_runtime_policy(loaded)


def _dockerfile_run_argv(text: str) -> list[list[str]]:
    return [
        json.loads(line[line.index("[") :])
        for line in text.splitlines()
        if line.startswith("RUN ")
    ]


def _generated_python_check(text: str, needle: str) -> str:
    return next(
        argv[2]
        for argv in _dockerfile_run_argv(text)
        if argv[:2] == ["python", "-c"] and needle in argv[2]
    )


def _candidate_build_plan(root: Path):
    project = make_graph_project(root)
    wheel = project / "candidate.whl"
    digest = _write_candidate_wheel(wheel)
    return plan_container_image(
        config_path=project / "agentseek.json",
        runtime_artifact=candidate_runtime_artifact(wheel, digest),
    )


def test_candidate_renderer_stages_a_pep427_valid_wheel_filename(
    tmp_path: Path,
) -> None:
    text = render_build_dockerfile(_candidate_build_plan(tmp_path)).decode("utf-8")
    canonical = "agentseek_api-0.3.0-py3-none-any.whl"

    assert f'"/opt/agentseek/runtime/{canonical}"' in text
    assert f'"/opt/agentseek/runtime/{canonical}[embedded]"' in text
    assert "/opt/agentseek/runtime/agentseek-api-0.3.0.whl" not in text


def test_windows_binary_flag_is_used_for_bundle_file_io(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binary_flag = 0x8000
    monkeypatch.setattr(container_build.os, "O_BINARY", binary_flag, raising=False)

    assert container_build._output_file_flags() & binary_flag
    assert container_build._source_file_flags() & binary_flag


def test_dockerfile_uses_manifest_labels_and_buildkit_pip_secret(
    tmp_path: Path,
) -> None:
    plan = build_plan_fixture(tmp_path)
    dockerfile = render_build_dockerfile(plan)
    text = dockerfile.decode("utf-8")
    bundle = materialize_build_bundle(
        plan,
        dockerfile_bytes=dockerfile,
        output_root=tmp_path / "bundle",
    )
    assert "COPY app /deps/agent" in text
    assert "COPY manifest.v1.json /opt/agentseek/manifest.v1.json" in text
    assert "COPY runtime-constraints.txt /opt/agentseek/runtime-constraints.txt" in text
    assert "org.agentseek.environment-contract=preloaded-v1" in text
    assert "org.agentseek.runtime-manifest=/opt/agentseek/manifest.v1.json" in text
    assert "org.agentseek.runtime-distribution=agentseek-api" in text
    assert "org.agentseek.runtime-version=0.3.0" in text
    assert "agentseek-api[embedded]==0.3.0" in text
    assert 'python", "-m", "pip", "check' in text
    assert "importlib.metadata" in text
    assert "sysconfig.get_paths" in text
    assert "--mount=type=secret,id=pip_config,target=/etc/pip.conf" in text
    assert "pip.conf" not in "\n".join(
        line for line in text.splitlines() if line.startswith("COPY")
    )
    assert "ENTRYPOINT []" in text
    assert (
        '"serve", "--environment-mode", "preloaded-v1", "--host", '
        '"0.0.0.0", "--port", "2024"' in text
    )
    assert bundle.dockerfile.read_bytes() == dockerfile
    assert (
        next(
            item.sha256
            for item in bundle.inventory
            if item.relative_path == "Dockerfile"
        )
        == hashlib.sha256(dockerfile).hexdigest()
    )
    assert read_archive_member(bundle.archive_bytes(), "Dockerfile") == dockerfile


def test_generated_image_command_ignores_hostile_workdir_and_pythonpath(
    tmp_path: Path,
) -> None:
    text = render_build_dockerfile(build_plan_fixture(tmp_path)).decode("utf-8")
    command_line = next(line for line in text.splitlines() if line.startswith("CMD "))
    command = json.loads(command_line.removeprefix("CMD "))
    module_index = command.index("agentseek_api.cli")
    probe_command = [*command[: module_index + 1], "version"]
    hostile = tmp_path / "hostile"
    package = hostile / "agentseek_api"
    package.mkdir(parents=True)
    marker = tmp_path / "hostile-image-bootstrap"
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "cli.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('canary')\nraise SystemExit(23)\n",
        encoding="utf-8",
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(hostile)

    completed = subprocess.run(
        probe_command,
        cwd=hostile,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )

    assert completed.returncode == 0
    assert completed.stdout == "agentseek-api 0.3.0\n"
    assert marker.exists() is False


def test_package_only_plan_does_not_copy_missing_app_directory(
    tmp_path: Path,
) -> None:
    plan = package_only_build_plan_fixture(tmp_path)
    dockerfile = render_build_dockerfile(plan)
    text = dockerfile.decode("utf-8")

    assert "COPY app /deps/agent" not in text
    assert "WORKDIR /deps/agent" in text
    materialize_build_bundle(
        plan,
        dockerfile_bytes=dockerfile,
        output_root=tmp_path / "package-only-bundle",
    )


@pytest.mark.parametrize(
    "requirement",
    [
        "https://user:password@example.invalid/pkg.whl",
        "package @ https://user:password@example.invalid/pkg.whl",
        "git+https://token@example.invalid/repo.git",
        "package @ git+https://user%40name:secret@example.invalid/repo.git",
    ],
)
def test_dependency_url_credentials_are_rejected(requirement: str) -> None:
    with pytest.raises(ContainerBuildError, match="pip_config_file"):
        validate_dependency_specification(requirement)


@pytest.mark.parametrize(
    "requirement",
    [
        "https://example.invalid/pkg.whl",
        "package @ https://example.invalid/pkg.whl",
        "package @ https://example.invalid/pkg.whl#sha256=" + "a" * 64,
        "package @ https://example.invalid/pkg.whl#sha256="
        + "B" * 64
        + "&subdirectory=python/pkg",
        "ordinary-package[extra]>=1; python_version >= '3.12'",
        "./local dependency",
        "../local",
        "C:\\workspace\\local",
    ],
)
def test_dependency_specification_accepts_v1_inputs(requirement: str) -> None:
    validate_dependency_specification(requirement)


@pytest.mark.parametrize(
    "requirement",
    [
        "http://example.invalid/pkg.whl",
        "file:///tmp/pkg.whl",
        "git+file:///tmp/repo",
        "git+https://example.invalid/repo.git",
        "ftp://example.invalid/pkg.whl",
        "ssh://example.invalid/repo",
        "hg+ssh://example.invalid/repo",
        "git@example.invalid:repo.git",
        "custom+scheme://example.invalid/pkg",
        "https://example.invalid/pkg.whl?token=secret",
        "https://example.invalid/pkg.whl?%74oken=secret",
        "https://example.invalid/pkg.whl%3Fauth=secret",
        "https://example.invalid/pkg.whl#token=secret",
        "https://example.invalid/pkg.whl#sha256=bad",
        "https://example.invalid/pkg.whl#sha256=" + "a" * 64 + "&sha256=" + "b" * 64,
        "https://example.invalid/pkg.whl#subdirectory=../escape",
        "https://example.invalid/pkg.whl#subdirectory=not/./normalized",
        "https://example.invalid/pkg.whl#subdirectory=wheel%26token=secret",
        "https://example.invalid/pkg.whl%23token=secret",
    ],
)
def test_dependency_specification_rejects_non_v1_urls(requirement: str) -> None:
    with pytest.raises(ContainerBuildError, match="pip_config_file"):
        validate_dependency_specification(requirement)


def test_external_pip_config_is_identity_only_secret_source(tmp_path: Path) -> None:
    project = make_graph_project(tmp_path)
    external = tmp_path / "private-pip.conf"
    external.write_text("password=external-canary\n", encoding="utf-8")
    config = project / "agentseek.json"
    payload = json.loads(config.read_text(encoding="utf-8"))
    payload["pip_config_file"] = str(external)
    config.write_text(json.dumps(payload), encoding="utf-8")

    plan = plan_container_image(config_path=config)

    assert plan.pip_config_file == external
    assert plan.pip_config_identity is not None
    assert external not in (
        source.source_path for source in plan.selected_sources.values()
    )
    assert "external-canary" not in repr(plan)


@pytest.mark.parametrize("kind", ["missing", "symlink", "directory", "unreadable"])
def test_pip_config_requires_readable_regular_nonsymlink_file(
    tmp_path: Path, kind: str
) -> None:
    project = make_graph_project(tmp_path)
    pip_config = tmp_path / "pip.conf"
    if kind == "symlink":
        target = tmp_path / "target.conf"
        target.write_text("[global]\n", encoding="utf-8")
        pip_config.symlink_to(target)
    elif kind == "directory":
        pip_config.mkdir()
    elif kind == "unreadable":
        pip_config.write_text("[global]\n", encoding="utf-8")
        pip_config.chmod(0)
    config = project / "agentseek.json"
    payload = json.loads(config.read_text(encoding="utf-8"))
    payload["pip_config_file"] = str(pip_config)
    config.write_text(json.dumps(payload), encoding="utf-8")

    try:
        with pytest.raises(ContainerBuildError, match="readable regular file"):
            plan_container_image(config_path=config)
    finally:
        if kind == "unreadable":
            pip_config.chmod(0o600)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFO is POSIX-only")
def test_pip_config_rejects_special_file_without_opening_it(tmp_path: Path) -> None:
    project = make_graph_project(tmp_path)
    pip_config = tmp_path / "pip.conf"
    os.mkfifo(pip_config)
    config = project / "agentseek.json"
    payload = json.loads(config.read_text(encoding="utf-8"))
    payload["pip_config_file"] = str(pip_config)
    config.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ContainerBuildError, match="readable regular file"):
        plan_container_image(config_path=config)


def test_renderer_preserves_plan_fields_and_authoritative_order(tmp_path: Path) -> None:
    plan = build_plan_fixture(tmp_path)
    plan = replace(
        plan,
        base_image="python:3.13-slim-bookworm",
        python_version="3.13",
        image_distro="bookworm",
        dockerfile_lines=('RUN ["python", "-c", "print(\'trusted\')"]',),
    )

    text = render_build_dockerfile(plan).decode("utf-8")

    assert "FROM python:3.13-slim-bookworm" in text
    assert '# agentseek-python-version="3.13"' in text
    assert '# agentseek-image-distro="bookworm"' in text
    user_install = text.index("/deps/agent")
    custom = text.index("print('trusted')")
    runtime_install = text.rindex("agentseek-api[embedded]==0.3.0")
    manifest = text.index("COPY manifest.v1.json")
    labels = text.index("LABEL org.agentseek.environment-contract")
    assert user_install < custom < runtime_install < manifest < labels


@pytest.mark.parametrize("candidate", [False, True])
def test_exact_runtime_install_forces_selected_artifact_replacement(
    tmp_path: Path, candidate: bool
) -> None:
    plan = (
        _candidate_build_plan(tmp_path) if candidate else build_plan_fixture(tmp_path)
    )

    commands = _dockerfile_run_argv(render_build_dockerfile(plan).decode())
    runtime_install = next(
        argv
        for argv in commands
        if argv[:4] == ["python", "-m", "pip", "install"]
        and any("agentseek-api" in operand.replace("_", "-") for operand in argv)
    )

    assert "--force-reinstall" in runtime_install


def _run_generated_check(script: str, setup: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-O", "-c", setup + "\n" + f"exec({script!r})"],
        capture_output=True,
        check=False,
        text=True,
    )


@pytest.mark.parametrize("valid", [True, False])
def test_candidate_hash_verifier_has_success_and_failure_paths_under_optimize(
    tmp_path: Path, valid: bool
) -> None:
    plan = _candidate_build_plan(tmp_path)
    script = _generated_python_check(
        render_build_dockerfile(plan).decode(),
        "agentseek_api-0.3.0-py3-none-any.whl",
    )
    assert plan.runtime_artifact.candidate_wheel is not None
    candidate = tmp_path / "candidate-check.whl"
    candidate.write_bytes(
        plan.runtime_artifact.candidate_wheel.read_bytes()
        if valid
        else b"independently-invalid-candidate"
    )
    setup = (
        "import pathlib\n"
        "OriginalPath=pathlib.Path\n"
        f"actual=OriginalPath({str(candidate)!r})\n"
        "pathlib.Path=lambda value: actual if value=="
        "'/opt/agentseek/runtime/agentseek_api-0.3.0-py3-none-any.whl' "
        "else OriginalPath(value)"
    )

    completed = _run_generated_check(script, setup)

    assert (completed.returncode == 0) is valid, completed.stderr


_VALID_MANIFEST = (
    b'{"dependencies":["/deps/agent"],"graphs":{"chat":"chat.graph:graph"},'
    b'"runtime":{"contract":"preloaded-v1","distribution":"agentseek-api",'
    b'"version":"0.3.0"},"schema_version":1}\n'
)


@pytest.mark.parametrize("check", ["hash", "canonical", "parser"])
@pytest.mark.parametrize("valid", [True, False])
def test_manifest_verifier_has_success_and_failure_paths_under_optimize(
    tmp_path: Path, check: str, valid: bool
) -> None:
    script = _generated_python_check(
        render_build_dockerfile(build_plan_fixture(tmp_path)).decode(),
        "manifest.v1.json",
    )
    manifest = tmp_path / f"manifest-{check}.json"
    invalid = {
        "hash": b'{"schema_version":1}\n',
        "canonical": json.dumps(
            json.loads(_VALID_MANIFEST), indent=2, sort_keys=False
        ).encode()
        + b"\n",
        "parser": b"not-json\n",
    }
    manifest.write_bytes(_VALID_MANIFEST if valid else invalid[check])
    setup = (
        "import pathlib\n"
        "OriginalPath=pathlib.Path\n"
        f"actual=OriginalPath({str(manifest)!r})\n"
        "pathlib.Path=lambda value: actual if value=="
        "'/opt/agentseek/manifest.v1.json' else OriginalPath(value)"
    )

    completed = _run_generated_check(script, setup)

    assert (completed.returncode == 0) is valid, completed.stderr


@pytest.mark.parametrize("check", ["ownership", "version", "site-packages", "python"])
@pytest.mark.parametrize("valid", [True, False])
def test_runtime_verifier_has_success_and_failure_paths_under_optimize(
    tmp_path: Path, check: str, valid: bool
) -> None:
    script = _generated_python_check(
        render_build_dockerfile(build_plan_fixture(tmp_path)).decode(),
        "importlib.metadata",
    )
    site_packages = tmp_path / "site-packages"
    distribution_root = (
        tmp_path / "outside"
        if check == "site-packages" and not valid
        else site_packages
    )
    module = distribution_root / "agentseek_api" / "cli.py"
    module.parent.mkdir(parents=True)
    module.write_text("", encoding="utf-8")
    owned_file = (
        "agentseek_api/not-cli.py"
        if check == "ownership" and not valid
        else "agentseek_api/cli.py"
    )
    version = "9.9.9" if check == "version" and not valid else "0.3.0"
    python_version = "(3,11,0)" if check == "python" and not valid else "(3,12,0)"
    setup = "\n".join(
        (
            "import agentseek_api.cli,importlib.metadata,pathlib,sys,sysconfig",
            f"site=pathlib.Path({str(site_packages)!r})",
            f"distribution_root=pathlib.Path({str(distribution_root)!r})",
            f"agentseek_api.cli.__file__={str(module)!r}",
            "class Distribution:",
            f" version={version!r}",
            f" files=(importlib.metadata.PackagePath({owned_file!r}),)",
            " def locate_file(self,item): return distribution_root/item",
            "importlib.metadata.distribution=lambda name:Distribution()",
            "sysconfig.get_paths=lambda:{'purelib':str(site),'platlib':str(site)}",
            f"sys.version_info={python_version}",
        )
    )

    completed = _run_generated_check(script, setup)

    assert (completed.returncode == 0) is valid, completed.stderr


def test_renderer_json_escapes_install_operands_and_candidate_source(
    tmp_path: Path,
) -> None:
    project = make_graph_project(tmp_path)
    local = project / "local dependency"
    local.mkdir()
    (local / "requirements.txt").write_text("httpx>=0.27\n", encoding="utf-8")
    config = project / "agentseek.json"
    payload = json.loads(config.read_text(encoding="utf-8"))
    payload["dependencies"] = [
        "./local dependency",
        "package @ https://example.invalid/pkg.whl#sha256=" + "a" * 64,
    ]
    config.write_text(json.dumps(payload), encoding="utf-8")
    wheel = project / "candidate runtime.whl"
    digest = _write_candidate_wheel(wheel)
    plan = plan_container_image(
        config_path=config,
        runtime_artifact=candidate_runtime_artifact(wheel, digest),
    )

    text = render_build_dockerfile(plan).decode("utf-8")

    assert '"--requirement", "/deps/agent/local dependency/requirements.txt"' in text
    assert '"package @ https://example.invalid/pkg.whl#sha256=' in text
    assert 'COPY ["runtime/candidate runtime.whl", "/opt/agentseek/runtime/' in text
    assert text.index("candidate runtime.whl") < text.index(
        "agentseek_api-0.3.0-py3-none-any.whl[embedded]"
    )


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

    absent_plan, absent_patch = plan_generated_up_auth(plan, None)
    assert absent_patch is None
    assert all(
        SourceReason.AUTH not in source.reasons
        for source in absent_plan.selected_sources.values()
    )

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
    manifest_path = project / "manifest.v1.json"
    manifest_path.write_bytes(manifest.to_json_bytes())
    loaded = load_container_runtime_manifest_v1(manifest_path)

    assert interpret_host_runtime_policy(
        host_payload, config_path=config
    ) == interpret_manifest_runtime_policy(loaded)


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


# Task 4 Fix Round 1 regressions


def test_ordinary_selected_source_same_bytes_replacement_is_rejected(
    tmp_path: Path,
) -> None:
    project = make_graph_project(tmp_path)
    source = project / "asset.txt"
    source.write_text("stable", encoding="utf-8")
    plan = plan_container_image(
        config_path=project / "agentseek.json", build_include=("asset.txt",)
    )
    replacement = project / "replacement.txt"
    replacement.write_bytes(source.read_bytes())
    replacement.replace(source)

    with pytest.raises(Exception, match="identity"):
        materialize_build_bundle(
            plan,
            dockerfile_bytes=b"FROM scratch\n",
            output_root=tmp_path / "bundle",
        )

    assert not (tmp_path / "bundle").exists()


def test_ordinary_selected_source_hash_is_rechecked_when_metadata_is_restored(
    tmp_path: Path,
) -> None:
    project = make_graph_project(tmp_path)
    source = project / "asset.txt"
    source.write_text("safe-content", encoding="utf-8")
    plan = plan_container_image(
        config_path=project / "agentseek.json", build_include=("asset.txt",)
    )
    frozen = source.stat()
    source.write_text("evil-content", encoding="utf-8")
    os.utime(source, ns=(frozen.st_atime_ns, frozen.st_mtime_ns))

    with pytest.raises(Exception, match="hash|changed"):
        materialize_build_bundle(
            plan,
            dockerfile_bytes=b"FROM scratch\n",
            output_root=tmp_path / "bundle",
        )

    assert not (tmp_path / "bundle").exists()


def test_intermediate_directory_symlink_swap_is_rejected_without_reading_canary(
    tmp_path: Path,
) -> None:
    project = make_graph_project(tmp_path)
    selected_dir = project / "assets"
    selected_dir.mkdir()
    selected_file = selected_dir / "value.txt"
    selected_file.write_text("safe", encoding="utf-8")
    plan = plan_container_image(
        config_path=project / "agentseek.json", build_include=("assets/value.txt",)
    )
    moved = project / "assets-original"
    selected_dir.rename(moved)
    outside = tmp_path / "outside-assets"
    outside.mkdir()
    (outside / "value.txt").write_text("external-directory-canary", encoding="utf-8")
    selected_dir.symlink_to(outside, target_is_directory=True)

    with pytest.raises(Exception, match="project root|symlink|identity") as caught:
        materialize_build_bundle(
            plan,
            dockerfile_bytes=b"FROM scratch\n",
            output_root=tmp_path / "bundle",
        )

    assert "external-directory-canary" not in str(caught.value)
    assert not (tmp_path / "bundle").exists()


@pytest.mark.parametrize("vcs_name", [".git", ".hg", ".svn", ".bzr"])
def test_all_supported_vcs_metadata_is_recursively_excluded(
    tmp_path: Path, vcs_name: str
) -> None:
    project = make_graph_project(tmp_path)
    vcs = project / vcs_name
    vcs.mkdir()
    (vcs / "secret").write_text("vcs-canary", encoding="utf-8")

    plan = plan_container_image(config_path=project / "agentseek.json")
    bundle = materialize_build_bundle(
        plan,
        dockerfile_bytes=b"FROM scratch\n",
        output_root=tmp_path / "bundle",
    )

    assert b"vcs-canary" not in bundle.archive_bytes()
    with pytest.raises(Exception, match="VCS"):
        plan_container_image(
            config_path=project / "agentseek.json", build_include=(vcs_name,)
        )


def test_public_manifest_mappings_are_deep_frozen(tmp_path: Path) -> None:
    project = make_graph_project(tmp_path)
    schemes = {"key": {"type": "apiKey", "name": "x-key", "in": "header"}}
    config = project / "agentseek.json"
    config.write_text(
        json.dumps(
            {
                "graphs": {"chat": "chat.graph:graph"},
                "auth": {
                    "openapi": {
                        "securitySchemes": schemes,
                        "security": [{"key": []}],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    manifest = plan_container_image(config_path=config).manifest
    before = manifest.to_json_bytes()
    schemes["key"]["name"] = "mutated"
    with pytest.raises(TypeError):
        manifest.auth.openapi.security_schemes["key"]["name"] = "mutated"  # type: ignore[index]
    assert manifest.to_json_bytes() == before


@pytest.mark.parametrize(
    "payload",
    [
        {"store": {"ttl": {"default_ttl": float("nan")}}},
        {"store": {"ttl": {"default_ttl": float("inf")}}},
        {
            "graphs": {
                "chat": {
                    "graph": "chat.graph:graph",
                    "input_schema": {"bad": float("nan")},
                }
            }
        },
    ],
)
def test_non_finite_manifest_numbers_are_rejected(
    tmp_path: Path, payload: dict[str, object]
) -> None:
    project = make_graph_project(tmp_path)
    config = project / "agentseek.json"
    document = {"graphs": {"chat": "chat.graph:graph"}, **payload}
    config.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(Exception, match="finite"):
        plan_container_image(config_path=config)


@pytest.mark.parametrize(
    "scheme",
    [
        {"name": "x-key", "in": "header"},
        {"type": "apiKey", "in": "header"},
        {"type": "http"},
        {"type": "oauth2", "flows": {}},
        {
            "type": "oauth2",
            "flows": {"authorizationCode": {"scopes": {}}},
        },
        {"type": "openIdConnect"},
    ],
)
def test_openapi_scheme_type_specific_required_fields_fail_closed(
    tmp_path: Path, scheme: dict[str, object]
) -> None:
    project = make_graph_project(tmp_path)
    config = project / "agentseek.json"
    config.write_text(
        json.dumps(
            {
                "graphs": {"chat": "chat.graph:graph"},
                "auth": {
                    "openapi": {"securitySchemes": {"bad": scheme}, "security": []}
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(Exception, match="required|type|flow"):
        plan_container_image(config_path=config)


@pytest.mark.parametrize(
    "scheme",
    [
        {"type": "http", "scheme": "bearer", "bearerFormat": "JWT"},
        {
            "type": "openIdConnect",
            "openIdConnectUrl": "https://id.example/.well-known/openid-configuration",
        },
        {
            "type": "oauth2",
            "flows": {
                "implicit": {
                    "authorizationUrl": "https://id.example/authorize",
                    "scopes": {},
                },
                "password": {
                    "tokenUrl": "https://id.example/token",
                    "scopes": {},
                },
                "clientCredentials": {
                    "tokenUrl": "https://id.example/token",
                    "refreshUrl": "https://id.example/refresh",
                    "scopes": {},
                },
            },
        },
    ],
)
def test_openapi_scheme_type_specific_valid_fields_round_trip(
    tmp_path: Path, scheme: dict[str, object]
) -> None:
    project = make_graph_project(tmp_path)
    config = project / "agentseek.json"
    config.write_text(
        json.dumps(
            {
                "graphs": {"chat": "chat.graph:graph"},
                "auth": {
                    "openapi": {"securitySchemes": {"valid": scheme}, "security": []}
                },
            }
        ),
        encoding="utf-8",
    )

    document = plan_container_image(config_path=config).manifest.to_json_object()

    assert document["auth"]["openapi"]["securitySchemes"]["valid"] == scheme  # type: ignore[index]


@pytest.mark.parametrize(
    "url",
    [
        "https://id.example/config?token=openapi-canary",
        "https://id.example/config#access_token=openapi-canary",
    ],
)
def test_openapi_query_and_unsafe_fragment_urls_are_value_free(
    tmp_path: Path, url: str
) -> None:
    project = make_graph_project(tmp_path)
    config = project / "agentseek.json"
    config.write_text(
        json.dumps(
            {
                "graphs": {"chat": "chat.graph:graph"},
                "auth": {
                    "openapi": {
                        "securitySchemes": {
                            "bad": {
                                "type": "openIdConnect",
                                "openIdConnectUrl": url,
                            }
                        },
                        "security": [],
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(Exception, match="query|fragment") as caught:
        plan_container_image(config_path=config)
    assert "openapi-canary" not in str(caught.value)


@pytest.mark.parametrize(
    "url",
    [
        "https://packages.example/fixture.whl?token=query-canary",
        "https://packages.example/fixture.whl#access_token=fragment-canary",
    ],
)
def test_credential_query_and_unsafe_fragment_urls_are_rejected_value_free(
    tmp_path: Path, url: str
) -> None:
    project = make_graph_project(tmp_path)
    config = project / "agentseek.json"
    config.write_text(
        json.dumps({"graphs": {"chat": "chat.graph:graph"}, "dependencies": [url]}),
        encoding="utf-8",
    )
    with pytest.raises(Exception, match="query|fragment|credential") as caught:
        plan_container_image(config_path=config)
    assert "canary" not in str(caught.value)


def test_config_dotenv_is_config_parent_relative_and_recursively_excluded(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    config_dir = project / "config"
    package = project / "chat"
    config_dir.mkdir(parents=True)
    package.mkdir()
    (project / "pyproject.toml").write_text(
        '[project]\nname="fixture"\nversion="1.0"\n', encoding="utf-8"
    )
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "graph.py").write_text("graph = object()\n", encoding="utf-8")
    dotenv = config_dir / ".env"
    dotenv.write_text("TOKEN=recursive-dotenv-canary\n", encoding="utf-8")
    config = config_dir / "agentseek.json"
    config.write_text(
        json.dumps(
            {
                "graphs": {"chat": "../chat/graph.py:graph"},
                "dependencies": [".."],
                "env": ".env",
            }
        ),
        encoding="utf-8",
    )

    plan = plan_container_image(config_path=config, invocation_cwd=project)
    bundle = materialize_build_bundle(
        plan,
        dockerfile_bytes=b"FROM scratch\n",
        output_root=tmp_path / "bundle",
    )

    assert dotenv.resolve() in plan.excluded_paths
    assert b"recursive-dotenv-canary" not in bundle.archive_bytes()


@pytest.mark.parametrize(
    "source_kind",
    ["config-mapping", "config-dotenv", "cli-dotenv", "launch"],
)
def test_relative_final_auth_uses_invocation_cwd_for_non_auth_origins(
    tmp_path: Path, source_kind: str
) -> None:
    project = tmp_path / "project"
    config_dir = project / "config"
    config_dir.mkdir(parents=True)
    (project / "pyproject.toml").write_text(
        '[project]\nname="fixture"\nversion="1.0"\n', encoding="utf-8"
    )
    (project / "auth.py").write_text("auth = 'cwd'\n", encoding="utf-8")
    (config_dir / "auth.py").write_text("auth = 'config'\n", encoding="utf-8")
    config = config_dir / "agentseek.json"
    config.write_text(
        json.dumps({"graphs": {"chat": "installed.graph:graph"}}),
        encoding="utf-8",
    )
    plan = plan_container_image(config_path=config, invocation_cwd=project)

    updated, patch = plan_generated_up_auth(
        plan,
        FinalAuthSelection(
            "auth.py:auth", EnvironmentOrigin(source_kind, "fixture.env")
        ),
    )

    assert patch == AuthPayloadPatch("/deps/agent/auth.py:auth")
    assert updated.selected_sources["app/auth.py"].source_path == project / "auth.py"


def test_relative_dedicated_auth_origin_uses_config_parent(tmp_path: Path) -> None:
    project = tmp_path / "project"
    config_dir = project / "config"
    config_dir.mkdir(parents=True)
    (project / "pyproject.toml").write_text(
        '[project]\nname="fixture"\nversion="1.0"\n', encoding="utf-8"
    )
    (project / "auth.py").write_text("auth = 'cwd'\n", encoding="utf-8")
    selected = config_dir / "auth.py"
    selected.write_text("auth = 'config'\n", encoding="utf-8")
    config = config_dir / "agentseek.json"
    config.write_text(
        json.dumps({"graphs": {"chat": "installed.graph:graph"}}),
        encoding="utf-8",
    )
    plan = plan_container_image(config_path=config, invocation_cwd=project)

    updated, patch = plan_generated_up_auth(
        plan,
        FinalAuthSelection("auth.py:auth", EnvironmentOrigin("auth", str(config))),
    )

    assert patch == AuthPayloadPatch("/deps/agent/config/auth.py:auth")
    assert updated.selected_sources["app/config/auth.py"].source_path == selected


@pytest.mark.parametrize("cwd_kind", ["missing", "regular-file"])
def test_invocation_cwd_must_be_an_existing_directory(
    tmp_path: Path, cwd_kind: str
) -> None:
    project = make_graph_project(tmp_path)
    invocation_cwd = tmp_path / cwd_kind
    if cwd_kind == "regular-file":
        invocation_cwd.write_text("not a directory", encoding="utf-8")

    with pytest.raises(Exception, match="invocation cwd"):
        plan_container_image(
            config_path=project / "agentseek.json", invocation_cwd=invocation_cwd
        )


@pytest.mark.skipif(
    Path(tempfile.gettempdir()).absolute() == Path(tempfile.gettempdir()).resolve(),
    reason="platform temporary directory has no lexical/canonical alias",
)
def test_planner_normalizes_macos_temporary_directory_alias() -> None:
    with tempfile.TemporaryDirectory(prefix="agentseek-container-plan-") as directory:
        lexical_root = Path(directory)
        project = make_graph_project(lexical_root)

        plan = plan_container_image(config_path=project / "agentseek.json")

        assert plan.config_path == (project / "agentseek.json").resolve()
        assert plan.project_root == project.resolve()


def test_planner_alias_normalization_does_not_accept_project_symlink(
    tmp_path: Path,
) -> None:
    project = make_graph_project(tmp_path)
    alias = project / "config-alias.json"
    alias.symlink_to(project / "agentseek.json")

    with pytest.raises(Exception, match="project root|unsafe intermediate"):
        plan_container_image(config_path=alias)


def test_planner_rejects_missing_config(tmp_path: Path) -> None:
    with pytest.raises(Exception, match="config source is missing"):
        plan_container_image(config_path=tmp_path / "missing.json")


def test_planner_rejects_vcs_metadata_config(tmp_path: Path) -> None:
    project = make_graph_project(tmp_path)
    vcs_config = project / ".git" / "agentseek.json"
    vcs_config.parent.mkdir()
    vcs_config.write_text("{}", encoding="utf-8")
    with pytest.raises(Exception, match="excluded VCS metadata"):
        plan_container_image(config_path=vcs_config)


def test_output_writer_rejects_regular_file_ancestor(tmp_path: Path) -> None:
    ancestor = tmp_path / "not-a-directory"
    ancestor.write_text("must-survive", encoding="utf-8")

    with pytest.raises(Exception, match="output ancestor.*unsafe"):
        container_build._write_file(ancestor / "artifact", b"blocked")  # noqa: SLF001

    assert ancestor.read_text(encoding="utf-8") == "must-survive"


def test_materialization_rejects_nested_output_ancestor_swap_without_external_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = make_graph_project(tmp_path)
    output = tmp_path / "bundle"
    captured = tmp_path / "captured-app"
    outside = tmp_path / "outside"
    outside.mkdir()
    original_write = container_build._write_file  # noqa: SLF001
    calls = 0

    def swap_nested_parent_after_first_write(path: Path, data: bytes) -> None:
        nonlocal calls
        original_write(path, data)
        calls += 1
        if calls == 1:
            app = output / "context" / "app"
            app.rename(captured)
            app.symlink_to(outside, target_is_directory=True)

    monkeypatch.setattr(
        container_build, "_write_file", swap_nested_parent_after_first_write
    )

    with pytest.raises(Exception, match="output.*(ancestor|directory|symlink|changed)"):
        materialize_build_bundle(
            plan_container_image(config_path=project / "agentseek.json"),
            dockerfile_bytes=b"FROM scratch\n",
            output_root=output,
        )

    assert calls == 1
    assert not output.exists()
    assert list(outside.iterdir()) == []
    assert (captured / "chat" / "__init__.py").is_file()


def test_materialization_rejects_nested_output_ancestor_inode_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = make_graph_project(tmp_path)
    output = tmp_path / "bundle"
    captured = tmp_path / "captured-app"
    original_write = container_build._write_file  # noqa: SLF001
    calls = 0

    def swap_nested_parent_after_first_write(path: Path, data: bytes) -> None:
        nonlocal calls
        original_write(path, data)
        calls += 1
        if calls == 1:
            app = output / "context" / "app"
            app.rename(captured)
            app.mkdir(mode=0o700)
            (app / "chat").mkdir(mode=0o700)

    monkeypatch.setattr(
        container_build, "_write_file", swap_nested_parent_after_first_write
    )

    with pytest.raises(Exception, match="output.*ancestor.*identity"):
        materialize_build_bundle(
            plan_container_image(config_path=project / "agentseek.json"),
            dockerfile_bytes=b"FROM scratch\n",
            output_root=output,
        )

    assert calls == 1
    assert not output.exists()
    assert (captured / "chat" / "__init__.py").is_file()


def test_materialization_rejects_output_context_swap_without_deleting_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = make_graph_project(tmp_path)
    output = tmp_path / "bundle"
    captured = tmp_path / "captured-context"
    original_write = container_build._write_file  # noqa: SLF001
    calls = 0

    def swap_after_first_write(path: Path, data: bytes) -> None:
        nonlocal calls
        original_write(path, data)
        calls += 1
        if calls == 1:
            (output / "context").rename(captured)
            (output / "context").mkdir(mode=0o700)
            (output / "context" / "replacement-canary").write_text(
                "must-survive", encoding="utf-8"
            )

    monkeypatch.setattr(container_build, "_write_file", swap_after_first_write)

    with pytest.raises(Exception, match="identity|changed"):
        materialize_build_bundle(
            plan_container_image(config_path=project / "agentseek.json"),
            dockerfile_bytes=b"FROM scratch\n",
            output_root=output,
        )

    assert (output / "context" / "replacement-canary").read_text(
        encoding="utf-8"
    ) == "must-survive"
    assert captured.is_dir()


def test_materialization_rejects_output_root_swap_without_deleting_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = make_graph_project(tmp_path)
    output = tmp_path / "bundle"
    captured = tmp_path / "captured-root"
    original_write = container_build._write_file  # noqa: SLF001
    calls = 0

    def swap_after_first_write(path: Path, data: bytes) -> None:
        nonlocal calls
        original_write(path, data)
        calls += 1
        if calls == 1:
            output.rename(captured)
            output.mkdir(mode=0o700)
            (output / "replacement-canary").write_text("must-survive", encoding="utf-8")

    monkeypatch.setattr(container_build, "_write_file", swap_after_first_write)

    with pytest.raises(Exception, match="root identity"):
        materialize_build_bundle(
            plan_container_image(config_path=project / "agentseek.json"),
            dockerfile_bytes=b"FROM scratch\n",
            output_root=output,
        )

    assert (output / "replacement-canary").read_text(encoding="utf-8") == (
        "must-survive"
    )
    assert captured.is_dir()


def test_materialization_rejects_project_root_inode_swap(tmp_path: Path) -> None:
    project = make_graph_project(tmp_path)
    plan = plan_container_image(config_path=project / "agentseek.json")
    captured = tmp_path / "captured-project"
    project.rename(captured)
    project.mkdir()

    with pytest.raises(Exception, match="project root identity"):
        materialize_build_bundle(
            plan,
            dockerfile_bytes=b"FROM scratch\n",
            output_root=tmp_path / "bundle",
        )

    assert not (tmp_path / "bundle").exists()


def test_materialization_rejects_vanished_project_root(tmp_path: Path) -> None:
    project = make_graph_project(tmp_path)
    plan = plan_container_image(config_path=project / "agentseek.json")
    project.rename(tmp_path / "captured-project")

    with pytest.raises(Exception, match="project root changed"):
        materialize_build_bundle(
            plan,
            dockerfile_bytes=b"FROM scratch\n",
            output_root=tmp_path / "bundle",
        )

    assert not (tmp_path / "bundle").exists()


def test_materialization_fails_closed_when_private_output_creation_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agentseek_api.secure_temp import SecureArtifactError

    project = make_graph_project(tmp_path)

    def reject_output(_path: Path) -> None:
        raise SecureArtifactError("private output unavailable")

    monkeypatch.setattr(container_build, "create_private_directory", reject_output)

    with pytest.raises(Exception, match="could not be created"):
        materialize_build_bundle(
            plan_container_image(config_path=project / "agentseek.json"),
            dockerfile_bytes=b"FROM scratch\n",
            output_root=tmp_path / "bundle",
        )


def test_materialization_fails_closed_when_output_root_cannot_be_inspected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = make_graph_project(tmp_path)
    plan = plan_container_image(config_path=project / "agentseek.json")
    output = tmp_path / "bundle"
    original_lstat = Path.lstat

    def unreadable_output(path: Path) -> os.stat_result:
        if path == output:
            raise PermissionError("blocked")
        return original_lstat(path)

    monkeypatch.setattr(Path, "lstat", unreadable_output)

    with pytest.raises(Exception, match="output root could not be verified"):
        materialize_build_bundle(
            plan,
            dockerfile_bytes=b"FROM scratch\n",
            output_root=output,
        )

    assert not output.exists()


def test_materialization_fails_closed_when_os_write_makes_no_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = make_graph_project(tmp_path)
    output = tmp_path / "bundle"
    monkeypatch.setattr(container_build.os, "write", lambda _fd, _data: 0)

    with pytest.raises(Exception, match="Could not write the build bundle"):
        materialize_build_bundle(
            plan_container_image(config_path=project / "agentseek.json"),
            dockerfile_bytes=b"FROM scratch\n",
            output_root=output,
        )

    assert not output.exists()


def test_materialization_writes_every_byte_when_os_write_is_partial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = make_graph_project(tmp_path)
    original_write = os.write

    def partial_write(fd: int, data: bytes | memoryview) -> int:
        return original_write(fd, bytes(data[: max(1, len(data) // 2)]))

    monkeypatch.setattr(container_build.os, "write", partial_write)
    bundle = materialize_build_bundle(
        plan_container_image(config_path=project / "agentseek.json"),
        dockerfile_bytes=b"FROM scratch\n",
        output_root=tmp_path / "bundle",
    )

    assert bundle.dockerfile.read_bytes() == b"FROM scratch\n"
    assert bundle.manifest.read_bytes().endswith(b"\n")
    assert bundle.archive_bytes()


def test_materialization_computes_each_inventory_digest_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = make_graph_project(tmp_path)
    output = tmp_path / "bundle"
    original_read_bytes = Path.read_bytes

    def reject_output_reopen(path: Path) -> bytes:
        if output / "context" in (path, *path.parents):
            raise AssertionError("frozen output bytes must drive inventory")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", reject_output_reopen)
    bundle = materialize_build_bundle(
        plan_container_image(config_path=project / "agentseek.json"),
        dockerfile_bytes=b"FROM scratch\n",
        output_root=output,
    )

    assert bundle.inventory
