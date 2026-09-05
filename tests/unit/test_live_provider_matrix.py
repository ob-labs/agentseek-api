import importlib.util
from pathlib import Path

import pytest


def _load_module():
    module_path = Path(__file__).resolve().parents[2] / "scripts" / "build_live_provider_matrix.py"
    spec = importlib.util.spec_from_file_location("build_live_provider_matrix", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_schedule_builds_full_matrix() -> None:
    module = _load_module()

    matrix = module.build_matrix(
        event_name="schedule",
        run_openai_compatible=False,
        run_anthropic_compatible=False,
        backend_tier="mysql",
    )

    assert len(matrix["include"]) == 8
    assert {lane["provider_kind"] for lane in matrix["include"]} == {"openai", "anthropic"}
    assert {lane["backend_name"] for lane in matrix["include"]} == {"seekdb", "oceanbase", "mysql", "postgresql-metadata"}


def test_workflow_dispatch_filters_provider_and_backend() -> None:
    module = _load_module()

    matrix = module.build_matrix(
        event_name="workflow_dispatch",
        run_openai_compatible=True,
        run_anthropic_compatible=False,
        backend_tier="mysql",
    )

    assert matrix == {
        "include": [
            {
                "provider_kind": "openai",
                "provider_label": "OpenAI-Compatible",
                "backend_name": "mysql",
            }
        ]
    }


def test_workflow_dispatch_requires_at_least_one_provider() -> None:
    module = _load_module()

    with pytest.raises(ValueError, match="At least one provider"):
        module.build_matrix(
            event_name="workflow_dispatch",
            run_openai_compatible=False,
            run_anthropic_compatible=False,
            backend_tier="all",
        )


@pytest.mark.parametrize(
    ("backend_name", "expected"),
    [
        ("seekdb", "streaming,store,mcp,hitl"),
        ("oceanbase", "streaming,store,mcp,hitl"),
        ("mysql", "streaming,hitl"),
        ("postgresql-metadata", "streaming,mcp"),
    ],
)
def test_backend_capabilities_match_tier_contract(backend_name: str, expected: str) -> None:
    module = _load_module()

    assert module.capability_set_for_backend(backend_name) == expected


@pytest.mark.parametrize(
    ("backend_name", "embedded_available", "expected"),
    [
        ("seekdb", True, {"SEEKDB_MODE": "embed", "SEEKDB_DOCKER_BACKEND": "seekdb"}),
        ("seekdb", False, {"SEEKDB_MODE": "docker", "SEEKDB_DOCKER_BACKEND": "seekdb"}),
        ("oceanbase", True, {"SEEKDB_MODE": "docker", "SEEKDB_DOCKER_BACKEND": "oceanbase"}),
        ("mysql", True, {"SEEKDB_MODE": "docker", "SEEKDB_DOCKER_BACKEND": "mysql"}),
        ("postgresql-metadata", True, {"SEEKDB_MODE": "docker", "SEEKDB_DOCKER_BACKEND": "seekdb"}),
    ],
)
def test_local_backend_launcher_matches_selected_tier(
    backend_name: str,
    embedded_available: bool,
    expected: dict[str, str],
) -> None:
    module = _load_module()

    assert module.local_backend_env_for_tier(backend_name, embedded_available=embedded_available) == expected
