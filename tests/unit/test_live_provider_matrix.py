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
                "docker_backend": "mysql",
                "image": "mysql:8.4",
                "mode": "",
                "user": "root",
                "password": "root",
                "port": "3306",
                "db_name": "seekdb",
                "url": "mysql+aiomysql://root:root@127.0.0.1:3306/seekdb",
                "capabilities": "streaming,hitl",
                "metadata_db_url": "",
                "metadata_db_backend": "",
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
