import importlib.util
import json
from pathlib import Path


EXAMPLES_ROOT = Path(__file__).resolve().parents[2] / "examples"


def test_minimal_agentseek_json_example_is_valid() -> None:
    config = json.loads((EXAMPLES_ROOT / "minimal_agentseek" / "agentseek.json").read_text(encoding="utf-8"))

    assert config["graphs"] == {"chat": "./graph.py:graph"}


def test_auth_examples_import_backends_and_app() -> None:
    custom_spec = importlib.util.spec_from_file_location(
        "custom_auth_example",
        EXAMPLES_ROOT / "auth" / "custom_backend.py",
    )
    assert custom_spec is not None
    assert custom_spec.loader is not None
    custom_module = importlib.util.module_from_spec(custom_spec)
    custom_spec.loader.exec_module(custom_module)
    assert hasattr(custom_module, "backend")

    app_spec = importlib.util.spec_from_file_location(
        "custom_routes_example",
        EXAMPLES_ROOT / "custom_routes" / "app.py",
    )
    assert app_spec is not None
    assert app_spec.loader is not None
    app_module = importlib.util.module_from_spec(app_spec)
    app_spec.loader.exec_module(app_module)
    assert hasattr(app_module, "app")


def test_assistant_config_example_documents_config_context_and_metadata() -> None:
    config = json.loads((EXAMPLES_ROOT / "assistant_config" / "agentseek.json").read_text(encoding="utf-8"))

    assert config["graphs"] == {"assistant_config": "./graph.py:graph"}
    assert config["env"]["ASSISTANT_EXAMPLE_MODE"] == "config"
