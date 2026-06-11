import json
import sys

import pytest

from agentseek_api.core.store_config import (
    _active_config_path,
    _apply_config_dependencies,
    _load_embedding_function,
    _looks_like_python_reference,
    load_store_config,
)


def test_load_store_config_reads_ttl_and_python_embedding_function(tmp_path, monkeypatch) -> None:
    monkeypatch.syspath_prepend(str(tmp_path))
    helper_mod_name = f"embedding_helpers_{id(tmp_path):x}"
    helper_file = tmp_path / f"{helper_mod_name}.py"
    helper_file.write_text(
        """
def vector_for_text(text: str) -> list[float]:
    return [1.0 if "db" in text.lower() else 0.0, 0.0]
""".strip(),
        encoding="utf-8",
    )
    embeddings_file = tmp_path / "embeddings.py"
    embeddings_file.write_text(
        f"""
from {helper_mod_name} import vector_for_text


def embed_texts(texts: list[str]) -> list[list[float]]:
    return [vector_for_text(text) for text in texts]
""".strip(),
        encoding="utf-8",
    )
    config_path = tmp_path / "agentseek.json"
    config_path.write_text(
        json.dumps(
            {
                "dependencies": ["."],
                "graphs": {"chat": "chat.graph:graph"},
                "store": {
                    "ttl": {
                        "refresh_on_read": False,
                        "default_ttl": 60,
                        "sweep_interval_minutes": 5,
                    },
                    "index": {
                        "embed": f"{embeddings_file}:embed_texts",
                        "dims": 2,
                        "fields": ["text", "summary"],
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    config = load_store_config(agentseek_graphs=str(config_path))

    assert config.ttl.to_runtime_config() == {
        "refresh_on_read": False,
        "default_ttl": 60.0,
        "sweep_interval_minutes": 5,
    }
    assert config.index.fields == ["text", "summary"]
    assert config.index.dims == 2
    assert callable(config.index.embed)
    assert config.index.embed(["db entry"]) == [[1.0, 0.0]]


def test_load_store_config_preserves_provider_string_embedding_reference(tmp_path) -> None:
    config_path = tmp_path / "langgraph.json"
    config_path.write_text(
        json.dumps(
            {
                "dependencies": ["."],
                "graphs": {"chat": "chat.graph:graph"},
                "store": {
                    "index": {
                        "embed": "openai:text-embedding-3-small",
                        "dims": 1536,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    config = load_store_config(agentseek_graphs=str(config_path))

    assert config.index.to_runtime_config() == {
        "embed": "openai:text-embedding-3-small",
        "dims": 1536,
    }


def test_load_store_config_raises_for_invalid_python_embedding_reference(tmp_path) -> None:
    config_path = tmp_path / "agentseek.json"
    config_path.write_text(
        json.dumps(
            {
                "dependencies": ["."],
                "graphs": {"chat": "chat.graph:graph"},
                "store": {
                    "index": {
                        "embed": "./embeddings.py:missing_embedder",
                        "dims": 2,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="store.index.embed"):
        load_store_config(agentseek_graphs=str(config_path))


def test_active_config_path_returns_none_when_no_file(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert _active_config_path(None) is None


def test_active_config_path_returns_none_for_nonexistent_explicit_path(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert _active_config_path(str(tmp_path / "missing.json")) is None


def test_active_config_path_falls_back_to_agentseek_json(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "agentseek.json").write_text("{}", encoding="utf-8")
    result = _active_config_path(None)
    assert result is not None
    assert result.name == "agentseek.json"


def test_looks_like_python_reference_no_colon() -> None:
    assert _looks_like_python_reference("no_colon_here") is False


def test_looks_like_python_reference_empty_parts() -> None:
    assert _looks_like_python_reference(":symbol") is False
    assert _looks_like_python_reference("module:") is False


def test_looks_like_python_reference_valid() -> None:
    assert _looks_like_python_reference("my_module:embed_fn") is True


def test_looks_like_python_reference_non_identifier_symbol() -> None:
    assert _looks_like_python_reference("openai:text-embedding-3-small") is False


def test_load_embedding_function_invalid_reference(tmp_path) -> None:
    config_path = tmp_path / "agentseek.json"
    config_path.write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Invalid store.index.embed reference"):
        _load_embedding_function(":bad", config_path=config_path)


def test_load_embedding_function_module_import_error(tmp_path) -> None:
    config_path = tmp_path / "agentseek.json"
    config_path.write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Could not import"):
        _load_embedding_function("nonexistent_module_xyz:fn", config_path=config_path)


def test_load_embedding_function_not_callable(tmp_path, monkeypatch) -> None:
    monkeypatch.syspath_prepend(str(tmp_path))
    mod = tmp_path / "embed_mod_notcallable.py"
    mod.write_text("my_var = 42\n", encoding="utf-8")
    config_path = tmp_path / "agentseek.json"
    config_path.write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Could not resolve callable"):
        _load_embedding_function("embed_mod_notcallable:my_var", config_path=config_path)


def test_apply_config_dependencies_non_list(tmp_path) -> None:
    config_path = tmp_path / "agentseek.json"
    config_path.write_text("{}", encoding="utf-8")
    _apply_config_dependencies({"dependencies": "not-a-list"}, config_path=config_path)


def test_apply_config_dependencies_skips_non_string(tmp_path) -> None:
    config_path = tmp_path / "agentseek.json"
    config_path.write_text("{}", encoding="utf-8")
    _apply_config_dependencies({"dependencies": [123, None]}, config_path=config_path)


def test_apply_config_dependencies_adds_relative_path(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "agentseek.json"
    config_path.write_text("{}", encoding="utf-8")
    subdir = tmp_path / "libs"
    subdir.mkdir()
    original_path = list(sys.path)
    try:
        _apply_config_dependencies({"dependencies": ["libs"]}, config_path=config_path)
        assert str(subdir.resolve()) in sys.path
    finally:
        sys.path[:] = original_path


def test_load_store_config_returns_default_for_no_store_key(tmp_path) -> None:
    config_path = tmp_path / "agentseek.json"
    config_path.write_text(json.dumps({"graphs": {}}), encoding="utf-8")
    config = load_store_config(agentseek_graphs=str(config_path))
    assert config.ttl.to_runtime_config() is None
    assert config.index.to_runtime_config() is None


def test_load_store_config_returns_default_for_invalid_json(tmp_path) -> None:
    config_path = tmp_path / "agentseek.json"
    config_path.write_text("NOT VALID JSON", encoding="utf-8")
    config = load_store_config(agentseek_graphs=str(config_path))
    assert config.ttl.to_runtime_config() is None


def test_load_store_config_returns_default_when_no_config_found(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    config = load_store_config(agentseek_graphs=None)
    assert config.ttl.to_runtime_config() is None
    assert config.index.to_runtime_config() is None
