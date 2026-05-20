import json

import pytest

from agentseek_api.core.store_config import load_store_config


def test_load_store_config_reads_ttl_and_python_embedding_function(tmp_path, monkeypatch) -> None:
    helper_file = tmp_path / "embedding_helpers.py"
    helper_file.write_text(
        """
def vector_for_text(text: str) -> list[float]:
    return [1.0 if "db" in text.lower() else 0.0, 0.0]
""".strip(),
        encoding="utf-8",
    )
    embeddings_file = tmp_path / "embeddings.py"
    embeddings_file.write_text(
        """
from embedding_helpers import vector_for_text


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
