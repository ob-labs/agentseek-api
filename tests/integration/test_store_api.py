from datetime import UTC, datetime, timedelta

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from agentseek_api.api import store as store_module
from agentseek_api.models.api import StoreDeleteRequest, StoreListNamespacesRequest, StorePutRequest, StoreSearchRequest
from agentseek_api.models.auth import User
from agentseek_api.settings import settings


def test_store_put_get_update_delete_and_user_isolation(client: TestClient) -> None:
    put = client.put(
        "/store/items",
        json={"namespace": ["memories", "users"], "key": "profile", "value": {"name": "Ada"}},
        headers={"x-user-id": "u1"},
    )
    assert put.status_code == 200
    assert put.json()["value"] == {"name": "Ada"}

    updated = client.put(
        "/store/items",
        json={"namespace": ["memories", "users"], "key": "profile", "value": {"name": "Ada", "level": 2}},
        headers={"x-user-id": "u1"},
    )
    assert updated.status_code == 200
    assert updated.json()["created_at"] == put.json()["created_at"]
    assert updated.json()["updated_at"] != put.json()["updated_at"]

    fetched = client.get(
        "/store/items",
        params=[("namespace", "memories"), ("namespace", "users"), ("key", "profile")],
        headers={"x-user-id": "u1"},
    )
    assert fetched.status_code == 200
    assert fetched.json()["value"] == {"name": "Ada", "level": 2}

    hidden = client.get(
        "/store/items",
        params=[("namespace", "memories"), ("namespace", "users"), ("key", "profile")],
        headers={"x-user-id": "u2"},
    )
    assert hidden.status_code == 404

    deleted = client.request(
        "DELETE",
        "/store/items",
        json={"namespace": ["memories", "users"], "key": "profile"},
        headers={"x-user-id": "u1"},
    )
    assert deleted.status_code == 204
    assert client.get(
        "/store/items",
        params=[("namespace", "memories"), ("namespace", "users"), ("key", "profile")],
        headers={"x-user-id": "u1"},
    ).status_code == 404


def test_store_search_namespaces_and_info_flag(client: TestClient) -> None:
    items = [
        (["memories", "users"], "profile", {"kind": "person", "name": "Ada"}),
        (["memories", "projects"], "agentseek", {"kind": "project", "name": "AgentSeek"}),
        (["scratch"], "note", {"kind": "note", "name": "temp"}),
    ]
    for namespace, key, value in items:
        response = client.put(
            "/store/items",
            json={"namespace": namespace, "key": key, "value": value},
            headers={"x-user-id": "u1"},
        )
        assert response.status_code == 200

    search = client.post(
        "/store/items/search",
        json={"namespace_prefix": ["memories"], "filter": {"kind": "project"}, "limit": 10, "offset": 0},
        headers={"x-user-id": "u1"},
    )
    assert search.status_code == 200
    assert [item["key"] for item in search.json()["items"]] == ["agentseek"]

    paged = client.post(
        "/store/items/search",
        json={"namespace_prefix": ["memories"], "limit": 1, "offset": 1},
        headers={"x-user-id": "u1"},
    )
    assert paged.status_code == 200
    assert len(paged.json()["items"]) == 1

    namespaces = client.post(
        "/store/namespaces",
        json={"prefix": ["memories"], "max_depth": 2, "limit": 10, "offset": 0},
        headers={"x-user-id": "u1"},
    )
    assert namespaces.status_code == 200
    assert namespaces.json() == [["memories", "projects"], ["memories", "users"]]

    assert client.post(
        "/store/items/search",
        json={"namespace_prefix": ["memories"], "limit": 10, "offset": 0},
        headers={"x-user-id": "u2"},
    ).json() == {"items": []}

    info = client.get("/info")
    assert info.status_code == 200
    assert info.json()["flags"]["store"] is True


def test_store_ttl_config_from_agentseek_json_expires_and_refreshes_items(
    client: TestClient,
    monkeypatch,
    tmp_path,
) -> None:
    from agentseek_api.api import store as store_module

    config_path = tmp_path / "agentseek.json"
    config_path.write_text(
        """
{
  "$schema": "https://langgra.ph/schema.json",
  "dependencies": ["."],
  "graphs": {
    "memory_agent": "./agent/graph.py:graph"
  },
  "store": {
    "ttl": {
      "refresh_on_read": true,
      "sweep_interval_minutes": 1,
      "default_ttl": 10
    }
  }
}
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setattr(settings, "AGENTSEEK_GRAPHS", str(config_path))
    now = datetime(2026, 5, 18, 12, 0, tzinfo=UTC)
    monkeypatch.setattr(store_module, "_utc_now", lambda: now)

    put = client.put(
        "/store/items",
        json={"namespace": ["memories"], "key": "ttl", "value": {"kind": "note"}},
        headers={"x-user-id": "u1"},
    )
    assert put.status_code == 200

    later = now + timedelta(minutes=5)
    monkeypatch.setattr(store_module, "_utc_now", lambda: later)
    read = client.get(
        "/store/items",
        params=[("namespace", "memories"), ("key", "ttl")],
        headers={"x-user-id": "u1"},
    )
    assert read.status_code == 200

    refreshed = later + timedelta(minutes=9)
    monkeypatch.setattr(store_module, "_utc_now", lambda: refreshed)
    still_present = client.post(
        "/store/items/search",
        json={"namespace_prefix": ["memories"], "limit": 10, "offset": 0},
        headers={"x-user-id": "u1"},
    )
    assert [item["key"] for item in still_present.json()["items"]] == ["ttl"]

    expired = refreshed + timedelta(minutes=11)
    monkeypatch.setattr(store_module, "_utc_now", lambda: expired)
    missing = client.get(
        "/store/items",
        params=[("namespace", "memories"), ("key", "ttl")],
        headers={"x-user-id": "u1"},
    )
    assert missing.status_code == 404


def test_store_semantic_search_uses_custom_embedding_function(client: TestClient, monkeypatch, tmp_path) -> None:
    helper_file = tmp_path / "embedding_helpers.py"
    helper_file.write_text(
        """
def vector_for_text(text: str) -> list[float]:
    lower = text.lower()
    return [
        1.0 if "oceanbase" in lower or "database" in lower else 0.0,
        1.0 if "frontend" in lower or "ui" in lower else 0.0,
    ]
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
    config_path = tmp_path / "langgraph.json"
    config_path.write_text(
        f"""
{{
  "dependencies": ["."],
  "graphs": {{
    "memory_agent": "./agent/graph.py:graph"
  }},
  "store": {{
    "index": {{
      "embed": "{embeddings_file}:embed_texts",
      "dims": 2,
      "fields": ["text", "summary"]
    }}
  }}
}}
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setattr(settings, "AGENTSEEK_GRAPHS", str(config_path))

    for key, value in {
        "db": {"text": "OceanBase checkpoint database memory", "summary": "persistent database"},
        "ui": {"text": "Frontend UI polish", "summary": "layout work"},
    }.items():
        response = client.put(
            "/store/items",
            json={"namespace": ["memories"], "key": key, "value": value},
            headers={"x-user-id": "u1"},
        )
        assert response.status_code == 200

    search = client.post(
        "/store/items/search",
        json={"namespace_prefix": ["memories"], "query": "database memory", "limit": 10, "offset": 0},
        headers={"x-user-id": "u1"},
    )

    assert search.status_code == 200
    assert [item["key"] for item in search.json()["items"]] == ["db", "ui"]


def test_store_provider_embedding_config_fails_deterministically(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    config_path = tmp_path / "agentseek.json"
    config_path.write_text(
        """
{
  "dependencies": ["."],
  "graphs": {"memory_agent": "./agent/graph.py:graph"},
  "store": {
    "index": {
      "embed": "openai:text-embedding-3-small",
      "dims": 1536
    }
  }
}
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setattr(settings, "AGENTSEEK_GRAPHS", str(config_path))

    response = client.put(
        "/store/items",
        json={"namespace": ["memories"], "key": "provider", "value": {"text": "needs embedding"}},
        headers={"x-user-id": "u1"},
    )

    assert response.status_code == 422
    assert "provider strings are not supported" in response.json()["detail"]


def test_store_namespace_max_depth_truncates_results(client: TestClient) -> None:
    response = client.put(
        "/store/items",
        json={"namespace": ["a", "b", "c", "d"], "key": "deep", "value": {"kind": "note"}},
        headers={"x-user-id": "u1"},
    )
    assert response.status_code == 200

    namespaces = client.post(
        "/store/namespaces",
        json={"prefix": ["a"], "max_depth": 3, "limit": 10, "offset": 0},
        headers={"x-user-id": "u1"},
    )

    assert namespaces.status_code == 200
    assert namespaces.json() == [["a", "b", "c"]]


@pytest.mark.asyncio
async def test_store_route_functions_cover_database_mutations(client: TestClient) -> None:
    _ = client
    user = User(identity="direct-user", is_authenticated=True)
    namespace = ["direct", "store"]

    created = await store_module.put_item(
        StorePutRequest(namespace=namespace, key="profile", value={"kind": "profile", "level": 1}),
        user=user,
    )
    assert created.value == {"kind": "profile", "level": 1}

    updated = await store_module.put_item(
        StorePutRequest(namespace=namespace, key="profile", value={"kind": "profile", "level": 2}),
        user=user,
    )
    assert updated.created_at == created.created_at
    assert updated.value == {"kind": "profile", "level": 2}

    fetched = await store_module.get_item(namespace=namespace, key="profile", user=user)
    assert fetched.value == {"kind": "profile", "level": 2}

    searched = await store_module.search_items(
        StoreSearchRequest(namespace_prefix=["direct"], filter={"kind": "profile"}, limit=10, offset=0),
        user=user,
    )
    assert [item.key for item in searched.items] == ["profile"]

    deep_namespace = ["direct", "store", "child"]
    await store_module.put_item(
        StorePutRequest(namespace=deep_namespace, key="nested", value={"kind": "profile", "level": 3}),
        user=user,
    )

    namespaces = await store_module.list_namespaces(
        StoreListNamespacesRequest(prefix=["direct"], suffix=["store"], max_depth=2),
        user=user,
    )
    assert namespaces == [namespace]

    truncated = await store_module.list_namespaces(
        StoreListNamespacesRequest(prefix=["direct"], max_depth=2),
        user=user,
    )
    assert truncated == [namespace]

    response = await store_module.delete_item(StoreDeleteRequest(namespace=namespace, key="profile"), user=user)
    assert response.status_code == 204

    with pytest.raises(HTTPException) as exc_info:
        await store_module.get_item(namespace=namespace, key="profile", user=user)
    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_store_route_functions_cover_expiry_and_config_edges(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    _ = client
    user = User(identity="expiry-user", is_authenticated=True)
    config_path = tmp_path / "agentseek.json"
    config_path.write_text(
        """
{
  "dependencies": [".", 123, "./missing"],
  "graphs": {"memory_agent": "./agent/graph.py:graph"},
  "store": {
    "ttl": {
      "refresh_on_read": false,
      "sweep_interval_minutes": 1,
      "default_ttl": 1
    }
  }
}
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setattr(settings, "AGENTSEEK_GRAPHS", str(config_path))
    monkeypatch.setattr(store_module, "_last_sweep_at", None)
    now = datetime(2026, 5, 18, 12, 0, tzinfo=UTC)
    monkeypatch.setattr(store_module, "_utc_now", lambda: now)

    created = await store_module.put_item(
        StorePutRequest(namespace=["expiry"], key="note", value={"text": "short lived"}),
        user=user,
    )
    assert created.key == "note"

    monkeypatch.setattr(store_module, "_utc_now", lambda: now + timedelta(minutes=2))
    with pytest.raises(HTTPException) as exc_info:
        await store_module.get_item(namespace=["expiry"], key="note", user=user)
    assert exc_info.value.status_code == 404

    with pytest.raises(HTTPException) as invalid_namespace:
        store_module._namespace_path(["valid", ""])
    assert invalid_namespace.value.status_code == 422

    assert store_module._load_embedding_function("missing.module:embed", config_path=config_path) is None
    assert store_module._load_embedding_function("not-a-reference", config_path=config_path) is None
    assert store_module._load_embedding_function(":missing", config_path=config_path) is None
    assert store_module._extract_field_text(
        {
            "metadata": {"title": "Nested"},
            "chapters": [{"content": "first"}, {"content": "second"}],
            "authors": [{"name": "Ada"}],
        },
        ["metadata.title", "chapters[*].content", "authors[0].name"],
    ) == "Nested\nfirst\nsecond\nAda"
