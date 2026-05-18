from fastapi.testclient import TestClient


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
