from fastapi.testclient import TestClient


def test_assistants_search_and_count(client: TestClient) -> None:
    first = client.post("/assistants", json={"name": "first", "graph_id": "default"})
    second = client.post("/assistants", json={"name": "second", "graph_id": "react_agent"})
    assert first.status_code == 200
    assert second.status_code == 200

    search = client.post("/assistants/search", json={"graph_id": "default"})
    assert search.status_code == 200
    body = search.json()
    assert len(body) == 1
    assert body[0]["name"] == "first"

    count = client.post("/assistants/count", json={"graph_id": "default"})
    assert count.status_code == 200
    assert count.json() == 1


def test_create_assistant_persists_langsmith_fields(client: TestClient) -> None:
    created = client.post(
        "/assistants",
        json={
            "name": "rich",
            "graph_id": "default",
            "metadata": {"suite": "compat"},
            "config": {"configurable": {"temperature": 0}},
            "context": {"tenant": "mysql-family"},
            "description": "assistant description",
        },
    )
    assert created.status_code == 200
    assert created.json()["metadata"] == {"suite": "compat"}
    assert created.json()["config"] == {"tags": [], "configurable": {"temperature": 0}}
    assert created.json()["context"] == {"tenant": "mysql-family"}
    assert created.json()["description"] == "assistant description"


def test_patch_and_delete_assistant(client: TestClient) -> None:
    created = client.post(
        "/assistants",
        json={"name": "before", "graph_id": "default", "metadata": {"existing": True}},
    )
    assert created.status_code == 200
    assistant_id = created.json()["assistant_id"]

    patched = client.patch(
        f"/assistants/{assistant_id}",
        json={"name": "after", "graph_id": "react_agent", "metadata": {"team": "api"}},
    )
    assert patched.status_code == 200
    assert patched.json()["name"] == "after"
    assert patched.json()["graph_id"] == "react_agent"
    assert patched.json()["metadata"] == {"existing": True, "team": "api"}

    deleted = client.delete(f"/assistants/{assistant_id}")
    assert deleted.status_code == 204

    fetched = client.get(f"/assistants/{assistant_id}")
    assert fetched.status_code == 404
