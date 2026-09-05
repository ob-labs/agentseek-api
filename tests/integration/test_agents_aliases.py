from fastapi.testclient import TestClient


def test_agents_alias_crud_matches_assistants_resource(client: TestClient) -> None:
    created = client.post("/agents", json={"name": "agent-1", "graph_id": "default"})
    assert created.status_code == 200
    body = created.json()
    assert body["name"] == "agent-1"
    assert body["graph_id"] == "default"

    assistant_id = body["assistant_id"]

    listed = client.post("/agents/search", json={})
    assert listed.status_code == 200
    assert any(item["assistant_id"] == assistant_id for item in listed.json())

    fetched_via_assistants = client.get(f"/assistants/{assistant_id}")
    assert fetched_via_assistants.status_code == 200
    assert fetched_via_assistants.json()["assistant_id"] == assistant_id

    patched = client.patch(
        f"/agents/{assistant_id}",
        json={"name": "agent-1b", "graph_id": "react_agent"},
    )
    assert patched.status_code == 200
    assert patched.json()["name"] == "agent-1b"
    assert patched.json()["graph_id"] == "react_agent"

    search = client.post("/agents/search", json={"graph_id": "react_agent"})
    assert search.status_code == 200
    assert [item["assistant_id"] for item in search.json()] == [assistant_id]

    count = client.post("/agents/count", json={"graph_id": "react_agent"})
    assert count.status_code == 200
    assert count.json() == 1

    deleted = client.delete(f"/agents/{assistant_id}")
    assert deleted.status_code == 204
    assert client.get(f"/assistants/{assistant_id}").status_code == 404


def test_agents_alias_exposes_graph_schema_and_version_helpers(client: TestClient) -> None:
    created = client.post(
        "/assistants",
        json={
            "name": "agent-helpers",
            "graph_id": "default",
            "description": "Custom helper description",
        },
    )
    assert created.status_code == 200
    assistant_id = created.json()["assistant_id"]

    graph = client.get(f"/agents/{assistant_id}/graph")
    assert graph.status_code == 200
    assert "nodes" in graph.json()
    assert "edges" in graph.json()

    schemas = client.get(f"/agents/{assistant_id}/schemas")
    assert schemas.status_code == 200
    assert schemas.json()["graph_id"] == "default"
    assert schemas.json()["input_schema"]["type"] == "object"

    latest = client.post(f"/agents/{assistant_id}/latest?version=1")
    assert latest.status_code == 200
    assert latest.json()["assistant_id"] == assistant_id
    assert latest.json()["version"] == 1
