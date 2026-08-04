"""langgraph-api parity: assistant config/context mirroring on create/patch.

Matches langgraph-api ops-layer ``consolidate_config_and_context``:
- both ``config.configurable`` and ``context`` non-empty -> 400
- only ``config.configurable`` -> ``context`` mirrors it
- only ``context`` -> ``config.configurable`` mirrors it
"""

from fastapi.testclient import TestClient


def test_create_mirrors_context_to_config(client: TestClient) -> None:
    created = client.post(
        "/assistants",
        json={"name": "ctx-sync", "graph_id": "default", "context": {"tenant": "acme"}},
    )
    assert created.status_code == 200
    body = created.json()
    assert body["config"]["configurable"] == {"tenant": "acme"}
    assert body["context"] == {"tenant": "acme"}


def test_create_mirrors_configurable_to_context(client: TestClient) -> None:
    created = client.post(
        "/assistants",
        json={"name": "cfg-sync", "graph_id": "default", "config": {"configurable": {"temperature": 0}}},
    )
    assert created.status_code == 200
    body = created.json()
    assert body["config"]["configurable"] == {"temperature": 0}
    assert body["context"] == {"temperature": 0}


def test_create_rejects_both_configurable_and_context(client: TestClient) -> None:
    created = client.post(
        "/assistants",
        json={
            "name": "both",
            "graph_id": "default",
            "config": {"configurable": {"temperature": 0}},
            "context": {"tenant": "acme"},
        },
    )
    assert created.status_code == 400
    assert "Cannot specify both configurable and context" in created.json()["detail"]


def test_create_empty_params_stays_empty(client: TestClient) -> None:
    created = client.post(
        "/assistants",
        json={"name": "empty", "graph_id": "default"},
    )
    assert created.status_code == 200
    body = created.json()
    assert body["config"]["configurable"] == {}
    assert body["context"] == {}


def test_patch_mirrors_context_to_config(client: TestClient) -> None:
    created = client.post("/assistants", json={"name": "p1", "graph_id": "default"})
    assert created.status_code == 200
    assistant_id = created.json()["assistant_id"]

    patched = client.patch(
        f"/assistants/{assistant_id}",
        json={"context": {"tenant": "acme"}},
    )
    assert patched.status_code == 200
    body = patched.json()
    assert body["config"]["configurable"] == {"tenant": "acme"}
    assert body["context"] == {"tenant": "acme"}


def test_patch_mirrors_configurable_to_context(client: TestClient) -> None:
    created = client.post("/assistants", json={"name": "p2", "graph_id": "default"})
    assert created.status_code == 200
    assistant_id = created.json()["assistant_id"]

    patched = client.patch(
        f"/assistants/{assistant_id}",
        json={"config": {"configurable": {"temperature": 0}}},
    )
    assert patched.status_code == 200
    body = patched.json()
    assert body["config"]["configurable"] == {"temperature": 0}
    assert body["context"] == {"temperature": 0}


def test_patch_rejects_both_configurable_and_context(client: TestClient) -> None:
    created = client.post("/assistants", json={"name": "p3", "graph_id": "default"})
    assert created.status_code == 200
    assistant_id = created.json()["assistant_id"]

    patched = client.patch(
        f"/assistants/{assistant_id}",
        json={
            "config": {"configurable": {"temperature": 0}},
            "context": {"tenant": "acme"},
        },
    )
    assert patched.status_code == 400
    assert "Cannot specify both configurable and context" in patched.json()["detail"]
