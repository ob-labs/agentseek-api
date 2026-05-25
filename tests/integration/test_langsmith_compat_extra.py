from fastapi.testclient import TestClient


def _create_assistant(client: TestClient, *, graph_id: str = "default") -> str:
    response = client.post("/assistants", json={"name": f"{graph_id}-assistant", "graph_id": graph_id})
    assert response.status_code == 200
    return response.json()["assistant_id"]


def _create_thread(client: TestClient) -> str:
    response = client.post("/threads", json={"metadata": {"extra": True}})
    assert response.status_code == 200
    return response.json()["thread_id"]


def test_assistant_graph_schema_and_version_endpoints(client: TestClient) -> None:
    response = client.post(
        "/assistants",
        json={
            "name": "finance-bot",
            "graph_id": "react_agent",
            "description": "Answers finance questions",
        },
    )
    assert response.status_code == 200
    assistant_id = response.json()["assistant_id"]

    graph = client.get(f"/assistants/{assistant_id}/graph")
    assert graph.status_code == 200
    assert graph.json()["graph_id"] == "react_agent"

    schemas = client.get(f"/assistants/{assistant_id}/schemas")
    assert schemas.status_code == 200
    assert schemas.json()["name"] == "finance-bot"
    assert schemas.json()["description"] == "Answers finance questions"
    assert schemas.json()["input_schema"] == {"type": "object"}
    assert schemas.json()["output_schema"] == {"type": "object"}

    subgraphs = client.get(f"/assistants/{assistant_id}/subgraphs")
    assert subgraphs.status_code == 501
    assert subgraphs.json()["detail"] == "Assistant subgraph inspection is not supported"

    namespaced = client.get(f"/assistants/{assistant_id}/subgraphs/root")
    assert namespaced.status_code == 501
    assert namespaced.json()["detail"] == "Assistant subgraph inspection is not supported"

    versioned = client.post(f"/assistants/{assistant_id}/versions")
    assert versioned.status_code == 200
    assert versioned.json() == {
        "assistant_id": assistant_id,
        "current_version": 1,
        "latest_version": 1,
        "available_versions": [1],
        "supports_version_history": False,
    }

    latest = client.post(f"/assistants/{assistant_id}/latest")
    assert latest.status_code == 409
    assert latest.json()["detail"] == "Assistant version promotion is not supported"


def test_run_join_and_delete_endpoints(client: TestClient) -> None:
    assistant_id = _create_assistant(client)
    thread_id = _create_thread(client)
    created = client.post(
        f"/threads/{thread_id}/runs",
        json={"assistant_id": assistant_id, "input": {"message": "join me"}},
    )
    assert created.status_code == 200
    run_id = created.json()["run_id"]

    joined = client.get(f"/threads/{thread_id}/runs/{run_id}/join")
    assert joined.status_code == 200
    assert joined.json()["run_id"] == run_id

    deleted = client.delete(f"/threads/{thread_id}/runs/{run_id}")
    assert deleted.status_code == 204

    fetched = client.get(f"/threads/{thread_id}/runs/{run_id}")
    assert fetched.status_code == 404


def test_checkpoint_state_thread_stream_and_protocol_endpoints(client: TestClient) -> None:
    assistant_id = _create_assistant(client)
    thread_id = _create_thread(client)

    created = client.post(
        f"/threads/{thread_id}/runs",
        json={"assistant_id": assistant_id, "input": {"message": "protocol"}},
    )
    assert created.status_code == 200

    state = client.get(f"/threads/{thread_id}/state")
    assert state.status_code == 200
    checkpoint_id = state.json()["checkpoint"]["checkpoint_id"]
    assert state.json()["values"]["output"] == {"echo": {"message": "protocol"}}

    checkpoint_state = client.post(
        f"/threads/{thread_id}/state/checkpoint",
        json={"checkpoint_id": checkpoint_id},
    )
    assert checkpoint_state.status_code == 200
    assert checkpoint_state.json()["checkpoint"]["checkpoint_id"] == checkpoint_id

    updated_state = client.post(f"/threads/{thread_id}/state", json={"values": {"manual": True}})
    assert updated_state.status_code == 200
    assert updated_state.json()["values"]["manual"] is True
    manual_checkpoint_id = updated_state.json()["checkpoint"]["checkpoint_id"]

    latest_state = client.get(f"/threads/{thread_id}/state")
    assert latest_state.status_code == 200
    assert latest_state.json()["values"]["manual"] is True

    manual_checkpoint = client.post(
        f"/threads/{thread_id}/state/checkpoint",
        json={"checkpoint_id": manual_checkpoint_id},
    )
    assert manual_checkpoint.status_code == 200
    assert manual_checkpoint.json()["values"]["manual"] is True

    checkpointed = client.get(f"/threads/{thread_id}/state/{manual_checkpoint_id}")
    assert checkpointed.status_code == 200
    assert checkpointed.json()["checkpoint"]["thread_id"] == thread_id
    assert checkpointed.json()["values"]["manual"] is True

    thread_stream = client.get(f"/threads/{thread_id}/stream")
    assert thread_stream.status_code == 200
    assert thread_stream.headers["content-type"].startswith("text/event-stream")

    command = client.post(
        f"/threads/{thread_id}/commands",
        json={
            "id": 7,
            "method": "run.start",
            "params": {"assistant_id": assistant_id, "input": {"message": "from command"}},
        },
    )
    assert command.status_code == 200
    assert command.json()["type"] == "success"
    assert command.json()["id"] == 7
    assert command.json()["result"]["run_id"]

    events = client.post(f"/threads/{thread_id}/stream/events", json={"channels": ["messages"]})
    assert events.status_code == 200
    assert events.headers["content-type"].startswith("text/event-stream")


def test_empty_thread_state_returns_empty_payload(client: TestClient) -> None:
    thread_id = _create_thread(client)

    state = client.get(f"/threads/{thread_id}/state")
    assert state.status_code == 200
    assert state.json()["values"] == {}
    assert state.json()["checkpoint"]["thread_id"] == thread_id
    assert state.json()["checkpoint"]["checkpoint_id"] == thread_id
    assert state.json()["metadata"]["status"] == "idle"

    checkpoint = client.get(f"/threads/{thread_id}/state/{thread_id}")
    assert checkpoint.status_code == 200
    assert checkpoint.json()["values"] == {}

    checkpoint_post = client.post(
        f"/threads/{thread_id}/state/checkpoint",
        json={"checkpoint_id": thread_id},
    )
    assert checkpoint_post.status_code == 200
    assert checkpoint_post.json()["checkpoint"]["checkpoint_id"] == thread_id
