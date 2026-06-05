from fastapi.testclient import TestClient


def _create_thread(
    client: TestClient,
    *,
    user_id: str,
    metadata: dict[str, object],
) -> str:
    payload: dict[str, object] = {"metadata": metadata}
    response = client.post("/threads", json=payload, headers={"x-user-id": user_id})
    assert response.status_code == 200
    return response.json()["thread_id"]


def test_threads_search_count_patch_copy_and_prune(client: TestClient) -> None:
    assistant = client.post("/assistants", json={"name": "copy-source", "graph_id": "default"})
    assert assistant.status_code == 200
    assistant_id = assistant.json()["assistant_id"]

    thread_id = _create_thread(
        client,
        user_id="u1",
        metadata={"topic": "alpha", "tag": "keep"},
    )
    _create_thread(client, user_id="u1", metadata={"topic": "beta"})
    _create_thread(client, user_id="u2", metadata={"topic": "alpha"})

    run = client.post(
        f"/threads/{thread_id}/runs",
        json={"assistant_id": assistant_id, "input": {"message": "copy history"}},
        headers={"x-user-id": "u1"},
    )
    assert run.status_code == 200

    search = client.post("/threads/search", json={"metadata": {"topic": "alpha"}}, headers={"x-user-id": "u1"})
    assert search.status_code == 200
    search_body = search.json()
    assert len(search_body) == 1
    assert search_body[0]["thread_id"] == thread_id

    count = client.post("/threads/count", json={"metadata": {"topic": "alpha"}}, headers={"x-user-id": "u1"})
    assert count.status_code == 200
    assert count.json() == 1

    patched = client.patch(f"/threads/{thread_id}", json={"metadata": {"tag": "patched"}}, headers={"x-user-id": "u1"})
    assert patched.status_code == 200
    assert patched.json()["metadata"] == {"topic": "alpha", "tag": "patched", "graph_id": "default"}
    assert patched.json()["config"] == {}

    unsupported_patch = client.patch(
        f"/threads/{thread_id}",
        json={"config": {"retention": "long"}},
        headers={"x-user-id": "u1"},
    )
    assert unsupported_patch.status_code == 422

    copied = client.post(f"/threads/{thread_id}/copy", headers={"x-user-id": "u1"})
    assert copied.status_code == 200
    assert copied.json()["thread_id"] != thread_id
    assert copied.json()["metadata"]["topic"] == "alpha"

    copied_history = client.get(f"/threads/{copied.json()['thread_id']}/history", headers={"x-user-id": "u1"})
    assert copied_history.status_code == 200
    assert len(copied_history.json()) >= 1

    pruned = client.post("/threads/prune", json={"thread_ids": [copied.json()["thread_id"]], "strategy": "delete"}, headers={"x-user-id": "u1"})
    assert pruned.status_code == 200
    assert pruned.json()["pruned_count"] == 1

    copied_get = client.get(f"/threads/{copied.json()['thread_id']}", headers={"x-user-id": "u1"})
    assert copied_get.status_code == 404


def test_thread_state_and_history_endpoints(client: TestClient) -> None:
    assistant = client.post("/assistants", json={"name": "stateful", "graph_id": "default"})
    assert assistant.status_code == 200
    assistant_id = assistant.json()["assistant_id"]

    thread_id = _create_thread(client, user_id="u1", metadata={"state": True})
    run = client.post(
        f"/threads/{thread_id}/runs",
        json={"assistant_id": assistant_id, "input": {"message": "hello"}},
        headers={"x-user-id": "u1"},
    )
    assert run.status_code == 200

    state = client.get(f"/threads/{thread_id}/state", headers={"x-user-id": "u1"})
    assert state.status_code == 200
    state_body = state.json()
    assert "values" in state_body
    assert "checkpoint" in state_body
    assert state_body["checkpoint"]["thread_id"] == thread_id
    assert state_body["values"]["input"] == {"message": "hello"}
    assert state_body["values"]["output"] == {"echo": {"message": "hello"}}

    history = client.get(f"/threads/{thread_id}/history", headers={"x-user-id": "u1"})
    assert history.status_code == 200
    history_body = history.json()
    assert len(history_body) >= 1
    assert history_body[0]["checkpoint"]["thread_id"] == thread_id


def test_threads_prune_keep_latest_removes_older_runs_but_keeps_thread(client: TestClient) -> None:
    assistant = client.post("/assistants", json={"name": "keep-latest", "graph_id": "default"})
    assert assistant.status_code == 200
    assistant_id = assistant.json()["assistant_id"]

    thread_id = _create_thread(client, user_id="u1", metadata={"topic": "prune"})

    first = client.post(
        f"/threads/{thread_id}/runs",
        json={"assistant_id": assistant_id, "input": {"message": "first"}},
        headers={"x-user-id": "u1"},
    )
    second = client.post(
        f"/threads/{thread_id}/runs",
        json={"assistant_id": assistant_id, "input": {"message": "second"}},
        headers={"x-user-id": "u1"},
    )
    assert first.status_code == 200
    assert second.status_code == 200

    pruned = client.post(
        "/threads/prune",
        json={"thread_ids": [thread_id], "strategy": "keep_latest"},
        headers={"x-user-id": "u1"},
    )
    assert pruned.status_code == 200
    assert pruned.json()["pruned_count"] == 1

    history = client.get(f"/threads/{thread_id}/history", headers={"x-user-id": "u1"})
    assert history.status_code == 200
    assert len(history.json()) == 1

    fetched = client.get(f"/threads/{thread_id}", headers={"x-user-id": "u1"})
    assert fetched.status_code == 200
