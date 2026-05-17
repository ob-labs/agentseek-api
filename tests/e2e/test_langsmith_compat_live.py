import json
from uuid import uuid4

import httpx
import pytest


def _user_headers(user_id: str) -> dict[str, str]:
    return {"x-user-id": user_id}


async def _create_assistant(
    client: httpx.AsyncClient,
    *,
    name: str,
    graph_id: str = "default",
    metadata: dict[str, object] | None = None,
    config: dict[str, object] | None = None,
    context: dict[str, object] | None = None,
    description: str | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {"name": name, "graph_id": graph_id}
    if metadata is not None:
        payload["metadata"] = metadata
    if config is not None:
        payload["config"] = config
    if context is not None:
        payload["context"] = context
    if description is not None:
        payload["description"] = description
    response = await client.post("/assistants", json=payload)
    assert response.status_code == 200
    return response.json()


async def _create_thread(
    client: httpx.AsyncClient,
    *,
    user_id: str,
    metadata: dict[str, object],
    config: dict[str, object] | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {"metadata": metadata}
    if config is not None:
        payload["config"] = config
    response = await client.post("/threads", json=payload, headers=_user_headers(user_id))
    assert response.status_code == 200
    return response.json()


async def _create_thread_run(
    client: httpx.AsyncClient,
    *,
    thread_id: str,
    assistant_id: str,
    payload: object,
    user_id: str,
) -> dict[str, object]:
    response = await client.post(
        f"/threads/{thread_id}/runs",
        json={"assistant_id": assistant_id, "input": payload},
        headers=_user_headers(user_id),
    )
    assert response.status_code == 200
    return response.json()


async def _wait_for_run(
    client: httpx.AsyncClient,
    *,
    thread_id: str,
    run_id: str,
    user_id: str,
) -> dict[str, object]:
    response = await client.get(
        f"/threads/{thread_id}/runs/{run_id}/wait",
        headers=_user_headers(user_id),
    )
    assert response.status_code == 200
    return response.json()


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_live_system_and_assistant_endpoints(e2e_base_url: str) -> None:
    async with httpx.AsyncClient(base_url=e2e_base_url, timeout=30.0, trust_env=False) as client:
        health = await client.get("/health")
        assert health.status_code == 200
        assert health.json() == {"status": "healthy"}

        ok = await client.get("/ok")
        assert ok.status_code == 200
        assert ok.json() == {"ok": True}

        info = await client.get("/info")
        assert info.status_code == 200
        info_body = info.json()
        assert info_body["flags"]["assistants"] is True
        assert info_body["metadata"]["checkpoint_backend"] == "langchain-oceanbase"
        assert info_body["metadata"]["checkpoint_backend_version"] == "0.4.0"

        metrics = await client.get("/metrics")
        assert metrics.status_code == 200
        assert metrics.headers["content-type"].startswith("text/plain")
        assert "agentseek_api_info" in metrics.text

        metrics_json = await client.get("/metrics?format=json")
        assert metrics_json.status_code == 200
        assert metrics_json.json()["checks"]["checkpointer"] == "ok"

        default_assistant = await _create_assistant(
            client,
            name="live-default",
            graph_id="default",
            metadata={"suite": "live-create"},
            config={"temperature": 0},
            context={"tenant": "mysql-family"},
            description="live assistant create",
        )
        react_assistant = await _create_assistant(client, name="live-react", graph_id="react_agent")
        assert default_assistant["metadata"] == {"suite": "live-create"}
        assert default_assistant["config"] == {"temperature": 0}
        assert default_assistant["context"] == {"tenant": "mysql-family"}
        assert default_assistant["description"] == "live assistant create"

        listed = await client.get("/assistants")
        assert listed.status_code == 200
        listed_ids = {item["assistant_id"] for item in listed.json()}
        assert default_assistant["assistant_id"] in listed_ids
        assert react_assistant["assistant_id"] in listed_ids

        searched = await client.post("/assistants/search", json={"graph_id": "default", "limit": 20, "offset": 0})
        assert searched.status_code == 200
        assert any(item["assistant_id"] == default_assistant["assistant_id"] for item in searched.json())

        counted = await client.post("/assistants/count", json={"graph_id": "default"})
        assert counted.status_code == 200
        assert counted.json() >= 1

        fetched = await client.get(f"/assistants/{default_assistant['assistant_id']}")
        assert fetched.status_code == 200
        assert fetched.json()["assistant_id"] == default_assistant["assistant_id"]

        patched = await client.patch(
            f"/assistants/{default_assistant['assistant_id']}",
            json={
                "name": "live-default-patched",
                "graph_id": "stress_test",
                "metadata": {"suite": "live"},
                "config": {"temperature": 0},
                "context": {"tenant": "mysql-family"},
                "description": "live assistant",
            },
        )
        assert patched.status_code == 200
        patched_body = patched.json()
        assert patched_body["graph_id"] == "stress_test"
        assert patched_body["metadata"]["suite"] == "live"
        assert patched_body["description"] == "live assistant"

        graph = await client.get(f"/assistants/{default_assistant['assistant_id']}/graph")
        assert graph.status_code == 200
        assert graph.json()["graph_id"] == "stress_test"

        schemas = await client.get(f"/assistants/{default_assistant['assistant_id']}/schemas")
        assert schemas.status_code == 200
        assert "input_schema" in schemas.json()

        subgraphs = await client.get(f"/assistants/{default_assistant['assistant_id']}/subgraphs")
        assert subgraphs.status_code == 200
        assert isinstance(subgraphs.json(), list)

        namespaced = await client.get(f"/assistants/{default_assistant['assistant_id']}/subgraphs/root")
        assert namespaced.status_code == 200
        assert isinstance(namespaced.json(), list)

        versions = await client.post(f"/assistants/{default_assistant['assistant_id']}/versions")
        assert versions.status_code == 200
        assert versions.json()["version"] >= 1

        latest = await client.post(f"/assistants/{default_assistant['assistant_id']}/latest")
        assert latest.status_code == 200
        assert latest.json()["assistant_id"] == default_assistant["assistant_id"]

        unsupported_delete = await client.delete(
            f"/assistants/{default_assistant['assistant_id']}?delete_threads=true",
        )
        assert unsupported_delete.status_code == 400

        deleted = await client.delete(f"/assistants/{react_assistant['assistant_id']}")
        assert deleted.status_code == 204

        deleted_get = await client.get(f"/assistants/{react_assistant['assistant_id']}")
        assert deleted_get.status_code == 404


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_live_thread_endpoints(e2e_base_url: str) -> None:
    async with httpx.AsyncClient(base_url=e2e_base_url, timeout=60.0, trust_env=False) as client:
        assistant = await _create_assistant(client, name="thread-live-default", graph_id="default")
        user_id = "thread-live-user"
        suite_id = f"thread-live-{uuid4()}"

        alpha_thread = await _create_thread(
            client,
            user_id=user_id,
            metadata={"topic": "alpha", "tag": "keep", "suite": suite_id},
            config={"retention": "short"},
        )
        beta_thread = await _create_thread(client, user_id=user_id, metadata={"topic": "beta"})
        _ = await _create_thread(client, user_id="thread-live-other", metadata={"topic": "alpha"})

        first_run = await _create_thread_run(
            client,
            thread_id=str(alpha_thread["thread_id"]),
            assistant_id=str(assistant["assistant_id"]),
            payload={"message": "first"},
            user_id=user_id,
        )
        await _wait_for_run(
            client,
            thread_id=str(alpha_thread["thread_id"]),
            run_id=str(first_run["run_id"]),
            user_id=user_id,
        )

        listed = await client.get("/threads", headers=_user_headers(user_id))
        assert listed.status_code == 200
        listed_ids = {item["thread_id"] for item in listed.json()}
        assert str(alpha_thread["thread_id"]) in listed_ids
        assert str(beta_thread["thread_id"]) in listed_ids

        searched = await client.post(
            "/threads/search",
            json={"metadata": {"suite": suite_id}, "limit": 20, "offset": 0},
            headers=_user_headers(user_id),
        )
        assert searched.status_code == 200
        assert [item["thread_id"] for item in searched.json()] == [alpha_thread["thread_id"]]

        counted = await client.post(
            "/threads/count",
            json={"metadata": {"suite": suite_id}},
            headers=_user_headers(user_id),
        )
        assert counted.status_code == 200
        assert counted.json() == 1

        fetched = await client.get(f"/threads/{alpha_thread['thread_id']}", headers=_user_headers(user_id))
        assert fetched.status_code == 200
        assert fetched.json()["thread_id"] == alpha_thread["thread_id"]
        assert fetched.json()["config"] == {"retention": "short"}

        patched = await client.patch(
            f"/threads/{alpha_thread['thread_id']}",
            json={"metadata": {"tag": "patched"}},
            headers=_user_headers(user_id),
        )
        assert patched.status_code == 200
        assert patched.json()["metadata"]["tag"] == "patched"
        assert patched.json()["metadata"]["topic"] == "alpha"

        state = await client.get(f"/threads/{alpha_thread['thread_id']}/state", headers=_user_headers(user_id))
        assert state.status_code == 200
        run_checkpoint_id = state.json()["checkpoint"]["checkpoint_id"]
        assert "output" in state.json()["values"]

        history = await client.get(f"/threads/{alpha_thread['thread_id']}/history", headers=_user_headers(user_id))
        assert history.status_code == 200
        assert len(history.json()) >= 1

        history_post = await client.post(f"/threads/{alpha_thread['thread_id']}/history", headers=_user_headers(user_id))
        assert history_post.status_code == 200
        assert len(history_post.json()) >= 1

        run_checkpoint = await client.get(
            f"/threads/{alpha_thread['thread_id']}/state/{run_checkpoint_id}",
            headers=_user_headers(user_id),
        )
        assert run_checkpoint.status_code == 200
        assert run_checkpoint.json()["checkpoint"]["checkpoint_id"] == run_checkpoint_id

        manual_state = await client.post(
            f"/threads/{alpha_thread['thread_id']}/state",
            json={"values": {"manual": True}},
            headers=_user_headers(user_id),
        )
        assert manual_state.status_code == 200
        manual_checkpoint_id = manual_state.json()["checkpoint"]["checkpoint_id"]

        manual_checkpoint = await client.get(
            f"/threads/{alpha_thread['thread_id']}/state/{manual_checkpoint_id}",
            headers=_user_headers(user_id),
        )
        assert manual_checkpoint.status_code == 200
        assert manual_checkpoint.json()["values"]["manual"] is True

        snapshotted = await client.post(
            f"/threads/{alpha_thread['thread_id']}/state/checkpoint",
            json={"checkpoint_id": manual_checkpoint_id},
            headers=_user_headers(user_id),
        )
        assert snapshotted.status_code == 200
        assert snapshotted.json()["checkpoint"]["thread_id"] == alpha_thread["thread_id"]
        assert snapshotted.json()["values"]["manual"] is True

        copied = await client.post(f"/threads/{alpha_thread['thread_id']}/copy", headers=_user_headers(user_id))
        assert copied.status_code == 200
        copied_thread_id = copied.json()["thread_id"]
        assert copied_thread_id != alpha_thread["thread_id"]

        copied_history = await client.get(f"/threads/{copied_thread_id}/history", headers=_user_headers(user_id))
        assert copied_history.status_code == 200
        assert len(copied_history.json()) >= 1

        deleted_copy = await client.post(
            "/threads/prune",
            json={"thread_ids": [copied_thread_id], "strategy": "delete"},
            headers=_user_headers(user_id),
        )
        assert deleted_copy.status_code == 200
        assert deleted_copy.json()["pruned_count"] == 1

        second_run = await _create_thread_run(
            client,
            thread_id=str(alpha_thread["thread_id"]),
            assistant_id=str(assistant["assistant_id"]),
            payload={"message": "second"},
            user_id=user_id,
        )
        await _wait_for_run(
            client,
            thread_id=str(alpha_thread["thread_id"]),
            run_id=str(second_run["run_id"]),
            user_id=user_id,
        )
        second_manual = await client.post(
            f"/threads/{alpha_thread['thread_id']}/state",
            json={"values": {"manual": "second"}},
            headers=_user_headers(user_id),
        )
        assert second_manual.status_code == 200

        kept_latest = await client.post(
            "/threads/prune",
            json={"thread_ids": [alpha_thread["thread_id"]], "strategy": "keep_latest"},
            headers=_user_headers(user_id),
        )
        assert kept_latest.status_code == 200
        assert kept_latest.json()["pruned_count"] == 1

        pruned_history = await client.get(f"/threads/{alpha_thread['thread_id']}/history", headers=_user_headers(user_id))
        assert pruned_history.status_code == 200
        assert len(pruned_history.json()) == 1
        assert pruned_history.json()[0]["values"]["manual"] == "second"

        thread_stream = await client.get(f"/threads/{alpha_thread['thread_id']}/stream", headers=_user_headers(user_id))
        assert thread_stream.status_code == 200
        assert thread_stream.headers["content-type"].startswith("text/event-stream")

        command = await client.post(
            f"/threads/{alpha_thread['thread_id']}/commands",
            json={"method": "run.start", "params": {"assistant_id": assistant["assistant_id"], "input": {"message": "command"}}},
            headers=_user_headers(user_id),
        )
        assert command.status_code == 200
        assert command.json()["ok"] is True

        unsupported_command = await client.post(
            f"/threads/{alpha_thread['thread_id']}/commands",
            json={"method": "not-supported", "params": {}},
            headers=_user_headers(user_id),
        )
        assert unsupported_command.status_code == 200
        assert unsupported_command.json()["ok"] is False

        event_stream = await client.post(
            f"/threads/{alpha_thread['thread_id']}/stream/events",
            json={"channels": ["messages"]},
            headers=_user_headers(user_id),
        )
        assert event_stream.status_code == 200
        assert event_stream.headers["content-type"].startswith("text/event-stream")

        deleted_thread = await client.delete(f"/threads/{beta_thread['thread_id']}", headers=_user_headers(user_id))
        assert deleted_thread.status_code == 204

        deleted_thread_get = await client.get(f"/threads/{beta_thread['thread_id']}", headers=_user_headers(user_id))
        assert deleted_thread_get.status_code == 404


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_live_run_and_stateless_endpoints(e2e_base_url: str) -> None:
    async with httpx.AsyncClient(base_url=e2e_base_url, timeout=60.0, trust_env=False) as client:
        user_id = "run-live-user"
        default_assistant = await _create_assistant(client, name="run-live-default", graph_id="default")
        hitl_assistant = await _create_assistant(client, name="run-live-hitl", graph_id="subgraph_hitl_agent")
        stress_assistant = await _create_assistant(client, name="run-live-stress", graph_id="stress_test")

        thread = await _create_thread(client, user_id=user_id, metadata={"suite": "runs"})
        thread_id = str(thread["thread_id"])

        waited_create = await client.post(
            f"/threads/{thread_id}/runs/wait",
            json={"assistant_id": default_assistant["assistant_id"], "input": {"message": "wait-create"}},
            headers=_user_headers(user_id),
        )
        assert waited_create.status_code == 200
        assert waited_create.json()["status"] == "success"

        streamed_create = await client.post(
            f"/threads/{thread_id}/runs/stream",
            json={"assistant_id": default_assistant["assistant_id"], "input": {"message": "stream-create"}},
            headers=_user_headers(user_id),
        )
        assert streamed_create.status_code == 200
        assert streamed_create.headers["content-type"].startswith("text/event-stream")
        assert "event: end" in streamed_create.text

        created = await _create_thread_run(
            client,
            thread_id=thread_id,
            assistant_id=str(default_assistant["assistant_id"]),
            payload={"message": "fetch me"},
            user_id=user_id,
        )
        run_id = str(created["run_id"])

        fetched = await client.get(f"/threads/{thread_id}/runs/{run_id}", headers=_user_headers(user_id))
        assert fetched.status_code == 200
        assert fetched.json()["run_id"] == run_id

        listed = await client.get(f"/threads/{thread_id}/runs", headers=_user_headers(user_id))
        assert listed.status_code == 200
        assert any(item["run_id"] == run_id for item in listed.json())

        joined = await client.get(f"/threads/{thread_id}/runs/{run_id}/join", headers=_user_headers(user_id))
        assert joined.status_code == 200
        assert joined.json()["status"] == "success"

        streamed = await client.get(f"/threads/{thread_id}/runs/{run_id}/stream", headers=_user_headers(user_id))
        assert streamed.status_code == 200
        assert streamed.headers["content-type"].startswith("text/event-stream")
        assert "event: end" in streamed.text

        interrupted = await _create_thread_run(
            client,
            thread_id=thread_id,
            assistant_id=str(hitl_assistant["assistant_id"]),
            payload={"foo": "hello "},
            user_id=user_id,
        )
        interrupted_wait = await _wait_for_run(
            client,
            thread_id=thread_id,
            run_id=str(interrupted["run_id"]),
            user_id=user_id,
        )
        assert interrupted_wait["status"] == "interrupted"

        resumed = await client.post(
            f"/threads/{thread_id}/runs/{interrupted['run_id']}/resume",
            json={"resume": "world"},
            headers=_user_headers(user_id),
        )
        assert resumed.status_code == 200
        resumed_wait = await _wait_for_run(
            client,
            thread_id=thread_id,
            run_id=str(interrupted["run_id"]),
            user_id=user_id,
        )
        assert resumed_wait["status"] == "success"

        cancellable = await _create_thread_run(
            client,
            thread_id=thread_id,
            assistant_id=str(stress_assistant["assistant_id"]),
            payload={"delay": 0.25, "steps": 12},
            user_id=user_id,
        )
        cancelled = await client.post(
            f"/threads/{thread_id}/runs/{cancellable['run_id']}/cancel",
            headers=_user_headers(user_id),
        )
        assert cancelled.status_code == 200
        cancelled_wait = await _wait_for_run(
            client,
            thread_id=thread_id,
            run_id=str(cancellable["run_id"]),
            user_id=user_id,
        )
        assert cancelled_wait["status"] == "error"
        assert cancelled_wait["last_error"] == "Run cancelled"

        deletable = await _create_thread_run(
            client,
            thread_id=thread_id,
            assistant_id=str(default_assistant["assistant_id"]),
            payload={"message": "delete me"},
            user_id=user_id,
        )
        deleted = await client.delete(
            f"/threads/{thread_id}/runs/{deletable['run_id']}",
            headers=_user_headers(user_id),
        )
        assert deleted.status_code == 204

        stateless = await client.post(
            "/runs",
            json={"assistant_id": default_assistant["assistant_id"], "input": {"mode": "stateless"}},
            headers=_user_headers(user_id),
        )
        assert stateless.status_code == 200

        stateless_wait = await client.post(
            "/runs/wait",
            json={"assistant_id": default_assistant["assistant_id"], "input": {"mode": "wait"}},
            headers=_user_headers(user_id),
        )
        assert stateless_wait.status_code == 200
        assert stateless_wait.json()["status"] == "success"

        stateless_stream = await client.post(
            "/runs/stream",
            json={"assistant_id": default_assistant["assistant_id"], "input": {"mode": "stream"}},
            headers=_user_headers(user_id),
        )
        assert stateless_stream.status_code == 200
        assert stateless_stream.headers["content-type"].startswith("text/event-stream")

        batch = await client.post(
            "/runs/batch",
            json=[
                {"assistant_id": default_assistant["assistant_id"], "input": {"batch": 1}},
                {"assistant_id": default_assistant["assistant_id"], "input": {"batch": 2}},
            ],
            headers=_user_headers(user_id),
        )
        assert batch.status_code == 200
        batch_body = batch.json()
        assert len(batch_body) == 2
        for item in batch_body:
            waited_batch = await _wait_for_run(
                client,
                thread_id=str(item["thread_id"]),
                run_id=str(item["run_id"]),
                user_id=user_id,
            )
            assert waited_batch["status"] == "success"

        stateless_cancellable = await client.post(
            "/runs",
            json={"assistant_id": stress_assistant["assistant_id"], "input": {"delay": 0.25, "steps": 12}},
            headers=_user_headers(user_id),
        )
        assert stateless_cancellable.status_code == 200
        stateless_cancellable_body = stateless_cancellable.json()

        cancelled_many = await client.post(
            "/runs/cancel",
            json={
                "thread_id": stateless_cancellable_body["thread_id"],
                "run_ids": [stateless_cancellable_body["run_id"]],
            },
            headers=_user_headers(user_id),
        )
        assert cancelled_many.status_code == 204

        cancelled_many_wait = await _wait_for_run(
            client,
            thread_id=str(stateless_cancellable_body["thread_id"]),
            run_id=str(stateless_cancellable_body["run_id"]),
            user_id=user_id,
        )
        assert cancelled_many_wait["status"] == "error"
        assert cancelled_many_wait["last_error"] == "Run cancelled"

        final_stream = await client.get(
            f"/threads/{thread_id}/runs/{run_id}/stream",
            headers=_user_headers(user_id),
        )
        assert final_stream.status_code == 200
        payload_lines = [line for line in final_stream.text.splitlines() if line.startswith("data: ")]
        assert payload_lines
        payload = json.loads(payload_lines[-1].replace("data: ", "", 1))
        assert payload["run_id"] == run_id
