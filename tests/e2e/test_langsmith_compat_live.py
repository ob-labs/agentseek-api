import asyncio
import json
import os
from uuid import uuid4

import httpx
import pytest
from langchain_oceanbase.store import OceanBaseStore
from sqlalchemy.engine import URL

from agentseek_api.core.runtime_store import make_user_store_namespace


def _user_headers(user_id: str) -> dict[str, str]:
    return {"x-user-id": user_id}


def _seekdb_url() -> str:
    return URL.create(
        drivername="mysql+aiomysql",
        username=os.getenv("OCEANBASE_USER", "root@test"),
        password=os.getenv("OCEANBASE_PASSWORD", ""),
        host=os.getenv("OCEANBASE_HOST", "127.0.0.1"),
        port=int(os.getenv("OCEANBASE_PORT", "2881")),
        database=os.getenv("OCEANBASE_DB_NAME", "seekdb"),
    ).render_as_string(hide_password=False)
async def _fetch_store_item_from_backend(
    *,
    user_id: str,
    namespace: list[str],
    key: str,
) -> dict[str, object] | None:
    store = OceanBaseStore(
        connection_args={
            "host": os.getenv("OCEANBASE_HOST", "127.0.0.1"),
            "port": os.getenv("OCEANBASE_PORT", "2881"),
            "user": os.getenv("OCEANBASE_USER", "root@test"),
            "password": os.getenv("OCEANBASE_PASSWORD", ""),
            "db_name": os.getenv("OCEANBASE_DB_NAME", "seekdb"),
        }
    )
    try:
        item = await store.aget(make_user_store_namespace(user_id=user_id, namespace=tuple(namespace)), key)
        if item is None:
            return None
        return {
            "namespace": list(item.namespace[2:]),
            "value": dict(item.value),
        }
    finally:
        store.obvector.engine.dispose()


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
) -> dict[str, object]:
    payload: dict[str, object] = {"metadata": metadata}
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
        assert info_body["flags"]["protocol_v2"] is True
        assert info_body["metadata"]["checkpoint_backend"] == "langchain-oceanbase"
        assert info_body["metadata"]["checkpoint_backend_version"] == "0.5.0"

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
            config={"configurable": {"temperature": 0}},
            context={"tenant": "mysql-family"},
            description="live assistant create",
        )
        react_assistant = await _create_assistant(client, name="live-react", graph_id="react_agent")
        assert default_assistant["metadata"] == {"suite": "live-create"}
        assert default_assistant["config"] == {"tags": [], "recursion_limit": None, "configurable": {"temperature": 0}}
        assert default_assistant["context"] == {"tenant": "mysql-family"}
        assert default_assistant["description"] == "live assistant create"

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
                "config": {"configurable": {"temperature": 0}},
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
        assert "nodes" in graph.json()

        schemas = await client.get(f"/assistants/{default_assistant['assistant_id']}/schemas")
        assert schemas.status_code == 200
        assert "input_schema" in schemas.json()

        subgraphs = await client.get(f"/assistants/{default_assistant['assistant_id']}/subgraphs")
        assert subgraphs.status_code == 200

        namespaced = await client.get(f"/assistants/{default_assistant['assistant_id']}/subgraphs/root")
        assert namespaced.status_code == 200

        versions = await client.post(f"/assistants/{default_assistant['assistant_id']}/versions")
        assert versions.status_code == 200
        assert versions.json() == {
            "assistant_id": default_assistant["assistant_id"],
            "current_version": 1,
            "latest_version": 1,
            "available_versions": [1],
            "supports_version_history": False,
        }

        latest = await client.post(f"/assistants/{default_assistant['assistant_id']}/latest")
        assert latest.status_code == 409
        assert latest.json()["detail"] == "Assistant version promotion is not supported"

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
async def test_live_store_endpoints_use_mysql_family_backend(e2e_base_url: str) -> None:
    user_id = f"store-user-{uuid4()}"
    other_user_id = f"store-other-{uuid4()}"
    namespace = ["e2e", "store", uuid4().hex]

    async with httpx.AsyncClient(base_url=e2e_base_url, timeout=30.0, trust_env=False) as client:
        info = await client.get("/info")
        assert info.status_code == 200
        info_body = info.json()
        assert info_body["flags"]["store"] is True
        assert info_body["metadata"]["checkpoint_backend"] == "langchain-oceanbase"

        created = await client.put(
            "/store/items",
            json={
                "namespace": namespace,
                "key": "profile",
                "value": {"kind": "profile", "name": "Ada"},
            },
            headers=_user_headers(user_id),
        )
        assert created.status_code == 200
        created_body = created.json()
        assert created_body["namespace"] == namespace
        assert created_body["key"] == "profile"
        assert created_body["value"] == {"kind": "profile", "name": "Ada"}

        updated = await client.put(
            "/store/items",
            json={
                "namespace": namespace,
                "key": "profile",
                "value": {"kind": "profile", "name": "Ada", "level": 2},
            },
            headers=_user_headers(user_id),
        )
        assert updated.status_code == 200
        updated_body = updated.json()
        assert updated_body["created_at"] == created_body["created_at"]
        assert updated_body["value"] == {"kind": "profile", "name": "Ada", "level": 2}

        backend_row = await _fetch_store_item_from_backend(user_id=user_id, namespace=namespace, key="profile")
        assert backend_row == {
            "namespace": namespace,
            "value": {"kind": "profile", "name": "Ada", "level": 2},
        }
        assert await _fetch_store_item_from_backend(user_id=other_user_id, namespace=namespace, key="profile") is None

        fetched = await client.get(
            "/store/items",
            params=[("key", "profile"), *(("namespace", part) for part in namespace)],
            headers=_user_headers(user_id),
        )
        assert fetched.status_code == 200
        assert fetched.json()["value"] == {"kind": "profile", "name": "Ada", "level": 2}

        isolated = await client.get(
            "/store/items",
            params=[("key", "profile"), *(("namespace", part) for part in namespace)],
            headers=_user_headers(other_user_id),
        )
        assert isolated.status_code == 404

        await client.put(
            "/store/items",
            json={
                "namespace": namespace[:-1] + ["scratch"],
                "key": "note",
                "value": {"kind": "note", "name": "temporary"},
            },
            headers=_user_headers(user_id),
        )

        searched = await client.post(
            "/store/items/search",
            json={
                "namespace_prefix": namespace[:2],
                "filter": {"kind": "profile"},
                "limit": 10,
                "offset": 0,
            },
            headers=_user_headers(user_id),
        )
        assert searched.status_code == 200
        assert [item["key"] for item in searched.json()["items"]] == ["profile"]

        namespaces = await client.post(
            "/store/namespaces",
            json={"prefix": namespace[:2], "max_depth": 3, "limit": 10, "offset": 0},
            headers=_user_headers(user_id),
        )
        assert namespaces.status_code == 200
        assert namespace in namespaces.json()

        deleted = await client.request(
            "DELETE",
            "/store/items",
            json={"namespace": namespace, "key": "profile"},
            headers=_user_headers(user_id),
        )
        assert deleted.status_code == 204

        missing = await client.get(
            "/store/items",
            params=[("key", "profile"), *(("namespace", part) for part in namespace)],
            headers=_user_headers(user_id),
        )
        assert missing.status_code == 404
        assert await _fetch_store_item_from_backend(user_id=user_id, namespace=namespace, key="profile") is None


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_live_store_ttl_from_manifest_expires_items_on_mysql_family_backend(e2e_base_url: str) -> None:
    user_id = f"store-ttl-user-{uuid4()}"
    namespace = ["e2e", "ttl", uuid4().hex]

    async with httpx.AsyncClient(base_url=e2e_base_url, timeout=30.0, trust_env=False) as client:
        created = await client.put(
            "/store/items",
            json={
                "namespace": namespace,
                "key": "ephemeral",
                "value": {"kind": "note", "name": "expires-from-config"},
            },
            headers=_user_headers(user_id),
        )
        assert created.status_code == 200

        immediate = await client.get(
            "/store/items",
            params=[("key", "ephemeral"), *(("namespace", part) for part in namespace)],
            headers=_user_headers(user_id),
        )
        assert immediate.status_code == 200

        await asyncio.sleep(4.0)

        expired = await client.get(
            "/store/items",
            params=[("key", "ephemeral"), *(("namespace", part) for part in namespace)],
            headers=_user_headers(user_id),
        )
        assert expired.status_code == 404

        searched = await client.post(
            "/store/items/search",
            json={"namespace_prefix": namespace[:2], "limit": 10, "offset": 0},
            headers=_user_headers(user_id),
        )
        assert searched.status_code == 200
        assert searched.json() == {"items": []}


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_live_store_graph_uses_injected_mysql_family_backend(e2e_base_url: str) -> None:
    user_id = f"graph-store-user-{uuid4()}"
    other_user_id = f"graph-store-other-{uuid4()}"
    namespace = ["graph", "memory"]
    key = f"memory-{uuid4().hex}"
    value = {"text": "Ada from graph", "kind": "profile"}

    async with httpx.AsyncClient(base_url=e2e_base_url, timeout=60.0, trust_env=False) as client:
        assistant = await _create_assistant(client, name="live-store-memory", graph_id="store_memory")
        thread = await _create_thread(
            client,
            user_id=user_id,
            metadata={"suite": "live-store-graph"},
        )
        run = await _create_thread_run(
            client,
            thread_id=str(thread["thread_id"]),
            assistant_id=str(assistant["assistant_id"]),
            payload={"memory_key": key, "memory_value": value},
            user_id=user_id,
        )

        waited = await _wait_for_run(
            client,
            thread_id=str(thread["thread_id"]),
            run_id=str(run["run_id"]),
            user_id=user_id,
        )

        assert waited["status"] == "success"
        assert waited["output"] == {
            "namespace": namespace,
            "key": key,
            "value": value,
        }

        backend_row = await _fetch_store_item_from_backend(user_id=user_id, namespace=namespace, key=key)
        assert backend_row == {
            "namespace": namespace,
            "value": value,
        }
        assert await _fetch_store_item_from_backend(user_id=other_user_id, namespace=namespace, key=key) is None


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

        listed = await client.post("/threads/search", json={}, headers=_user_headers(user_id))
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
        assert fetched.json()["config"] == {}

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

        async with client.stream("GET", f"/threads/{alpha_thread['thread_id']}/stream", headers=_user_headers(user_id)) as thread_stream:
            assert thread_stream.status_code == 200
            assert thread_stream.headers["content-type"].startswith("text/event-stream")

            async def collect_lifecycle_states() -> list[str]:
                lifecycle_states: list[str] = []
                event_name: str | None = None
                event_data: dict[str, object] | None = None
                async for line in thread_stream.aiter_lines():
                    if line.startswith("event: "):
                        event_name = line.removeprefix("event: ")
                    elif line.startswith("data: "):
                        event_data = json.loads(line.removeprefix("data: "))
                    elif not line:
                        if event_name == "lifecycle" and isinstance(event_data, dict):
                            data = event_data.get("params", {}).get("data", {})
                            if isinstance(data, dict):
                                state = data.get("event")
                                if isinstance(state, str):
                                    lifecycle_states.append(state)
                                    if {"started", "completed"} <= set(lifecycle_states):
                                        return lifecycle_states
                        event_name = None
                        event_data = None
                return lifecycle_states

            stream_task = asyncio.create_task(collect_lifecycle_states())

            command = await client.post(
                f"/threads/{alpha_thread['thread_id']}/commands",
                json={
                    "id": 21,
                    "method": "run.start",
                    "params": {"assistant_id": assistant["assistant_id"], "input": {"message": "command"}},
                },
                headers=_user_headers(user_id),
            )
            assert command.status_code == 200
            assert command.json()["type"] == "success"
            assert command.json()["id"] == 21
            assert command.json()["result"]["run_id"]

            lifecycle_states = await asyncio.wait_for(stream_task, timeout=15.0)
            assert "started" in lifecycle_states
            assert "completed" in lifecycle_states

        unsupported_command = await client.post(
            f"/threads/{alpha_thread['thread_id']}/commands",
            json={"id": 22, "method": "not-supported", "params": {}},
            headers=_user_headers(user_id),
        )
        assert unsupported_command.status_code == 400
        assert unsupported_command.json() == {
            "type": "error",
            "id": 22,
            "error": "unknown_command",
            "message": "Unsupported command 'not-supported'",
        }

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
        assert waited_create.json()["output"] == {"echo": {"message": "wait-create"}}

        streamed_create = await client.post(
            f"/threads/{thread_id}/runs/stream",
            json={"assistant_id": default_assistant["assistant_id"], "input": {"message": "stream-create"}},
            headers=_user_headers(user_id),
        )
        assert streamed_create.status_code == 200
        assert streamed_create.headers["content-type"].startswith("text/event-stream")
        assert "event: metadata" in streamed_create.text
        assert "event: values" in streamed_create.text

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
        assert joined.headers["content-location"] == f"/threads/{thread_id}/runs/{run_id}"
        assert joined.json()["output"] == {"echo": {"message": "fetch me"}}

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
        assert stateless_wait.json()["output"] == {"echo": {"mode": "wait"}}

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
