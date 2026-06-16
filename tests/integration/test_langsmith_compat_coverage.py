import asyncio

from fastapi import HTTPException
from fastapi.testclient import TestClient
from sqlalchemy import select

from agentseek_api.core.database import db_manager
from agentseek_api.core.orm import Run
from agentseek_api.models.api import RunRead


def _create_assistant(client: TestClient, *, name: str = "assistant", graph_id: str = "default") -> dict[str, object]:
    response = client.post("/assistants", json={"name": name, "graph_id": graph_id})
    assert response.status_code == 200
    return response.json()


def _create_thread(
    client: TestClient,
    *,
    user_id: str = "default_user",
    metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    response = client.post("/threads", json={"metadata": metadata or {}}, headers={"x-user-id": user_id})
    assert response.status_code == 200
    return response.json()


async def _update_run_status(run_id: str, *, status: str) -> None:
    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        row = await session.scalar(select(Run).where(Run.run_id == run_id))
        assert row is not None
        row.status = status
        row.last_error = None
        await session.commit()


def test_assistant_routes_cover_list_patch_and_missing_paths(client: TestClient) -> None:
    created = _create_assistant(client, name="before", graph_id="default")
    assistant_id = str(created["assistant_id"])

    listed = client.post("/assistants/search", json={})
    assert listed.status_code == 200
    assert any(item["assistant_id"] == assistant_id for item in listed.json())

    patched = client.patch(
        f"/assistants/{assistant_id}",
        json={
            "name": "after",
            "graph_id": "react_agent",
            "metadata": {"team": "api"},
            "config": {"configurable": {"temperature": 0}},
            "context": {"tenant": "compat"},
            "description": "updated",
        },
    )
    assert patched.status_code == 200
    patched_body = patched.json()
    assert patched_body["name"] == "after"
    assert patched_body["graph_id"] == "react_agent"
    assert patched_body["metadata"] == {"team": "api"}
    assert patched_body["config"] == {"tags": [], "configurable": {"temperature": 0}}
    assert patched_body["context"] == {"tenant": "compat"}
    assert patched_body["description"] == "updated"

    mismatched_name = client.post("/assistants/search", json={"name": "missing"})
    assert mismatched_name.status_code == 200
    assert mismatched_name.json() == []

    mismatched_metadata = client.post("/assistants/search", json={"metadata": {"team": "other"}})
    assert mismatched_metadata.status_code == 200
    assert mismatched_metadata.json() == []

    missing_id = "00000000-0000-0000-0000-000000000000"
    assert client.get(f"/assistants/{missing_id}").status_code == 404
    assert client.patch(f"/assistants/{missing_id}", json={"name": "nope"}).status_code == 404
    assert client.delete(f"/assistants/{missing_id}").status_code == 404


def test_thread_routes_cover_search_prune_delete_and_missing_paths(client: TestClient) -> None:
    assistant_id = str(_create_assistant(client)["assistant_id"])
    keep_thread = _create_thread(client, user_id="u1", metadata={"topic": "alpha"})
    delete_thread = _create_thread(client, user_id="u1", metadata={"topic": "beta"})

    created_run = client.post(
        f"/threads/{delete_thread['thread_id']}/runs",
        json={"assistant_id": assistant_id, "input": {"message": "delete me"}},
        headers={"x-user-id": "u1"},
    )
    assert created_run.status_code == 200
    delete_run_id = created_run.json()["run_id"]
    # Run exists before the thread is deleted.
    assert client.get(
        f"/threads/{delete_thread['thread_id']}/runs/{delete_run_id}",
        headers={"x-user-id": "u1"},
    ).status_code == 200

    listed = client.post("/threads/search", json={}, headers={"x-user-id": "u1"})
    assert listed.status_code == 200
    assert {item["thread_id"] for item in listed.json()} == {keep_thread["thread_id"], delete_thread["thread_id"]}

    id_filtered = client.post(
        "/threads/search",
        json={"ids": [keep_thread["thread_id"]], "status": "idle"},
        headers={"x-user-id": "u1"},
    )
    assert id_filtered.status_code == 200
    assert [item["thread_id"] for item in id_filtered.json()] == [keep_thread["thread_id"]]

    status_miss = client.post(
        "/threads/search",
        json={"ids": [keep_thread["thread_id"]], "status": "busy"},
        headers={"x-user-id": "u1"},
    )
    assert status_miss.status_code == 200
    assert status_miss.json() == []

    pruned = client.post(
        "/threads/prune",
        json={"thread_ids": [keep_thread["thread_id"]], "strategy": "keep_latest"},
        headers={"x-user-id": "u1"},
    )
    assert pruned.status_code == 200
    assert pruned.json()["pruned_count"] == 1
    assert client.get(f"/threads/{keep_thread['thread_id']}", headers={"x-user-id": "u1"}).status_code == 200

    deleted = client.delete(f"/threads/{delete_thread['thread_id']}", headers={"x-user-id": "u1"})
    assert deleted.status_code == 204
    assert client.get(f"/threads/{delete_thread['thread_id']}", headers={"x-user-id": "u1"}).status_code == 404
    # The cascade must also remove the thread's runs, not just the thread row.
    assert client.get(
        f"/threads/{delete_thread['thread_id']}/runs/{delete_run_id}",
        headers={"x-user-id": "u1"},
    ).status_code == 404

    missing_id = "00000000-0000-0000-0000-000000000001"
    missing_headers = {"x-user-id": "u1"}
    assert client.patch(f"/threads/{missing_id}", json={"metadata": {"x": 1}}, headers=missing_headers).status_code == 404
    assert client.delete(f"/threads/{missing_id}", headers=missing_headers).status_code == 404
    assert client.post(f"/threads/{missing_id}/copy", headers=missing_headers).status_code == 404
    assert client.get(f"/threads/{missing_id}/state", headers=missing_headers).status_code == 404
    assert client.get(f"/threads/{missing_id}/history", headers=missing_headers).status_code == 404
    assert client.post(f"/threads/{missing_id}/history", json={}, headers=missing_headers).status_code == 404
    missing_checkpoint = client.post(
        f"/threads/{missing_id}/state/checkpoint",
        json={"checkpoint_id": "missing"},
        headers=missing_headers,
    )
    assert missing_checkpoint.status_code == 404
    assert client.post(f"/threads/{missing_id}/state", json={"values": {}}, headers=missing_headers).status_code == 404
    assert client.get(f"/threads/{missing_id}/stream", headers=missing_headers).status_code == 404

    bad_checkpoint = client.post(
        f"/threads/{keep_thread['thread_id']}/state/checkpoint",
        json={"checkpoint_id": "00000000-0000-0000-0000-000000000002"},
        headers=missing_headers,
    )
    assert bad_checkpoint.status_code == 404
    assert bad_checkpoint.json()["detail"] == "Checkpoint not found"

    unsupported = client.post(
        f"/threads/{keep_thread['thread_id']}/commands",
        json={"id": 99, "method": "not-supported", "params": {}},
        headers=missing_headers,
    )
    assert unsupported.status_code == 400
    assert unsupported.json() == {
        "type": "error",
        "id": 99,
        "error": "unknown_command",
        "message": "Unsupported command 'not-supported'",
    }


def test_run_routes_cover_list_wait_resume_cancel_and_missing_paths(
    client: TestClient,
    monkeypatch,
) -> None:
    assistant_id = str(_create_assistant(client)["assistant_id"])
    thread_id = str(_create_thread(client)["thread_id"])

    created = client.post(
        f"/threads/{thread_id}/runs",
        json={"assistant_id": assistant_id, "input": {"message": "coverage"}},
    )
    assert created.status_code == 200
    created_body = created.json()
    run_id = str(created_body["run_id"])

    listed = client.get(f"/threads/{thread_id}/runs")
    assert listed.status_code == 200
    assert [item["run_id"] for item in listed.json()] == [run_id]

    assert client.get(f"/threads/{thread_id}/runs/00000000-0000-0000-0000-000000000003").status_code == 404
    assert client.get(f"/threads/{thread_id}/runs/00000000-0000-0000-0000-000000000003/wait").status_code == 404
    assert client.get(f"/threads/{thread_id}/runs/00000000-0000-0000-0000-000000000003/join").status_code == 404
    assert client.get(f"/threads/{thread_id}/runs/00000000-0000-0000-0000-000000000003/stream").status_code == 404
    assert client.delete(f"/threads/{thread_id}/runs/00000000-0000-0000-0000-000000000003").status_code == 404

    missing_resume = client.post(
        f"/threads/{thread_id}/runs/00000000-0000-0000-0000-000000000003/resume",
        json={"resume": {"approved": True}},
    )
    assert missing_resume.status_code == 404

    wrong_state_resume = client.post(
        f"/threads/{thread_id}/runs/{run_id}/resume",
        json={"resume": {"approved": True}},
    )
    assert wrong_state_resume.status_code == 409

    asyncio.run(_update_run_status(run_id, status="running"))
    cancelled = client.post(f"/threads/{thread_id}/runs/{run_id}/cancel")
    assert cancelled.status_code == 200

    fetched = client.get(f"/threads/{thread_id}/runs/{run_id}")
    assert fetched.status_code == 200
    assert fetched.json()["status"] == "error"
    assert fetched.json()["last_error"] == "Run cancelled"

    pending_created = RunRead.model_validate(
        {
            "run_id": "pending-run",
            "thread_id": thread_id,
            "assistant_id": assistant_id,
            "status": "pending",
            "output": None,
            "metadata": {},
            "kwargs": {},
            "multitask_strategy": "enqueue",
        }
    )
    final_created = pending_created.model_copy(update={"status": "success", "output": {"ok": True}})

    async def fake_create_run(*args, **kwargs):
        return pending_created

    async def fake_wait_run(*args, **kwargs):
        return final_created

    async def fake_wait_response_payload(*args, **kwargs):
        return {"ok": True}

    monkeypatch.setattr("agentseek_api.api.runs.create_run", fake_create_run)
    monkeypatch.setattr("agentseek_api.api.runs._wait_run_terminal", fake_wait_run)
    monkeypatch.setattr("agentseek_api.api.runs._wait_response_payload", fake_wait_response_payload)

    waited = client.post(
        f"/threads/{thread_id}/runs/wait",
        json={"assistant_id": assistant_id, "input": {"message": "wait later"}},
    )
    assert waited.status_code == 200
    assert waited.json() == {"ok": True}


def test_create_run_wait_keeps_join_open_across_internal_wait_windows(client: TestClient, monkeypatch) -> None:
    assistant_id = str(_create_assistant(client)["assistant_id"])
    thread_id = str(_create_thread(client, metadata={"case": "wait-keepalive"})["thread_id"])

    pending_created = RunRead.model_validate(
        {
            "run_id": "slow-run",
            "thread_id": thread_id,
            "assistant_id": assistant_id,
            "status": "pending",
            "output": None,
            "metadata": {},
            "kwargs": {},
            "multitask_strategy": "enqueue",
        }
    )
    final_created = pending_created.model_copy(update={"status": "success", "output": {"ok": True}})
    wait_calls = {"count": 0}

    async def fake_create_run(*args, **kwargs):
        return pending_created

    async def fake_wait_run(*args, **kwargs):
        wait_calls["count"] += 1
        if wait_calls["count"] == 1:
            raise HTTPException(status_code=408, detail="Run wait timeout")
        return final_created

    async def fake_wait_response_payload(*args, **kwargs):
        return {"ok": True}

    monkeypatch.setattr("agentseek_api.api.runs.create_run", fake_create_run)
    monkeypatch.setattr("agentseek_api.api.runs._wait_run_terminal", fake_wait_run)
    monkeypatch.setattr("agentseek_api.api.runs._wait_response_payload", fake_wait_response_payload)

    waited = client.post(
        f"/threads/{thread_id}/runs/wait",
        json={"assistant_id": assistant_id, "input": {"message": "wait through keepalive"}},
    )

    assert waited.status_code == 200
    assert waited.json() == {"ok": True}
    assert wait_calls["count"] == 2


def test_stateless_routes_cover_wait_and_cancel_status_branch(client: TestClient, monkeypatch) -> None:
    assistant_id = str(_create_assistant(client)["assistant_id"])

    pending_run = RunRead.model_validate(
        {
            "run_id": "pending-stateless",
            "thread_id": "stateless-thread",
            "assistant_id": assistant_id,
            "status": "pending",
            "output": None,
            "metadata": {},
            "kwargs": {},
            "multitask_strategy": "enqueue",
        }
    )
    finished_run = pending_run.model_copy(update={"status": "success", "output": {"done": True}})

    async def fake_create_stateless_run(*args, **kwargs):
        return pending_run

    async def fake_wait_run(*args, **kwargs):
        return finished_run

    async def fake_wait_response_payload(*args, **kwargs):
        return {"done": True}

    monkeypatch.setattr("agentseek_api.api.stateless_runs.create_stateless_run", fake_create_stateless_run)
    monkeypatch.setattr("agentseek_api.api.runs._wait_run_terminal", fake_wait_run)
    monkeypatch.setattr("agentseek_api.api.runs._wait_response_payload", fake_wait_response_payload)

    waited = client.post("/runs/wait", json={"assistant_id": assistant_id, "input": {"message": "later"}})
    assert waited.status_code == 200
    assert waited.json() == {"done": True}

    created = client.post("/runs", json={"assistant_id": assistant_id, "input": {"message": "cancel by status"}})
    assert created.status_code == 200
    created_body = created.json()
    asyncio.run(_update_run_status(str(created_body["run_id"]), status="running"))

    cancelled = client.post("/runs/cancel", json={"status": "running"})
    assert cancelled.status_code == 204

    fetched = client.get(f"/threads/{created_body['thread_id']}/runs/{created_body['run_id']}")
    assert fetched.status_code == 200
    assert fetched.json()["status"] == "error"
    assert fetched.json()["last_error"] == "Run cancelled"


def test_stateless_cancel_scopes_by_run_ids_and_thread_id(client: TestClient) -> None:
    assistant_id = str(_create_assistant(client)["assistant_id"])

    first = client.post("/runs", json={"assistant_id": assistant_id, "input": {"message": "first"}})
    second = client.post("/runs", json={"assistant_id": assistant_id, "input": {"message": "second"}})
    third = client.post("/runs", json={"assistant_id": assistant_id, "input": {"message": "third"}})
    assert first.status_code == 200
    assert second.status_code == 200
    assert third.status_code == 200

    first_body = first.json()
    second_body = second.json()
    third_body = third.json()

    asyncio.run(_update_run_status(str(first_body["run_id"]), status="running"))
    asyncio.run(_update_run_status(str(second_body["run_id"]), status="running"))
    asyncio.run(_update_run_status(str(third_body["run_id"]), status="running"))

    cancel_selected = client.post("/runs/cancel", json={"run_ids": [first_body["run_id"]]})
    assert cancel_selected.status_code == 204

    first_run = client.get(f"/threads/{first_body['thread_id']}/runs/{first_body['run_id']}")
    second_run = client.get(f"/threads/{second_body['thread_id']}/runs/{second_body['run_id']}")
    assert first_run.status_code == 200
    assert second_run.status_code == 200
    assert first_run.json()["status"] == "error"
    assert second_run.json()["status"] == "running"

    cancel_thread = client.post("/runs/cancel", json={"thread_id": second_body["thread_id"]})
    assert cancel_thread.status_code == 204

    second_run_after = client.get(f"/threads/{second_body['thread_id']}/runs/{second_body['run_id']}")
    third_run_after = client.get(f"/threads/{third_body['thread_id']}/runs/{third_body['run_id']}")
    assert second_run_after.status_code == 200
    assert third_run_after.status_code == 200
    assert second_run_after.json()["status"] == "error"
    assert third_run_after.json()["status"] == "running"
