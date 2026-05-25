import asyncio

from sqlalchemy import select
from fastapi.testclient import TestClient

from agentseek_api.core.database import db_manager
from agentseek_api.core.orm import CronJob


def _create_assistant(client: TestClient) -> str:
    response = client.post("/assistants", json={"name": "cron-assistant", "graph_id": "default"})
    assert response.status_code == 200
    return response.json()["assistant_id"]


def _create_thread(client: TestClient, user_id: str = "default_user") -> str:
    response = client.post("/threads", json={"metadata": {"scope": "cron"}}, headers={"x-user-id": user_id})
    assert response.status_code == 200
    return response.json()["thread_id"]


async def _fetch_cron(cron_id: str) -> CronJob | None:
    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        return await session.scalar(select(CronJob).where(CronJob.cron_id == cron_id))


def test_create_stateless_cron_persists_and_returns_resource(client: TestClient) -> None:
    assistant_id = _create_assistant(client)

    response = client.post(
        "/runs/crons",
        json={
            "assistant_id": assistant_id,
            "schedule": "0 * * * *",
            "input": ["stateless-cron", {"kind": "list-payload"}],
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["assistant_id"] == assistant_id
    assert body["thread_id"] is None
    assert body["enabled"] is True
    assert body["schedule"] == "0 * * * *"
    assert body["next_run_at"] is not None
    assert body["cron_id"]

    persisted = asyncio.run(_fetch_cron(body["cron_id"]))
    assert persisted is not None
    assert persisted.assistant_id == assistant_id
    assert persisted.thread_id is None
    assert persisted.schedule == "0 * * * *"
    assert persisted.input_json == ["stateless-cron", {"kind": "list-payload"}]
    assert persisted.next_run_at is not None
    assert persisted.next_run_at.isoformat() == body["next_run_at"]


def test_create_thread_cron_persists_thread_and_user_binding(client: TestClient) -> None:
    assistant_id = _create_assistant(client)
    thread_id = _create_thread(client, user_id="owner")

    response = client.post(
        f"/threads/{thread_id}/runs/crons",
        json={
            "assistant_id": assistant_id,
            "schedule": "0 * * * *",
            "input": {"kind": "thread-cron"},
        },
        headers={"x-user-id": "owner"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["assistant_id"] == assistant_id
    assert body["thread_id"] == thread_id

    persisted = asyncio.run(_fetch_cron(body["cron_id"]))
    assert persisted is not None
    assert persisted.thread_id == thread_id
    assert persisted.user_id == "owner"
    assert persisted.assistant_id == assistant_id


def test_create_thread_cron_missing_thread_returns_not_found(client: TestClient) -> None:
    assistant_id = _create_assistant(client)

    response = client.post(
        "/threads/does-not-exist/runs/crons",
        json={
            "assistant_id": assistant_id,
            "schedule": "0 * * * *",
            "input": {"kind": "thread-cron"},
        },
    )

    assert response.status_code == 404
    assert response.json() == {"detail": "Thread not found"}
