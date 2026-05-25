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
            "schedule": "FREQ=MINUTELY;INTERVAL=5",
            "input": ["stateless-cron", {"kind": "list-payload"}],
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["assistant_id"] == assistant_id
    assert body["thread_id"] is None
    assert body["enabled"] is True
    assert body["schedule"] == "FREQ=MINUTELY;INTERVAL=5"
    assert body["next_run_at"] is not None
    assert body["cron_id"]

    persisted = asyncio.run(_fetch_cron(body["cron_id"]))
    assert persisted is not None
    assert persisted.assistant_id == assistant_id
    assert persisted.thread_id is None
    assert persisted.schedule == "FREQ=MINUTELY;INTERVAL=5"
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
            "schedule": "FREQ=HOURLY;INTERVAL=1",
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
            "schedule": "FREQ=DAILY;INTERVAL=1",
            "input": {"kind": "thread-cron"},
        },
    )

    assert response.status_code == 404
    assert response.json() == {"detail": "Thread not found"}


def test_search_count_get_patch_and_delete_crons(client: TestClient) -> None:
    assistant_id = _create_assistant(client)

    first = client.post(
        "/runs/crons",
        json={
            "assistant_id": assistant_id,
            "schedule": "FREQ=MINUTELY;INTERVAL=5",
            "input": {"kind": "search-match"},
            "enabled": True,
        },
    )
    assert first.status_code == 200
    first_body = first.json()

    second = client.post(
        "/runs/crons",
        json={
            "assistant_id": assistant_id,
            "schedule": "FREQ=HOURLY;INTERVAL=1",
            "input": {"kind": "search-disabled"},
            "enabled": False,
        },
    )
    assert second.status_code == 200

    search_response = client.post(
        "/runs/crons/search",
        json={"assistant_id": assistant_id, "enabled": True, "limit": 10, "offset": 0},
    )
    assert search_response.status_code == 200
    search_body = search_response.json()
    assert [item["cron_id"] for item in search_body["items"]] == [first_body["cron_id"]]

    count_response = client.post(
        "/runs/crons/count",
        json={"assistant_id": assistant_id, "enabled": True},
    )
    assert count_response.status_code == 200
    assert count_response.json() == {"count": 1}

    get_response = client.get(f"/runs/crons/{first_body['cron_id']}")
    assert get_response.status_code == 200
    assert get_response.json()["cron_id"] == first_body["cron_id"]

    patch_response = client.patch(
        f"/runs/crons/{first_body['cron_id']}",
        json={"schedule": "FREQ=MINUTELY;INTERVAL=1", "enabled": False, "input": {"kind": "patched"}},
    )
    assert patch_response.status_code == 200
    patch_body = patch_response.json()
    assert patch_body["schedule"] == "FREQ=MINUTELY;INTERVAL=1"
    assert patch_body["enabled"] is False

    delete_response = client.delete(f"/runs/crons/{first_body['cron_id']}")
    assert delete_response.status_code == 204

    missing_response = client.get(f"/runs/crons/{first_body['cron_id']}")
    assert missing_response.status_code == 404


def test_patch_cron_rejects_explicit_null_input(client: TestClient) -> None:
    assistant_id = _create_assistant(client)
    created = client.post(
        "/runs/crons",
        json={
            "assistant_id": assistant_id,
            "schedule": "FREQ=MINUTELY;INTERVAL=5",
            "input": {"kind": "original"},
            "enabled": True,
        },
    )
    assert created.status_code == 200
    cron_id = created.json()["cron_id"]

    response = client.patch(f"/runs/crons/{cron_id}", json={"input": None})

    assert response.status_code == 400
    assert response.json() == {"detail": "input cannot be null"}


def test_search_crons_rejects_negative_limit_and_offset(client: TestClient) -> None:
    assistant_id = _create_assistant(client)

    response = client.post(
        "/runs/crons/search",
        json={"assistant_id": assistant_id, "enabled": True, "limit": -1, "offset": -1},
    )

    assert response.status_code == 422
