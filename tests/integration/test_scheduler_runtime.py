from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

from sqlalchemy import select
from fastapi.testclient import TestClient

from agentseek_api.core.database import db_manager
from agentseek_api.core.orm import CronJob, Run, Thread


def _as_utc(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


async def _fetch_cron(cron_id: str) -> CronJob | None:
    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        return await session.scalar(select(CronJob).where(CronJob.cron_id == cron_id))


async def _mark_cron_due(cron_id: str, *, when: datetime) -> None:
    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        cron = await session.scalar(select(CronJob).where(CronJob.cron_id == cron_id))
        assert cron is not None
        cron.next_run_at = when
        await session.commit()


async def _mark_thread_busy(thread_id: str) -> None:
    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        thread = await session.scalar(select(Thread).where(Thread.thread_id == thread_id))
        assert thread is not None
        thread.status = "busy"
        await session.commit()


async def _list_threads_for_user(user_id: str) -> list[Thread]:
    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        return list((await session.scalars(select(Thread).where(Thread.user_id == user_id))).all())


async def _list_runs_for_thread(thread_id: str) -> list[Run]:
    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        return list((await session.scalars(select(Run).where(Run.thread_id == thread_id))).all())


def _create_assistant(client: TestClient) -> str:
    response = client.post("/assistants", json={"name": "scheduler-assistant", "graph_id": "default"})
    assert response.status_code == 200
    return response.json()["assistant_id"]


def _create_thread(client: TestClient, *, user_id: str) -> str:
    response = client.post("/threads", json={"metadata": {"scope": "scheduler"}}, headers={"x-user-id": user_id})
    assert response.status_code == 200
    return response.json()["thread_id"]


def test_dispatch_due_crons_creates_stateless_run_and_skips_busy_thread(client: TestClient) -> None:
    from agentseek_api.services.cron_scheduler import dispatch_due_crons

    assistant_id = _create_assistant(client)
    thread_id = _create_thread(client, user_id="owner")

    stateless = client.post(
        "/runs/crons",
        json={
            "assistant_id": assistant_id,
            "schedule": "FREQ=MINUTELY;INTERVAL=1",
            "input": {"kind": "stateless"},
        },
        headers={"x-user-id": "owner"},
    )
    assert stateless.status_code == 200

    thread_bound = client.post(
        f"/threads/{thread_id}/runs/crons",
        json={
            "assistant_id": assistant_id,
            "schedule": "FREQ=MINUTELY;INTERVAL=1",
            "input": {"kind": "thread-bound"},
        },
        headers={"x-user-id": "owner"},
    )
    assert thread_bound.status_code == 200

    due_at = datetime.now(UTC) - timedelta(minutes=1)
    asyncio.run(_mark_cron_due(stateless.json()["cron_id"], when=due_at))
    asyncio.run(_mark_cron_due(thread_bound.json()["cron_id"], when=due_at))
    asyncio.run(_mark_thread_busy(thread_id))

    results = asyncio.run(dispatch_due_crons(limit=10, scheduler_id="scheduler-1", now=due_at))

    assert len(results) == 2
    queued = [result for result in results if result.status == "queued"]
    skipped = [result for result in results if result.status == "skipped"]
    assert len(queued) == 1
    assert len(skipped) == 1
    assert skipped[0].thread_id == thread_id
    assert skipped[0].skip_reason == "thread_busy"

    user_threads = asyncio.run(_list_threads_for_user("owner"))
    stateless_threads = [thread for thread in user_threads if thread.metadata_json.get("cron_id") == stateless.json()["cron_id"]]
    assert len(stateless_threads) == 1

    created_runs = asyncio.run(_list_runs_for_thread(stateless_threads[0].thread_id))
    assert len(created_runs) == 1
    assert created_runs[0].status == "success"

    busy_thread_runs = asyncio.run(_list_runs_for_thread(thread_id))
    assert busy_thread_runs == []

    persisted_stateless = asyncio.run(_fetch_cron(stateless.json()["cron_id"]))
    persisted_thread_bound = asyncio.run(_fetch_cron(thread_bound.json()["cron_id"]))
    assert persisted_stateless is not None
    assert persisted_thread_bound is not None
    assert _as_utc(persisted_stateless.next_run_at) > due_at
    assert _as_utc(persisted_thread_bound.next_run_at) > due_at
