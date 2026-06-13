from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

from sqlalchemy import select
from fastapi.testclient import TestClient

from agentseek_api.core.database import db_manager
from agentseek_api.core.orm import CronJob, CronTick, CronWebhookAttempt, Run, Thread
from agentseek_api.models.auth import User


def _as_utc(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


async def _fetch_cron(cron_id: str) -> CronJob | None:
    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        return await session.scalar(select(CronJob).where(CronJob.cron_id == cron_id))


async def _list_ticks_for_cron(cron_id: str) -> list[CronTick]:
    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        return list((await session.scalars(select(CronTick).where(CronTick.cron_id == cron_id).order_by(CronTick.id.asc()))).all())


async def _list_webhook_attempts_for_tick(tick_id: int) -> list[CronWebhookAttempt]:
    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        return list(
            (
                await session.scalars(
                    select(CronWebhookAttempt)
                    .where(CronWebhookAttempt.tick_id == tick_id)
                    .order_by(CronWebhookAttempt.attempt_number.asc())
                )
            ).all()
        )


async def _mark_cron_due(cron_id: str, *, when: datetime) -> None:
    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        cron = await session.scalar(select(CronJob).where(CronJob.cron_id == cron_id))
        assert cron is not None
        cron.next_run_at = when
        await session.commit()


async def _age_tick(*, cron_id: str, stale_at: datetime) -> None:
    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        tick = await session.scalar(select(CronTick).where(CronTick.cron_id == cron_id))
        assert tick is not None
        tick.created_at = stale_at
        tick.updated_at = stale_at
        await session.commit()


async def _set_tick_delivery_state(
    *,
    cron_id: str,
    delivery_status: str,
    updated_at: datetime,
) -> None:
    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        tick = await session.scalar(select(CronTick).where(CronTick.cron_id == cron_id))
        assert tick is not None
        tick.webhook_delivery_status = delivery_status
        tick.updated_at = updated_at
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


async def _set_cron_webhook(cron_id: str, *, webhook: str, max_attempts: int = 3) -> None:
    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        cron = await session.scalar(select(CronJob).where(CronJob.cron_id == cron_id))
        assert cron is not None
        cron.webhook = webhook
        cron.max_webhook_attempts = max_attempts
        await session.commit()


async def _set_cron_end_time(cron_id: str, *, end_time: datetime | None) -> None:
    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        cron = await session.scalar(select(CronJob).where(CronJob.cron_id == cron_id))
        assert cron is not None
        cron.end_time = end_time
        await session.commit()


def _create_assistant(client: TestClient) -> str:
    response = client.post("/assistants", json={"name": "scheduler-assistant", "graph_id": "default"})
    assert response.status_code == 200
    return response.json()["assistant_id"]


def _create_thread(client: TestClient, *, user_id: str) -> str:
    response = client.post("/threads", json={"metadata": {"scope": "scheduler"}}, headers={"x-user-id": user_id})
    assert response.status_code == 200
    return response.json()["thread_id"]


def test_dispatch_due_crons_creates_stateless_run_and_skips_busy_thread(client: TestClient) -> None:
    from agentseek_api.services import cron_scheduler as cron_scheduler_module

    assistant_id = _create_assistant(client)
    thread_id = _create_thread(client, user_id="owner")

    stateless = client.post(
        "/runs/crons",
        json={
            "assistant_id": assistant_id,
            "schedule": "FREQ=MINUTELY;INTERVAL=1",
            "input": {"kind": "stateless"},
            "metadata": {"source": "scheduler-runtime"},
            "config": {"model": "gpt-test"},
            "context": {"tenant": "acme"},
            "on_run_completed": "keep",
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

    results = asyncio.run(cron_scheduler_module.dispatch_due_crons(limit=10, scheduler_id="scheduler-1", now=due_at))

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
    persisted_metadata = created_runs[0].metadata_json
    assert persisted_metadata is not None
    assert persisted_metadata.get("source") == "scheduler-runtime"
    assert persisted_metadata.get("cron_id") == stateless.json()["cron_id"]
    assert persisted_metadata.get("scheduled_for") == due_at.isoformat()
    assert isinstance(persisted_metadata.get("__agentseek_checkpoint_id"), str)
    assert created_runs[0].kwargs_json == {"config": {"model": "gpt-test"}, "context": {"tenant": "acme"}, "stream_modes": ["values"]}

    busy_thread_runs = asyncio.run(_list_runs_for_thread(thread_id))
    assert busy_thread_runs == []

    persisted_stateless = asyncio.run(_fetch_cron(stateless.json()["cron_id"]))
    persisted_thread_bound = asyncio.run(_fetch_cron(thread_bound.json()["cron_id"]))
    stateless_ticks = asyncio.run(_list_ticks_for_cron(stateless.json()["cron_id"]))
    thread_bound_ticks = asyncio.run(_list_ticks_for_cron(thread_bound.json()["cron_id"]))
    assert persisted_stateless is not None
    assert persisted_thread_bound is not None
    assert _as_utc(persisted_stateless.next_run_at) > due_at
    assert _as_utc(persisted_thread_bound.next_run_at) > due_at
    assert persisted_stateless.last_tick_status == "success"
    assert persisted_stateless.last_run_at is not None
    assert persisted_stateless.last_error is None
    assert _as_utc(persisted_stateless.last_run_at) >= _as_utc(stateless_ticks[0].updated_at)
    assert persisted_thread_bound.last_tick_status == "skipped"
    assert persisted_thread_bound.last_run_at is None
    assert persisted_thread_bound.last_error is None
    assert len(stateless_ticks) == 1
    assert stateless_ticks[0].status == "success"
    assert stateless_ticks[0].run_id == created_runs[0].run_id
    assert stateless_ticks[0].skip_reason is None
    assert len(thread_bound_ticks) == 1
    assert thread_bound_ticks[0].status == "skipped"
    assert thread_bound_ticks[0].run_id is None
    assert thread_bound_ticks[0].skip_reason == "thread_busy"


def test_dispatch_due_crons_persists_submission_error_and_continues(client: TestClient, monkeypatch) -> None:
    from agentseek_api.services import cron_scheduler as cron_scheduler_module

    assistant_id = _create_assistant(client)
    failing_thread_id = _create_thread(client, user_id="owner")

    failing = client.post(
        f"/threads/{failing_thread_id}/runs/crons",
        json={
            "assistant_id": assistant_id,
            "schedule": "FREQ=MINUTELY;INTERVAL=1",
            "input": {"kind": "will-fail"},
        },
        headers={"x-user-id": "owner"},
    )
    assert failing.status_code == 200

    stateless = client.post(
        "/runs/crons",
        json={
            "assistant_id": assistant_id,
            "schedule": "FREQ=MINUTELY;INTERVAL=1",
            "input": {"kind": "will-pass"},
        },
        headers={"x-user-id": "owner"},
    )
    assert stateless.status_code == 200

    due_at = datetime.now(UTC) - timedelta(minutes=1)
    asyncio.run(_mark_cron_due(failing.json()["cron_id"], when=due_at))
    asyncio.run(_mark_cron_due(stateless.json()["cron_id"], when=due_at))

    original_submit = cron_scheduler_module.submit_existing_run

    async def flaky_submit_existing_run(*, run_id: str, **kwargs):
        session_factory = db_manager.get_session_factory()
        async with session_factory() as session:
            run = await session.scalar(select(Run).where(Run.run_id == run_id))
        assert run is not None
        if run.thread_id == failing_thread_id:
            raise RuntimeError("submit boom")
        return await original_submit(run_id=run_id, **kwargs)

    monkeypatch.setattr(cron_scheduler_module, "submit_existing_run", flaky_submit_existing_run)

    results = asyncio.run(cron_scheduler_module.dispatch_due_crons(limit=10, scheduler_id="scheduler-1", now=due_at))

    assert len(results) == 2
    error_results = [result for result in results if result.status == "error"]
    queued_results = [result for result in results if result.status == "queued"]
    assert len(error_results) == 1
    assert error_results[0].thread_id == failing_thread_id
    assert len(queued_results) == 1

    failing_ticks = asyncio.run(_list_ticks_for_cron(failing.json()["cron_id"]))
    passing_ticks = asyncio.run(_list_ticks_for_cron(stateless.json()["cron_id"]))
    assert len(failing_ticks) == 1
    assert failing_ticks[0].status == "error"
    assert failing_ticks[0].run_id is not None
    assert failing_ticks[0].skip_reason == "submit boom"
    persisted_failing = asyncio.run(_fetch_cron(failing.json()["cron_id"]))
    assert persisted_failing is not None
    assert persisted_failing.last_error == "submit boom"
    assert len(passing_ticks) == 1
    assert passing_ticks[0].status == "success"


def test_dispatch_due_crons_reconciles_terminal_ticks_and_persists_webhook_attempts(
    client: TestClient,
    monkeypatch,
) -> None:
    from agentseek_api.services import cron_scheduler as cron_scheduler_module

    class _FakeResponse:
        def __init__(self, status_code: int) -> None:
            self.status_code = status_code

    class FakeWebhookClient:
        def __init__(self) -> None:
            self.calls_by_url: dict[str, int] = {}

        async def post(self, url: str, json: dict[str, object]) -> _FakeResponse:
            attempt = self.calls_by_url.get(url, 0) + 1
            self.calls_by_url[url] = attempt
            if url.endswith("/success") and attempt < 3:
                return _FakeResponse(500)
            return _FakeResponse(200)

    async def _no_sleep(_: float) -> None:
        return None

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

    asyncio.run(_set_cron_webhook(stateless.json()["cron_id"], webhook="https://example.com/success", max_attempts=3))
    asyncio.run(_set_cron_webhook(thread_bound.json()["cron_id"], webhook="https://example.com/skipped", max_attempts=2))

    due_at = datetime.now(UTC) - timedelta(minutes=1)
    asyncio.run(_mark_cron_due(stateless.json()["cron_id"], when=due_at))
    asyncio.run(_mark_cron_due(thread_bound.json()["cron_id"], when=due_at))
    asyncio.run(_mark_thread_busy(thread_id))

    fake_http_client = FakeWebhookClient()
    monkeypatch.setattr(cron_scheduler_module, "get_webhook_http_client", lambda: fake_http_client)

    results = asyncio.run(
        cron_scheduler_module.dispatch_due_crons(
            limit=10,
            scheduler_id="scheduler-1",
            now=due_at,
            webhook_sleep=_no_sleep,
        )
    )

    assert len(results) == 2

    stateless_ticks = asyncio.run(_list_ticks_for_cron(stateless.json()["cron_id"]))
    thread_bound_ticks = asyncio.run(_list_ticks_for_cron(thread_bound.json()["cron_id"]))
    assert len(stateless_ticks) == 1
    assert stateless_ticks[0].status == "success"
    assert stateless_ticks[0].webhook_delivery_status == "delivered"
    assert stateless_ticks[0].webhook_attempt_count == 3
    assert stateless_ticks[0].webhook_last_status_code == 200
    assert len(thread_bound_ticks) == 1
    assert thread_bound_ticks[0].status == "skipped"
    assert thread_bound_ticks[0].webhook_delivery_status == "delivered"
    assert thread_bound_ticks[0].webhook_attempt_count == 1
    assert thread_bound_ticks[0].webhook_last_status_code == 200

    stateless_attempts = asyncio.run(_list_webhook_attempts_for_tick(stateless_ticks[0].id))
    skipped_attempts = asyncio.run(_list_webhook_attempts_for_tick(thread_bound_ticks[0].id))
    assert [attempt.attempt_number for attempt in stateless_attempts] == [1, 2, 3]
    assert [attempt.status_code for attempt in stateless_attempts] == [500, 500, 200]
    assert [attempt.attempt_number for attempt in skipped_attempts] == [1]
    assert [attempt.status_code for attempt in skipped_attempts] == [200]


def test_claim_due_crons_reowned_tick_cannot_be_immediately_reclaimed_again(
    client: TestClient,
) -> None:
    from agentseek_api.services import cron_scheduler as cron_scheduler_module

    assistant_id = _create_assistant(client)
    created = client.post(
        "/runs/crons",
        json={
            "assistant_id": assistant_id,
            "schedule": "FREQ=MINUTELY;INTERVAL=1",
            "input": {"kind": "reclaim"},
        },
        headers={"x-user-id": "owner"},
    )
    assert created.status_code == 200

    due_at = datetime.now(UTC).replace(second=0, microsecond=0) - timedelta(minutes=1)
    asyncio.run(_mark_cron_due(created.json()["cron_id"], when=due_at))

    first = asyncio.run(
        cron_scheduler_module.claim_due_crons(limit=10, scheduler_id="scheduler-a", now=due_at)
    )
    assert len(first) == 1

    stale_at = due_at - timedelta(seconds=31)
    asyncio.run(_age_tick(cron_id=created.json()["cron_id"], stale_at=stale_at))

    reclaimed_at = due_at + timedelta(seconds=31)
    second = asyncio.run(
        cron_scheduler_module.claim_due_crons(limit=10, scheduler_id="scheduler-b", now=reclaimed_at)
    )
    third = asyncio.run(
        cron_scheduler_module.claim_due_crons(limit=10, scheduler_id="scheduler-c", now=reclaimed_at)
    )

    assert [item.tick_id for item in second] == [first[0].tick_id]
    assert third == []

    ticks = asyncio.run(_list_ticks_for_cron(created.json()["cron_id"]))
    assert len(ticks) == 1
    assert ticks[0].scheduler_id == "scheduler-b"
    assert _as_utc(ticks[0].updated_at) >= reclaimed_at


def test_dispatch_due_crons_reclaims_stale_queued_tick_without_creating_duplicate_run(
    client: TestClient,
) -> None:
    from agentseek_api.services import cron_scheduler as cron_scheduler_module
    from agentseek_api.services.run_preparation import prepare_run

    assistant_id = _create_assistant(client)
    thread_id = _create_thread(client, user_id="owner")
    created = client.post(
        f"/threads/{thread_id}/runs/crons",
        json={
            "assistant_id": assistant_id,
            "schedule": "FREQ=MINUTELY;INTERVAL=1",
            "input": {"kind": "reclaim-queued"},
        },
        headers={"x-user-id": "owner"},
    )
    assert created.status_code == 200

    due_at = datetime.now(UTC).replace(second=0, microsecond=0) - timedelta(minutes=1)
    asyncio.run(_mark_cron_due(created.json()["cron_id"], when=due_at))
    claim = asyncio.run(cron_scheduler_module.claim_due_crons(limit=10, scheduler_id="scheduler-a", now=due_at))[0]

    run, _graph_id = asyncio.run(
        prepare_run(
            thread_id=thread_id,
            assistant_id=assistant_id,
            payload=claim.input_json,
            user=User(identity="owner", is_authenticated=True),
            metadata={
                **claim.metadata_json,
                "cron_id": claim.cron_id,
                "scheduled_for": due_at.isoformat(),
            },
            kwargs=claim.kwargs_json,
            tick_id=claim.tick_id,
        )
    )

    stale_at = due_at - timedelta(seconds=31)
    asyncio.run(_age_tick(cron_id=created.json()["cron_id"], stale_at=stale_at))

    results = asyncio.run(
        cron_scheduler_module.dispatch_due_crons(
            limit=10,
            scheduler_id="scheduler-b",
            now=due_at + timedelta(seconds=31),
        )
    )

    assert len(results) == 1
    assert results[0].status == "queued"
    assert results[0].run_id == run.run_id

    ticks = asyncio.run(_list_ticks_for_cron(created.json()["cron_id"]))
    runs = asyncio.run(_list_runs_for_thread(thread_id))
    assert len(ticks) == 1
    assert ticks[0].run_id == run.run_id
    assert ticks[0].status == "success"
    assert len(runs) == 1
    assert runs[0].run_id == run.run_id


def test_reconcile_terminal_ticks_reserves_webhook_delivery_before_retry_loop(
    client: TestClient,
    monkeypatch,
) -> None:
    from agentseek_api.services import cron_scheduler as cron_scheduler_module

    class _FakeResponse:
        def __init__(self, status_code: int) -> None:
            self.status_code = status_code

    class FakeWebhookClient:
        def __init__(self) -> None:
            self.post_calls = 0

        async def post(self, url: str, json: dict[str, object]) -> _FakeResponse:
            self.post_calls += 1
            return _FakeResponse(200)

    async def _no_sleep(_: float) -> None:
        return None

    assistant_id = _create_assistant(client)
    created = client.post(
        "/runs/crons",
        json={
            "assistant_id": assistant_id,
            "schedule": "FREQ=MINUTELY;INTERVAL=1",
            "input": {"kind": "idempotent-webhook"},
        },
        headers={"x-user-id": "owner"},
    )
    assert created.status_code == 200

    due_at = datetime.now(UTC) - timedelta(minutes=1)
    asyncio.run(_mark_cron_due(created.json()["cron_id"], when=due_at))
    asyncio.run(
        cron_scheduler_module.dispatch_due_crons(limit=10, scheduler_id="scheduler-1", now=due_at)
    )
    asyncio.run(_set_cron_webhook(created.json()["cron_id"], webhook="https://example.com/once", max_attempts=2))

    fake_http_client = FakeWebhookClient()
    real_deliver = cron_scheduler_module.deliver_webhook_with_retries
    delivery_calls: list[int] = []

    async def wrapped_deliver(**kwargs):
        delivery_calls.append(kwargs["tick_id"])
        if len(delivery_calls) == 1:
            await cron_scheduler_module._reconcile_terminal_ticks(
                http_client=kwargs["http_client"],
                sleep=_no_sleep,
            )
        return await real_deliver(**kwargs)

    monkeypatch.setattr(cron_scheduler_module, "get_webhook_http_client", lambda: fake_http_client)
    monkeypatch.setattr(cron_scheduler_module, "deliver_webhook_with_retries", wrapped_deliver)

    asyncio.run(
        cron_scheduler_module._reconcile_terminal_ticks(
            http_client=fake_http_client,
            sleep=_no_sleep,
        )
    )

    ticks = asyncio.run(_list_ticks_for_cron(created.json()["cron_id"]))
    attempts = asyncio.run(_list_webhook_attempts_for_tick(ticks[0].id))
    assert delivery_calls == [ticks[0].id]
    assert fake_http_client.post_calls == 1
    assert len(attempts) == 1
    assert attempts[0].attempt_number == 1
    assert attempts[0].status_code == 200


def test_reconcile_terminal_ticks_reclaims_stale_delivering_webhook_reservation(
    client: TestClient,
    monkeypatch,
) -> None:
    from agentseek_api.services import cron_scheduler as cron_scheduler_module

    class _FakeResponse:
        def __init__(self, status_code: int) -> None:
            self.status_code = status_code

    class FakeWebhookClient:
        def __init__(self) -> None:
            self.post_calls = 0

        async def post(self, url: str, json: dict[str, object]) -> _FakeResponse:
            self.post_calls += 1
            return _FakeResponse(200)

    async def _no_sleep(_: float) -> None:
        return None

    assistant_id = _create_assistant(client)
    created = client.post(
        "/runs/crons",
        json={
            "assistant_id": assistant_id,
            "schedule": "FREQ=MINUTELY;INTERVAL=1",
            "input": {"kind": "stale-delivering"},
        },
        headers={"x-user-id": "owner"},
    )
    assert created.status_code == 200

    due_at = datetime.now(UTC) - timedelta(minutes=1)
    asyncio.run(_mark_cron_due(created.json()["cron_id"], when=due_at))
    asyncio.run(
        cron_scheduler_module.dispatch_due_crons(limit=10, scheduler_id="scheduler-1", now=due_at)
    )
    stale_at = due_at - timedelta(seconds=31)
    asyncio.run(_set_cron_webhook(created.json()["cron_id"], webhook="https://example.com/reclaim", max_attempts=2))
    asyncio.run(
        _set_tick_delivery_state(
            cron_id=created.json()["cron_id"],
            delivery_status="delivering",
            updated_at=stale_at,
        )
    )

    fake_http_client = FakeWebhookClient()
    monkeypatch.setattr(cron_scheduler_module, "get_webhook_http_client", lambda: fake_http_client)

    asyncio.run(
        cron_scheduler_module._reconcile_terminal_ticks(
            http_client=fake_http_client,
            sleep=_no_sleep,
        )
    )

    ticks = asyncio.run(_list_ticks_for_cron(created.json()["cron_id"]))
    attempts = asyncio.run(_list_webhook_attempts_for_tick(ticks[0].id))
    assert fake_http_client.post_calls == 1
    assert ticks[0].webhook_delivery_status == "delivered"
    assert len(attempts) == 1
    assert attempts[0].attempt_number == 1


def test_dispatch_due_crons_skips_cron_past_end_time(client: TestClient) -> None:
    from agentseek_api.services import cron_scheduler as cron_scheduler_module

    assistant_id = _create_assistant(client)
    created = client.post(
        "/runs/crons",
        json={"assistant_id": assistant_id, "schedule": "FREQ=MINUTELY;INTERVAL=1", "input": {"kind": "past-end-time"}},
        headers={"x-user-id": "owner"},
    )
    assert created.status_code == 200
    cron_id = created.json()["cron_id"]

    due_at = datetime.now(UTC) - timedelta(minutes=1)
    asyncio.run(_mark_cron_due(cron_id, when=due_at))
    asyncio.run(_set_cron_end_time(cron_id, end_time=due_at - timedelta(minutes=5)))

    results = asyncio.run(cron_scheduler_module.dispatch_due_crons(limit=10, scheduler_id="scheduler-1", now=due_at))

    assert results == []
    ticks = asyncio.run(_list_ticks_for_cron(cron_id))
    assert ticks == []


def test_reclaim_started_tick_skipped_when_cron_past_end_time(client: TestClient) -> None:
    """A stale 'started' tick (dispatch never ran) must NOT be reclaimed and
    re-dispatched once the cron's end_time has passed."""
    from agentseek_api.services import cron_scheduler as cron_scheduler_module

    assistant_id = _create_assistant(client)
    created = client.post(
        "/runs/crons",
        json={"assistant_id": assistant_id, "schedule": "FREQ=MINUTELY;INTERVAL=1", "input": {"kind": "reclaim-end"}},
        headers={"x-user-id": "owner"},
    )
    assert created.status_code == 200
    cron_id = created.json()["cron_id"]

    due_at = datetime.now(UTC).replace(second=0, microsecond=0) - timedelta(minutes=1)
    asyncio.run(_mark_cron_due(cron_id, when=due_at))

    first = asyncio.run(cron_scheduler_module.claim_due_crons(limit=10, scheduler_id="scheduler-a", now=due_at))
    assert len(first) == 1  # creates a 'started' tick

    # Tick goes stale, and the cron's end_time passes before dispatch reclaims it.
    asyncio.run(_age_tick(cron_id=cron_id, stale_at=due_at - timedelta(seconds=31)))
    asyncio.run(_set_cron_end_time(cron_id, end_time=due_at - timedelta(minutes=5)))

    reclaimed_at = due_at + timedelta(seconds=31)
    second = asyncio.run(cron_scheduler_module.claim_due_crons(limit=10, scheduler_id="scheduler-b", now=reclaimed_at))

    assert second == []
    ticks = asyncio.run(_list_ticks_for_cron(cron_id))
    assert len(ticks) == 1
    assert ticks[0].status == "skipped"
    assert ticks[0].skip_reason == "past_end_time"


def test_dispatch_due_crons_disables_cron_when_next_run_crosses_end_time(client: TestClient) -> None:
    from agentseek_api.services import cron_scheduler as cron_scheduler_module

    assistant_id = _create_assistant(client)
    created = client.post(
        "/runs/crons",
        json={"assistant_id": assistant_id, "schedule": "FREQ=MINUTELY;INTERVAL=1", "input": {"kind": "cross-end-time"}},
        headers={"x-user-id": "owner"},
    )
    assert created.status_code == 200
    cron_id = created.json()["cron_id"]

    due_at = datetime.now(UTC).replace(second=0, microsecond=0) - timedelta(minutes=1)
    asyncio.run(_mark_cron_due(cron_id, when=due_at))
    # end_time is after this fire (so it fires now) but before the next computed run_at.
    asyncio.run(_set_cron_end_time(cron_id, end_time=due_at + timedelta(seconds=30)))

    results = asyncio.run(cron_scheduler_module.dispatch_due_crons(limit=10, scheduler_id="scheduler-1", now=due_at))

    assert len(results) == 1
    assert results[0].status == "queued"

    persisted = asyncio.run(_fetch_cron(cron_id))
    assert persisted is not None
    assert persisted.enabled is False
    assert _as_utc(persisted.next_run_at) > _as_utc(persisted.end_time)


def test_dispatch_due_crons_passes_multitask_strategy_to_thread_run(client: TestClient) -> None:
    from agentseek_api.services import cron_scheduler as cron_scheduler_module

    assistant_id = _create_assistant(client)
    thread_id = _create_thread(client, user_id="owner")

    created = client.post(
        f"/threads/{thread_id}/runs/crons",
        json={
            "assistant_id": assistant_id,
            "schedule": "FREQ=MINUTELY;INTERVAL=1",
            "input": {"kind": "thread-bound"},
            "multitask_strategy": "interrupt",
        },
        headers={"x-user-id": "owner"},
    )
    assert created.status_code == 200

    due_at = datetime.now(UTC) - timedelta(minutes=1)
    asyncio.run(_mark_cron_due(created.json()["cron_id"], when=due_at))

    results = asyncio.run(cron_scheduler_module.dispatch_due_crons(limit=10, scheduler_id="scheduler-1", now=due_at))

    assert len(results) == 1
    assert results[0].status == "queued"
    assert results[0].thread_id == thread_id

    runs = asyncio.run(_list_runs_for_thread(thread_id))
    assert len(runs) == 1
    assert runs[0].multitask_strategy == "interrupt"


async def _fetch_thread(thread_id: str) -> Thread | None:
    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        return await session.scalar(select(Thread).where(Thread.thread_id == thread_id))


def test_dispatch_due_crons_deletes_stateless_thread_on_run_completed_delete(client: TestClient) -> None:
    from agentseek_api.services import cron_scheduler as cron_scheduler_module

    assistant_id = _create_assistant(client)
    created = client.post(
        "/runs/crons",
        json={
            "assistant_id": assistant_id,
            "schedule": "FREQ=MINUTELY;INTERVAL=1",
            "input": {"kind": "stateless-delete"},
            "on_run_completed": "delete",
        },
        headers={"x-user-id": "owner"},
    )
    assert created.status_code == 200
    cron_id = created.json()["cron_id"]

    due_at = datetime.now(UTC) - timedelta(minutes=1)
    asyncio.run(_mark_cron_due(cron_id, when=due_at))

    results = asyncio.run(cron_scheduler_module.dispatch_due_crons(limit=10, scheduler_id="scheduler-1", now=due_at))
    assert len(results) == 1
    assert results[0].status == "queued"

    user_threads = asyncio.run(_list_threads_for_user("owner"))
    stateless_threads = [t for t in user_threads if t.metadata_json.get("cron_id") == cron_id]
    assert stateless_threads == []

    ticks = asyncio.run(_list_ticks_for_cron(cron_id))
    assert len(ticks) == 1
    assert ticks[0].status == "success"


def test_dispatch_due_crons_keeps_stateless_thread_on_run_completed_keep(client: TestClient) -> None:
    from agentseek_api.services import cron_scheduler as cron_scheduler_module

    assistant_id = _create_assistant(client)
    created = client.post(
        "/runs/crons",
        json={
            "assistant_id": assistant_id,
            "schedule": "FREQ=MINUTELY;INTERVAL=1",
            "input": {"kind": "stateless-keep"},
            "on_run_completed": "keep",
        },
        headers={"x-user-id": "owner"},
    )
    assert created.status_code == 200
    cron_id = created.json()["cron_id"]

    due_at = datetime.now(UTC) - timedelta(minutes=1)
    asyncio.run(_mark_cron_due(cron_id, when=due_at))

    results = asyncio.run(cron_scheduler_module.dispatch_due_crons(limit=10, scheduler_id="scheduler-1", now=due_at))
    assert len(results) == 1
    assert results[0].status == "queued"

    user_threads = asyncio.run(_list_threads_for_user("owner"))
    stateless_threads = [t for t in user_threads if t.metadata_json.get("cron_id") == cron_id]
    assert len(stateless_threads) == 1
    persisted = asyncio.run(_fetch_thread(stateless_threads[0].thread_id))
    assert persisted is not None


def test_reconcile_does_not_delete_interrupted_stateless_run(client: TestClient) -> None:
    """on_run_completed='delete' must NOT delete a thread whose run is
    interrupted: interrupted runs are resumable, and deleting them makes the
    human-in-the-loop workflow unrecoverable."""
    from agentseek_api.services import cron_scheduler as cron_scheduler_module

    assistant_id = _create_assistant(client)
    created = client.post(
        "/runs/crons",
        json={
            "assistant_id": assistant_id,
            "schedule": "FREQ=MINUTELY;INTERVAL=1",
            "input": {"kind": "stateless-interrupt"},
            "on_run_completed": "delete",
        },
        headers={"x-user-id": "owner"},
    )
    assert created.status_code == 200
    cron_id = created.json()["cron_id"]

    # Seed a stateless ephemeral thread + interrupted run + a queued tick that
    # points at it, mirroring what dispatch produces when a graph interrupts.
    async def _seed() -> tuple[str, str]:
        session_factory = db_manager.get_session_factory()
        async with session_factory() as session:
            thread = Thread(
                user_id="owner",
                metadata_json={"stateless": True, "cron_id": cron_id},
                status="interrupted",
            )
            session.add(thread)
            await session.flush()
            run = Run(
                thread_id=thread.thread_id,
                assistant_id=assistant_id,
                user_id="owner",
                status="interrupted",
                input_json={"kind": "stateless-interrupt"},
            )
            session.add(run)
            await session.flush()
            tick = CronTick(
                cron_id=cron_id,
                thread_id=thread.thread_id,
                run_id=run.run_id,
                scheduler_id="scheduler-1",
                scheduled_for=datetime.now(UTC) - timedelta(minutes=1),
                status="queued",
            )
            session.add(tick)
            await session.commit()
            return thread.thread_id, run.run_id

    thread_id, run_id = asyncio.run(_seed())

    async def _no_sleep(_: float) -> None:
        return None

    asyncio.run(cron_scheduler_module._reconcile_terminal_ticks(sleep=_no_sleep))

    # Tick reconciled to interrupted, but the thread + run must survive.
    ticks = asyncio.run(_list_ticks_for_cron(cron_id))
    assert len(ticks) == 1
    assert ticks[0].status == "interrupted"
    persisted_thread = asyncio.run(_fetch_thread(thread_id))
    assert persisted_thread is not None
    runs = asyncio.run(_list_runs_for_thread(thread_id))
    assert [r.run_id for r in runs] == [run_id]


def test_dispatch_due_crons_never_deletes_caller_owned_thread(client: TestClient) -> None:
    from agentseek_api.services import cron_scheduler as cron_scheduler_module

    assistant_id = _create_assistant(client)
    thread_id = _create_thread(client, user_id="owner")
    created = client.post(
        f"/threads/{thread_id}/runs/crons",
        json={"assistant_id": assistant_id, "schedule": "FREQ=MINUTELY;INTERVAL=1", "input": {"kind": "thread-bound"}},
        headers={"x-user-id": "owner"},
    )
    assert created.status_code == 200
    cron_id = created.json()["cron_id"]

    due_at = datetime.now(UTC) - timedelta(minutes=1)
    asyncio.run(_mark_cron_due(cron_id, when=due_at))

    results = asyncio.run(cron_scheduler_module.dispatch_due_crons(limit=10, scheduler_id="scheduler-1", now=due_at))
    assert len(results) == 1
    assert results[0].status == "queued"

    persisted = asyncio.run(_fetch_thread(thread_id))
    assert persisted is not None
