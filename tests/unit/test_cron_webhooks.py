from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import select

from agentseek_api.core.database import db_manager
from agentseek_api.core.orm import Assistant, CronJob, CronTick, CronWebhookAttempt


class _FakeCheckpointer:
    def __init__(self, connection_args: dict[str, str]) -> None:
        self.connection_args = connection_args

    def setup(self) -> None:
        return None


class _FakeResponse:
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code


class FakeWebhookClient:
    def __init__(self) -> None:
        self.failures_before_success = 0
        self.calls: list[tuple[str, dict[str, object]]] = []

    async def post(self, url: str, json: dict[str, object]) -> _FakeResponse:
        self.calls.append((url, json))
        if self.failures_before_success > 0:
            self.failures_before_success -= 1
            return _FakeResponse(500)
        return _FakeResponse(200)


async def _no_sleep(_: float) -> None:
    return None


@pytest.fixture
async def persisted_tick(monkeypatch: pytest.MonkeyPatch, tmp_path):
    from agentseek_api.settings import settings

    monkeypatch.setattr(settings, "SEEKDB_URL", f"sqlite+aiosqlite:///{tmp_path}/cron-webhooks.db")
    monkeypatch.setattr("agentseek_api.core.database.OceanBaseCheckpointSaver", _FakeCheckpointer)
    await db_manager.close()
    await db_manager.initialize()

    session_factory = db_manager.get_session_factory()
    due_at = datetime.now(UTC) - timedelta(minutes=1)
    async with session_factory() as session:
        assistant = Assistant(name="cron-webhooks", graph_id="default")
        session.add(assistant)
        await session.flush()
        cron = CronJob(
            assistant_id=assistant.assistant_id,
            thread_id=None,
            user_id="owner",
            schedule="FREQ=MINUTELY;INTERVAL=1",
            enabled=True,
            input_json={"kind": "unit"},
            next_run_at=due_at,
            webhook="https://example.com/hook",
        )
        session.add(cron)
        await session.flush()
        tick = CronTick(
            cron_id=cron.cron_id,
            thread_id=None,
            run_id="run-1",
            scheduler_id="scheduler-1",
            scheduled_for=due_at,
            status="success",
        )
        session.add(tick)
        await session.commit()
        await session.refresh(tick)

    try:
        yield tick
    finally:
        await db_manager.close()


@pytest.mark.asyncio
async def test_deliver_webhook_with_retries_records_each_attempt(persisted_tick: CronTick) -> None:
    from agentseek_api.services.cron_webhooks import deliver_webhook_with_retries

    fake_http_client = FakeWebhookClient()
    fake_http_client.failures_before_success = 2

    result = await deliver_webhook_with_retries(
        webhook_url="https://example.com/hook",
        payload={"cron_id": "c1", "status": "success"},
        tick_id=persisted_tick.id,
        max_attempts=3,
        http_client=fake_http_client,
        sleep=_no_sleep,
    )

    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        attempts = list(
            (
                await session.scalars(
                    select(CronWebhookAttempt)
                    .where(CronWebhookAttempt.tick_id == persisted_tick.id)
                    .order_by(CronWebhookAttempt.attempt_number.asc())
                )
            ).all()
        )
        tick = await session.scalar(select(CronTick).where(CronTick.id == persisted_tick.id))

    assert result.delivered is True
    assert result.attempt_count == 3
    assert result.status_code == 200
    assert len(attempts) == 3
    assert [attempt.attempt_number for attempt in attempts] == [1, 2, 3]
    assert [attempt.status_code for attempt in attempts] == [500, 500, 200]
    assert tick is not None
    assert tick.webhook_delivery_status == "delivered"
    assert tick.webhook_attempt_count == 3
    assert tick.webhook_last_status_code == 200


@pytest.mark.asyncio
async def test_deliver_webhook_with_retries_persists_terminal_failure_metadata(
    persisted_tick: CronTick,
) -> None:
    from agentseek_api.services.cron_webhooks import deliver_webhook_with_retries

    fake_http_client = FakeWebhookClient()
    fake_http_client.failures_before_success = 3

    result = await deliver_webhook_with_retries(
        webhook_url="https://example.com/hook",
        payload={"cron_id": "c1", "status": "success"},
        tick_id=persisted_tick.id,
        max_attempts=3,
        http_client=fake_http_client,
        sleep=_no_sleep,
    )

    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        attempts = list(
            (
                await session.scalars(
                    select(CronWebhookAttempt)
                    .where(CronWebhookAttempt.tick_id == persisted_tick.id)
                    .order_by(CronWebhookAttempt.attempt_number.asc())
                )
            ).all()
        )
        tick = await session.scalar(select(CronTick).where(CronTick.id == persisted_tick.id))

    assert result.delivered is False
    assert result.attempt_count == 3
    assert result.status_code == 500
    assert result.error == "HTTP 500"
    assert len(attempts) == 3
    assert [attempt.attempt_number for attempt in attempts] == [1, 2, 3]
    assert [attempt.status_code for attempt in attempts] == [500, 500, 500]
    assert attempts[-1].error == "HTTP 500"
    assert tick is not None
    assert tick.webhook_delivery_status == "failed"
    assert tick.webhook_attempt_count == 3
    assert tick.webhook_last_status_code == 500
    assert tick.webhook_last_error == "HTTP 500"
