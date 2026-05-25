from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol

import httpx
from sqlalchemy import func, select

from agentseek_api.core.database import db_manager
from agentseek_api.core.orm import CronJob, CronTick, CronWebhookAttempt


class AsyncWebhookClient(Protocol):
    async def post(self, url: str, json: dict[str, object]) -> Any:
        ...


@dataclass(frozen=True, slots=True)
class WebhookDeliveryResult:
    delivered: bool
    attempt_count: int
    status_code: int | None
    error: str | None = None


class HttpxWebhookClient:
    def __init__(self, *, timeout: float = 10.0) -> None:
        self._timeout = timeout

    async def post(self, url: str, json: dict[str, object]) -> httpx.Response:
        async with httpx.AsyncClient(timeout=self._timeout, trust_env=False) as client:
            return await client.post(url, json=json)


def get_webhook_http_client() -> AsyncWebhookClient:
    return HttpxWebhookClient()


def build_webhook_payload(*, cron: CronJob, tick: CronTick) -> dict[str, object]:
    payload: dict[str, object] = {
        "cron_id": cron.cron_id,
        "tick_id": tick.id,
        "status": tick.status,
        "scheduled_for": tick.scheduled_for.isoformat(),
    }
    if tick.run_id is not None:
        payload["run_id"] = tick.run_id
    if tick.thread_id is not None:
        payload["thread_id"] = tick.thread_id
    if tick.skip_reason is not None:
        payload["skip_reason"] = tick.skip_reason
    return payload


async def _persist_webhook_attempt(
    *,
    tick_id: int,
    attempt_number: int,
    status_code: int | None,
    error: str | None,
) -> None:
    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        session.add(
            CronWebhookAttempt(
                tick_id=tick_id,
                attempt_number=attempt_number,
                status_code=status_code,
                error=error,
            )
        )
        tick = await session.scalar(select(CronTick).where(CronTick.id == tick_id))
        if tick is None:
            raise RuntimeError(f"Cron tick {tick_id} not found")
        tick.webhook_attempt_count = attempt_number
        tick.webhook_last_status_code = status_code
        tick.webhook_last_error = error
        await session.commit()


async def _persist_webhook_result(
    *,
    tick_id: int,
    delivered: bool,
    status_code: int | None,
    error: str | None,
    attempt_count: int,
) -> None:
    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        tick = await session.scalar(select(CronTick).where(CronTick.id == tick_id))
        if tick is None:
            raise RuntimeError(f"Cron tick {tick_id} not found")
        tick.webhook_delivery_status = "delivered" if delivered else "failed"
        tick.webhook_attempt_count = attempt_count
        tick.webhook_last_status_code = status_code
        tick.webhook_last_error = error
        tick.webhook_delivered_at = datetime.now(UTC) if delivered else None
        await session.commit()


async def _load_existing_attempt_count(*, tick_id: int) -> int:
    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        count = await session.scalar(
            select(func.max(CronWebhookAttempt.attempt_number)).where(CronWebhookAttempt.tick_id == tick_id)
        )
    return int(count or 0)


async def deliver_webhook_with_retries(
    *,
    webhook_url: str,
    payload: dict[str, object],
    tick_id: int,
    max_attempts: int,
    http_client: AsyncWebhookClient,
    sleep=asyncio.sleep,
) -> WebhookDeliveryResult:
    final_status_code: int | None = None
    final_error: str | None = None
    bounded_attempts = max(1, max_attempts)
    existing_attempts = await _load_existing_attempt_count(tick_id=tick_id)
    next_attempt_number = existing_attempts + 1
    if next_attempt_number > bounded_attempts:
        await _persist_webhook_result(
            tick_id=tick_id,
            delivered=False,
            status_code=None,
            error="Webhook delivery attempts exhausted",
            attempt_count=existing_attempts,
        )
        return WebhookDeliveryResult(
            delivered=False,
            attempt_count=existing_attempts,
            status_code=None,
            error="Webhook delivery attempts exhausted",
        )

    for attempt_number in range(next_attempt_number, bounded_attempts + 1):
        try:
            response = await http_client.post(webhook_url, json=payload)
            final_status_code = int(response.status_code)
            final_error = None if 200 <= final_status_code < 300 else f"HTTP {final_status_code}"
            await _persist_webhook_attempt(
                tick_id=tick_id,
                attempt_number=attempt_number,
                status_code=final_status_code,
                error=final_error,
            )
            if 200 <= final_status_code < 300:
                await _persist_webhook_result(
                    tick_id=tick_id,
                    delivered=True,
                    status_code=final_status_code,
                    error=None,
                    attempt_count=attempt_number,
                )
                return WebhookDeliveryResult(
                    delivered=True,
                    attempt_count=attempt_number,
                    status_code=final_status_code,
                )
        except Exception as exc:  # noqa: BLE001
            final_status_code = None
            final_error = str(exc)
            await _persist_webhook_attempt(
                tick_id=tick_id,
                attempt_number=attempt_number,
                status_code=None,
                error=final_error,
            )

        if attempt_number < bounded_attempts:
            await sleep(min(2 ** (attempt_number - 1), 8))

    await _persist_webhook_result(
        tick_id=tick_id,
        delivered=False,
        status_code=final_status_code,
        error=final_error,
        attempt_count=bounded_attempts,
    )
    return WebhookDeliveryResult(
        delivered=False,
        attempt_count=bounded_attempts,
        status_code=final_status_code,
        error=final_error,
    )
