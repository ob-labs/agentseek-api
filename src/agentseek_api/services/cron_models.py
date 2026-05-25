from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

from agentseek_api.core.orm import CronJob

DispatchStatus = Literal["queued", "skipped"]


@dataclass(frozen=True, slots=True)
class ClaimedCron:
    tick_id: int
    cron_id: str
    assistant_id: str
    thread_id: str | None
    user_id: str
    schedule: str
    input_json: Any
    scheduled_for: datetime

    @classmethod
    def from_row(cls, row: CronJob, *, tick_id: int, scheduled_for: datetime) -> "ClaimedCron":
        return cls(
            tick_id=tick_id,
            cron_id=row.cron_id,
            assistant_id=row.assistant_id,
            thread_id=row.thread_id,
            user_id=row.user_id,
            schedule=row.schedule,
            input_json=row.input_json,
            scheduled_for=scheduled_for,
        )


@dataclass(frozen=True, slots=True)
class CronDispatchResult:
    cron_id: str
    status: DispatchStatus
    thread_id: str | None
    run_id: str | None = None
    skip_reason: str | None = None
