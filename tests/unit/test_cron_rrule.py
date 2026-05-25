from datetime import UTC, datetime

import pytest

from agentseek_api.services.cron_rrule import compute_next_run_at, validate_schedule


def test_validate_schedule_rejects_unsupported_clause() -> None:
    with pytest.raises(ValueError, match="Unsupported RRULE clause: BYSETPOS"):
        validate_schedule("FREQ=MONTHLY;BYSETPOS=1", timezone_name="UTC")


def test_validate_schedule_rejects_count_clause() -> None:
    with pytest.raises(ValueError, match="Unsupported RRULE clause: COUNT"):
        validate_schedule("FREQ=DAILY;COUNT=2", timezone_name="UTC")


def test_compute_next_run_at_returns_future_utc_datetime() -> None:
    now = datetime.now(UTC)

    next_run = compute_next_run_at("FREQ=MINUTELY;INTERVAL=5", timezone_name="UTC")

    assert next_run.tzinfo == UTC
    assert next_run > now
