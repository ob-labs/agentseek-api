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


@pytest.mark.parametrize(
    ("schedule", "message"),
    [
        ("", "Malformed RRULE"),
        ("FREQ=DAILY;FREQ=WEEKLY", "Duplicate RRULE clause: FREQ"),
        ("INTERVAL=5", "RRULE must include FREQ"),
        ("FREQ=YEARLY", "Unsupported RRULE frequency: YEARLY"),
        ("FREQ=DAILY;INTERVAL=0", "INTERVAL must be greater than 0"),
        ("FREQ=DAILY;BYHOUR=24", "BYHOUR must be between 0 and 23"),
        ("FREQ=DAILY;BYMINUTE=60", "BYMINUTE must be between 0 and 59"),
        ("FREQ=MONTHLY;BYMONTHDAY=0", "BYMONTHDAY must be between 1 and 31"),
        ("FREQ=WEEKLY;BYDAY=XX", "Unsupported BYDAY value: XX"),
        ("FREQ=DAILY;UNTIL=20260525", "UNTIL must use YYYYMMDDTHHMMSS or YYYYMMDDTHHMMSSZ"),
    ],
)
def test_validate_schedule_rejects_invalid_values(schedule: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        validate_schedule(schedule, timezone_name="UTC")


def test_validate_schedule_rejects_invalid_timezone() -> None:
    with pytest.raises(ValueError, match="Invalid timezone: Mars/Olympus"):
        validate_schedule("FREQ=DAILY;INTERVAL=1", timezone_name="Mars/Olympus")


def test_compute_next_run_at_supports_hourly_daily_weekly_and_monthly_rules() -> None:
    now = datetime(2026, 5, 25, 10, 17, tzinfo=UTC)

    hourly = compute_next_run_at("FREQ=HOURLY;INTERVAL=2;BYMINUTE=30", timezone_name="UTC", now=now)
    daily = compute_next_run_at("FREQ=DAILY;INTERVAL=1;BYHOUR=9;BYMINUTE=15", timezone_name="UTC", now=now)
    weekly = compute_next_run_at(
        "FREQ=WEEKLY;INTERVAL=1;BYDAY=WE,FR;BYHOUR=8;BYMINUTE=0",
        timezone_name="UTC",
        now=now,
    )
    monthly = compute_next_run_at(
        "FREQ=MONTHLY;INTERVAL=1;BYMONTHDAY=31;BYHOUR=12;BYMINUTE=0",
        timezone_name="UTC",
        now=datetime(2026, 4, 29, 15, 0, tzinfo=UTC),
    )

    assert hourly == datetime(2026, 5, 25, 10, 30, tzinfo=UTC)
    assert daily == datetime(2026, 5, 26, 9, 15, tzinfo=UTC)
    assert weekly == datetime(2026, 5, 27, 8, 0, tzinfo=UTC)
    assert monthly == datetime(2026, 4, 30, 12, 0, tzinfo=UTC)


def test_compute_next_run_at_respects_timezone_conversion() -> None:
    now = datetime(2026, 5, 25, 0, 30, tzinfo=UTC)

    next_run = compute_next_run_at(
        "FREQ=DAILY;INTERVAL=1;BYHOUR=9;BYMINUTE=0",
        timezone_name="Asia/Shanghai",
        now=now,
    )

    assert next_run == datetime(2026, 5, 25, 1, 0, tzinfo=UTC)


def test_compute_next_run_at_rejects_rrule_without_future_occurrences() -> None:
    with pytest.raises(ValueError, match="RRULE has no future occurrences"):
        compute_next_run_at(
            "FREQ=DAILY;INTERVAL=1;BYHOUR=9;BYMINUTE=0;UNTIL=20260525T080000Z",
            timezone_name="UTC",
            now=datetime(2026, 5, 25, 8, 30, tzinfo=UTC),
        )
