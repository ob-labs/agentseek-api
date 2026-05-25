from calendar import monthrange
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

SUPPORTED_RRULE_KEYS = {
    "COUNT",
    "FREQ",
    "INTERVAL",
    "UNTIL",
    "BYDAY",
    "BYHOUR",
    "BYMINUTE",
    "BYMONTHDAY",
}
SUPPORTED_FREQUENCIES = {"MINUTELY", "HOURLY", "DAILY", "WEEKLY", "MONTHLY"}
WEEKDAY_INDEX = {
    "MO": 0,
    "TU": 1,
    "WE": 2,
    "TH": 3,
    "FR": 4,
    "SA": 5,
    "SU": 6,
}


def _parse_schedule(schedule: str) -> dict[str, str]:
    parts = [chunk.strip() for chunk in schedule.split(";") if chunk.strip()]
    if not parts:
        raise ValueError("Malformed RRULE")

    parsed: dict[str, str] = {}
    for part in parts:
        key, separator, value = part.partition("=")
        if separator != "=" or not key or not value:
            raise ValueError("Malformed RRULE")
        normalized_key = key.upper()
        if normalized_key not in SUPPORTED_RRULE_KEYS:
            raise ValueError(f"Unsupported RRULE clause: {normalized_key}")
        if normalized_key in parsed:
            raise ValueError(f"Duplicate RRULE clause: {normalized_key}")
        parsed[normalized_key] = value
    return parsed


def _parse_positive_int(value: str, *, clause: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{clause} must be an integer") from exc
    if parsed <= 0:
        raise ValueError(f"{clause} must be greater than 0")
    return parsed


def _parse_bounded_int(value: str, *, clause: str, lower: int, upper: int) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{clause} must be an integer") from exc
    if parsed < lower or parsed > upper:
        raise ValueError(f"{clause} must be between {lower} and {upper}")
    return parsed


def _require_timezone(timezone_name: str) -> ZoneInfo:
    try:
        return ZoneInfo(timezone_name or "UTC")
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Invalid timezone: {timezone_name}") from exc


def validate_schedule(schedule: str, *, timezone_name: str = "UTC") -> None:
    parsed = _parse_schedule(schedule)
    _require_timezone(timezone_name)

    frequency = parsed.get("FREQ")
    if frequency is None:
        raise ValueError("RRULE must include FREQ")
    if frequency.upper() not in SUPPORTED_FREQUENCIES:
        raise ValueError(f"Unsupported RRULE frequency: {frequency.upper()}")

    if "INTERVAL" in parsed:
        _parse_positive_int(parsed["INTERVAL"], clause="INTERVAL")
    if "COUNT" in parsed:
        _parse_positive_int(parsed["COUNT"], clause="COUNT")
    if "BYHOUR" in parsed:
        _parse_bounded_int(parsed["BYHOUR"], clause="BYHOUR", lower=0, upper=23)
    if "BYMINUTE" in parsed:
        _parse_bounded_int(parsed["BYMINUTE"], clause="BYMINUTE", lower=0, upper=59)
    if "BYMONTHDAY" in parsed:
        _parse_bounded_int(parsed["BYMONTHDAY"], clause="BYMONTHDAY", lower=1, upper=31)
    if "BYDAY" in parsed:
        for token in parsed["BYDAY"].split(","):
            normalized = token.strip().upper()
            if normalized not in WEEKDAY_INDEX:
                raise ValueError(f"Unsupported BYDAY value: {normalized}")
    if "UNTIL" in parsed:
        _parse_until(parsed["UNTIL"])


def _parse_until(value: str) -> datetime:
    try:
        if value.endswith("Z"):
            return datetime.strptime(value, "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
        return datetime.strptime(value, "%Y%m%dT%H%M%S").replace(tzinfo=UTC)
    except ValueError as exc:
        raise ValueError("UNTIL must use YYYYMMDDTHHMMSS or YYYYMMDDTHHMMSSZ") from exc


def _aligned_minute(now: datetime, interval: int) -> datetime:
    candidate = now.replace(second=0, microsecond=0) + timedelta(minutes=1)
    minute_offset = candidate.minute % interval
    if minute_offset:
        candidate += timedelta(minutes=interval - minute_offset)
    return candidate


def _aligned_hour(now: datetime, interval: int, minute: int) -> datetime:
    candidate = now.replace(minute=minute, second=0, microsecond=0)
    if candidate <= now:
        candidate += timedelta(hours=1)
    hour_offset = candidate.hour % interval
    if hour_offset:
        candidate += timedelta(hours=interval - hour_offset)
    return candidate


def _aligned_day(now: datetime, interval: int, hour: int, minute: int) -> datetime:
    candidate = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if candidate <= now:
        candidate += timedelta(days=1)
    days_since_anchor = (candidate.date() - now.date()).days
    remainder = days_since_anchor % interval
    if remainder:
        candidate += timedelta(days=interval - remainder)
    return candidate


def _aligned_week(now: datetime, interval: int, hour: int, minute: int, byday: str | None) -> datetime:
    weekdays = [WEEKDAY_INDEX[token.strip().upper()] for token in byday.split(",")] if byday else [now.weekday()]
    weekdays = sorted(set(weekdays))
    base = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    for offset in range(0, 7 * max(interval, 1) + 7):
        candidate = base + timedelta(days=offset)
        if candidate.weekday() not in weekdays:
            continue
        if candidate <= now:
            continue
        weeks_since_now = (candidate.date() - now.date()).days // 7
        if weeks_since_now % interval == 0:
            return candidate
    raise ValueError("Unable to compute next weekly run")


def _aligned_month(now: datetime, interval: int, hour: int, minute: int, month_day: int | None) -> datetime:
    year = now.year
    month = now.month
    target_day = month_day or now.day
    for offset in range(0, 24):
        candidate_month = month + offset
        candidate_year = year + (candidate_month - 1) // 12
        normalized_month = ((candidate_month - 1) % 12) + 1
        if offset % interval != 0:
            continue
        last_day = monthrange(candidate_year, normalized_month)[1]
        candidate = datetime(
            candidate_year,
            normalized_month,
            min(target_day, last_day),
            hour,
            minute,
            tzinfo=now.tzinfo,
        )
        if candidate > now:
            return candidate
    raise ValueError("Unable to compute next monthly run")


def compute_next_run_at(schedule: str, *, timezone_name: str = "UTC", now: datetime | None = None) -> datetime:
    validate_schedule(schedule, timezone_name=timezone_name)
    parsed = _parse_schedule(schedule)
    timezone = _require_timezone(timezone_name)
    local_now = (now or datetime.now(UTC)).astimezone(timezone)

    interval = _parse_positive_int(parsed.get("INTERVAL", "1"), clause="INTERVAL")
    frequency = parsed["FREQ"].upper()
    hour = int(parsed.get("BYHOUR", local_now.strftime("%H")))
    minute = int(parsed.get("BYMINUTE", "0" if frequency in {"HOURLY", "DAILY", "WEEKLY", "MONTHLY"} else local_now.strftime("%M")))
    month_day = int(parsed["BYMONTHDAY"]) if "BYMONTHDAY" in parsed else None

    if frequency == "MINUTELY":
        candidate = _aligned_minute(local_now, interval)
    elif frequency == "HOURLY":
        candidate = _aligned_hour(local_now, interval, minute)
    elif frequency == "DAILY":
        candidate = _aligned_day(local_now, interval, hour, minute)
    elif frequency == "WEEKLY":
        candidate = _aligned_week(local_now, interval, hour, minute, parsed.get("BYDAY"))
    else:
        candidate = _aligned_month(local_now, interval, hour, minute, month_day)

    until = _parse_until(parsed["UNTIL"]) if "UNTIL" in parsed else None
    if until is not None and candidate.astimezone(UTC) > until:
        raise ValueError("RRULE has no future occurrences")
    return candidate.astimezone(UTC)
