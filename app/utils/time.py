"""Time helpers (UTC-first, with KSA / US-Eastern conveniences)."""
from __future__ import annotations

from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

UTC = timezone.utc
KSA = ZoneInfo("Asia/Riyadh")
US_EASTERN = ZoneInfo("America/New_York")


def utcnow() -> datetime:
    return datetime.now(UTC)


def to_ksa(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(KSA)


def to_eastern(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(US_EASTERN)


def start_of_week(dt: datetime) -> datetime:
    """Monday 00:00 (in the datetime's own tz) for the week containing dt."""
    monday = dt - timedelta(days=dt.weekday())
    return monday.replace(hour=0, minute=0, second=0, microsecond=0)


def start_of_month(dt: datetime) -> datetime:
    return dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)


def parse_hhmm(value: str) -> tuple[int, int]:
    """Parse ``"01:15"`` into ``(1, 15)``."""
    hh, mm = value.split(":")
    return int(hh), int(mm)
