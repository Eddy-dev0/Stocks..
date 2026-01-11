"""Centralized time source for live vs. as-of execution."""

from __future__ import annotations

from datetime import date, datetime, time as dt_time, timedelta, tzinfo
from typing import Iterable


class AppClock:
    """Provide effective timestamps for live and test-date runs."""

    def __init__(self) -> None:
        self._override_date: date | None = None
        self._override_time: dt_time = dt_time(16, 0)
        self._override_timezone: tzinfo | None = None

    @property
    def is_override(self) -> bool:
        return self._override_date is not None

    @property
    def override_date(self) -> date | None:
        return self._override_date

    def set_test_date(
        self,
        value: date,
        *,
        tz: tzinfo | None = None,
        time_of_day: dt_time | None = None,
    ) -> None:
        """Activate as-of mode for the supplied date."""

        self._override_date = value
        if time_of_day is not None:
            self._override_time = time_of_day
        if tz is not None:
            self._override_timezone = tz

    def clear_test_date(self) -> None:
        """Return to real system time."""

        self._override_date = None
        self._override_timezone = None

    def now(self, tz: tzinfo | None = None) -> datetime:
        if self._override_date is None:
            return datetime.now(tz) if tz is not None else datetime.now()
        base = datetime.combine(self._override_date, self._override_time)
        tzinfo = tz or self._override_timezone
        if tzinfo is None:
            return base
        if base.tzinfo is None:
            return base.replace(tzinfo=tzinfo)
        return base.astimezone(tzinfo)

    def today(self, tz: tzinfo | None = None) -> date:
        return self.now(tz).date()

    def system_now(self, tz: tzinfo | None = None) -> datetime:
        return datetime.now(tz) if tz is not None else datetime.now()

    def system_today(self, tz: tzinfo | None = None) -> date:
        return self.system_now(tz).date()


def resolve_asof_trading_day(
    requested: date | datetime | None,
    available: Iterable[date | datetime],
) -> date | None:
    """Find the latest available trading day on or before the requested date."""

    available_dates = {_coerce_date(value) for value in available if value is not None}
    available_dates = {value for value in available_dates if value is not None}
    if not available_dates:
        return None

    latest = max(available_dates)
    if requested is None:
        return latest

    requested_date = _coerce_date(requested)
    if requested_date is None:
        return latest
    if requested_date >= latest:
        return latest
    if requested_date in available_dates:
        return requested_date

    earliest = min(available_dates)
    cursor = requested_date
    while cursor >= earliest:
        cursor = cursor - timedelta(days=1)
        if cursor in available_dates:
            return cursor
    return None


def _coerce_date(value: date | datetime) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return None


app_clock = AppClock()


__all__ = ["AppClock", "app_clock", "resolve_asof_trading_day"]
