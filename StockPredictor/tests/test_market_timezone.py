"""Tests for resolving market timezones from localized names."""

from __future__ import annotations

from datetime import timedelta, tzinfo
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stock_predictor.core.config import PredictorConfig
from stock_predictor.core import pipeline


def test_resolve_market_timezone_accepts_german_alias() -> None:
    config = PredictorConfig(ticker="AAPL", market_timezone="Mitteleuropäische Zeit")

    tz = pipeline.resolve_market_timezone(config)

    assert tz is not None
    assert tz.key == "Europe/Berlin"


def test_resolve_market_timezone_uses_localized_host_timezone(monkeypatch) -> None:
    class DummyTimezone(tzinfo):
        def utcoffset(self, dt):  # type: ignore[override]
            return timedelta(hours=1)

        def dst(self, dt):  # type: ignore[override]
            return timedelta(0)

        def tzname(self, dt):  # type: ignore[override]
            return "Heure d’Europe centrale"

    class DummyDateTime:
        def __init__(self, tz: tzinfo):
            self.tzinfo = tz

        def astimezone(self, tz=None):  # noqa: D401 - match datetime signature
            if tz is None:
                return self
            return DummyDateTime(tz)

    class DummyDateModule:
        @staticmethod
        def now(tz=None):  # noqa: D401 - mimic datetime.now signature
            tzinfo = tz or DummyTimezone()
            return DummyDateTime(tzinfo)

    monkeypatch.setattr(pipeline, "datetime", DummyDateModule)

    tz = pipeline.resolve_market_timezone(None)

    assert tz.key == "Europe/Paris"
    assert tz.key != pipeline.DEFAULT_MARKET_TIMEZONE.key
