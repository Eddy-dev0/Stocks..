"""Tests for the centralized application clock."""

from datetime import date

from stock_predictor.core.clock import resolve_asof_trading_day


def test_resolve_asof_trading_day_weekend() -> None:
    available = [date(2024, 5, 2), date(2024, 5, 3), date(2024, 5, 6)]
    requested = date(2024, 5, 4)
    assert resolve_asof_trading_day(requested, available) == date(2024, 5, 3)


def test_resolve_asof_trading_day_future_clamps() -> None:
    available = [date(2024, 5, 2), date(2024, 5, 3), date(2024, 5, 6)]
    requested = date(2024, 6, 1)
    assert resolve_asof_trading_day(requested, available) == date(2024, 5, 6)
