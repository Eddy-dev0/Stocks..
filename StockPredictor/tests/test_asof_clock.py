from __future__ import annotations

from datetime import date

from stock_predictor.core.clock import resolve_asof_trading_day


def test_resolve_asof_trading_day_rolls_backwards() -> None:
    available = [
        date(2024, 5, 1),
        date(2024, 5, 3),
        date(2024, 5, 6),
    ]

    resolved = resolve_asof_trading_day(date(2024, 5, 4), available)

    assert resolved == date(2024, 5, 3)
