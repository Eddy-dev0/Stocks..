from __future__ import annotations

from datetime import datetime, timedelta

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stock_predictor.screener.pattern_engine.trade_quality import TradeQualityOptions, calculate_trade_quality
from stock_predictor.screener.pattern_engine.types import Candle


def _trend_candles(n: int = 260) -> list[Candle]:
    base = datetime(2024, 1, 1)
    candles: list[Candle] = []
    px = 100.0
    for i in range(n):
        px += (0.35 if i % 12 < 7 else -0.25)
        candles.append(
            Candle(
                timestamp=base + timedelta(hours=i),
                open=px - 0.2,
                high=px + 1.0,
                low=px - 1.0,
                close=px,
                volume=2_000_000,
            )
        )
    return candles


def test_trade_quality_returns_shape() -> None:
    candles = _trend_candles()
    result = calculate_trade_quality(
        candles,
        "Channel Up",
        "bullish",
        TradeQualityOptions(forward_bars=10, min_occurrences=5, use_walk_forward=True),
    )
    assert hasattr(result, "rating")
    assert result.occurrences >= 0
    assert result.average_move_percent >= 0
