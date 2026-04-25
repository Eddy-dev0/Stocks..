from __future__ import annotations

from ..types import Candle, PatternDetection
from .common import make_detection


def detect_diamond(candles: list[Candle]) -> PatternDetection | None:
    if len(candles) < 30:
        return None
    first = candles[-30:-15]
    second = candles[-15:]
    first_range = max(c.high for c in first) - min(c.low for c in first)
    second_range = max(c.high for c in second) - min(c.low for c in second)
    if first_range <= second_range:
        return None
    direction = "bullish" if candles[-1].close > candles[-3].close else "bearish"
    breakout = max(c.high for c in second) if direction == "bullish" else min(c.low for c in second)
    invalidation = min(c.low for c in second) if direction == "bullish" else max(c.high for c in second)
    return make_detection("Diamond", candles, [], breakout, invalidation, direction, 60.0)
