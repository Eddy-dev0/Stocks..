from __future__ import annotations

from ..types import Candle, PatternDetection
from .common import make_detection


def detect_cup_and_handle(candles: list[Candle]) -> PatternDetection | None:
    if len(candles) < 40:
        return None
    cup = candles[-35:-8]
    handle = candles[-8:]
    left = cup[0].close
    right = cup[-1].close
    bottom = min(c.low for c in cup)
    if abs(left - right) / max(right, 1e-9) > 0.03:
        return None
    if (min(left, right) - bottom) / max(right, 1e-9) < 0.05:
        return None
    if handle[-1].close < handle[0].close * 0.96:
        return None
    breakout = max(left, right)
    invalidation = min(c.low for c in handle)
    return make_detection("Cup and Handle", candles, [], breakout, invalidation, "bullish", 66.0, breakout)
