from __future__ import annotations

from ..types import Candle, PatternDetection
from .common import make_detection


def _momentum(candles: list[Candle], bars: int = 10) -> float:
    if len(candles) < bars + 1:
        return 0.0
    start = candles[-bars - 1].close
    end = candles[-1].close
    return (end - start) / max(start, 1e-9)


def detect_flag(candles: list[Candle]) -> PatternDetection | None:
    if len(candles) < 15:
        return None
    pole = _momentum(candles[-15:-7], bars=6)
    pullback = _momentum(candles[-7:], bars=5)
    if pole <= 0.03 or pullback >= 0:
        return None
    breakout = max(c.high for c in candles[-10:])
    invalidation = min(c.low for c in candles[-10:])
    return make_detection("Flag", candles, [], breakout, invalidation, "bullish", 64.0)


def detect_bearish_flag(candles: list[Candle]) -> PatternDetection | None:
    if len(candles) < 15:
        return None
    pole = _momentum(candles[-15:-7], bars=6)
    pullback = _momentum(candles[-7:], bars=5)
    if pole >= -0.03 or pullback <= 0:
        return None
    breakdown = min(c.low for c in candles[-10:])
    invalidation = max(c.high for c in candles[-10:])
    return make_detection("Bearish Flag", candles, [], breakdown, invalidation, "bearish", 64.0)


def detect_pennant(candles: list[Candle]) -> PatternDetection | None:
    if len(candles) < 20:
        return None
    highs = [c.high for c in candles[-10:]]
    lows = [c.low for c in candles[-10:]]
    if max(highs) - min(highs) < 0.01 * candles[-1].close:
        return None
    direction = "bullish" if candles[-1].close >= candles[-12].close else "bearish"
    breakout = max(highs) if direction == "bullish" else min(lows)
    invalidation = min(lows) if direction == "bullish" else max(highs)
    return make_detection("Pennant", candles, [], breakout, invalidation, direction, 61.0)
