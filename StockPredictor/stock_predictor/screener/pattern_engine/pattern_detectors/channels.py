from __future__ import annotations

from ..types import Candle, PatternDetection
from .common import make_detection


def _slope(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    return (values[-1] - values[0]) / n


def detect_channel(candles: list[Candle]) -> PatternDetection | None:
    if len(candles) < 30:
        return None
    highs = [c.high for c in candles[-20:]]
    lows = [c.low for c in candles[-20:]]
    width = (max(highs) - min(lows)) / max(candles[-1].close, 1e-9)
    if width > 0.2:
        return None
    direction = "neutral"
    if _slope([c.close for c in candles[-20:]]) > 0:
        direction = "bullish"
    elif _slope([c.close for c in candles[-20:]]) < 0:
        direction = "bearish"
    return make_detection("Channel", candles, [], max(highs), min(lows), direction, 60.0)


def detect_channel_up(candles: list[Candle]) -> PatternDetection | None:
    if len(candles) < 20:
        return None
    close = [c.close for c in candles[-20:]]
    if _slope(close) <= 0:
        return None
    return make_detection("Channel Up", candles, [], max(c.high for c in candles[-20:]), min(c.low for c in candles[-20:]), "bullish", 62.0)


def detect_channel_down(candles: list[Candle]) -> PatternDetection | None:
    if len(candles) < 20:
        return None
    close = [c.close for c in candles[-20:]]
    if _slope(close) >= 0:
        return None
    return make_detection("Channel Down", candles, [], min(c.low for c in candles[-20:]), max(c.high for c in candles[-20:]), "bearish", 62.0)
