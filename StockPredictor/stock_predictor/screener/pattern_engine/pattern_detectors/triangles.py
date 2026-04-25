from __future__ import annotations

from ..swing_points import tolerance
from ..types import Candle, PatternDetection, SwingPoint
from .common import make_detection


def detect_ascending_triangle(candles: list[Candle], swings: list[SwingPoint]) -> PatternDetection | None:
    highs = [s for s in swings if s.kind == "high"][-3:]
    lows = [s for s in swings if s.kind == "low"][-3:]
    if len(highs) < 2 or len(lows) < 2:
        return None
    tol = tolerance(candles)
    resistance = sum(h.price for h in highs) / len(highs)
    if any(abs(h.price - resistance) > tol * 1.5 for h in highs):
        return None
    if not (lows[-1].price > lows[0].price):
        return None
    return make_detection("Ascending Triangle", candles, lows + highs, resistance, lows[0].price - tol, "bullish", 68.0, resistance)


def detect_descending_triangle(candles: list[Candle], swings: list[SwingPoint]) -> PatternDetection | None:
    lows = [s for s in swings if s.kind == "low"][-3:]
    highs = [s for s in swings if s.kind == "high"][-3:]
    if len(highs) < 2 or len(lows) < 2:
        return None
    tol = tolerance(candles)
    support = sum(l.price for l in lows) / len(lows)
    if any(abs(l.price - support) > tol * 1.5 for l in lows):
        return None
    if not (highs[-1].price < highs[0].price):
        return None
    return make_detection("Descending Triangle", candles, highs + lows, support, highs[0].price + tol, "bearish", 68.0, support)
