from __future__ import annotations

from ..swing_points import compute_atr, find_swing_points
from ..types import Candle, PatternDetection
from .common import make_detection


def detect_diamond(candles: list[Candle]) -> PatternDetection | None:
    if len(candles) < 40:
        return None
    window = candles[-120:] if len(candles) >= 120 else candles
    swings = find_swing_points(window, left_bars=2, right_bars=2, min_move_atr=0.2)
    if len(swings) < 5:
        return None
    pivots = swings[-6:]
    highs = [p for p in pivots if p.kind == "high"]
    lows = [p for p in pivots if p.kind == "low"]
    if len(highs) < 3 or len(lows) < 3:
        return None
    h1, h2, h3 = highs[-3:]
    l1, l2, l3 = lows[-3:]
    range1 = h1.price - l1.price
    range2 = h2.price - l2.price
    if range2 <= range1 * 1.15:
        return None
    if not (h3.price < h2.price and l3.price > l2.price):
        return None
    atr = compute_atr(window)
    mid = (h2.index + l2.index) / 2
    rel = (mid - pivots[0].index) / max(1, pivots[-1].index - pivots[0].index)
    if not (0.3 <= rel <= 0.7):
        return None
    right_upper = max(h2.price, h3.price)
    right_lower = min(l2.price, l3.price)
    direction = "bullish" if window[-1].close >= window[-2].close else "bearish"
    breakout = right_upper if direction == "bullish" else right_lower
    invalidation = (right_lower - atr * 0.5) if direction == "bullish" else (right_upper + atr * 0.5)
    return make_detection(
        "Diamond",
        window,
        pivots,
        breakout,
        invalidation,
        direction,
        82.0,
        explanation="Price volatility expanded first and then contracted into a diamond structure.",
    )
