from __future__ import annotations

from ..swing_points import compute_atr
from ..types import Candle, PatternDetection, SwingPoint
from .common import dynamic_tolerance, make_detection, pretrend_ok


def detect_double_top(candles: list[Candle], swings: list[SwingPoint]) -> PatternDetection | None:
    highs = [s for s in swings if s.kind == "high"]
    lows = [s for s in swings if s.kind == "low"]
    if len(highs) < 2:
        return None
    h1, h2 = highs[-2], highs[-1]
    bars_between = h2.index - h1.index
    if bars_between < 5 or bars_between > 80:
        return None
    between_lows = [l for l in lows if h1.index < l.index < h2.index]
    if not between_lows:
        return None
    l = min(between_lows, key=lambda p: p.price)
    atr = compute_atr(candles)
    avg_high = (h1.price + h2.price) / 2
    tol = dynamic_tolerance(avg_high, atr, percent=0.015, atr_mult=1.0)
    if abs(h1.price - h2.price) > tol or h2.price > h1.price + tol:
        return None
    neckline = l.price
    if neckline >= avg_high - max(atr * 1.5, avg_high * 0.02):
        return None

    score = 20 + 15 + 15
    if pretrend_ok(candles, h1.index, bullish_reversal=False):
        score += 10
    if h2.price <= h1.price:
        score += 10
    invalidation = max(h1.price, h2.price) + atr * 0.2
    return make_detection(
        "Double Top",
        candles,
        [h1, l, h2],
        neckline,
        invalidation,
        "bearish",
        score,
        neckline=neckline,
        support_level=neckline,
        resistance_level=max(h1.price, h2.price),
        explanation="Price rejected the same resistance twice and broke below the neckline.",
    )
