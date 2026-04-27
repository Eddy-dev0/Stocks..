from __future__ import annotations

from ..swing_points import compute_atr
from ..types import Candle, PatternDetection, SwingPoint
from .common import dynamic_tolerance, make_detection, pretrend_ok


def detect_double_bottom(candles: list[Candle], swings: list[SwingPoint]) -> PatternDetection | None:
    lows = [s for s in swings if s.kind == "low"]
    highs = [s for s in swings if s.kind == "high"]
    if len(lows) < 2:
        return None
    l1, l2 = lows[-2], lows[-1]
    bars_between = l2.index - l1.index
    if bars_between < 5 or bars_between > 80:
        return None
    between_highs = [h for h in highs if l1.index < h.index < l2.index]
    if not between_highs:
        return None
    h = max(between_highs, key=lambda p: p.price)
    atr = compute_atr(candles)
    avg_low = (l1.price + l2.price) / 2
    tol = dynamic_tolerance(avg_low, atr, percent=0.015, atr_mult=1.0)
    if abs(l1.price - l2.price) > tol or l2.price < l1.price - tol:
        return None
    neckline = h.price
    if neckline <= avg_low + max(atr * 1.5, avg_low * 0.02):
        return None

    score = 20 + 15 + 15
    if pretrend_ok(candles, l1.index, bullish_reversal=True):
        score += 10
    if l2.price >= l1.price:
        score += 10
    invalidation = min(l1.price, l2.price) - atr * 0.2
    return make_detection(
        "Double Bottom",
        candles,
        [l1, h, l2],
        neckline,
        invalidation,
        "bullish",
        score,
        neckline=neckline,
        support_level=min(l1.price, l2.price),
        resistance_level=neckline,
        explanation="Two lows held near the same support and price reclaimed the neckline.",
    )
