from __future__ import annotations

from ..swing_points import tolerance
from ..types import Candle, PatternDetection, SwingPoint
from .common import make_detection


def detect_double_bottom(candles: list[Candle], swings: list[SwingPoint]) -> PatternDetection | None:
    lows = [s for s in swings if s.kind == "low"]
    highs = [s for s in swings if s.kind == "high"]
    if len(lows) < 2 or not highs:
        return None
    a, b = lows[-2], lows[-1]
    if b.index <= a.index:
        return None
    between = [h for h in highs if a.index < h.index < b.index]
    if not between:
        return None
    neckline = max(x.price for x in between)
    tol = tolerance(candles)
    if abs(a.price - b.price) > tol:
        return None
    score = 70.0 + max(0.0, 15.0 - (abs(a.price - b.price) / max(tol, 1e-9)) * 15)
    return make_detection("Double Bottom", candles, [a, b], neckline, min(a.price, b.price) - tol, "bullish", score, neckline)
