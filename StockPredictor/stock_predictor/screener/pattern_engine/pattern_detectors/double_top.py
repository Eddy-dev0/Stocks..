from __future__ import annotations

from ..swing_points import tolerance
from ..types import Candle, PatternDetection, SwingPoint
from .common import make_detection


def detect_double_top(candles: list[Candle], swings: list[SwingPoint]) -> PatternDetection | None:
    highs = [s for s in swings if s.kind == "high"]
    lows = [s for s in swings if s.kind == "low"]
    if len(highs) < 2 or not lows:
        return None
    a, b = highs[-2], highs[-1]
    between = [l for l in lows if a.index < l.index < b.index]
    if not between:
        return None
    neckline = min(x.price for x in between)
    tol = tolerance(candles)
    if abs(a.price - b.price) > tol:
        return None
    score = 70.0 + max(0.0, 15.0 - (abs(a.price - b.price) / max(tol, 1e-9)) * 15)
    return make_detection("Double Top", candles, [a, b], neckline, max(a.price, b.price) + tol, "bearish", score, neckline)
