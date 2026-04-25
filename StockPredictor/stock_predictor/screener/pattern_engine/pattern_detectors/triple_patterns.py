from __future__ import annotations

from ..swing_points import tolerance
from ..types import Candle, PatternDetection, SwingPoint
from .common import make_detection


def detect_triple_bottom(candles: list[Candle], swings: list[SwingPoint]) -> PatternDetection | None:
    lows = [s for s in swings if s.kind == "low"]
    highs = [s for s in swings if s.kind == "high"]
    if len(lows) < 3:
        return None
    pts = lows[-3:]
    tol = tolerance(candles)
    ref = sum(p.price for p in pts) / 3
    if any(abs(p.price - ref) > tol * 1.5 for p in pts):
        return None
    neckline = max((h.price for h in highs if pts[0].index < h.index < pts[-1].index), default=0.0)
    if neckline <= 0:
        return None
    return make_detection("Triple Bottom", candles, pts, neckline, min(p.price for p in pts) - tol, "bullish", 72.0, neckline)


def detect_triple_top(candles: list[Candle], swings: list[SwingPoint]) -> PatternDetection | None:
    highs = [s for s in swings if s.kind == "high"]
    lows = [s for s in swings if s.kind == "low"]
    if len(highs) < 3:
        return None
    pts = highs[-3:]
    tol = tolerance(candles)
    ref = sum(p.price for p in pts) / 3
    if any(abs(p.price - ref) > tol * 1.5 for p in pts):
        return None
    neckline = min((l.price for l in lows if pts[0].index < l.index < pts[-1].index), default=0.0)
    if neckline <= 0:
        return None
    return make_detection("Triple Top", candles, pts, neckline, max(p.price for p in pts) + tol, "bearish", 72.0, neckline)
