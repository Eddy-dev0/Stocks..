from __future__ import annotations

from ..swing_points import tolerance
from ..types import Candle, PatternDetection, SwingPoint
from .common import make_detection


def detect_head_and_shoulders(candles: list[Candle], swings: list[SwingPoint]) -> PatternDetection | None:
    highs = [s for s in swings if s.kind == "high"]
    lows = [s for s in swings if s.kind == "low"]
    if len(highs) < 3 or len(lows) < 2:
        return None
    l_sh, head, r_sh = highs[-3], highs[-2], highs[-1]
    if not (head.price > l_sh.price and head.price > r_sh.price):
        return None
    tol = tolerance(candles)
    if abs(l_sh.price - r_sh.price) > tol * 2:
        return None
    neckline_lows = [x for x in lows if l_sh.index < x.index < r_sh.index]
    if len(neckline_lows) < 2:
        return None
    neckline = sum(x.price for x in neckline_lows[-2:]) / 2
    score = 75.0
    return make_detection("Head and Shoulders", candles, [l_sh, head, r_sh], neckline, head.price + tol, "bearish", score, neckline)


def detect_inverted_head_and_shoulders(candles: list[Candle], swings: list[SwingPoint]) -> PatternDetection | None:
    lows = [s for s in swings if s.kind == "low"]
    highs = [s for s in swings if s.kind == "high"]
    if len(lows) < 3 or len(highs) < 2:
        return None
    l_sh, head, r_sh = lows[-3], lows[-2], lows[-1]
    if not (head.price < l_sh.price and head.price < r_sh.price):
        return None
    tol = tolerance(candles)
    if abs(l_sh.price - r_sh.price) > tol * 2:
        return None
    neckline_highs = [x for x in highs if l_sh.index < x.index < r_sh.index]
    if len(neckline_highs) < 2:
        return None
    neckline = sum(x.price for x in neckline_highs[-2:]) / 2
    score = 75.0
    return make_detection("Inverted Head and Shoulders", candles, [l_sh, head, r_sh], neckline, head.price - tol, "bullish", score, neckline)
