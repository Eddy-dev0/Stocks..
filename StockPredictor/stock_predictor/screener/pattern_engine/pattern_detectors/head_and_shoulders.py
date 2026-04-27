from __future__ import annotations

from ..swing_points import compute_atr
from ..types import Candle, PatternDetection, SwingPoint
from .common import fit_trendline, make_detection, pretrend_ok


def detect_head_and_shoulders(candles: list[Candle], swings: list[SwingPoint]) -> PatternDetection | None:
    highs = [s for s in swings if s.kind == "high"]
    lows = [s for s in swings if s.kind == "low"]
    if len(highs) < 3 or len(lows) < 2:
        return None
    ls, head, rs = highs[-3], highs[-2], highs[-1]
    t1 = min((x for x in lows if ls.index < x.index < head.index), default=None, key=lambda p: p.price)
    t2 = min((x for x in lows if head.index < x.index < rs.index), default=None, key=lambda p: p.price)
    if t1 is None or t2 is None:
        return None
    atr = compute_atr(candles)
    if not (head.price > ls.price + max(atr, ls.price * 0.015) and head.price > rs.price + max(atr, rs.price * 0.015)):
        return None
    shoulder_tol = max(((ls.price + rs.price) / 2) * 0.04, atr * 1.5)
    if abs(ls.price - rs.price) > shoulder_tol:
        return None
    if rs.price >= head.price - max(atr * 0.5, head.price * 0.01):
        return None
    left_dur, right_dur = head.index - ls.index, rs.index - head.index
    balance = min(left_dur, right_dur) / max(left_dur, right_dur)
    if balance < 0.25:
        return None
    neck = fit_trendline([t1, t2])
    if neck is None:
        return None
    score = 20 + 15 + 15 + 10
    if pretrend_ok(candles, ls.index, bullish_reversal=False):
        score += 10
    if balance >= 0.5:
        score += 10
    invalidation = rs.price + atr * 0.3
    return make_detection(
        "Head and Shoulders",
        candles,
        [ls, t1, head, t2, rs],
        neck.end_price,
        invalidation,
        "bearish",
        score,
        neckline=neck.end_price,
        trendline_neckline=neck,
        explanation="Price formed a higher head between two shoulders and broke below the neckline.",
    )


def detect_inverted_head_and_shoulders(candles: list[Candle], swings: list[SwingPoint]) -> PatternDetection | None:
    lows = [s for s in swings if s.kind == "low"]
    highs = [s for s in swings if s.kind == "high"]
    if len(lows) < 3 or len(highs) < 2:
        return None
    ls, head, rs = lows[-3], lows[-2], lows[-1]
    p1 = max((x for x in highs if ls.index < x.index < head.index), default=None, key=lambda p: p.price)
    p2 = max((x for x in highs if head.index < x.index < rs.index), default=None, key=lambda p: p.price)
    if p1 is None or p2 is None:
        return None
    atr = compute_atr(candles)
    if not (head.price < ls.price - max(atr, ls.price * 0.015) and head.price < rs.price - max(atr, rs.price * 0.015)):
        return None
    shoulder_tol = max(((ls.price + rs.price) / 2) * 0.04, atr * 1.5)
    if abs(ls.price - rs.price) > shoulder_tol:
        return None
    if rs.price <= head.price + max(atr * 0.5, head.price * 0.01):
        return None
    left_dur, right_dur = head.index - ls.index, rs.index - head.index
    balance = min(left_dur, right_dur) / max(left_dur, right_dur)
    if balance < 0.25:
        return None
    neck = fit_trendline([p1, p2])
    if neck is None:
        return None
    score = 20 + 15 + 15 + 10
    if pretrend_ok(candles, ls.index, bullish_reversal=True):
        score += 10
    if balance >= 0.5:
        score += 10
    invalidation = rs.price - atr * 0.3
    return make_detection(
        "Inverted Head and Shoulders",
        candles,
        [ls, p1, head, p2, rs],
        neck.end_price,
        invalidation,
        "bullish",
        score,
        neckline=neck.end_price,
        trendline_neckline=neck,
        explanation="Price formed a lower head between two shoulders and broke above the neckline.",
    )
