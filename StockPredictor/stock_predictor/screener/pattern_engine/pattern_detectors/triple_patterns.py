from __future__ import annotations

from ..swing_points import compute_atr
from ..types import Candle, PatternDetection, SwingPoint
from .common import dynamic_tolerance, make_detection, median_price, pretrend_ok


def detect_triple_bottom(candles: list[Candle], swings: list[SwingPoint]) -> PatternDetection | None:
    lows = [s for s in swings if s.kind == "low"]
    highs = [s for s in swings if s.kind == "high"]
    if len(lows) < 3:
        return None
    l1, l2, l3 = lows[-3:]
    if min(l2.index - l1.index, l3.index - l2.index) < 4:
        return None
    if not (12 <= l3.index - l1.index <= 140):
        return None
    h1 = max((h for h in highs if l1.index < h.index < l2.index), default=None, key=lambda p: p.price)
    h2 = max((h for h in highs if l2.index < h.index < l3.index), default=None, key=lambda p: p.price)
    if h1 is None or h2 is None:
        return None
    atr = compute_atr(candles)
    support = median_price([l1, l2, l3])
    tol = dynamic_tolerance(support, atr, percent=0.018, atr_mult=1.2)
    if max(abs(x.price - support) for x in [l1, l2, l3]) > tol:
        return None
    if min(h1.price, h2.price) <= support + max(atr * 1.2, support * 0.015):
        return None
    neckline = max(h1.price, h2.price)
    score = 25 + 15 + 15
    if pretrend_ok(candles, l1.index, bullish_reversal=True):
        score += 10
    if l3.price >= l1.price:
        score += 10
    invalidation = min(l1.price, l2.price, l3.price) - atr * 0.3
    return make_detection(
        "Triple Bottom",
        candles,
        [l1, h1, l2, h2, l3],
        neckline,
        invalidation,
        "bullish",
        score,
        neckline=neckline,
        support_level=support,
        resistance_level=neckline,
        explanation="Price tested the same support area three times and broke above resistance.",
    )


def detect_triple_top(candles: list[Candle], swings: list[SwingPoint]) -> PatternDetection | None:
    highs = [s for s in swings if s.kind == "high"]
    lows = [s for s in swings if s.kind == "low"]
    if len(highs) < 3:
        return None
    h1, h2, h3 = highs[-3:]
    if min(h2.index - h1.index, h3.index - h2.index) < 4:
        return None
    if not (12 <= h3.index - h1.index <= 140):
        return None
    l1 = min((l for l in lows if h1.index < l.index < h2.index), default=None, key=lambda p: p.price)
    l2 = min((l for l in lows if h2.index < l.index < h3.index), default=None, key=lambda p: p.price)
    if l1 is None or l2 is None:
        return None
    atr = compute_atr(candles)
    resistance = median_price([h1, h2, h3])
    tol = dynamic_tolerance(resistance, atr, percent=0.018, atr_mult=1.2)
    if max(abs(x.price - resistance) for x in [h1, h2, h3]) > tol:
        return None
    if max(l1.price, l2.price) >= resistance - max(atr * 1.2, resistance * 0.015):
        return None
    neckline = min(l1.price, l2.price)
    score = 25 + 15 + 15
    if pretrend_ok(candles, h1.index, bullish_reversal=False):
        score += 10
    if h3.price <= h1.price:
        score += 10
    invalidation = max(h1.price, h2.price, h3.price) + atr * 0.3
    return make_detection(
        "Triple Top",
        candles,
        [h1, l1, h2, l2, h3],
        neckline,
        invalidation,
        "bearish",
        score,
        neckline=neckline,
        support_level=neckline,
        resistance_level=resistance,
        explanation="Price failed three times at resistance and broke below support.",
    )
