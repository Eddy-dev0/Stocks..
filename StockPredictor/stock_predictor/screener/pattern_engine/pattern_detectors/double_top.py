from __future__ import annotations

from ..swing_points import compute_atr
from ..types import Candle, PatternDetection, ScoreBreakdown, SwingPoint
from .common import make_detection, pretrend_ok, price_tolerance


def detect_double_top(candles: list[Candle], swings: list[SwingPoint], sensitivity: str = "normal") -> PatternDetection | None:
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
    tol = price_tolerance(avg_high, atr, sensitivity)
    high_diff = abs(h1.price - h2.price) / max(avg_high, 1e-9)
    if h2.price > h1.price + tol:
        return None
    neckline = l.price
    neckline_distance = avg_high - neckline
    if neckline_distance <= max(atr * 0.8, avg_high * 0.01):
        return None

    structure = 20
    geometry = 0.0
    symmetry = 0.0
    trend_ctx = 0.0
    neckline_score = 0.0
    penalties = 0.0
    if high_diff <= 0.01:
        symmetry += 20
    elif high_diff <= 0.025:
        symmetry += 15
    elif high_diff <= 0.04:
        symmetry += 8
    else:
        return None
    if neckline_distance >= atr * 2:
        neckline_score += 15
    elif neckline_distance >= atr:
        neckline_score += 8
    else:
        geometry += 2
    if pretrend_ok(candles, h1.index, bullish_reversal=False):
        trend_ctx += 10
    if h2.price <= h1.price:
        geometry += 8
    else:
        geometry += 3
    if bars_between > 45:
        penalties -= 4
    score = structure + geometry + symmetry + trend_ctx + neckline_score + penalties
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
        explanation="Two nearby highs formed resistance with a neckline pivot; breakdown confirms, otherwise forming.",
        score_breakdown=ScoreBreakdown(
            structure=structure,
            geometry=geometry,
            trendContext=trend_ctx,
            symmetry=symmetry,
            neckline=neckline_score,
            penalties=penalties,
            total=score,
        ),
        sensitivity=sensitivity,
    )
