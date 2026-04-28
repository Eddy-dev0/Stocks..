from __future__ import annotations

from ..swing_points import compute_atr
from ..types import Candle, PatternDetection, ScoreBreakdown, SwingPoint
from .common import make_detection, pretrend_ok, price_tolerance


def detect_double_bottom(candles: list[Candle], swings: list[SwingPoint], sensitivity: str = "normal") -> PatternDetection | None:
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
    tol = price_tolerance(avg_low, atr, sensitivity)
    low_diff = abs(l1.price - l2.price) / max(avg_low, 1e-9)
    if l2.price < l1.price - tol:
        return None
    neckline = h.price
    neckline_distance = neckline - avg_low
    if neckline_distance <= max(atr * 0.8, avg_low * 0.01):
        return None

    structure = 20
    geometry = 0.0
    symmetry = 0.0
    trend_ctx = 0.0
    neckline_score = 0.0
    penalties = 0.0
    if low_diff <= 0.01:
        symmetry += 20
    elif low_diff <= 0.025:
        symmetry += 15
    elif low_diff <= 0.04:
        symmetry += 8
    else:
        return None
    if neckline_distance >= atr * 2:
        neckline_score += 15
    elif neckline_distance >= atr:
        neckline_score += 8
    else:
        geometry += 2
    if pretrend_ok(candles, l1.index, bullish_reversal=True):
        trend_ctx += 10
    if l2.price >= l1.price:
        geometry += 8
    else:
        geometry += 3
    if bars_between > 45:
        penalties -= 4
    score = structure + geometry + symmetry + trend_ctx + neckline_score + penalties
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
        explanation="Two nearby lows formed support with a neckline pivot; breakout confirms, otherwise forming.",
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
