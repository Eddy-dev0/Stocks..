from __future__ import annotations

from ..swing_points import compute_atr
from ..types import Candle, PatternDetection, SwingPoint
from .common import dynamic_tolerance, fit_trendline, make_detection, median_price


def detect_ascending_triangle(candles: list[Candle], swings: list[SwingPoint]) -> PatternDetection | None:
    highs = [s for s in swings if s.kind == "high"][-4:]
    lows = [s for s in swings if s.kind == "low"][-4:]
    if len(highs) < 2 or len(lows) < 2:
        return None
    atr = compute_atr(candles)
    resistance = median_price(highs)
    tol = dynamic_tolerance(resistance, atr, percent=0.01, atr_mult=1.0)
    if sum(1 for h in highs if abs(h.price - resistance) <= tol) < 2:
        return None
    low_line = fit_trendline(lows)
    if low_line is None:
        return None
    min_positive_slope = max(atr * 0.02, candles[-1].close * 0.0002)
    if low_line.slope <= min_positive_slope:
        return None
    start_i, end_i = lows[0].index, highs[-1].index
    if not (15 <= end_i - start_i <= 160):
        return None
    range_start = resistance - (low_line.slope * start_i + low_line.intercept)
    range_end = resistance - (low_line.slope * end_i + low_line.intercept)
    if range_end >= range_start * 0.75:
        return None
    score = 20 + 20 + 15 + 10
    return make_detection(
        "Ascending Triangle",
        candles,
        lows + highs,
        resistance,
        (low_line.slope * end_i + low_line.intercept) - atr * 0.5,
        "bullish",
        score,
        resistance_level=resistance,
        support_level=(low_line.slope * end_i + low_line.intercept),
        trendline_lower=low_line,
        explanation="Resistance stayed flat while lows moved higher into a breakout area.",
    )


def detect_descending_triangle(candles: list[Candle], swings: list[SwingPoint]) -> PatternDetection | None:
    lows = [s for s in swings if s.kind == "low"][-4:]
    highs = [s for s in swings if s.kind == "high"][-4:]
    if len(highs) < 2 or len(lows) < 2:
        return None
    atr = compute_atr(candles)
    support = median_price(lows)
    tol = dynamic_tolerance(support, atr, percent=0.01, atr_mult=1.0)
    if sum(1 for l in lows if abs(l.price - support) <= tol) < 2:
        return None
    up_line = fit_trendline(highs)
    if up_line is None:
        return None
    if up_line.slope >= -max(atr * 0.02, candles[-1].close * 0.0002):
        return None
    start_i, end_i = highs[0].index, lows[-1].index
    if not (15 <= end_i - start_i <= 160):
        return None
    range_start = (up_line.slope * start_i + up_line.intercept) - support
    range_end = (up_line.slope * end_i + up_line.intercept) - support
    if range_end >= range_start * 0.75:
        return None
    score = 20 + 20 + 15 + 10
    return make_detection(
        "Descending Triangle",
        candles,
        highs + lows,
        support,
        (up_line.slope * end_i + up_line.intercept) + atr * 0.5,
        "bearish",
        score,
        support_level=support,
        resistance_level=(up_line.slope * end_i + up_line.intercept),
        trendline_upper=up_line,
        explanation="Support stayed flat while highs moved lower into a breakdown area.",
    )
