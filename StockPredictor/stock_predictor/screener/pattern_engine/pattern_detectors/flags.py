from __future__ import annotations

from ..swing_points import compute_atr, find_swing_points
from ..types import Candle, PatternDetection
from .common import fit_trendline, make_detection, slope


def detect_pennant(candles: list[Candle]) -> PatternDetection | None:
    if len(candles) < 25:
        return None
    atr = compute_atr(candles)
    pole_start = len(candles) - 24
    pole_end = len(candles) - 12
    pole_move = candles[pole_end].close - candles[pole_start].close
    pole_move_pct = abs(pole_move) / max(candles[pole_start].close, 1e-9)
    if pole_move_pct < 0.04 and abs(pole_move) < atr * 4:
        return None
    cons = candles[pole_end + 1 :]
    if not (5 <= len(cons) <= 40):
        return None
    swings = find_swing_points(cons, left_bars=1, right_bars=1, min_move_atr=0.1)
    highs = [s for s in swings if s.kind == "high"]
    lows = [s for s in swings if s.kind == "low"]
    if len(highs) < 2 or len(lows) < 2:
        return None
    upper = fit_trendline(highs)
    lower = fit_trendline(lows)
    if upper is None or lower is None or not (upper.slope < 0 < lower.slope):
        return None
    range_start = upper.start_price - lower.start_price
    range_end = upper.end_price - lower.end_price
    if range_end >= range_start * 0.75:
        return None
    direction = "bullish" if pole_move > 0 else "bearish"
    breakout = upper.end_price if direction == "bullish" else lower.end_price
    invalidation = lower.end_price - atr * 0.5 if direction == "bullish" else upper.end_price + atr * 0.5
    return make_detection(
        "Pennant",
        candles,
        [],
        breakout,
        invalidation,
        direction,
        75.0,
        trendline_upper=upper,
        trendline_lower=lower,
        explanation="Strong impulse move followed by a small converging consolidation.",
    )


def detect_flag(candles: list[Candle]) -> PatternDetection | None:
    if len(candles) < 20:
        return None
    atr = compute_atr(candles)
    pole_start = len(candles) - 20
    pole_end = len(candles) - 10
    pole_move = candles[pole_end].close - candles[pole_start].close
    if pole_move <= 0 or (pole_move / candles[pole_start].close < 0.04 and pole_move < atr * 4):
        return None
    flag = candles[pole_end + 1 :]
    if not (5 <= len(flag) <= 50):
        return None
    highs = [c.high for c in flag]
    lows = [c.low for c in flag]
    upper_s = slope(highs)
    lower_s = slope(lows)
    parallel_diff = abs(upper_s - lower_s)
    if parallel_diff > max(abs(upper_s), abs(lower_s), 1e-9) * 0.35:
        return None
    retrace = (candles[pole_end].close - min(lows)) / max(pole_move, 1e-9)
    if retrace > 0.5:
        return None
    breakout = max(highs)
    invalidation = min(lows) - atr * 0.5
    return make_detection(
        "Flag",
        candles,
        [],
        breakout,
        invalidation,
        "bullish",
        80.0,
        explanation="Strong upward impulse followed by a controlled pullback channel.",
    )


def detect_bearish_flag(candles: list[Candle]) -> PatternDetection | None:
    if len(candles) < 20:
        return None
    atr = compute_atr(candles)
    pole_start = len(candles) - 20
    pole_end = len(candles) - 10
    pole_move = candles[pole_start].close - candles[pole_end].close
    if pole_move <= 0 or (pole_move / candles[pole_start].close < 0.04 and pole_move < atr * 4):
        return None
    flag = candles[pole_end + 1 :]
    if not (5 <= len(flag) <= 50):
        return None
    highs = [c.high for c in flag]
    lows = [c.low for c in flag]
    upper_s = slope(highs)
    lower_s = slope(lows)
    parallel_diff = abs(upper_s - lower_s)
    if parallel_diff > max(abs(upper_s), abs(lower_s), 1e-9) * 0.35:
        return None
    retrace = (max(highs) - candles[pole_end].close) / max(pole_move, 1e-9)
    if retrace > 0.5:
        return None
    breakdown = min(lows)
    invalidation = max(highs) + atr * 0.5
    return make_detection(
        "Bearish Flag",
        candles,
        [],
        breakdown,
        invalidation,
        "bearish",
        80.0,
        explanation="Strong downward impulse followed by a controlled upward consolidation.",
    )
