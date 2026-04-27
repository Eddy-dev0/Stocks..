from __future__ import annotations

from ..swing_points import compute_atr, find_swing_points
from ..types import Candle, PatternDetection
from .common import fit_trendline, make_detection


def _detect_channel_core(candles: list[Candle], mode: str) -> PatternDetection | None:
    if len(candles) < 30:
        return None
    window = candles[-60:] if len(candles) >= 60 else candles
    swings = find_swing_points(window, left_bars=2, right_bars=2, min_move_atr=0.2)
    highs = [s for s in swings if s.kind == "high"]
    lows = [s for s in swings if s.kind == "low"]
    if len(highs) < 2 or len(lows) < 2:
        return None
    upper = fit_trendline(highs)
    lower = fit_trendline(lows)
    if upper is None or lower is None:
        return None
    parallel_diff = abs(upper.slope - lower.slope)
    if parallel_diff > max(abs(upper.slope), abs(lower.slope), 1e-9) * 0.35:
        return None
    atr = compute_atr(window)
    inside = 0
    width_sum = 0.0
    for i, c in enumerate(window):
        up = upper.slope * i + upper.intercept
        lo = lower.slope * i + lower.intercept
        width_sum += up - lo
        if lo <= c.close <= up:
            inside += 1
    inside_ratio = inside / len(window)
    width = width_sum / len(window)
    if inside_ratio < 0.7 or width < atr * 2:
        return None

    if mode == "up" and not (upper.slope > 0 and lower.slope > 0):
        return None
    if mode == "down" and not (upper.slope < 0 and lower.slope < 0):
        return None

    direction = "neutral"
    ptype = "Channel"
    explanation = "Price is respecting parallel upper and lower trend boundaries."
    if mode == "up":
        direction = "bullish"
        ptype = "Channel Up"
        explanation = "Price is forming higher highs and higher lows in a rising parallel channel."
    elif mode == "down":
        direction = "bearish"
        ptype = "Channel Down"
        explanation = "Price is forming lower highs and lower lows in a falling parallel channel."

    return make_detection(
        ptype,
        window,
        highs[-2:] + lows[-2:],
        upper.end_price,
        lower.end_price,
        direction,
        80.0,
        trendline_upper=upper,
        trendline_lower=lower,
        support_level=lower.end_price,
        resistance_level=upper.end_price,
        explanation=explanation,
    )


def detect_channel(candles: list[Candle]) -> PatternDetection | None:
    return _detect_channel_core(candles, "neutral")


def detect_channel_up(candles: list[Candle]) -> PatternDetection | None:
    return _detect_channel_core(candles, "up")


def detect_channel_down(candles: list[Candle]) -> PatternDetection | None:
    return _detect_channel_core(candles, "down")
