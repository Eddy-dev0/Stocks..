from __future__ import annotations

from ..types import Candle, PatternDetection, PatternDirection, PatternStatus, PatternType, SwingPoint


def _status_from_levels(
    candles: list[Candle],
    breakout: float,
    invalidation: float,
    direction: PatternDirection,
) -> PatternStatus:
    close = candles[-1].close
    if direction == "bullish":
        if close > breakout:
            return "confirmed"
        if close < invalidation:
            return "failed"
    elif direction == "bearish":
        if close < breakout:
            return "confirmed"
        if close > invalidation:
            return "failed"
    return "forming"


def make_detection(
    pattern_type: PatternType,
    candles: list[Candle],
    points: list[SwingPoint],
    breakout: float,
    invalidation: float,
    direction: PatternDirection,
    score: float,
    neckline: float | None = None,
) -> PatternDetection:
    status = _status_from_levels(candles, breakout, invalidation, direction)
    start_index = points[0].index if points else max(0, len(candles) - 20)
    end_index = points[-1].index if points else len(candles) - 1
    return PatternDetection(
        pattern_type=pattern_type,
        status=status,
        direction=direction,
        score=max(0.0, min(100.0, score)),
        start_index=start_index,
        end_index=end_index,
        breakout_level=breakout,
        invalidation_level=invalidation,
        neckline_level=neckline,
    )
