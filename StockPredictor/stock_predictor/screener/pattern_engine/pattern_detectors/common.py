from __future__ import annotations

from statistics import mean, median

from ..swing_points import calculate_atr_series, compute_atr, get_tolerance
from ..types import (
    Candle,
    PatternDetection,
    PatternDirection,
    PatternKeyPoint,
    PatternStatus,
    PatternType,
    SwingPoint,
    Trendline,
)


def slope(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mean_x = (n - 1) / 2
    mean_y = sum(values) / n
    denom = sum((i - mean_x) ** 2 for i in range(n))
    if denom == 0:
        return 0.0
    num = sum((i - mean_x) * (v - mean_y) for i, v in enumerate(values))
    return num / denom


def fit_trendline(points: list[SwingPoint]) -> Trendline | None:
    if len(points) < 2:
        return None
    xs = [float(p.index) for p in points]
    ys = [p.price for p in points]
    mean_x = mean(xs)
    mean_y = mean(ys)
    denom = sum((x - mean_x) ** 2 for x in xs)
    if denom == 0:
        return None
    m = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denom
    b = mean_y - m * mean_x
    start = points[0].index
    end = points[-1].index
    return Trendline(
        start_index=start,
        start_price=m * start + b,
        end_index=end,
        end_price=m * end + b,
        slope=m,
        intercept=b,
    )


def value_at(line: Trendline, index: int) -> float:
    return line.slope * index + line.intercept


def breakout_buffer(close: float, atr: float) -> float:
    return max(close * 0.0015, atr * 0.1)


def volume_score(candles: list[Candle], breakout_index: int | None) -> tuple[float, float]:
    if breakout_index is None or breakout_index < 20 or breakout_index >= len(candles):
        return 0.0, 1.0
    avg20 = mean(c.volume for c in candles[breakout_index - 20 : breakout_index])
    ratio = candles[breakout_index].volume / max(avg20, 1e-9)
    if ratio >= 1.5:
        return 12.0, ratio
    if ratio >= 1.3:
        return 8.0, ratio
    if ratio < 0.8:
        return -8.0, ratio
    return 0.0, ratio


def pretrend_ok(candles: list[Candle], index: int, bullish_reversal: bool, bars: int = 30) -> bool:
    if index - bars < 0:
        return False
    segment = [c.close for c in candles[index - bars : index + 1]]
    s = slope(segment)
    if bullish_reversal:
        return s < 0 or candles[index].close < candles[index - bars].close * 0.97
    return s > 0 or candles[index].close > candles[index - bars].close * 1.03


def determine_status(
    candles: list[Candle],
    direction: PatternDirection,
    breakout_level: float,
    invalidation_level: float,
    breakout_index: int | None,
    scan_start: int,
) -> tuple[PatternStatus, int | None]:
    atr_series = calculate_atr_series(candles)
    failed = any(c.close < invalidation_level for c in candles[scan_start:]) if direction == "bullish" else any(c.close > invalidation_level for c in candles[scan_start:])
    if failed:
        return "failed", None
    rng = range(scan_start, len(candles)) if breakout_index is None else range(breakout_index, len(candles))
    for i in rng:
        b = breakout_buffer(candles[i].close, atr_series[i])
        if direction == "bullish" and candles[i].close > breakout_level + b:
            return "confirmed", i
        if direction == "bearish" and candles[i].close < breakout_level - b:
            return "confirmed", i
    return "forming", None


def make_detection(
    pattern_type: PatternType,
    candles: list[Candle],
    points: list[SwingPoint],
    breakout: float,
    invalidation: float,
    direction: PatternDirection,
    score: float,
    neckline: float | None = None,
    explanation: str = "",
    trendline_upper: Trendline | None = None,
    trendline_lower: Trendline | None = None,
    trendline_neckline: Trendline | None = None,
    support_level: float | None = None,
    resistance_level: float | None = None,
    breakout_index: int | None = None,
) -> PatternDetection:
    start_index = points[0].index if points else max(0, len(candles) - 40)
    end_index = points[-1].index if points else len(candles) - 1
    status, resolved_breakout = determine_status(
        candles,
        direction,
        breakout,
        invalidation,
        breakout_index=breakout_index,
        scan_start=end_index,
    )
    vs, _ = volume_score(candles, resolved_breakout)
    final_score = max(0.0, min(100.0, score + vs))
    kp = tuple(PatternKeyPoint(index=p.index, price=p.price, type=p.kind) for p in points)
    return PatternDetection(
        pattern_type=pattern_type,
        status=status,
        direction=direction,
        score=final_score,
        start_index=start_index,
        end_index=end_index,
        breakout_level=breakout,
        invalidation_level=invalidation,
        neckline_level=neckline,
        signal_index=end_index,
        breakout_index=resolved_breakout,
        support_level=support_level,
        resistance_level=resistance_level,
        trendline_upper=trendline_upper,
        trendline_lower=trendline_lower,
        trendline_neckline=trendline_neckline,
        key_points=kp,
        explanation=explanation,
    )


def dedupe_detections(detections: list[PatternDetection]) -> list[PatternDetection]:
    if not detections:
        return []

    def rank(d: PatternDetection) -> tuple[int, float, int]:
        status_rank = {"confirmed": 3, "forming": 2, "failed": 1, "expired": 0}.get(d.status, 0)
        return status_rank, d.score, d.end_index

    ordered = sorted(detections, key=rank, reverse=True)
    kept: list[PatternDetection] = []
    for det in ordered:
        overlap = False
        for ex in kept:
            inter = max(0, min(det.end_index, ex.end_index) - max(det.start_index, ex.start_index) + 1)
            short = max(1, min(det.end_index - det.start_index + 1, ex.end_index - ex.start_index + 1))
            if inter / short > 0.70:
                overlap = True
                break
        if not overlap:
            kept.append(det)
    return sorted(kept, key=lambda d: d.end_index)


def median_price(points: list[SwingPoint]) -> float:
    return float(median(p.price for p in points)) if points else 0.0


def dynamic_tolerance(price: float, atr: float, percent: float = 0.015, atr_mult: float = 1.0) -> float:
    return get_tolerance(price, atr, percent_tolerance=percent, atr_multiplier=atr_mult)
