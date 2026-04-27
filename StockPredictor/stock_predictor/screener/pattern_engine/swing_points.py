from __future__ import annotations

from statistics import mean

from .types import Candle, SwingPoint


def calculate_atr_series(candles: list[Candle], period: int = 14) -> list[float]:
    if not candles:
        return []
    trs: list[float] = [candles[0].high - candles[0].low]
    for idx in range(1, len(candles)):
        prev_close = candles[idx - 1].close
        cur = candles[idx]
        tr = max(cur.high - cur.low, abs(cur.high - prev_close), abs(cur.low - prev_close))
        trs.append(tr)
    atr: list[float] = []
    for idx in range(len(trs)):
        start = max(0, idx - period + 1)
        atr.append(float(mean(trs[start : idx + 1])))
    return atr


def compute_atr(candles: list[Candle], period: int = 14) -> float:
    atr_series = calculate_atr_series(candles, period=period)
    return atr_series[-1] if atr_series else 0.0


def get_tolerance(price: float, atr: float, percent_tolerance: float = 0.01, atr_multiplier: float = 1.0) -> float:
    return max(price * percent_tolerance, atr * atr_multiplier)


def find_swing_points(
    candles: list[Candle],
    left_bars: int = 3,
    right_bars: int = 3,
    min_move_atr: float = 0.5,
    min_move_percent: float = 0.003,
) -> list[SwingPoint]:
    if len(candles) < left_bars + right_bars + 1:
        return []
    atr_series = calculate_atr_series(candles)
    swings: list[SwingPoint] = []
    last_price: float | None = None
    for i in range(left_bars, len(candles) - right_bars):
        c = candles[i]
        left = candles[i - left_bars : i]
        right = candles[i + 1 : i + 1 + right_bars]
        is_high = all(c.high > item.high for item in left) and all(c.high > item.high for item in right)
        is_low = all(c.low < item.low for item in left) and all(c.low < item.low for item in right)
        if not (is_high or is_low):
            continue
        kind = "high" if is_high else "low"
        price = c.high if is_high else c.low
        if last_price is not None:
            min_move = max(atr_series[i] * min_move_atr, abs(last_price) * min_move_percent)
            if abs(price - last_price) < min_move:
                continue
        swings.append(SwingPoint(index=i, timestamp=c.timestamp, price=price, kind=kind))
        last_price = price
    return swings


def tolerance(candles: list[Candle], percent_tolerance: float = 0.008, atr_multiplier: float = 0.8) -> float:
    if not candles:
        return 0.0
    atr = compute_atr(candles)
    close = candles[-1].close
    return get_tolerance(close, atr, percent_tolerance=percent_tolerance, atr_multiplier=atr_multiplier)
