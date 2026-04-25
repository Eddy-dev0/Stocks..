from __future__ import annotations

from statistics import mean

from .types import Candle, SwingPoint


def compute_atr(candles: list[Candle], period: int = 14) -> float:
    if len(candles) < 2:
        return 0.0
    trs: list[float] = []
    for idx in range(1, len(candles)):
        prev_close = candles[idx - 1].close
        cur = candles[idx]
        tr = max(cur.high - cur.low, abs(cur.high - prev_close), abs(cur.low - prev_close))
        trs.append(tr)
    if not trs:
        return 0.0
    return float(mean(trs[-period:]))


def find_swing_points(
    candles: list[Candle],
    left_bars: int = 3,
    right_bars: int = 3,
    min_move_atr: float = 0.5,
) -> list[SwingPoint]:
    if len(candles) < left_bars + right_bars + 1:
        return []
    atr = compute_atr(candles)
    swings: list[SwingPoint] = []
    last_price: float | None = None
    for i in range(left_bars, len(candles) - right_bars):
        win = candles[i - left_bars : i + right_bars + 1]
        c = candles[i]
        is_high = all(c.high >= item.high for item in win)
        is_low = all(c.low <= item.low for item in win)
        if not (is_high or is_low):
            continue
        price = c.high if is_high else c.low
        if last_price is not None and atr > 0:
            if abs(price - last_price) < atr * min_move_atr:
                continue
        swings.append(
            SwingPoint(
                index=i,
                timestamp=c.timestamp,
                price=price,
                kind="high" if is_high else "low",
            )
        )
        last_price = price
    return swings


def tolerance(candles: list[Candle]) -> float:
    if not candles:
        return 0.0
    atr = compute_atr(candles)
    close = candles[-1].close
    return max(0.5 * atr, close * 0.003)
