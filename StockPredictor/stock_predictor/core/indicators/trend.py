"""Trend oriented indicator implementations."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import IndicatorInputs, TA_LIB_AVAILABLE, talib
from .volatility import average_true_range


def supertrend(
    inputs: IndicatorInputs,
    *,
    period: int = 10,
    multiplier: float = 3.0,
    column_prefix: str = "Supertrend",
) -> pd.DataFrame:
    """Compute the Supertrend indicator.

    Parameters
    ----------
    inputs:
        Input OHLC data container.
    period:
        ATR lookback window used in the Supertrend calculation.
    multiplier:
        Multiplier applied to the ATR when computing the bands.
    column_prefix:
        Prefix used for the output column names.
    """

    atr = average_true_range(inputs, period=period)[f"ATR_{period}"]
    hl2 = (inputs.high + inputs.low) / 2

    basic_upper = hl2 + multiplier * atr
    basic_lower = hl2 - multiplier * atr

    final_upper = basic_upper.copy()
    final_lower = basic_lower.copy()

    for i in range(1, len(inputs.close)):
        if inputs.close.iloc[i - 1] > final_upper.iloc[i - 1]:
            final_upper.iloc[i] = min(basic_upper.iloc[i], final_upper.iloc[i - 1])
        else:
            final_upper.iloc[i] = basic_upper.iloc[i]

        if inputs.close.iloc[i - 1] < final_lower.iloc[i - 1]:
            final_lower.iloc[i] = max(basic_lower.iloc[i], final_lower.iloc[i - 1])
        else:
            final_lower.iloc[i] = basic_lower.iloc[i]

    direction = np.where(inputs.close > final_upper.shift(1), 1, np.nan)
    direction = np.where(inputs.close < final_lower.shift(1), -1, direction)
    direction = pd.Series(direction, index=inputs.close.index).ffill().fillna(1)

    trend = pd.Series(np.nan, index=inputs.close.index)
    for i in range(len(inputs.close)):
        if direction.iloc[i] == 1:
            trend.iloc[i] = final_lower.iloc[i]
        else:
            trend.iloc[i] = final_upper.iloc[i]

    result = pd.DataFrame(
        {
            f"{column_prefix}_{period}": trend,
            f"{column_prefix}_Direction_{period}": direction,
        }
    )
    return result


def ichimoku(
    inputs: IndicatorInputs,
    *,
    conversion_period: int = 9,
    base_period: int = 26,
    span_b_period: int = 52,
    displacement: int = 26,
) -> pd.DataFrame:
    """Compute Ichimoku Cloud components."""

    high = inputs.high
    low = inputs.low
    close = inputs.close

    conversion = ((high.rolling(conversion_period).max() + low.rolling(conversion_period).min()) / 2).rename(
        "Ichimoku_Tenkan"
    )
    base = ((high.rolling(base_period).max() + low.rolling(base_period).min()) / 2).rename("Ichimoku_Kijun")
    span_a = ((conversion + base) / 2).shift(displacement).rename("Ichimoku_Senkou_A")
    span_b = (
        (
            high.rolling(span_b_period).max()
            + low.rolling(span_b_period).min()
        )
        / 2
    ).shift(displacement).rename("Ichimoku_Senkou_B")
    lagging = close.shift(-displacement).rename("Ichimoku_Chikou")

    return pd.concat([conversion, base, span_a, span_b, lagging], axis=1)


def adx_dmi(
    inputs: IndicatorInputs,
    *,
    period: int = 14,
) -> pd.DataFrame:
    """Compute Average Directional Index and directional indicators."""

    high = inputs.high
    low = inputs.low
    close = inputs.close

    if TA_LIB_AVAILABLE:  # pragma: no branch - runtime check
        adx = talib.ADX(high, low, close, timeperiod=period)
        plus_di = talib.PLUS_DI(high, low, close, timeperiod=period)
        minus_di = talib.MINUS_DI(high, low, close, timeperiod=period)
    else:
        up_move = high.diff()
        down_move = -low.diff()

        plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
        minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

        tr = average_true_range(inputs, period=1)["ATR_1"]
        atr_smoothed = tr.rolling(window=period, min_periods=1).sum()

        plus_di = 100 * (plus_dm.rolling(window=period, min_periods=1).sum() / atr_smoothed.replace(0, np.nan))
        minus_di = 100 * (minus_dm.rolling(window=period, min_periods=1).sum() / atr_smoothed.replace(0, np.nan))
        dx = (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)) * 100
        adx = dx.rolling(window=period, min_periods=1).mean()

    return pd.DataFrame(
        {
            f"ADX_{period}": adx,
            f"Plus_DI_{period}": plus_di,
            f"Minus_DI_{period}": minus_di,
        }
    )


def parabolic_sar(
    inputs: IndicatorInputs,
    *,
    acceleration: float = 0.02,
    maximum: float = 0.2,
) -> pd.DataFrame:
    """Compute the Parabolic SAR indicator."""

    high = inputs.high
    low = inputs.low

    if TA_LIB_AVAILABLE:  # pragma: no branch
        sar = talib.SAR(high, low, acceleration=acceleration, maximum=maximum)
    else:
        sar = pd.Series(np.nan, index=inputs.close.index)
        ep = high.iloc[0]
        af = acceleration
        long_position = True
        sar.iloc[0] = low.iloc[0]
        for i in range(1, len(high)):
            prev_sar = sar.iloc[i - 1]
            if long_position:
                sar.iloc[i] = prev_sar + af * (ep - prev_sar)
                if low.iloc[i] < sar.iloc[i]:
                    long_position = False
                    sar.iloc[i] = ep
                    ep = low.iloc[i]
                    af = acceleration
                else:
                    if high.iloc[i] > ep:
                        ep = high.iloc[i]
                        af = min(af + acceleration, maximum)
            else:
                sar.iloc[i] = prev_sar + af * (ep - prev_sar)
                if high.iloc[i] > sar.iloc[i]:
                    long_position = True
                    sar.iloc[i] = ep
                    ep = high.iloc[i]
                    af = acceleration
                else:
                    if low.iloc[i] < ep:
                        ep = low.iloc[i]
                        af = min(af + acceleration, maximum)
    return pd.DataFrame({"Parabolic_SAR": sar})


__all__ = [
    "supertrend",
    "ichimoku",
    "adx_dmi",
    "parabolic_sar",
]
