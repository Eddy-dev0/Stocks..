"""Momentum oriented indicator functions."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import IndicatorInputs, TA_LIB_AVAILABLE, talib


def stochastic(
    inputs: IndicatorInputs,
    *,
    k_period: int = 14,
    d_period: int = 3,
    smooth_k: int = 3,
) -> pd.DataFrame:
    """Compute fast and slow stochastic oscillators."""

    high = inputs.high
    low = inputs.low
    close = inputs.close

    if TA_LIB_AVAILABLE:  # pragma: no branch
        fast_k, fast_d = talib.STOCHF(
            high,
            low,
            close,
            fastk_period=k_period,
            fastd_period=smooth_k,
            fastd_matype=0,
        )
        slow_k, slow_d = talib.STOCH(
            high,
            low,
            close,
            fastk_period=k_period,
            slowk_period=smooth_k,
            slowd_period=d_period,
            slowk_matype=0,
            slowd_matype=0,
        )
    else:
        lowest_low = low.rolling(window=k_period, min_periods=1).min()
        highest_high = high.rolling(window=k_period, min_periods=1).max()
        raw_k = 100 * (close - lowest_low) / (highest_high - lowest_low).replace(0, np.nan)
        fast_k = raw_k.rolling(window=smooth_k, min_periods=1).mean()
        fast_d = fast_k.rolling(window=d_period, min_periods=1).mean()
        slow_k = fast_k
        slow_d = fast_d

    result = pd.DataFrame(
        {
            f"Stoch_Fast_%K_{k_period}": fast_k,
            f"Stoch_Fast_%D_{d_period}": fast_d,
            f"Stoch_Slow_%K_{k_period}": slow_k,
            f"Stoch_Slow_%D_{d_period}": slow_d,
        }
    )
    return result


def wavetrend(
    inputs: IndicatorInputs,
    *,
    channel_length: int = 10,
    average_length: int = 21,
) -> pd.DataFrame:
    """Return the WaveTrend oscillator."""

    typical_price = inputs.typical_price
    esa = typical_price.ewm(span=channel_length, adjust=False).mean()
    deviation = (typical_price - esa).abs().ewm(span=channel_length, adjust=False).mean()
    ci = (typical_price - esa) / deviation.replace(0, np.nan)
    tci = ci.ewm(span=average_length, adjust=False).mean()
    wt1 = tci
    wt2 = wt1.rolling(window=4, min_periods=1).mean()
    wt_diff = wt1 - wt2
    return pd.DataFrame(
        {
            f"WaveTrend_{channel_length}_{average_length}": wt1,
            f"WaveTrend_Signal_{channel_length}_{average_length}": wt2,
            f"WaveTrend_Hist_{channel_length}_{average_length}": wt_diff,
        }
    )


def commodity_channel_index(inputs: IndicatorInputs, *, period: int = 20) -> pd.DataFrame:
    """Compute the Commodity Channel Index (CCI)."""

    high = inputs.high
    low = inputs.low
    close = inputs.close

    if TA_LIB_AVAILABLE:  # pragma: no branch
        cci = talib.CCI(high, low, close, timeperiod=period)
    else:
        typical_price = inputs.typical_price
        rolling_mean = typical_price.rolling(window=period, min_periods=1).mean()
        rolling_mean_dev = typical_price.rolling(window=period, min_periods=1).apply(
            lambda window: np.mean(np.abs(window - window.mean())), raw=False
        )
        denominator = 0.015 * rolling_mean_dev.replace(0, np.nan)
        cci = (typical_price - rolling_mean) / denominator
    return pd.DataFrame({f"CCI_{period}": cci})


def composite_score(indicators: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Combine provided indicator columns into a composite z-score."""

    if columns is None:
        candidates = [
            column
            for column in (
                "RSI_14",
                "MACD_12_26_9_Hist",
                "ADX_14",
                "WaveTrend_Hist_10_21",
            )
            if column in indicators
        ]
        columns = candidates

    if not columns:
        return pd.DataFrame({"Composite_Score": pd.Series(0.0, index=indicators.index)})

    zscores = []
    for column in columns:
        series = indicators[column]
        z = (series - series.rolling(window=63, min_periods=5).mean()) / series.rolling(window=63, min_periods=5).std()
        zscores.append(z)
    composite = pd.concat(zscores, axis=1).mean(axis=1)
    return pd.DataFrame({"Composite_Score": composite.fillna(0.0)})


__all__ = [
    "stochastic",
    "wavetrend",
    "commodity_channel_index",
    "composite_score",
]
