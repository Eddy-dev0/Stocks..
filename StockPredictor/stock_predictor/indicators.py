"""Technical indicator computations used throughout the project."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class IndicatorResult:
    """Container describing the indicators generated for a dataframe."""

    dataframe: pd.DataFrame
    columns: Sequence[str]


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=1).mean()


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def _bollinger_bands(
    series: pd.Series, window: int = 20, num_std: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    sma = series.rolling(window=window, min_periods=1).mean()
    std = series.rolling(window=window, min_periods=1).std(ddof=0)
    upper = sma + num_std * std
    lower = sma - num_std * std
    bandwidth = (upper - lower) / sma.replace(0, np.nan)
    percent_b = (series - lower) / (upper - lower).replace(0, np.nan)
    return sma, upper, lower, bandwidth, percent_b


def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    if volume.isna().all():
        return pd.Series(np.zeros(len(close)), index=close.index, name="OBV")
    direction = close.diff().fillna(0)
    sign = direction.apply(lambda value: 1 if value > 0 else (-1 if value < 0 else 0))
    obv = (sign * volume.fillna(0)).cumsum()
    return obv.rename("OBV")


def compute_indicators(df: pd.DataFrame) -> IndicatorResult:
    """Compute a broad set of technical indicators for ``df``."""

    if "Close" not in df.columns:
        raise ValueError("The dataframe must contain a 'Close' column to compute indicators.")

    close = pd.to_numeric(df["Close"], errors="coerce")
    volume = (
        pd.to_numeric(df.get("Volume"), errors="coerce")
        if "Volume" in df
        else pd.Series(0.0, index=df.index)
    )

    ema12 = _ema(close, 12)
    ema26 = _ema(close, 26)
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False, min_periods=1).mean()
    histogram = macd - signal

    sma20, bb_upper, bb_lower, bb_bandwidth, bb_percent_b = _bollinger_bands(close, window=20)

    indicators = pd.DataFrame(
        {
            "SMA_20": sma20,
            "SMA_50": close.rolling(window=50, min_periods=1).mean(),
            "SMA_200": close.rolling(window=200, min_periods=1).mean(),
            "EMA_12": ema12,
            "EMA_26": ema26,
            "MACD": macd,
            "MACD_Signal": signal,
            "MACD_Hist": histogram,
            "RSI_14": _rsi(close, period=14),
            "BB_Middle_20": sma20,
            "BB_Upper_20": bb_upper,
            "BB_Lower_20": bb_lower,
            "BB_Bandwidth": bb_bandwidth.fillna(0.0),
            "BB_Percent_B": bb_percent_b.fillna(0.5),
        },
        index=df.index,
    )

    if "Volume" in df.columns:
        indicators["OBV"] = _obv(close, volume)
        indicators["Volume_SMA_20"] = volume.rolling(window=20, min_periods=1).mean()
        indicators["Volume_EMA_20"] = _ema(volume, 20)

    indicators = indicators.fillna(method="ffill").fillna(method="bfill")

    return IndicatorResult(indicators, list(indicators.columns))
