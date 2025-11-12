"""High-level indicator assembly utilities."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from .indicators import (
    IndicatorInputs,
    adx_dmi,
    anchored_vwap,
    average_true_range,
    composite_score,
    ichimoku,
    liquidity_proxies,
    money_flow_index,
    on_balance_volume,
    parabolic_sar,
    pivot_points,
    stochastic,
    supertrend,
    volume_weighted_average_price,
    wavetrend,
)


DEFAULT_INDICATOR_CONFIG: dict[str, dict[str, object]] = {
    "supertrend": {"period": 10, "multiplier": 3.0},
    "ichimoku": {
        "conversion_period": 9,
        "base_period": 26,
        "span_b_period": 52,
        "displacement": 26,
    },
    "stochastic": {"k_period": 14, "d_period": 3, "smooth_k": 3},
    "atr": {"period": 14},
    "vwap": {},
    "anchored_vwap": {"anchor": None},
    "adx": {"period": 14},
    "mfi": {"period": 14},
    "parabolic_sar": {"acceleration": 0.02, "maximum": 0.2},
    "pivot_points": {"method": "classic"},
    "wavetrend": {"channel_length": 10, "average_length": 21},
    "liquidity": {"window": 20},
    "composite": {"columns": None},
}


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


def _bollinger(
    series: pd.Series, window: int = 20, num_std: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    sma = series.rolling(window=window, min_periods=1).mean()
    std = series.rolling(window=window, min_periods=1).std(ddof=0)
    upper = sma + num_std * std
    lower = sma - num_std * std
    bandwidth = (upper - lower) / sma.replace(0, np.nan)
    percent_b = (series - lower) / (upper - lower).replace(0, np.nan)
    return sma, upper, lower, bandwidth, percent_b


def compute_indicators(
    df: pd.DataFrame,
    config: Mapping[str, Mapping[str, object]] | None = None,
) -> IndicatorResult:
    """Compute an enriched set of technical indicators for ``df``.

    Parameters
    ----------
    df:
        Price dataframe containing at least ``High``, ``Low``, ``Close`` columns.
    config:
        Optional mapping overriding defaults for specific indicators.
    """

    if "Close" not in df.columns:
        raise ValueError("The dataframe must contain a 'Close' column to compute indicators.")

    config_map = {
        name: dict(DEFAULT_INDICATOR_CONFIG[name])
        for name in DEFAULT_INDICATOR_CONFIG
    }
    if config:
        for name, overrides in config.items():
            if name in config_map:
                config_map[name].update(overrides)

    close = pd.to_numeric(df["Close"], errors="coerce")
    volume = pd.to_numeric(df.get("Volume"), errors="coerce") if "Volume" in df else None
    high = pd.to_numeric(df.get("High", close), errors="coerce")
    low = pd.to_numeric(df.get("Low", close), errors="coerce")
    open_ = pd.to_numeric(df.get("Open", close), errors="coerce")

    date_index = None
    if "Date" in df.columns:
        date_index = pd.to_datetime(df["Date"], errors="coerce")
        if date_index.notna().any():
            close = close.set_axis(date_index)
            high = high.set_axis(date_index)
            low = low.set_axis(date_index)
            open_ = open_.set_axis(date_index)
            if volume is not None:
                volume = volume.set_axis(date_index)

    inputs = IndicatorInputs(high=high, low=low, close=close, volume=volume, open=open_)

    ema12 = _ema(close, 12)
    ema26 = _ema(close, 26)
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False, min_periods=1).mean()
    histogram = macd - signal

    sma20, bb_upper, bb_lower, bb_bandwidth, bb_percent_b = _bollinger(close, window=20)

    index = close.index

    indicators = pd.DataFrame(
        {
            "SMA_20": sma20,
            "SMA_50": close.rolling(window=50, min_periods=1).mean(),
            "SMA_200": close.rolling(window=200, min_periods=1).mean(),
            "EMA_12": ema12,
            "EMA_26": ema26,
            "MACD_12_26_9_Line": macd,
            "MACD_12_26_9_Signal": signal,
            "MACD_12_26_9_Hist": histogram,
            "RSI_14": _rsi(close, period=14),
            "BB_Middle_20": sma20,
            "BB_Upper_20": bb_upper,
            "BB_Lower_20": bb_lower,
            "BB_Bandwidth": bb_bandwidth.fillna(0.0),
            "BB_Percent_B": bb_percent_b.fillna(0.5),
        },
        index=index,
    )

    atr_df = average_true_range(inputs, period=int(config_map["atr"]["period"]))
    indicators = indicators.join(atr_df, how="outer")

    supertrend_df = supertrend(
        inputs,
        period=int(config_map["supertrend"]["period"]),
        multiplier=float(config_map["supertrend"]["multiplier"]),
    )
    indicators = indicators.join(supertrend_df, how="outer")

    ichimoku_df = ichimoku(inputs, **config_map["ichimoku"])
    indicators = indicators.join(ichimoku_df, how="outer")

    stochastic_df = stochastic(inputs, **config_map["stochastic"])
    indicators = indicators.join(stochastic_df, how="outer")

    adx_df = adx_dmi(inputs, period=int(config_map["adx"]["period"]))
    indicators = indicators.join(adx_df, how="outer")

    parabolic_df = parabolic_sar(inputs, **config_map["parabolic_sar"])
    indicators = indicators.join(parabolic_df, how="outer")

    pivot_df = pivot_points(inputs, **config_map["pivot_points"])
    indicators = indicators.join(pivot_df, how="outer")

    wavetrend_df = wavetrend(inputs, **config_map["wavetrend"])
    indicators = indicators.join(wavetrend_df, how="outer")

    liquidity_df = liquidity_proxies(inputs, **config_map["liquidity"])
    indicators = indicators.join(liquidity_df, how="outer")

    vwap_df = volume_weighted_average_price(inputs)
    indicators = indicators.join(vwap_df, how="outer")

    anchor = config_map["anchored_vwap"].get("anchor")
    if anchor:
        indicators = indicators.join(anchored_vwap(inputs, anchor=anchor), how="outer")

    indicators = indicators.join(on_balance_volume(inputs), how="outer")
    indicators = indicators.join(money_flow_index(inputs, period=int(config_map["mfi"]["period"])) , how="outer")

    composite_df = composite_score(indicators, columns=config_map["composite"].get("columns"))
    indicators = indicators.join(composite_df, how="outer")

    indicators = indicators.ffill().bfill()
    return IndicatorResult(indicators, list(indicators.columns))


__all__ = ["IndicatorResult", "compute_indicators", "DEFAULT_INDICATOR_CONFIG"]
