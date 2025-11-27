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
    accumulation_distribution_line,
    average_true_range,
    chaikin_accumulation_distribution,
    commodity_channel_index,
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


@dataclass(frozen=True)
class ConfluenceAssessment:
    """Summary of whether multiple momentum signals align."""

    passed: bool
    score: float
    components: dict[str, float]


DEFAULT_INDICATOR_CONFIG: dict[str, dict[str, object]] = {
    "moving_averages": {"sma_periods": (10, 20, 50, 100, 200), "ema_periods": (10, 12, 26, 50, 100)},
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
    "cci": {"period": 20},
    "parabolic_sar": {"acceleration": 0.02, "maximum": 0.2},
    "pivot_points": {"method": "classic"},
    "wavetrend": {"channel_length": 10, "average_length": 21},
    "liquidity": {"window": 20},
    "composite": {"columns": None},
    "adl": {"enabled": True, "chaikin_enabled": False, "short_period": 3, "long_period": 10},
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

    ma_config = config_map.get("moving_averages", {})
    sma_periods = set(ma_config.get("sma_periods", (20, 50, 200))) | {20, 50, 200}
    ema_periods = set(ma_config.get("ema_periods", (12, 26))) | {12, 26}

    sma_values = {period: close.rolling(window=period, min_periods=1).mean() for period in sorted(sma_periods)}
    ema_values = {period: _ema(close, period) for period in sorted(ema_periods)}

    ema12 = ema_values[12]
    ema26 = ema_values[26]
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False, min_periods=1).mean()
    histogram = macd - signal

    sma20, bb_upper, bb_lower, bb_bandwidth, bb_percent_b = _bollinger(close, window=20)

    index = close.index

    indicators = pd.DataFrame(
        {
            **{f"SMA_{period}": sma for period, sma in sma_values.items()},
            **{f"EMA_{period}": ema_values[period] for period in sorted(ema_periods)},
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

    cci_df = commodity_channel_index(inputs, period=int(config_map["cci"]["period"]))
    indicators = indicators.join(cci_df, how="outer")

    anchor = config_map["anchored_vwap"].get("anchor")
    if anchor:
        indicators = indicators.join(anchored_vwap(inputs, anchor=anchor), how="outer")

    indicators = indicators.join(on_balance_volume(inputs), how="outer")
    indicators = indicators.join(money_flow_index(inputs, period=int(config_map["mfi"]["period"])) , how="outer")

    adl_config = config_map["adl"]
    if adl_config.get("enabled", True):
        indicators = indicators.join(accumulation_distribution_line(inputs), how="outer")
        if adl_config.get("chaikin_enabled", False):
            indicators = indicators.join(
                chaikin_accumulation_distribution(
                    inputs,
                    short_period=int(adl_config.get("short_period", 3)),
                    long_period=int(adl_config.get("long_period", 10)),
                ),
                how="outer",
            )

    composite_df = composite_score(indicators, columns=config_map["composite"].get("columns"))
    indicators = indicators.join(composite_df, how="outer")

    indicators = indicators.ffill().bfill()
    return IndicatorResult(indicators, list(indicators.columns))


def _parse_ema_periods(index: pd.Index) -> list[int]:
    """Return discovered EMA periods parsed from column names."""

    periods: list[int] = []
    for column in index:
        if not isinstance(column, str) or not column.startswith("EMA_"):
            continue
        try:
            period = int(column.split("_")[1])
        except (IndexError, ValueError):
            continue
        periods.append(period)
    return sorted(set(periods))


def _linear_slope(series: pd.Series) -> float:
    """Return the slope of a series using a simple linear fit."""

    if series.size < 2:
        return 0.0
    x = np.arange(series.size)
    coeffs = np.polyfit(x, series.values, 1)
    return float(coeffs[0])


def _trend_snapshot(
    close: pd.Series, *, label: str, slope_window: int = 5
) -> dict[str, object]:
    """Summarize trend bias and strength for a single timeframe."""

    if not isinstance(close, pd.Series) or close.empty:
        return {"label": label, "bias": "neutral", "strength": 0.0}

    sma_50 = close.rolling(window=50, min_periods=1).mean()
    sma_200 = close.rolling(window=200, min_periods=1).mean()

    window = max(2, min(int(slope_window), len(close)))
    slope_50 = _linear_slope(sma_50.tail(window))
    slope_200 = _linear_slope(sma_200.tail(window))

    rsi_series = _rsi(close, period=14)
    latest_rsi = float(rsi_series.iloc[-1]) if not rsi_series.empty else 50.0

    ema_fast = _ema(close, span=20).iloc[-1]
    ema_slow = _ema(close, span=50).iloc[-1]

    bullish_components: list[float] = []
    bearish_components: list[float] = []

    for slope in (slope_50, slope_200):
        if slope > 0:
            bullish_components.append(1.0)
        elif slope < 0:
            bearish_components.append(1.0)

    if latest_rsi > 55:
        bullish_components.append(min(1.0, (latest_rsi - 50.0) / 50.0))
    elif latest_rsi < 45:
        bearish_components.append(min(1.0, (50.0 - latest_rsi) / 50.0))

    if ema_fast > ema_slow:
        bullish_components.append(1.0)
    elif ema_fast < ema_slow:
        bearish_components.append(1.0)

    bullish_score = float(np.mean(bullish_components)) if bullish_components else 0.0
    bearish_score = float(np.mean(bearish_components)) if bearish_components else 0.0

    bias = "neutral"
    strength = max(bullish_score, bearish_score)
    if bullish_score >= max(0.5, bearish_score + 0.05):
        bias = "bullish"
        strength = bullish_score
    elif bearish_score >= max(0.5, bullish_score + 0.05):
        bias = "bearish"
        strength = bearish_score

    return {
        "label": label,
        "bias": bias,
        "strength": float(np.clip(strength, 0.0, 1.0)),
        "bullish_score": bullish_score,
        "bearish_score": bearish_score,
        "slopes": {"sma50": slope_50, "sma200": slope_200},
        "latest": {
            "rsi": latest_rsi,
            "ema_fast": float(ema_fast),
            "ema_slow": float(ema_slow),
            "sma_50": float(sma_50.iloc[-1]),
            "sma_200": float(sma_200.iloc[-1]),
        },
    }


def compute_multi_timeframe_trends(
    price_df: pd.DataFrame,
    *,
    timeframes: Mapping[str, str] | None = None,
    slope_window: int = 5,
    include_base: bool = True,
    base_label: str = "daily",
) -> dict[str, object]:
    """Assess trend alignment across multiple timeframes.

    The helper resamples close prices to weekly and monthly intervals by
    default, evaluates key indicators (50/200 SMA slopes, EMA stacking, RSI),
    and returns per-timeframe trend biases alongside aggregate alignment
    metadata suitable for user-facing explanations.
    """

    if price_df is None or price_df.empty:
        return {}

    if "Date" not in price_df.columns or "Close" not in price_df.columns:
        return {}

    close = pd.to_numeric(price_df["Close"], errors="coerce")
    dates = pd.to_datetime(price_df["Date"], errors="coerce")
    frame = pd.Series(close.values, index=dates).dropna()
    frame = frame.sort_index()
    if frame.empty:
        return {}

    rules = timeframes or {"weekly": "W", "monthly": "ME"}
    assessments: dict[str, dict[str, object]] = {}

    if include_base:
        assessments[base_label] = _trend_snapshot(frame, label=base_label, slope_window=slope_window)

    for label, rule in rules.items():
        resampled = frame.resample(rule).last().dropna()
        if resampled.empty:
            continue
        assessments[label] = _trend_snapshot(resampled, label=label, slope_window=slope_window)

    if not assessments:
        return {}

    base_bias = assessments.get(base_label, {}).get("bias")
    aligned: list[str] = []
    conflicts: list[str] = []

    for label, snapshot in assessments.items():
        if label == base_label:
            continue
        bias = snapshot.get("bias")
        if bias and base_bias and bias != "neutral" and base_bias != "neutral":
            if bias == base_bias:
                aligned.append(label)
            else:
                conflicts.append(label)

    overall_bias = max(
        assessments.values(),
        key=lambda snap: snap.get("strength", 0.0),
        default={"bias": "neutral"},
    ).get("bias", "neutral")

    alignment_notes: list[str] = []
    if base_bias:
        if aligned and not conflicts:
            labels = ", ".join(label.title() for label in aligned)
            alignment_notes.append(f"{labels} trend {base_bias} aligns with {base_label} bias.")
        elif conflicts and not aligned:
            labels = ", ".join(label.title() for label in conflicts)
            alignment_notes.append(
                f"{labels} trend conflicts with {base_label} bias {base_bias}; expect choppier conviction."
            )
        elif aligned and conflicts:
            labels = ", ".join(label.title() for label in aligned + conflicts)
            alignment_notes.append(
                f"Mixed alignment across {labels} versus {base_label} bias {base_bias}."
            )

    average_strength = float(
        np.nanmean([snap.get("strength", np.nan) for snap in assessments.values()])
    )

    return {
        "timeframes": assessments,
        "base_timeframe": base_label if include_base else None,
        "overall_bias": overall_bias,
        "alignment_notes": alignment_notes,
        "alignment": {
            "aligned": aligned,
            "conflicts": conflicts,
            "agreement": bool(aligned and not conflicts),
        },
        "average_strength": average_strength,
    }


def evaluate_signal_confluence(
    indicator_frame: pd.DataFrame,
    *,
    ema_periods: Sequence[int] = (12, 26, 50),
    rsi_column: str = "RSI_14",
    obv_column: str = "OBV",
    slope_window: int = 5,
    pass_threshold: float = 0.6,
) -> ConfluenceAssessment:
    """Assess whether several momentum signals align.

    The evaluator inspects stacked EMAs (uptrend bias), RSI strength and slope,
    and the On-Balance Volume (OBV) slope. It returns a composite score between
    0 and 1 alongside a boolean gate indicating whether confluence is
    considered present.
    """

    if not isinstance(indicator_frame, pd.DataFrame) or indicator_frame.empty:
        return ConfluenceAssessment(False, 0.0, {})

    window = max(2, int(slope_window))
    frame = indicator_frame.tail(window)
    latest = frame.iloc[-1]

    components: dict[str, float] = {}

    # EMA alignment: favour higher short-term EMAs relative to long-term EMAs.
    discovered_periods = _parse_ema_periods(latest.index)
    relevant_periods = [period for period in ema_periods if period in discovered_periods]
    ema_score = 0.0
    if len(relevant_periods) >= 2:
        ema_values: list[float] = []
        for period in sorted(relevant_periods):
            value = latest.get(f"EMA_{period}")
            numeric = float(value) if pd.notna(value) else np.nan
            ema_values.append(numeric)
        pairs = [(a, b) for a, b in zip(ema_values, ema_values[1:]) if np.isfinite(a) and np.isfinite(b)]
        if pairs:
            positive = sum(1 for short, long in pairs if short > long)
            ema_score = positive / len(pairs)
    components["ema_alignment"] = ema_score

    # RSI bullishness: reward values above 50 and positive slope.
    rsi_score = 0.0
    rsi_series = frame.get(rsi_column)
    if isinstance(rsi_series, pd.Series):
        rsi_tail = rsi_series.dropna().tail(window)
        if not rsi_tail.empty:
            latest_rsi = float(rsi_tail.iloc[-1])
            slope = _linear_slope(rsi_tail)
            above_neutral = max(0.0, (latest_rsi - 50.0) / 50.0)
            slope_bonus = 0.25 if slope > 0 else 0.0
            rsi_score = min(1.0, above_neutral + slope_bonus)
    components["rsi_bullish"] = rsi_score

    # OBV slope: prefer a rising accumulation profile.
    obv_score = 0.0
    obv_series = frame.get(obv_column)
    if isinstance(obv_series, pd.Series):
        obv_tail = obv_series.dropna().tail(window)
        if obv_tail.size >= 2:
            slope = _linear_slope(obv_tail)
            obv_score = 1.0 if slope > 0 else 0.0
    components["obv_slope"] = obv_score

    available_scores = [score for score in components.values() if np.isfinite(score)]
    if available_scores:
        score = float(np.mean(available_scores))
    else:
        score = 0.0

    passed = bool(score >= pass_threshold and ema_score > 0 and rsi_score > 0)

    return ConfluenceAssessment(passed, score, components)


__all__ = [
    "ConfluenceAssessment",
    "IndicatorResult",
    "compute_indicators",
    "compute_multi_timeframe_trends",
    "DEFAULT_INDICATOR_CONFIG",
    "evaluate_signal_confluence",
]
