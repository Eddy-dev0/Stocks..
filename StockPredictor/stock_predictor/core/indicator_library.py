"""Extended indicator computations and strategy adapters."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Iterable

import numpy as np
import pandas as pd

from .indicators import IndicatorInputs
from .indicators.volatility import average_true_range
from .strategies.lorentzian_classification import LorentzianClassificationStrategy
from .strategies.smart_money import SmartMoneyConceptsStrategy


INDICATOR_TOGGLE_DEFAULTS: dict[str, bool] = {
    "indicator_rsi": True,
    "indicator_moving_averages": True,
    "indicator_macd": True,
    "indicator_bollinger_bands": True,
    "indicator_fibonacci_retracement": True,
    "indicator_stochastic": True,
    "indicator_volume_profile": True,
    "indicator_williams_r_trend_exhaustion": True,
    "indicator_koncorde": True,
    "indicator_lorentzian_classification": True,
    "indicator_cm_williams_vix_fix": True,
    "indicator_smart_money_concepts": True,
    "indicator_hull_suite": True,
    "indicator_laguerre_multi_filter": True,
    "indicator_supertrend": True,
    "indicator_ichimoku": True,
    "indicator_atr": True,
    "indicator_vwap": True,
    "indicator_adx": True,
    "indicator_mfi": True,
    "indicator_cci": True,
    "indicator_parabolic_sar": True,
    "indicator_pivot_points": True,
    "indicator_wavetrend": True,
    "indicator_liquidity": True,
    "indicator_obv": True,
    "indicator_adl": True,
    "indicator_composite": True,
}


def default_indicator_toggles() -> dict[str, bool]:
    """Return default toggles for indicator-level feature flags."""

    return dict(INDICATOR_TOGGLE_DEFAULTS)


def resolve_indicator_toggles(toggles: Mapping[str, bool] | None) -> dict[str, bool]:
    """Merge indicator toggles with defaults."""

    resolved = default_indicator_toggles()
    if toggles:
        for key, value in toggles.items():
            name = str(key).strip().lower()
            if not name:
                continue
            if name in resolved:
                resolved[name] = bool(value)
                continue
            prefixed = f"indicator_{name}"
            if prefixed in resolved:
                resolved[prefixed] = bool(value)
    return resolved


def fibonacci_retracement(
    inputs: IndicatorInputs, *, window: int = 60
) -> pd.DataFrame:
    """Compute rolling Fibonacci retracement levels."""

    close = inputs.close
    high = inputs.high.rolling(window=window, min_periods=1).max()
    low = inputs.low.rolling(window=window, min_periods=1).min()
    span = (high - low).replace(0, np.nan)
    levels = {
        "FIB_23_6": low + span * 0.236,
        "FIB_38_2": low + span * 0.382,
        "FIB_61_8": low + span * 0.618,
        "FIB_Range_High": high,
        "FIB_Range_Low": low,
        "FIB_Close_Position": (close - low) / span,
    }
    return pd.DataFrame(levels, index=close.index)


def _volume_profile_window(
    close_slice: np.ndarray, volume_slice: np.ndarray, bins: int
) -> tuple[float, float]:
    if close_slice.size == 0:
        return np.nan, np.nan
    min_price = np.nanmin(close_slice)
    max_price = np.nanmax(close_slice)
    if not np.isfinite(min_price) or not np.isfinite(max_price):
        return np.nan, np.nan
    if min_price == max_price:
        total_volume = np.nansum(volume_slice)
        return float(total_volume), 1.0
    edges = np.linspace(min_price, max_price, bins + 1)
    bin_index = np.digitize(close_slice, edges, right=False) - 1
    bin_index = np.clip(bin_index, 0, bins - 1)
    volume_per_bin = np.bincount(bin_index, weights=volume_slice, minlength=bins)
    current_bin = np.digitize([close_slice[-1]], edges, right=False)[0] - 1
    current_bin = int(np.clip(current_bin, 0, bins - 1))
    volume_at_price = float(volume_per_bin[current_bin])
    max_volume = float(np.nanmax(volume_per_bin)) if volume_per_bin.size else np.nan
    ratio = volume_at_price / max_volume if max_volume else np.nan
    return volume_at_price, ratio


def volume_profile(
    inputs: IndicatorInputs,
    *,
    window: int = 50,
    bins: int = 20,
) -> pd.DataFrame:
    """Estimate volume profile features for the current price level."""

    close = inputs.close
    if inputs.volume is None:
        return pd.DataFrame(index=close.index)
    volume = inputs.volume.fillna(0.0)

    volume_at_price: list[float] = []
    volume_ratio: list[float] = []
    close_values = close.to_numpy()
    volume_values = volume.to_numpy()

    for idx in range(len(close_values)):
        start = max(0, idx - window + 1)
        slice_close = close_values[start : idx + 1]
        slice_volume = volume_values[start : idx + 1]
        v_at_price, ratio = _volume_profile_window(slice_close, slice_volume, bins)
        volume_at_price.append(v_at_price)
        volume_ratio.append(ratio)

    return pd.DataFrame(
        {
            "Volume_Profile_Volume": volume_at_price,
            "Volume_Profile_Ratio": volume_ratio,
        },
        index=close.index,
    )


def williams_r_trend_exhaustion(
    inputs: IndicatorInputs,
    *,
    period: int = 14,
    signal_period: int = 6,
) -> pd.DataFrame:
    """Compute a Williams %R based trend exhaustion oscillator."""

    high = inputs.high
    low = inputs.low
    close = inputs.close
    highest_high = high.rolling(window=period, min_periods=1).max()
    lowest_low = low.rolling(window=period, min_periods=1).min()
    denom = (highest_high - lowest_low).replace(0, np.nan)
    wpr = -100 * (highest_high - close) / denom
    signal = wpr.ewm(span=signal_period, adjust=False, min_periods=1).mean()
    exhaustion = wpr - signal
    return pd.DataFrame(
        {
            "WPR_Trend": wpr,
            "WPR_Signal": signal,
            "WPR_Exhaustion": exhaustion,
        },
        index=close.index,
    )


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=1).mean()


def _wma(series: pd.Series, window: int) -> pd.Series:
    weights = np.arange(1, window + 1)
    return series.rolling(window=window, min_periods=1).apply(
        lambda values: float(np.dot(values, weights[-len(values) :]) / weights[-len(values) :].sum()),
        raw=True,
    )


def hull_moving_average(series: pd.Series, window: int) -> pd.Series:
    half = max(1, window // 2)
    sqrt_window = max(1, int(np.sqrt(window)))
    wma_half = _wma(series, half)
    wma_full = _wma(series, window)
    return _wma(2 * wma_half - wma_full, sqrt_window)


def hull_suite(
    inputs: IndicatorInputs,
    *,
    fast: int = 9,
    slow: int = 21,
) -> pd.DataFrame:
    """Compute Hull moving averages and slope delta."""

    close = inputs.close
    fast_hma = hull_moving_average(close, fast)
    slow_hma = hull_moving_average(close, slow)
    delta = fast_hma - fast_hma.shift(2)
    return pd.DataFrame(
        {
            f"HMA_{fast}": fast_hma,
            f"HMA_{slow}": slow_hma,
            f"HMA_{fast}_Delta": delta,
        },
        index=close.index,
    )


def koncorde(
    inputs: IndicatorInputs,
    *,
    trend_periods: Iterable[int] = (3, 5, 8, 13),
    volume_window: int = 14,
) -> pd.DataFrame:
    """Approximate Koncorde components using trend and volume indices."""

    close = inputs.close
    volume = inputs.volume if inputs.volume is not None else pd.Series(0.0, index=close.index)
    returns = close.pct_change(fill_method=None).fillna(0.0)

    trend_components = {}
    for period in trend_periods:
        trend_components[f"Koncorde_Trend_{period}"] = _ema(returns, period)

    pvi = pd.Series(np.nan, index=close.index)
    nvi = pd.Series(np.nan, index=close.index)
    pvi.iloc[0] = 1000
    nvi.iloc[0] = 1000
    for idx in range(1, len(close)):
        if volume.iloc[idx] > volume.iloc[idx - 1]:
            pvi.iloc[idx] = pvi.iloc[idx - 1] * (1 + returns.iloc[idx])
            nvi.iloc[idx] = nvi.iloc[idx - 1]
        elif volume.iloc[idx] < volume.iloc[idx - 1]:
            nvi.iloc[idx] = nvi.iloc[idx - 1] * (1 + returns.iloc[idx])
            pvi.iloc[idx] = pvi.iloc[idx - 1]
        else:
            pvi.iloc[idx] = pvi.iloc[idx - 1]
            nvi.iloc[idx] = nvi.iloc[idx - 1]

    pvi_signal = _ema(pvi, volume_window)
    nvi_signal = _ema(nvi, volume_window)
    institutional = pvi - pvi_signal
    retail = nvi - nvi_signal
    avg_price = _ema(close, volume_window)

    data = {
        **trend_components,
        "Koncorde_IVP": pvi,
        "Koncorde_IVN": nvi,
        "Koncorde_Institutional": institutional,
        "Koncorde_Retail": retail,
        "Koncorde_Avg_Price": avg_price,
    }
    return pd.DataFrame(data, index=close.index)


def cm_williams_vix_fix(
    inputs: IndicatorInputs,
    *,
    period: int = 22,
    bollinger_period: int = 20,
    bollinger_std: float = 2.0,
    overbought: float = 90.0,
    oversold: float = 10.0,
) -> pd.DataFrame:
    """Compute the CM Williams Vix Fix indicator."""

    close = inputs.close
    low = inputs.low
    highest_close = close.rolling(window=period, min_periods=1).max()
    wvf = ((highest_close - low) / highest_close.replace(0, np.nan)) * 100
    mid = wvf.rolling(window=bollinger_period, min_periods=1).mean()
    std = wvf.rolling(window=bollinger_period, min_periods=1).std(ddof=0)
    upper = mid + bollinger_std * std
    lower = mid - bollinger_std * std
    return pd.DataFrame(
        {
            "WVF": wvf,
            "WVF_BB_Mid": mid,
            "WVF_BB_Upper": upper,
            "WVF_BB_Lower": lower,
            "WVF_Overbought": (wvf >= overbought).astype(float),
            "WVF_Oversold": (wvf <= oversold).astype(float),
        },
        index=close.index,
    )


def laguerre_filter(series: pd.Series, gamma: float) -> pd.Series:
    l0 = pd.Series(0.0, index=series.index)
    l1 = pd.Series(0.0, index=series.index)
    l2 = pd.Series(0.0, index=series.index)
    l3 = pd.Series(0.0, index=series.index)

    for idx in range(len(series)):
        price = series.iloc[idx]
        if idx == 0:
            l0.iloc[idx] = price
            l1.iloc[idx] = price
            l2.iloc[idx] = price
            l3.iloc[idx] = price
        else:
            l0.iloc[idx] = (1 - gamma) * price + gamma * l0.iloc[idx - 1]
            l1.iloc[idx] = -gamma * l0.iloc[idx] + l0.iloc[idx - 1] + gamma * l1.iloc[idx - 1]
            l2.iloc[idx] = -gamma * l1.iloc[idx] + l1.iloc[idx - 1] + gamma * l2.iloc[idx - 1]
            l3.iloc[idx] = -gamma * l2.iloc[idx] + l2.iloc[idx - 1] + gamma * l3.iloc[idx - 1]

    filt = (l0 + 2 * l1 + 2 * l2 + l3) / 6
    return filt


def _align_timeframe_series(series: pd.Series, rule: str) -> pd.Series:
    if not isinstance(series.index, pd.DatetimeIndex):
        return series
    resampled = series.resample(rule).last().dropna()
    return resampled.reindex(series.index, method="ffill")


def laguerre_multi_filter(
    inputs: IndicatorInputs,
    *,
    mode: str = "ribbon",
    filters: int = 18,
    gamma_min: float = 0.05,
    gamma_max: float = 0.9,
    band_atr_period: int = 14,
    band_atr_multiplier: float = 1.5,
    timeframes: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Compute Laguerre multi-filter ribbon or band mode."""

    close = inputs.close
    data: dict[str, pd.Series] = {}

    gamma_values = np.linspace(gamma_min, gamma_max, filters)
    if mode.lower() == "ribbon":
        for idx, gamma in enumerate(gamma_values, start=1):
            filt = laguerre_filter(close, float(gamma))
            data[f"Laguerre_{idx:02d}"] = filt
    else:
        avg_filter = laguerre_filter(close, float(np.mean(gamma_values)))
        atr_df = average_true_range(inputs, period=band_atr_period)
        atr = atr_df.iloc[:, 0] if not atr_df.empty else pd.Series(0.0, index=close.index)
        data["Laguerre_Avg"] = avg_filter
        data["Laguerre_Band_Upper"] = avg_filter + band_atr_multiplier * atr
        data["Laguerre_Band_Lower"] = avg_filter - band_atr_multiplier * atr

    direction = close.diff().fillna(0.0)
    data["Laguerre_Direction"] = np.where(direction >= 0, 1.0, -1.0)

    if timeframes:
        if not isinstance(close.index, pd.DatetimeIndex):
            return pd.DataFrame(data, index=close.index)
        for label, rule in timeframes.items():
            tf_series = _align_timeframe_series(close, rule)
            tf_inputs = IndicatorInputs(
                high=_align_timeframe_series(inputs.high, rule),
                low=_align_timeframe_series(inputs.low, rule),
                close=tf_series,
                volume=_align_timeframe_series(inputs.volume, rule)
                if inputs.volume is not None
                else None,
                open=_align_timeframe_series(inputs.open, rule) if inputs.open is not None else None,
            )
            tf_data = laguerre_multi_filter(
                tf_inputs,
                mode=mode,
                filters=filters,
                gamma_min=gamma_min,
                gamma_max=gamma_max,
                band_atr_period=band_atr_period,
                band_atr_multiplier=band_atr_multiplier,
                timeframes=None,
            )
            for column, series in tf_data.items():
                data[f"{column}_{label}"] = series

    return pd.DataFrame(data, index=close.index)


def lorentzian_classification(
    inputs: IndicatorInputs,
    *,
    ema_period: int = 200,
    atr_period: int = 14,
    atr_multiplier: float = 1.5,
    supertrend_period: int = 10,
    supertrend_multiplier: float = 3.0,
    lookback: int = 50,
    backtest: bool = False,
) -> pd.DataFrame:
    """Compute Lorentzian classification strategy signals."""

    strategy = LorentzianClassificationStrategy(
        ema_period=ema_period,
        atr_period=atr_period,
        atr_multiplier=atr_multiplier,
        supertrend_period=supertrend_period,
        supertrend_multiplier=supertrend_multiplier,
        lookback=lookback,
    )
    signals, _ = strategy.compute(inputs, backtest=backtest)
    return signals


def smart_money_concepts(
    inputs: IndicatorInputs,
    *,
    swing_window: int = 5,
    range_window: int = 50,
    timeframes: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Compute Smart Money Concepts features with optional multi-timeframe support."""

    strategy = SmartMoneyConceptsStrategy(
        swing_window=swing_window,
        range_window=range_window,
        timeframes=timeframes,
    )
    return strategy.compute(inputs)


__all__ = [
    "cm_williams_vix_fix",
    "default_indicator_toggles",
    "fibonacci_retracement",
    "hull_suite",
    "koncorde",
    "laguerre_multi_filter",
    "lorentzian_classification",
    "resolve_indicator_toggles",
    "smart_money_concepts",
    "volume_profile",
    "williams_r_trend_exhaustion",
]
