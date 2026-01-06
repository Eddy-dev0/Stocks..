"""Smart Money Concepts strategy features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..indicators import IndicatorInputs


def _rolling_extrema(series: pd.Series, window: int, mode: str) -> pd.Series:
    if mode == "max":
        return series.rolling(window=window, min_periods=1).max().shift(1)
    return series.rolling(window=window, min_periods=1).min().shift(1)


def _detect_order_blocks(
    close: pd.Series, open_: pd.Series, high: pd.Series, low: pd.Series, bos: pd.Series
) -> pd.Series:
    order_block = pd.Series(np.nan, index=close.index)
    last_bearish_idx = None
    last_bullish_idx = None
    for idx in range(len(close)):
        if open_.iloc[idx] > close.iloc[idx]:
            last_bearish_idx = idx
        if close.iloc[idx] > open_.iloc[idx]:
            last_bullish_idx = idx

        if bos.iloc[idx] > 0 and last_bearish_idx is not None:
            order_block.iloc[idx] = low.iloc[last_bearish_idx]
        elif bos.iloc[idx] < 0 and last_bullish_idx is not None:
            order_block.iloc[idx] = high.iloc[last_bullish_idx]
    return order_block


def _fair_value_gap(high: pd.Series, low: pd.Series) -> tuple[pd.Series, pd.Series]:
    upper = pd.Series(np.nan, index=high.index)
    lower = pd.Series(np.nan, index=high.index)
    for idx in range(2, len(high)):
        if low.iloc[idx] > high.iloc[idx - 2]:
            upper.iloc[idx] = low.iloc[idx]
            lower.iloc[idx] = high.iloc[idx - 2]
        elif high.iloc[idx] < low.iloc[idx - 2]:
            upper.iloc[idx] = low.iloc[idx - 2]
            lower.iloc[idx] = high.iloc[idx]
    return upper, lower


@dataclass(slots=True)
class SmartMoneyConceptsStrategy:
    """Compute Smart Money Concepts features."""

    swing_window: int = 5
    range_window: int = 50
    timeframes: dict[str, str] | None = None

    def _compute_frame(self, inputs: IndicatorInputs) -> pd.DataFrame:
        close = inputs.close
        high = inputs.high
        low = inputs.low
        open_ = inputs.open if inputs.open is not None else close

        swing_high = _rolling_extrema(high, self.swing_window, "max")
        swing_low = _rolling_extrema(low, self.swing_window, "min")

        bos = pd.Series(0.0, index=close.index)
        bos[close > swing_high] = 1.0
        bos[close < swing_low] = -1.0

        trend = bos.replace(0.0, np.nan).ffill().fillna(0.0)
        choch = pd.Series(0.0, index=close.index)
        trend_change = trend.diff().fillna(0.0)
        choch[(trend_change > 0) & (trend > 0)] = 1.0
        choch[(trend_change < 0) & (trend < 0)] = -1.0

        order_block = _detect_order_blocks(close, open_, high, low, bos)
        fvg_upper, fvg_lower = _fair_value_gap(high, low)

        range_high = high.rolling(window=self.range_window, min_periods=1).max()
        range_low = low.rolling(window=self.range_window, min_periods=1).min()
        mid = (range_high + range_low) / 2
        premium_discount = pd.Series(0.0, index=close.index)
        premium_discount[close > mid] = 1.0
        premium_discount[close < mid] = -1.0

        return pd.DataFrame(
            {
                "SMC_BOS": bos,
                "SMC_CHoCH": choch,
                "SMC_Order_Block": order_block,
                "SMC_FVG_Upper": fvg_upper,
                "SMC_FVG_Lower": fvg_lower,
                "SMC_Premium_Discount": premium_discount,
                "SMC_Range_High": range_high,
                "SMC_Range_Low": range_low,
            },
            index=close.index,
        )

    def compute(self, inputs: IndicatorInputs) -> pd.DataFrame:
        base = self._compute_frame(inputs)
        if not self.timeframes:
            return base

        if not isinstance(base.index, pd.DatetimeIndex):
            return base

        enriched = base.copy()
        for label, rule in self.timeframes.items():
            close = inputs.close.resample(rule).last().dropna()
            if close.empty:
                continue
            tf_inputs = IndicatorInputs(
                high=inputs.high.resample(rule).max().dropna(),
                low=inputs.low.resample(rule).min().dropna(),
                close=close,
                volume=inputs.volume.resample(rule).sum().dropna() if inputs.volume is not None else None,
                open=inputs.open.resample(rule).first().dropna() if inputs.open is not None else None,
            )
            tf_frame = self._compute_frame(tf_inputs)
            tf_frame = tf_frame.reindex(enriched.index, method="ffill")
            for column in tf_frame.columns:
                enriched[f"{column}_{label}"] = tf_frame[column]
        return enriched


__all__ = ["SmartMoneyConceptsStrategy"]
