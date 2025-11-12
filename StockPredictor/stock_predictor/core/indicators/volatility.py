"""Volatility related indicators."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import IndicatorInputs, TA_LIB_AVAILABLE, talib


def average_true_range(
    inputs: IndicatorInputs,
    *,
    period: int = 14,
) -> pd.DataFrame:
    """Return the Average True Range indicator."""

    high = inputs.high
    low = inputs.low
    close = inputs.close

    if TA_LIB_AVAILABLE:  # pragma: no branch - runtime check
        atr = talib.ATR(high, low, close, timeperiod=period)
    else:
        prev_close = close.shift(1)
        tr = pd.concat(
            [
                (high - low).abs(),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(window=period, min_periods=1).mean()
    return pd.DataFrame({f"ATR_{period}": atr})


__all__ = ["average_true_range"]
