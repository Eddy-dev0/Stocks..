"""Liquidity and sentiment proxy indicators."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import IndicatorInputs
from .volatility import average_true_range


def liquidity_proxies(
    inputs: IndicatorInputs,
    *,
    window: int = 20,
) -> pd.DataFrame:
    """Compute simple liquidity proxy indicators."""

    close = inputs.close
    volume = inputs.volume
    if volume is None:
        volume = pd.Series(np.nan, index=close.index)

    atr = average_true_range(inputs, period=max(window // 2, 5))[f"ATR_{max(window // 2, 5)}"]
    dollar_volume = close * volume
    volatility_of_volume = volume.rolling(window=window, min_periods=5).std()
    turnover = volume / volume.rolling(window=window, min_periods=5).mean()
    impact_proxy = atr / dollar_volume.replace(0, np.nan)

    sentiment_proxy = close.pct_change().rolling(window=window, min_periods=5).corr(volume.pct_change())

    return pd.DataFrame(
        {
            f"Liquidity_DollarVolume_{window}": dollar_volume.rolling(window=window, min_periods=1).mean(),
            f"Liquidity_VolumeVolatility_{window}": volatility_of_volume,
            f"Liquidity_Turnover_{window}": turnover,
            f"Liquidity_ImpactProxy_{window}": impact_proxy,
            f"Sentiment_VolumeCorrelation_{window}": sentiment_proxy,
        }
    )


__all__ = ["liquidity_proxies"]
