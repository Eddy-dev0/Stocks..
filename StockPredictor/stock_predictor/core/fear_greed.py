"""Computation helpers for fear and greed style indicators."""

from __future__ import annotations

import numpy as np
import pandas as pd

LOGGER_NAME = __name__


def _scale_to_index(series: pd.Series) -> pd.Series:
    """Normalise a series to the 0-100 fear/greed scale."""

    if series.empty:
        return series
    centered = series.fillna(0.0)
    # Clamp extreme outliers to stabilise the score before applying ``tanh``.
    std = centered.rolling(window=30, min_periods=5).std(ddof=0)
    std = std.replace(0.0, np.nan).fillna(centered.abs().median() or 1.0)
    z_score = centered / std.replace(0.0, np.nan)
    scaled = (np.tanh(z_score) + 1.0) * 50.0
    return scaled.clip(0.0, 100.0)


def compute_fear_greed_features(
    price_df: pd.DataFrame, *, window: int = 20
) -> pd.DataFrame:
    """Return a dataframe containing synthetic fear & greed scores.

    The implementation is intentionally light-weight so it can operate without
    external macroeconomic datasets. Momentum, realised volatility and volume
    acceleration are combined to produce a ticker-specific score. A smoothed
    market proxy is derived from the same components to provide a broader
    sentiment anchor when dedicated market data is unavailable.
    """

    if price_df.empty or "Close" not in price_df.columns:
        return pd.DataFrame(index=price_df.index)

    close = pd.to_numeric(price_df["Close"], errors="coerce")
    close = close.ffill().bfill()
    returns = close.pct_change().fillna(0.0)

    momentum = returns.rolling(window=window, min_periods=5).mean()
    volatility = returns.rolling(window=window, min_periods=5).std(ddof=0)
    volume_component = None
    if "Volume" in price_df.columns:
        volume = pd.to_numeric(price_df["Volume"], errors="coerce").fillna(0.0)
        volume_component = volume.pct_change().rolling(window=window, min_periods=5).mean()
    else:
        volume_component = pd.Series(0.0, index=close.index)

    momentum_score = _scale_to_index(momentum)
    volatility_score = 100.0 - _scale_to_index(volatility)
    volume_score = _scale_to_index(volume_component)

    composite = pd.concat(
        [momentum_score, volatility_score, volume_score], axis=1
    ).mean(axis=1)
    market_proxy = composite.rolling(window=window * 2, min_periods=window).mean()

    features = pd.DataFrame(
        {
            "FG_Momentum": momentum_score,
            "FG_Volatility": volatility_score,
            "FG_Volume": volume_score,
            "FG_Ticker": composite.clip(0.0, 100.0),
            "FG_Market": market_proxy.clip(0.0, 100.0),
        },
        index=close.index,
    )

    return features.ffill().bfill()


__all__ = ["compute_fear_greed_features"]
