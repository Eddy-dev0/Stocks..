"""Volume based indicators."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import IndicatorInputs, TA_LIB_AVAILABLE, talib


def volume_weighted_average_price(inputs: IndicatorInputs) -> pd.DataFrame:
    """Compute rolling VWAP."""

    typical = inputs.typical_price
    volume = inputs.volume
    if volume is None:
        vwap = pd.Series(np.nan, index=typical.index)
    else:
        vwap = (typical * volume).cumsum() / volume.cumsum().replace(0, np.nan)
    return pd.DataFrame({"VWAP": vwap})


def anchored_vwap(inputs: IndicatorInputs, anchor: pd.Timestamp | str) -> pd.DataFrame:
    """Compute anchored VWAP starting from ``anchor`` date (inclusive)."""

    typical = inputs.typical_price
    volume = inputs.volume
    if isinstance(anchor, str):
        anchor = pd.Timestamp(anchor)

    index = typical.index
    if isinstance(anchor, str):
        anchor = pd.Timestamp(anchor)

    if volume is None:
        result = pd.Series(np.nan, index=index)
    else:
        if isinstance(anchor, pd.Timestamp) and not isinstance(index, pd.DatetimeIndex):
            raise ValueError("Anchored VWAP requires datetime indexed price data when using timestamp anchors.")
        mask = index >= anchor
        mask = pd.Series(mask, index=index)
        cum_volume = volume.where(mask).cumsum()
        cum_price_volume = (typical * volume).where(mask).cumsum()
        result = (cum_price_volume / cum_volume.replace(0, np.nan)).where(mask)

    name = f"Anchored_VWAP_{pd.Timestamp(anchor).date()}"
    return pd.DataFrame({name: result})


def on_balance_volume(inputs: IndicatorInputs) -> pd.DataFrame:
    """Compute the On-Balance Volume indicator."""

    close = inputs.close
    volume = inputs.volume
    if volume is None:
        obv = pd.Series(0.0, index=close.index)
    else:
        direction = np.sign(close.diff().fillna(0))
        obv = (direction * volume.fillna(0)).cumsum()
    return pd.DataFrame({"OBV": obv})


def money_flow_index(
    inputs: IndicatorInputs,
    *,
    period: int = 14,
) -> pd.DataFrame:
    """Compute the Money Flow Index indicator."""

    high = inputs.high
    low = inputs.low
    close = inputs.close
    volume = inputs.volume

    if volume is None:
        return pd.DataFrame({f"MFI_{period}": pd.Series(np.nan, index=close.index)})

    if TA_LIB_AVAILABLE:  # pragma: no branch
        mfi = talib.MFI(high, low, close, volume, timeperiod=period)
    else:
        typical_price = (high + low + close) / 3
        raw_money_flow = typical_price * volume
        positive_flow = raw_money_flow.where(typical_price.diff() > 0, 0.0)
        negative_flow = raw_money_flow.where(typical_price.diff() < 0, 0.0)
        positive = positive_flow.rolling(window=period, min_periods=1).sum()
        negative = negative_flow.rolling(window=period, min_periods=1).sum()
        mfr = positive / negative.replace(0, np.nan)
        mfi = 100 - (100 / (1 + mfr))
    return pd.DataFrame({f"MFI_{period}": mfi})


def accumulation_distribution_line(inputs: IndicatorInputs) -> pd.DataFrame:
    """Compute the Accumulation/Distribution Line (ADL)."""

    high = inputs.high
    low = inputs.low
    close = inputs.close
    volume = inputs.volume

    if volume is None:
        adl = pd.Series(np.nan, index=close.index)
    else:
        price_range = (high - low).replace(0, np.nan)
        money_flow_multiplier = ((close - low) - (high - close)) / price_range
        money_flow_multiplier = money_flow_multiplier.fillna(0.0)
        money_flow_volume = money_flow_multiplier * volume.fillna(0.0)
        adl = money_flow_volume.cumsum()

    return pd.DataFrame({"ADL": adl})


def chaikin_accumulation_distribution(
    inputs: IndicatorInputs, *, short_period: int = 3, long_period: int = 10
) -> pd.DataFrame:
    """Compute the Chaikin Accumulation/Distribution oscillator."""

    adl = accumulation_distribution_line(inputs)["ADL"]
    short_ema = adl.ewm(span=short_period, adjust=False, min_periods=1).mean()
    long_ema = adl.ewm(span=long_period, adjust=False, min_periods=1).mean()
    chaikin = short_ema - long_ema
    name = f"Chaikin_AD_{short_period}_{long_period}"
    return pd.DataFrame({name: chaikin})


__all__ = [
    "volume_weighted_average_price",
    "anchored_vwap",
    "on_balance_volume",
    "money_flow_index",
    "accumulation_distribution_line",
    "chaikin_accumulation_distribution",
]
