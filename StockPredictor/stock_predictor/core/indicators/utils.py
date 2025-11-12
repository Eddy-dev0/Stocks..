"""Utility helpers for indicator computations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

try:  # pragma: no cover - optional dependency
    import talib  # type: ignore
except Exception:  # pragma: no cover - defensive import guard
    talib = None


@dataclass(frozen=True)
class IndicatorInputs:
    """Container bundling typical OHLCV inputs for indicators."""

    high: pd.Series
    low: pd.Series
    close: pd.Series
    volume: pd.Series | None = None
    open: pd.Series | None = None

    @property
    def typical_price(self) -> pd.Series:
        """Return the standard typical price series."""

        high = self.high.fillna(self.close)
        low = self.low.fillna(self.close)
        return (high + low + self.close.fillna(self.close)) / 3


TA_LIB_AVAILABLE = talib is not None


def ensure_series(data: Iterable[float] | pd.Series, index: pd.Index) -> pd.Series:
    """Ensure the provided ``data`` is represented as a pandas series."""

    if isinstance(data, pd.Series):
        return data.reindex(index)
    return pd.Series(list(data), index=index, dtype="float64")


__all__ = [
    "IndicatorInputs",
    "TA_LIB_AVAILABLE",
    "ensure_series",
    "talib",
]
