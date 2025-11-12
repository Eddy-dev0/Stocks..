"""Support and resistance related indicators."""

from __future__ import annotations

import pandas as pd

from .utils import IndicatorInputs


def pivot_points(inputs: IndicatorInputs, method: str = "classic") -> pd.DataFrame:
    """Compute classic daily pivot points."""

    high = inputs.high.shift(1)
    low = inputs.low.shift(1)
    close = inputs.close.shift(1)

    pivot = (high + low + close) / 3

    if method.lower() != "classic":  # pragma: no cover - defensive branch
        raise ValueError(f"Unsupported pivot point method: {method}")

    r1 = 2 * pivot - low
    s1 = 2 * pivot - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)
    r3 = high + 2 * (pivot - low)
    s3 = low - 2 * (high - pivot)

    return pd.DataFrame(
        {
            "Pivot_Point": pivot,
            "Resistance_1": r1,
            "Support_1": s1,
            "Resistance_2": r2,
            "Support_2": s2,
            "Resistance_3": r3,
            "Support_3": s3,
        }
    )


__all__ = ["pivot_points"]
