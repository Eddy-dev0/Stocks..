"""Utility helpers for extracting conservative support levels from indicator data."""

from __future__ import annotations

from typing import Any, Mapping, Tuple

import numpy as np
import pandas as pd

__all__ = ["indicator_support_floor"]


def _coerce_series(
    source: pd.DataFrame | pd.Series | Mapping[str, Any] | None,
) -> pd.Series | None:
    if source is None:
        return None
    if isinstance(source, pd.DataFrame):
        if source.empty:
            return None
        try:
            return source.iloc[-1]
        except (KeyError, IndexError):
            return None
    if isinstance(source, pd.Series):
        return source
    if isinstance(source, Mapping):
        if not source:
            return None
        return pd.Series(source)
    return None


def _is_support_column(name: str) -> bool:
    token = str(name or "").strip().lower()
    if not token:
        return False
    if token.startswith("bb_lower"):
        return True
    if token.startswith("supertrend") and "direction" not in token and "signal" not in token:
        return True
    if token.startswith("support_"):
        return True
    if token.startswith("pivot_support"):
        return True
    if token.endswith("_support"):
        return True
    if token == "swing_low" or token.endswith("swing_low"):
        return True
    return False


def _safe_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric):
        return None
    return numeric


def indicator_support_floor(
    source: pd.DataFrame | pd.Series | Mapping[str, Any] | None,
) -> Tuple[float | None, dict[str, float]]:
    """Return the tightest positive support level detected in *source* indicators."""

    series = _coerce_series(source)
    if series is None or series.empty:
        return None, {}

    candidates: dict[str, float] = {}
    for column, value in series.items():
        if not _is_support_column(column):
            continue
        numeric = _safe_float(value)
        if numeric is None or numeric <= 0:
            continue
        candidates[str(column)] = float(numeric)

    if not candidates:
        return None, {}

    floor_value = min(candidates.values())
    return float(floor_value), candidates
