"""Elliott wave detection utilities for the stock predictor package."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

LABEL_SEQUENCE: tuple[str, ...] = ("1", "2", "3", "4", "5", "A", "B", "C")
LABEL_TO_INDEX: dict[str, int] = {label: idx + 1 for idx, label in enumerate(LABEL_SEQUENCE)}
FIB_CONSTRAINTS: dict[str, tuple[float, float]] = {
    "2": (0.3, 0.8),  # Typical retracement of wave 1
    "3": (1.0, 2.618),  # Wave 3 usually extends beyond wave 1
    "4": (0.236, 0.618),
    "5": (0.382, 1.618),
    "A": (0.382, 0.786),
    "B": (0.382, 0.886),
    "C": (0.618, 1.618),
}
_EPSILON = 1e-9


@dataclass(slots=True)
class WaveSegment:
    """Represents a detected Elliott wave segment."""

    label: str
    start_index: int
    end_index: int
    start_price: float
    end_price: float
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    length: int
    price_change: float
    ratio: float | None

    @property
    def direction(self) -> int:
        """Return +1 for upward waves and -1 for downward waves."""

        return 1 if self.price_change >= 0 else -1


def detect_elliott_waves(
    prices: pd.Series,
    dates: Sequence[pd.Timestamp] | None = None,
    *,
    window: int = 5,
) -> list[WaveSegment]:
    """Detect Elliott wave segments for the provided price series.

    Parameters
    ----------
    prices:
        Ordered price series (typically closing prices).
    dates:
        Optional sequence aligning with ``prices`` that provides timestamps for
        the detected wave pivots. When omitted, ``NaT`` timestamps are used.
    window:
        Number of observations to use on each side when identifying swing
        highs/lows. Larger windows produce fewer, more significant swings.

    Returns
    -------
    list[WaveSegment]
        Labeled wave segments in chronological order.

    Raises
    ------
    ValueError
        If insufficient data is available to detect swings or the resulting
        segments violate the configured Fibonacci constraints.
    """

    if prices is None or prices.empty:
        raise ValueError("Price series is empty; cannot detect waves.")

    values = prices.to_numpy(dtype=float, copy=True)
    valid_mask = ~np.isnan(values)
    if valid_mask.sum() < max(window * 2 + 1, 6):
        raise ValueError("Insufficient valid prices for Elliott wave detection.")

    n = len(values)
    swings: list[dict[str, object]] = []
    last_type: str | None = None

    for idx in range(window, n - window):
        value = values[idx]
        if np.isnan(value):
            continue
        neighbourhood = values[idx - window : idx + window + 1]
        if np.isnan(neighbourhood).any():
            continue
        local_max = neighbourhood.max()
        local_min = neighbourhood.min()
        if np.isclose(value, local_max) and last_type != "H":
            swings.append({"pos": idx, "price": value, "type": "H"})
            last_type = "H"
        elif np.isclose(value, local_min) and last_type != "L":
            swings.append({"pos": idx, "price": value, "type": "L"})
            last_type = "L"

    if len(swings) < 6:
        raise ValueError("Insufficient swing pivots to construct Elliott waves.")

    if swings[0]["type"] == "H":
        swings = swings[1:]
    if len(swings) < 6:
        raise ValueError("Swing detection did not yield enough alternating pivots.")

    wave_segments: list[WaveSegment] = []
    max_segments = min(len(swings) - 1, len(LABEL_SEQUENCE))
    resolved_dates: list[pd.Timestamp] | None = None
    if dates is not None:
        resolved_dates = [pd.Timestamp(d) for d in dates]

    for seq_index in range(max_segments):
        start = swings[seq_index]
        end = swings[seq_index + 1]
        start_pos = int(start["pos"])
        end_pos = int(end["pos"])
        if end_pos <= start_pos:
            continue
        label = LABEL_SEQUENCE[seq_index]
        start_price = float(start["price"])
        end_price = float(end["price"])
        length = end_pos - start_pos
        price_change = end_price - start_price

        base_segment = _get_ratio_base(label, wave_segments)
        ratio = None
        if base_segment is not None and abs(base_segment.price_change) > _EPSILON:
            ratio = abs(price_change) / max(abs(base_segment.price_change), _EPSILON)

        if not _passes_ratio_constraint(label, ratio):
            ratio_msg = "nan" if ratio is None else f"{ratio:.3f}"
            raise ValueError(
                f"Wave {label} violates Fibonacci constraint with ratio {ratio_msg}."
            )

        start_date = (
            resolved_dates[start_pos]
            if resolved_dates is not None and 0 <= start_pos < len(resolved_dates)
            else pd.NaT
        )
        end_date = (
            resolved_dates[end_pos]
            if resolved_dates is not None and 0 <= end_pos < len(resolved_dates)
            else pd.NaT
        )

        wave_segments.append(
            WaveSegment(
                label=label,
                start_index=start_pos,
                end_index=end_pos,
                start_price=start_price,
                end_price=end_price,
                start_date=start_date,
                end_date=end_date,
                length=length,
                price_change=price_change,
                ratio=ratio,
            )
        )

    if not wave_segments:
        raise ValueError("No valid Elliott waves were detected.")

    return wave_segments


def apply_wave_features(
    df: pd.DataFrame,
    *,
    price_column: str = "Close",
    window: int = 5,
) -> tuple[pd.DataFrame, list[WaveSegment]]:
    """Attach Elliott wave derived features to a price dataframe."""

    if price_column not in df.columns:
        LOGGER.warning("Price column '%s' not present; skipping Elliott features.", price_column)
        enriched = df.copy()
        return _ensure_wave_feature_columns(enriched), []

    working = df.copy()
    if "Date" in working.columns:
        working = working.sort_values("Date")
    working = working.reset_index(drop=True)

    prices = working[price_column]
    dates: Sequence[pd.Timestamp] | None = None
    if "Date" in working.columns:
        dates = working["Date"].tolist()

    try:
        waves = detect_elliott_waves(prices, dates=dates, window=window)
    except ValueError as exc:
        LOGGER.info("Elliott wave detection skipped: %s", exc)
        return _ensure_wave_feature_columns(working), []

    working = _ensure_wave_feature_columns(working)
    for wave in waves:
        label_value = LABEL_TO_INDEX.get(wave.label, np.nan)
        indices = working.index[wave.start_index : wave.end_index + 1]
        working.loc[indices, "ElliottWaveNumeric"] = label_value
        working.loc[indices, "ElliottWaveRatio"] = wave.ratio

    working["ElliottWaveNumeric"] = working["ElliottWaveNumeric"].ffill().bfill()
    working["ElliottWaveRatio"] = working["ElliottWaveRatio"].ffill().bfill()

    helper_values = _compute_helper_features(waves)
    for key, value in helper_values.items():
        working[key] = value

    return working, waves


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_ratio_base(label: str, waves: Sequence[WaveSegment]) -> WaveSegment | None:
    if not waves:
        return None
    if label in {"2", "4"}:
        return waves[-1]
    if label == "3":
        return next((wave for wave in waves if wave.label == "1"), waves[0])
    if label == "5":
        return next((wave for wave in waves if wave.label == "1"), waves[-1])
    if label == "A":
        return next((wave for wave in waves if wave.label == "5"), waves[-1])
    if label in {"B", "C"}:
        return next((wave for wave in waves if wave.label == "A"), waves[-1])
    return waves[-1]


def _passes_ratio_constraint(label: str, ratio: float | None) -> bool:
    if ratio is None:
        return True
    bounds = FIB_CONSTRAINTS.get(label)
    if not bounds:
        return True
    lower, upper = bounds
    return lower - 1e-6 <= ratio <= upper + 1e-6


def _ensure_wave_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    default_columns = {
        "ElliottWaveNumeric": np.nan,
        "ElliottWaveRatio": np.nan,
        "ElliottCurrentWaveIndex": np.nan,
        "ElliottPrevWaveLength": np.nan,
        "ElliottPrevWaveRatio": np.nan,
        "ElliottImpulseAvgLength": np.nan,
        "ElliottCorrectiveAvgLength": np.nan,
        "ElliottImpulseCount": np.nan,
        "ElliottCorrectiveCount": np.nan,
    }
    for column, value in default_columns.items():
        if column not in enriched.columns:
            enriched[column] = value
    return enriched


def _compute_helper_features(waves: Iterable[WaveSegment]) -> dict[str, float]:
    waves_list = list(waves)
    if not waves_list:
        return {
            "ElliottCurrentWaveIndex": np.nan,
            "ElliottPrevWaveLength": np.nan,
            "ElliottPrevWaveRatio": np.nan,
            "ElliottImpulseAvgLength": np.nan,
            "ElliottCorrectiveAvgLength": np.nan,
            "ElliottImpulseCount": 0.0,
            "ElliottCorrectiveCount": 0.0,
        }

    last_wave = waves_list[-1]
    impulse_waves = [wave for wave in waves_list if wave.label in {"1", "3", "5"}]
    corrective_waves = [wave for wave in waves_list if wave.label in {"2", "4", "A", "B", "C"}]

    def _average_length(items: List[WaveSegment]) -> float:
        if not items:
            return float("nan")
        return float(np.mean([wave.length for wave in items]))

    return {
        "ElliottCurrentWaveIndex": float(LABEL_TO_INDEX.get(last_wave.label, np.nan)),
        "ElliottPrevWaveLength": float(last_wave.length),
        "ElliottPrevWaveRatio": float(last_wave.ratio) if last_wave.ratio is not None else float("nan"),
        "ElliottImpulseAvgLength": _average_length(impulse_waves),
        "ElliottCorrectiveAvgLength": _average_length(corrective_waves),
        "ElliottImpulseCount": float(len(impulse_waves)),
        "ElliottCorrectiveCount": float(len(corrective_waves)),
    }
