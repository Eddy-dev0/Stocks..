"""Utilities for evaluating directional prediction quality.

The UI needs quick access to horizon-level hit rates. This module centralises
the computation of direction counts so both training evaluation and walk-forward
backtests produce consistent metrics that can be persisted alongside the model
artefacts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

import numpy as np
import pandas as pd


@dataclass
class DirectionalStats:
    """Container for aggregating directional prediction results."""

    n_predictions: int = 0
    n_correct: int = 0
    n_wrong: int = 0

    def merge(self, other: "DirectionalStats") -> None:
        """Accumulate counts from another :class:`DirectionalStats` instance."""

        self.n_predictions += int(other.n_predictions)
        self.n_correct += int(other.n_correct)
        self.n_wrong += int(other.n_wrong)

    def as_summary(self, horizon: Optional[int] = None) -> Mapping[str, float | int]:
        """Return a serialisable summary with accuracy and error rate."""

        total = int(self.n_predictions)
        correct = int(self.n_correct)
        wrong = int(self.n_wrong)
        accuracy = float(correct / total) if total else 0.0
        error_rate = float(wrong / total) if total else 0.0

        return {
            "horizon": horizon,
            "n_predictions_h": total,
            "n_correct_h": correct,
            "n_wrong_h": wrong,
            "acc_h": accuracy,
            "error_rate_h": error_rate,
        }


def evaluate_directional_predictions(
    task: str,
    target: str,
    y_true: pd.Series,
    y_pred: np.ndarray,
    raw_features: Optional[pd.DataFrame] = None,
) -> DirectionalStats:
    """Compare predicted vs. actual direction for a single evaluation split."""

    stats = DirectionalStats()
    if y_pred is None or y_true is None:
        return stats

    actual = pd.Series(y_true).reset_index(drop=True)
    predicted = pd.Series(np.asarray(y_pred).reshape(-1))
    n = min(len(actual), len(predicted))
    if n == 0:
        return stats

    actual = actual.iloc[:n]
    predicted = predicted.iloc[:n]

    if task == "classification":
        mask = ~(actual.isna() | predicted.isna())
        stats.n_predictions = int(mask.sum())
        stats.n_correct = int((actual[mask].to_numpy() == predicted[mask].to_numpy()).sum())
        stats.n_wrong = int(stats.n_predictions - stats.n_correct)
        return stats

    baseline = np.zeros(n, dtype=float)
    if target == "close" and raw_features is not None and "Close_Current" in raw_features:
        baseline = raw_features["Close_Current"].to_numpy(dtype=float)[:n]

    predicted_direction = np.sign(predicted.to_numpy(dtype=float) - baseline)
    actual_direction = np.sign(actual.to_numpy(dtype=float) - baseline)
    mask = np.isfinite(predicted_direction) & np.isfinite(actual_direction)
    stats.n_predictions = int(mask.sum())
    stats.n_correct = int(np.sum((predicted_direction[mask] >= 0) == (actual_direction[mask] >= 0)))
    stats.n_wrong = int(stats.n_predictions - stats.n_correct)
    return stats
