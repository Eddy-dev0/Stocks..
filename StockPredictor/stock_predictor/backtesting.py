"""Backtesting utilities for the stock predictor models."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from .models import ModelFactory, classification_metrics, regression_metrics

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class BacktestResult:
    target: str
    splits: List[Dict[str, float]]
    aggregate: Dict[str, float]


class Backtester:
    """Perform rolling or expanding window backtests on prepared datasets."""

    def __init__(
        self,
        *,
        model_factory: ModelFactory,
        strategy: str,
        window: int,
        step: int,
    ) -> None:
        self.model_factory = model_factory
        self.strategy = strategy
        self.window = max(20, window)
        self.step = max(1, step)

    def run(self, X: pd.DataFrame, y: pd.Series, target: str) -> BacktestResult:
        task = "classification" if target == "direction" else "regression"
        splits = list(self._generate_splits(len(X)))
        split_metrics: list[Dict[str, float]] = []
        predictions: list[float] = []
        actuals: list[float] = []

        for index, (train_slice, test_slice) in enumerate(splits, start=1):
            X_train, y_train = X.iloc[train_slice], y.iloc[train_slice]
            X_test, y_test = X.iloc[test_slice], y.iloc[test_slice]
            if len(y_test) == 0:
                continue

            model = self.model_factory.create(task)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            if task == "classification":
                metrics = classification_metrics(y_test.to_numpy(), y_pred)
                metrics["directional_accuracy"] = metrics.get("accuracy", 0.0)
            else:
                metrics = regression_metrics(y_test.to_numpy(), y_pred)
                baseline = X_test["Close_Current"].to_numpy() if "Close_Current" in X_test else np.zeros_like(y_pred)
                predicted_direction = np.sign(y_pred - baseline)
                actual_direction = np.sign(y_test.to_numpy() - baseline)
                metrics["directional_accuracy"] = float(
                    np.mean((predicted_direction >= 0) == (actual_direction >= 0))
                )

            metrics["split"] = index
            split_metrics.append(metrics)
            predictions.extend(y_pred.tolist())
            actuals.extend(y_test.tolist())

        if not split_metrics:
            raise RuntimeError("Backtest did not produce any splits. Check dataset size or window configuration.")

        if target == "direction":
            aggregate = classification_metrics(np.array(actuals), np.array(predictions))
            aggregate["directional_accuracy"] = aggregate.get("accuracy", 0.0)
        else:
            aggregate = regression_metrics(np.array(actuals), np.array(predictions))
            aggregate["directional_accuracy"] = float(
                np.mean((np.array(predictions) >= 0) == (np.array(actuals) >= 0))
            )

        return BacktestResult(target=target, splits=split_metrics, aggregate=aggregate)

    def _generate_splits(self, n_samples: int) -> Iterable[Tuple[slice, slice]]:
        if self.strategy == "expanding":
            start = self.window
            while start < n_samples:
                train_slice = slice(0, start)
                test_end = min(n_samples, start + self.step)
                yield train_slice, slice(start, test_end)
                start += self.step
        else:  # rolling
            start = self.window
            while start < n_samples:
                train_slice = slice(max(0, start - self.window), start)
                test_end = min(n_samples, start + self.step)
                yield train_slice, slice(start, test_end)
                start += self.step
