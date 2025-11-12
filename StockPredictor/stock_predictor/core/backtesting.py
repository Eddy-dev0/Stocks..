"""Backtesting utilities for the stock predictor models."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from .models import ModelFactory, classification_metrics, regression_metrics
from sklearn.base import clone
from sklearn.pipeline import Pipeline

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

    def run(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        target: str,
        *,
        preprocessor_template: Optional[Pipeline] = None,
    ) -> BacktestResult:
        task = "classification" if target == "direction" else "regression"
        splits = list(self._generate_splits(len(X)))
        split_metrics: list[Dict[str, float]] = []

        for index, (train_slice, test_slice) in enumerate(splits, start=1):
            X_train, y_train = X.iloc[train_slice], y.iloc[train_slice]
            X_test, y_test = X.iloc[test_slice], y.iloc[test_slice]
            if len(y_test) == 0:
                continue

            pipeline = clone(preprocessor_template) if preprocessor_template is not None else None
            if pipeline is not None:
                pipeline.fit(X_train, y_train)
                X_train_transformed = pipeline.transform(X_train)
                X_test_transformed = pipeline.transform(X_test)
            else:
                X_train_transformed = X_train
                X_test_transformed = X_test

            model = self.model_factory.create(task, calibrate=(target == "direction"))
            model.fit(X_train_transformed, y_train)
            y_pred = model.predict(X_test_transformed)

            if task == "classification":
                metrics = classification_metrics(y_test.to_numpy(), y_pred)
                metrics["directional_accuracy"] = metrics.get("accuracy", 0.0)
            else:
                metrics = regression_metrics(y_test.to_numpy(), y_pred)
                baseline = (
                    X_test["Close_Current"].to_numpy()
                    if "Close_Current" in X_test
                    else np.zeros_like(y_pred)
                )
                predicted_direction = np.sign(y_pred - baseline)
                actual_direction = np.sign(y_test.to_numpy() - baseline)
                metrics["directional_accuracy"] = float(
                    np.mean((predicted_direction >= 0) == (actual_direction >= 0))
                )
                metrics["signed_error"] = float(np.mean(y_pred - y_test.to_numpy()))

            metrics["split"] = index
            metrics["train_size"] = int(len(y_train))
            metrics["test_size"] = int(len(y_test))
            split_metrics.append(metrics)

        if not split_metrics:
            raise RuntimeError("Backtest did not produce any splits. Check dataset size or window configuration.")

        aggregate = {}
        keys = {
            key
            for entry in split_metrics
            for key in entry.keys()
            if key not in {"split", "train_size", "test_size"}
        }
        for key in keys:
            values = [
                entry[key]
                for entry in split_metrics
                if isinstance(entry.get(key), (int, float, np.floating))
                and np.isfinite(entry.get(key))
            ]
            if values:
                aggregate[key] = float(np.mean(values))
        aggregate["test_rows"] = int(sum(entry.get("test_size", 0) for entry in split_metrics))
        aggregate["splits"] = int(len(split_metrics))

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
