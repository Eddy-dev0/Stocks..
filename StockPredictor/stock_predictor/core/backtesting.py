"""Backtesting utilities for the stock predictor models."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from .ml_preprocessing import get_feature_names_from_pipeline
from .models import (
    ModelFactory,
    classification_metrics,
    extract_feature_importance,
    regression_metrics,
)

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class BacktestResult:
    target: str
    splits: List[Dict[str, float]]
    aggregate: Dict[str, float]
    feature_importance: Dict[str, float]
    decision_thresholds: Dict[str, float]


class Backtester:
    """Perform walk-forward, event-driven backtests on prepared datasets."""

    def __init__(
        self,
        *,
        model_factory: ModelFactory,
        strategy: str,
        window: int,
        step: int,
        slippage_bps: float = 1.0,
        fee_bps: float = 1.0,
        neutral_threshold: float = 0.001,
        trading_days: int = 252,
        risk_free_rate: float = 0.0,
    ) -> None:
        self.model_factory = model_factory
        self.strategy = strategy
        self.window = max(20, window)
        self.step = max(1, step)
        self.slippage_bps = max(0.0, float(slippage_bps))
        self.fee_bps = max(0.0, float(fee_bps))
        self.neutral_threshold = float(neutral_threshold)
        self.trading_days = max(1, int(trading_days))
        self.risk_free_rate = float(risk_free_rate)
        self.cost_per_turn = (self.slippage_bps + self.fee_bps) / 10000.0

    def run(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        target: str,
        *,
        preprocessor_template: Optional[Pipeline] = None,
        auxiliary_targets: Optional[pd.DataFrame] = None,
    ) -> BacktestResult:
        task = "classification" if target == "direction" else "regression"
        splits = list(self._generate_splits(len(X)))
        split_metrics: list[Dict[str, float]] = []
        feature_totals: Dict[str, float] = {}
        feature_counts: Dict[str, int] = {}

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
            y_proba = None
            classes = None
            if task == "classification":
                try:
                    y_proba = model.predict_proba(X_test_transformed)
                except Exception:
                    LOGGER.debug("Model does not support predict_proba; skipping probability metrics.")
                estimator = getattr(model, "named_steps", {}).get("estimator")
                classes = getattr(estimator, "classes_", None)

            metrics = self._score_predictions(
                task, y_test, y_pred, X_test, y_proba=y_proba, classes=classes
            )
            auxiliary_test = (
                auxiliary_targets.iloc[test_slice]
                if auxiliary_targets is not None
                else None
            )
            trading_metrics = self._simulate_trading(
                target,
                y_pred,
                y_test,
                X_test,
                auxiliary_test,
                y_proba=y_proba,
                classes=classes,
            )
            metrics.update(trading_metrics)
            metrics["split"] = index
            metrics["train_size"] = int(len(y_train))
            metrics["test_size"] = int(len(y_test))
            split_metrics.append(metrics)

            feature_names = self._resolve_feature_names(pipeline, X_train_transformed)
            importance = extract_feature_importance(model, feature_names)
            for name, value in importance.items():
                feature_totals[name] = feature_totals.get(name, 0.0) + float(value)
                feature_counts[name] = feature_counts.get(name, 0) + 1

        if not split_metrics:
            raise RuntimeError("Backtest did not produce any splits. Check dataset size or window configuration.")

        aggregate = self._aggregate_metrics(split_metrics)
        aggregated_importance = self._normalise_feature_importance(feature_totals, feature_counts)
        threshold_summary = {
            key: value
            for key, value in aggregate.items()
            if "threshold" in key
        }

        return BacktestResult(
            target=target,
            splits=split_metrics,
            aggregate=aggregate,
            feature_importance=aggregated_importance,
            decision_thresholds=threshold_summary,
        )

    def _score_predictions(
        self,
        task: str,
        y_true: pd.Series,
        y_pred: np.ndarray,
        raw_X: pd.DataFrame,
        *,
        y_proba: np.ndarray | None = None,
        classes: np.ndarray | None = None,
    ) -> Dict[str, float]:
        if task == "classification":
            metrics = classification_metrics(
                y_true.to_numpy(), y_pred, y_proba=y_proba, classes=classes
            )
            metrics["directional_accuracy"] = metrics.get("accuracy", 0.0)

            calibration_metrics = self._calibration_by_bin(
                y_true.to_numpy(), y_proba, classes
            )
            metrics.update(calibration_metrics)
            return metrics

        metrics = regression_metrics(y_true.to_numpy(), y_pred)
        baseline = (
            raw_X["Close_Current"].to_numpy()
            if "Close_Current" in raw_X
            else np.zeros_like(y_pred)
        )
        predicted_direction = np.sign(y_pred - baseline)
        actual_direction = np.sign(y_true.to_numpy() - baseline)
        metrics["directional_accuracy"] = float(
            np.mean((predicted_direction >= 0) == (actual_direction >= 0))
        )
        metrics["signed_error"] = float(np.mean(y_pred - y_true.to_numpy()))
        return metrics

    def _calibration_by_bin(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray | None,
        classes: np.ndarray | None,
        *,
        bin_size: float = 0.1,
    ) -> Dict[str, float]:
        if y_proba is None or y_proba.size == 0:
            return {}

        positive_info = self._positive_class_proba(y_proba, classes)
        if positive_info is None:
            return {}

        positive_proba, positive_index = positive_info
        if positive_proba.size == 0:
            return {}

        y_binary = self._binary_targets(y_true, classes, positive_index)
        if y_binary.size == 0:
            return {}

        bins = np.arange(0.0, 1.0, bin_size)
        calibration: Dict[str, float] = {}
        for lower in bins:
            upper = round(min(1.0, lower + bin_size), 2)
            in_bin = (positive_proba >= lower) & (
                positive_proba < upper if upper < 1.0 else positive_proba <= upper
            )
            key = f"calibration_hit_rate_{lower:.1f}_{upper:.1f}"
            if not np.any(in_bin):
                calibration[key] = float("nan")
                continue
            calibration[key] = float(np.mean(y_binary[in_bin]))
        return calibration

    def _positive_class_proba(
        self, y_proba: np.ndarray, classes: np.ndarray | None
    ) -> tuple[np.ndarray, int] | None:
        proba_array = np.asarray(y_proba, dtype=float)
        if proba_array.ndim == 1:
            return np.clip(proba_array, 0.0, 1.0), 0
        if proba_array.shape[1] == 0:
            return None

        positive_index = 1 if proba_array.shape[1] > 1 else 0
        if classes is not None and len(classes) > positive_index:
            for idx, cls in enumerate(classes):
                if cls in {1, True, "1", "up"}:  # type: ignore[operator]
                    positive_index = idx
                    break
        positive_index = int(min(max(0, positive_index), proba_array.shape[1] - 1))
        return np.clip(proba_array[:, positive_index], 0.0, 1.0), positive_index

    def _binary_targets(
        self, y_true: np.ndarray, classes: np.ndarray | None, positive_index: int
    ) -> np.ndarray:
        if classes is not None and len(classes) > 0:
            mapping = {cls: idx for idx, cls in enumerate(classes)}
            positive_label = classes[min(positive_index, len(classes) - 1)]
            return np.array(
                [
                    1
                    if mapping.get(value, value)
                    == mapping.get(positive_label, positive_index)
                    else 0
                    for value in y_true
                ],
                dtype=float,
            )
        return np.array(
            [1 if value in {1, True, "1", "up"} else 0 for value in y_true],
            dtype=float,
        )

    def _simulate_trading(
        self,
        target: str,
        predictions: np.ndarray,
        y_test: pd.Series,
        raw_X: pd.DataFrame,
        auxiliary_test: Optional[pd.DataFrame],
        *,
        y_proba: np.ndarray | None = None,
        classes: np.ndarray | None = None,
    ) -> Dict[str, float]:
        actual_returns = self._actual_returns(target, y_test, raw_X, auxiliary_test)
        thresholds = self._optimise_thresholds(
            target,
            y_proba,
            classes,
            actual_returns,
        )
        if actual_returns.size == 0:
            return {
                "turnover": 0.0,
                "cagr": 0.0,
                "sharpe": 0.0,
                "sortino": 0.0,
                "max_drawdown": 0.0,
                "hit_rate": 0.0,
                "net_return": 0.0,
                **thresholds,
            }

        signals = self._signal_from_predictions(
            target,
            predictions,
            raw_X,
            y_proba=y_proba,
            classes=classes,
            thresholds=thresholds,
        )
        if signals.size == 0:
            return {
                "turnover": 0.0,
                "cagr": 0.0,
                "sharpe": 0.0,
                "sortino": 0.0,
                "max_drawdown": 0.0,
                "hit_rate": 0.0,
                "net_return": 0.0,
                **thresholds,
            }

        stats = self._trading_statistics(signals, actual_returns)
        stats.update(thresholds)
        return stats

    def _trading_statistics(
        self, signals: np.ndarray, actual_returns: np.ndarray
    ) -> Dict[str, float]:
        position_changes = np.diff(np.concatenate(([0.0], signals)))
        turnover = float(np.sum(np.abs(position_changes)))
        transaction_costs = np.abs(position_changes) * self.cost_per_turn
        gross_returns = signals * actual_returns
        net_returns = gross_returns - transaction_costs
        equity_curve = np.cumprod(1 + net_returns)

        stats = self._performance_statistics(net_returns, equity_curve)
        stats.update(
            {
                "turnover": turnover,
                "hit_rate": float(np.mean(net_returns > 0)) if net_returns.size else 0.0,
                "avg_position": float(np.mean(np.abs(signals))) if signals.size else 0.0,
                "trades": float(np.sum(np.abs(position_changes) > 0)),
                "gross_return": float(np.prod(1 + gross_returns) - 1) if gross_returns.size else 0.0,
                "net_return": float(equity_curve[-1] - 1) if equity_curve.size else 0.0,
            }
        )
        return stats

    def _performance_statistics(
        self,
        net_returns: np.ndarray,
        equity_curve: np.ndarray,
    ) -> Dict[str, float]:
        if net_returns.size == 0:
            return {"cagr": 0.0, "sharpe": 0.0, "sortino": 0.0, "max_drawdown": 0.0}

        cumulative = float(equity_curve[-1]) if equity_curve.size else 1.0
        periods = net_returns.size
        result: Dict[str, float] = {"cagr": 0.0, "sharpe": 0.0, "sortino": 0.0, "max_drawdown": 0.0}

        if cumulative > 0 and periods > 0:
            result["cagr"] = float(cumulative ** (self.trading_days / periods) - 1)

        excess = net_returns - (self.risk_free_rate / self.trading_days)
        std = float(np.std(excess, ddof=1)) if periods > 1 else 0.0
        if std > 0:
            result["sharpe"] = float(np.mean(excess) / std * np.sqrt(self.trading_days))

        downside = excess[excess < 0]
        downside_std = float(np.std(downside, ddof=1)) if downside.size > 1 else 0.0
        if downside.size == 1:
            downside_std = float(np.std(downside))
        if downside_std > 0:
            result["sortino"] = float(np.mean(excess) / downside_std * np.sqrt(self.trading_days))

        if equity_curve.size:
            running_max = np.maximum.accumulate(equity_curve)
            safe_running = np.where(running_max == 0, 1.0, running_max)
            drawdowns = equity_curve / safe_running - 1.0
            result["max_drawdown"] = float(np.min(drawdowns))

        return result

    def _actual_returns(
        self,
        target: str,
        y_test: pd.Series,
        raw_X: pd.DataFrame,
        auxiliary_test: Optional[pd.DataFrame],
    ) -> np.ndarray:
        if target == "return":
            return y_test.to_numpy(dtype=float)

        if target == "close":
            baseline = (
                raw_X["Close_Current"].to_numpy(dtype=float)
                if "Close_Current" in raw_X
                else np.zeros_like(y_test.to_numpy(dtype=float))
            )
            denominator = np.clip(np.abs(baseline), 1e-6, None)
            return (y_test.to_numpy(dtype=float) - baseline) / denominator

        if target == "direction":
            if auxiliary_test is not None and "return" in auxiliary_test:
                return auxiliary_test["return"].to_numpy(dtype=float)
            return np.where(y_test.to_numpy(dtype=float) > 0, 0.01, -0.01)

        if target == "volatility":
            if auxiliary_test is not None and "return" in auxiliary_test:
                return auxiliary_test["return"].to_numpy(dtype=float)
            return np.zeros_like(y_test.to_numpy(dtype=float))

        return y_test.to_numpy(dtype=float)

    def _signal_from_predictions(
        self,
        target: str,
        predictions: np.ndarray,
        raw_X: pd.DataFrame,
        *,
        y_proba: np.ndarray | None = None,
        classes: np.ndarray | None = None,
        thresholds: Dict[str, float] | None = None,
    ) -> np.ndarray:
        forecast = np.asarray(predictions, dtype=float)
        if forecast.size == 0:
            return np.array([])

        thresholds = thresholds or {}
        if target == "direction" and y_proba is not None and y_proba.size:
            positive_info = self._positive_class_proba(y_proba, classes)
            if positive_info is None:
                return np.where(forecast >= 0.5, 1.0, -1.0)

            positive_proba, _ = positive_info
            if positive_proba.size == 0:
                return np.where(forecast >= 0.5, 1.0, -1.0)

            long_threshold = max(0.5, float(thresholds.get("long_probability_threshold", 0.5)))
            short_threshold = float(thresholds.get("short_probability_threshold", 0.5))
            signals = np.zeros_like(positive_proba, dtype=float)
            signals[positive_proba >= long_threshold] = 1.0
            signals[positive_proba <= short_threshold] = -1.0
            return signals

        if target == "close" and "Close_Current" in raw_X:
            baseline = raw_X["Close_Current"].to_numpy(dtype=float)
            denominator = np.clip(np.abs(baseline), 1e-6, None)
            forecast = (forecast - baseline) / denominator

        threshold = float(thresholds.get("neutral_threshold", self.neutral_threshold))
        return np.where(
            forecast > threshold,
            1.0,
            np.where(forecast < -threshold, -1.0, 0.0),
        )

    def _optimise_thresholds(
        self,
        target: str,
        y_proba: np.ndarray | None,
        classes: np.ndarray | None,
        actual_returns: np.ndarray,
    ) -> Dict[str, float]:
        thresholds: Dict[str, float] = {
            "neutral_threshold": self.neutral_threshold,
            "long_probability_threshold": 0.5,
            "short_probability_threshold": 0.5,
        }
        if (
            target != "direction"
            or y_proba is None
            or y_proba.size == 0
            or actual_returns.size == 0
        ):
            return thresholds

        positive_info = self._positive_class_proba(y_proba, classes)
        if positive_info is None:
            return thresholds

        positive_proba, _ = positive_info
        if positive_proba.size == 0:
            return thresholds

        candidate_thresholds = sorted(
            set(
                float(value)
                for value in np.concatenate(
                    (
                        np.linspace(0.55, 0.9, 8),
                        np.quantile(positive_proba, [0.6, 0.7, 0.8, 0.9]),
                    )
                )
            )
        )
        best_threshold = 0.5
        best_score = -np.inf

        for threshold in candidate_thresholds:
            if threshold <= 0.5 or threshold >= 1.0:
                continue
            signals = np.zeros_like(positive_proba, dtype=float)
            signals[positive_proba >= threshold] = 1.0
            signals[positive_proba <= 1.0 - threshold] = -1.0
            stats = self._trading_statistics(signals, actual_returns)
            score = stats.get("net_return", -np.inf)
            if score > best_score:
                best_score = score
                best_threshold = float(threshold)

        thresholds["long_probability_threshold"] = best_threshold
        thresholds["short_probability_threshold"] = max(0.0, 1.0 - best_threshold)
        return thresholds

    def _aggregate_metrics(self, split_metrics: List[Dict[str, float]]) -> Dict[str, float]:
        aggregate: Dict[str, float] = {}
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
        return aggregate

    def _normalise_feature_importance(
        self,
        totals: Dict[str, float],
        counts: Dict[str, int],
    ) -> Dict[str, float]:
        if not totals:
            return {}
        averaged = {
            name: totals[name] / max(1, counts.get(name, 1))
            for name in totals
        }
        return dict(
            sorted(averaged.items(), key=lambda item: item[1], reverse=True)
        )

    def _resolve_feature_names(
        self,
        pipeline: Optional[Pipeline],
        transformed: pd.DataFrame | np.ndarray,
    ) -> List[str]:
        if pipeline is not None:
            names = get_feature_names_from_pipeline(pipeline)
            if names:
                return list(names)
        if isinstance(transformed, pd.DataFrame):
            return list(transformed.columns)
        if hasattr(transformed, "shape") and transformed.shape[1] > 0:
            return [f"feature_{idx}" for idx in range(transformed.shape[1])]
        return []

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
