"""Advanced backtesting utilities with walk-forward evaluation and caching.

This module replaces the previous thin delegate with a robust walk-forward
backtesting engine. It leverages the :class:`FeatureAssembler` via the
existing :class:`TrainingDatasetBuilder` to guarantee consistent feature
pipelines and strict time ordering.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
import copy
import json
import logging
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import mean_pinball_loss

from stock_predictor.core import PredictorConfig
from stock_predictor.core.features import FeatureToggles
from stock_predictor.core.models import ModelFactory, classification_metrics, regression_metrics
from stock_predictor.core.training_data import TrainingDataset, TrainingDatasetBuilder
from stock_predictor.core.directional import evaluate_directional_predictions


LOGGER = logging.getLogger(__name__)


SCHEMA_VERSION = "backtest/1.0"


@dataclass(slots=True)
class BacktestConfig:
    """Configuration for walk-forward backtesting runs."""

    n_runs: int = 1
    start_date: datetime | str | None = None
    end_date: datetime | str | None = None
    horizon: int | None = None
    targets: tuple[str, ...] | None = None
    toggles: FeatureToggles | Mapping[str, bool] | Iterable[str] | None = None
    training_window: int = 252
    step_size: int = 21
    embargo: int = 1
    random_seed: int | None = None
    parallelism: int = 1
    caching: bool = True
    cache_trained_windows: bool = True
    output_dir: Path | None = None
    schema_version: str = SCHEMA_VERSION


@dataclass(slots=True)
class BacktestResult:
    """Structured representation of backtest outputs."""

    config: BacktestConfig
    splits: list[dict[str, Any]]
    aggregate: dict[str, Any]
    calibration: dict[str, Any]
    reliability: dict[str, float]
    interval_coverage: dict[str, float]
    confidence_intervals: dict[str, tuple[float, float]]
    summary: dict[str, Any]
    warnings: list[str] = field(default_factory=list)
    outputs: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": self.config.schema_version,
            "config": self._serialise_config(),
            "splits": self.splits,
            "aggregate": self.aggregate,
            "calibration": self.calibration,
            "reliability": self.reliability,
            "interval_coverage": self.interval_coverage,
            "confidence_intervals": {
                name: {"low": low, "high": high}
                for name, (low, high) in self.confidence_intervals.items()
            },
            "summary": self.summary,
            "warnings": self.warnings,
            "outputs": self.outputs,
        }
        return payload

    def _serialise_config(self) -> dict[str, Any]:
        data = self.config.__dict__.copy()
        for key, value in list(data.items()):
            if isinstance(value, Path):
                data[key] = str(value)
        return data


class Backtester:
    """Perform rolling backtests using cached feature pipelines."""

    def __init__(self, config: PredictorConfig, trainer: Any) -> None:  # trainer kept for compatibility
        self.config = config
        self.trainer = trainer
        self.dataset_builder = TrainingDatasetBuilder(config)
        self._trained_window_cache: dict[tuple[str, int, int, int], tuple[Any, Any]] = {}
        self._feature_cache: dict[tuple[str, int, int, int], TrainingDataset] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(
        self,
        *,
        targets: Iterable[str] | None = None,
        backtest_config: BacktestConfig | None = None,
    ) -> BacktestResult:
        cfg = self._resolve_config(backtest_config, targets)
        dataset = self._prepare_dataset(cfg)
        result = self._run_evaluation(dataset, cfg)
        self._persist_outputs(result, cfg)
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _resolve_config(
        self, backtest_config: BacktestConfig | None, targets: Iterable[str] | None
    ) -> BacktestConfig:
        resolved = backtest_config or BacktestConfig()
        resolved_targets = tuple(targets) if targets is not None else resolved.targets
        if resolved_targets is None:
            resolved_targets = tuple(self.config.prediction_targets)
        resolved = replace(resolved, targets=tuple(resolved_targets))
        if resolved.horizon is None and self.config.prediction_horizons:
            resolved = replace(resolved, horizon=int(self.config.prediction_horizons[0]))
        if resolved.output_dir is None:
            resolved = replace(
                resolved,
                output_dir=(self.config.data_dir / "reports" / "backtests"),
            )
        return resolved

    def _prepare_dataset(self, cfg: BacktestConfig) -> TrainingDataset:
        if cfg.toggles is not None:
            try:
                toggles = cfg.toggles
                if not isinstance(toggles, FeatureToggles):
                    toggles = FeatureToggles.from_any(toggles)
                self.config.feature_toggles = toggles
            except Exception:  # pragma: no cover - defensive guard
                LOGGER.warning("Failed to apply custom feature toggles; using defaults.")
        toggle_fingerprint = json.dumps(self.config.feature_toggles.asdict(), sort_keys=True)
        cache_key = (
            self.config.ticker,
            int(cfg.horizon or 0),
            int(cfg.training_window or 0),
            hash(toggle_fingerprint),
        )
        if cfg.caching and cache_key in self._feature_cache:
            return self._feature_cache[cache_key]

        dataset = self.dataset_builder.build(force=not cfg.caching)
        if cfg.caching:
            self._feature_cache[cache_key] = dataset
        return dataset

    def _run_evaluation(self, dataset: TrainingDataset, cfg: BacktestConfig) -> BacktestResult:
        horizon = cfg.horizon if cfg.horizon is not None else int(self.config.prediction_horizons[0])
        features = self._slice_frame(dataset.features, cfg)
        targets = dataset.targets.get(int(horizon)) or {}

        splits: list[dict[str, Any]] = []
        calibration_pool: list[tuple[np.ndarray, np.ndarray]] = []
        regression_pool: list[tuple[np.ndarray, np.ndarray, Mapping[float, np.ndarray] | None]] = []

        for target_name, target_series in targets.items():
            if cfg.targets and target_name not in cfg.targets:
                continue
            target_series = self._align_target(target_series, features.index)
            if target_series.empty:
                continue
            split_metrics, calib_entries, reg_entries = self._evaluate_target(
                target_name,
                features,
                target_series,
                dataset,
                cfg,
                horizon=horizon,
            )
            splits.extend(split_metrics)
            calibration_pool.extend(calib_entries)
            regression_pool.extend(reg_entries)

        aggregate = self._aggregate_metrics(splits)
        calibration = self._build_calibration(calibration_pool)
        reliability = self._compute_reliability(calibration)
        interval_coverage = self._compute_interval_coverage(regression_pool)
        confidence_intervals = self._metric_confidence_intervals(splits, seed=cfg.random_seed)

        summary = {
            "classification": {
                "accuracy": aggregate.get("accuracy"),
                "precision": aggregate.get("precision"),
                "recall": aggregate.get("recall"),
                "f1": aggregate.get("f1"),
                "expected_calibration_error": reliability.get("expected_calibration_error"),
                "max_calibration_error": reliability.get("max_calibration_error"),
            },
            "regression": {
                "mae": aggregate.get("mae"),
                "rmse": aggregate.get("rmse"),
                "mape": aggregate.get("mape"),
                "residual_std": interval_coverage.get("residual_std", aggregate.get("residual_std")),
                "interval_coverage": interval_coverage.get("interval_coverage"),
            },
        }
        warnings = self._build_warnings(splits, reliability, interval_coverage)

        return BacktestResult(
            config=cfg,
            splits=splits,
            aggregate=aggregate,
            calibration=calibration,
            reliability=reliability,
            interval_coverage=interval_coverage,
            confidence_intervals=confidence_intervals,
            summary=summary,
            warnings=warnings,
        )

    def _evaluate_target(
        self,
        target: str,
        features: pd.DataFrame,
        target_series: pd.Series,
        dataset: TrainingDataset,
        cfg: BacktestConfig,
        *,
        horizon: int,
    ) -> tuple[
        list[dict[str, Any]],
        list[tuple[np.ndarray, np.ndarray]],
        list[tuple[np.ndarray, np.ndarray, Mapping[float, np.ndarray] | None]],
    ]:
        sorted_index = pd.DatetimeIndex(features.index).sort_values()
        aligned_features = features.loc[sorted_index]
        aligned_target = target_series.reindex(sorted_index).dropna()
        aligned_features = aligned_features.loc[aligned_target.index]

        indices = list(range(len(aligned_features)))
        splits = list(
            self._generate_walk_forward_splits(
                len(indices), cfg.training_window, cfg.step_size, cfg.embargo
            )
        )
        split_metrics: list[dict[str, Any]] = []
        calibration_entries: list[tuple[np.ndarray, np.ndarray]] = []
        regression_entries: list[tuple[np.ndarray, np.ndarray, Mapping[float, np.ndarray] | None]] = []
        template = dataset.preprocessors.get(int(horizon))
        model_params = {
            **self.config.model_params.get("global", {}),
            **self.config.model_params.get(target, {}),
        }
        calibration_params = self.config.model_params.get("calibration", {})
        factory = ModelFactory(self.config.model_type, model_params)

        for run_idx in range(max(1, cfg.n_runs)):
            for split_idx, (train_slice, test_slice) in enumerate(splits, start=1):
                train_idx = aligned_features.index[train_slice]
                test_idx = aligned_features.index[test_slice]
                X_train, y_train = aligned_features.loc[train_idx], aligned_target.loc[train_idx]
                X_test, y_test = aligned_features.loc[test_idx], aligned_target.loc[test_idx]
                if y_test.empty or y_train.empty:
                    continue

                pipeline = clone(template) if template is not None else None
                if pipeline is not None:
                    pipeline.fit(X_train, y_train)
                    X_train_processed = pipeline.transform(X_train)
                    X_test_processed = pipeline.transform(X_test)
                else:
                    X_train_processed = X_train
                    X_test_processed = X_test

                cache_key = (target, horizon, int(train_slice.start or 0), int(train_slice.stop or 0))
                model = None
                if cfg.cache_trained_windows and cache_key in self._trained_window_cache:
                    model, pipeline = self._clone_cached(cache_key)
                if model is None:
                    calibrate_flag = target == "direction"
                    model = factory.create(
                        "classification" if calibrate_flag else "regression",
                        calibrate=calibrate_flag,
                        calibration_params=calibration_params,
                    )
                    model.fit(X_train_processed, y_train)
                    if cfg.cache_trained_windows:
                        self._trained_window_cache[cache_key] = (
                            copy.deepcopy(model),
                            copy.deepcopy(pipeline),
                        )
                y_pred = model.predict(X_test_processed)
                metrics, quantiles = self._score_predictions(
                    target,
                    y_test,
                    y_pred,
                    X_test,
                    pipeline=pipeline,
                    model=model,
                    y_train=y_train,
                    train_features=X_train,
                )
                metrics.update(
                    {
                        "target": target,
                        "horizon": int(horizon),
                        "split": split_idx,
                        "run": run_idx,
                        "train_size": int(len(y_train)),
                        "test_size": int(len(y_test)),
                        "test_start": test_idx.min().isoformat(),
                        "test_end": test_idx.max().isoformat(),
                    }
                )
                split_metrics.append(metrics)

                if metrics.get("task") == "classification" and "proba_positive" in metrics:
                    calibration_entries.append((metrics["y_true"], metrics["proba_positive"]))
                else:
                    regression_entries.append((metrics["y_true"], metrics["y_pred"], quantiles))

        return split_metrics, calibration_entries, regression_entries

    def _score_predictions(
        self,
        target: str,
        y_true: pd.Series,
        y_pred: np.ndarray,
        raw_features: pd.DataFrame,
        *,
        pipeline: Any,
        model: Any,
        y_train: pd.Series,
        train_features: pd.DataFrame,
    ) -> dict[str, Any]:
        task = "classification" if target == "direction" else "regression"
        metrics: dict[str, Any]
        result: dict[str, Any] = {"task": task, "y_true": y_true.to_numpy(), "y_pred": y_pred}
        quantiles: Mapping[float, np.ndarray] | None = None

        if task == "classification":
            try:
                y_proba = model.predict_proba(pipeline.transform(raw_features) if pipeline else raw_features)[:, 1]
                result["proba_positive"] = y_proba
            except Exception:
                y_proba = None
            metrics = classification_metrics(y_true.to_numpy(), y_pred, y_proba=y_proba)
            reliability = evaluate_directional_predictions(y_true.to_numpy(), y_pred)
            metrics.update(reliability.as_summary())
        else:
            metrics = regression_metrics(y_true.to_numpy(), y_pred)
            baseline = raw_features.get("Close_Current") if "Close_Current" in raw_features else None
            if baseline is not None:
                predicted_direction = np.sign(y_pred - baseline.to_numpy(dtype=float))
                actual_direction = np.sign(y_true.to_numpy() - baseline.to_numpy(dtype=float))
                metrics["directional_accuracy"] = float(
                    np.mean((predicted_direction >= 0) == (actual_direction >= 0))
                )
            try:
                train_pred = model.predict(
                    pipeline.transform(train_features) if pipeline is not None else train_features
                )
            except Exception:
                train_pred = None
            residual_std = float(np.std(y_train.to_numpy() - train_pred)) if train_pred is not None else float("nan")
            metrics["residual_std"] = residual_std
            estimator = getattr(model, "named_steps", {}).get("estimator")
            if hasattr(estimator, "predict_quantiles"):
                try:
                    quantile_preds = estimator.predict_quantiles(
                        pipeline.transform(raw_features) if pipeline is not None else raw_features
                    )
                except Exception:
                    quantile_preds = {}
                if isinstance(quantile_preds, Mapping):
                    quantiles = {
                        float(key): np.asarray(value)
                        for key, value in quantile_preds.items()
                        if isinstance(key, (int, float))
                    }
                    for q, values in quantiles.items():
                        try:
                            metrics[f"pinball_q{int(q*100)}"] = float(
                                mean_pinball_loss(y_true.to_numpy(), values, alpha=q)
                            )
                        except Exception:
                            continue
        result.update(metrics)
        return result, quantiles

    def _generate_walk_forward_splits(
        self, n_samples: int, window: int, step: int, embargo: int
    ) -> Iterable[tuple[slice, slice]]:
        start = max(window, embargo)
        while start < n_samples:
            train_start = max(0, start - window)
            train_end = start - embargo
            test_start = start
            test_end = min(n_samples, start + step)
            yield slice(train_start, train_end), slice(test_start, test_end)
            start += step

    def _aggregate_metrics(self, splits: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        if not splits:
            return {}
        numeric_keys: dict[str, list[float]] = {}
        for entry in splits:
            for key, value in entry.items():
                if key in {"y_true", "y_pred", "proba_positive"}:
                    continue
                if isinstance(value, (int, float, np.number)) and np.isfinite(value):
                    numeric_keys.setdefault(key, []).append(float(value))
        return {key: float(np.mean(values)) for key, values in numeric_keys.items() if values}

    def _build_calibration(
        self, calibration_entries: Sequence[tuple[np.ndarray, np.ndarray]]
    ) -> dict[str, Any]:
        if not calibration_entries:
            return {}
        y_true = np.concatenate([entry[0] for entry in calibration_entries])
        y_proba = np.concatenate([entry[1] for entry in calibration_entries])
        frac_pos, mean_pred = calibration_curve(y_true, y_proba, n_bins=10, strategy="quantile")
        return {
            "fraction_positives": frac_pos.tolist(),
            "mean_predictions": mean_pred.tolist(),
        }

    def _compute_reliability(self, calibration: Mapping[str, Any]) -> dict[str, float]:
        frac = calibration.get("fraction_positives")
        mean_pred = calibration.get("mean_predictions")
        if not isinstance(frac, list) or not isinstance(mean_pred, list) or not frac:
            return {}
        frac_arr = np.asarray(frac)
        mean_arr = np.asarray(mean_pred)
        ece = float(np.mean(np.abs(frac_arr - mean_arr)))
        mce = float(np.max(np.abs(frac_arr - mean_arr)))
        return {"expected_calibration_error": ece, "max_calibration_error": mce}

    def _compute_interval_coverage(
        self,
        regression_entries: Sequence[tuple[np.ndarray, np.ndarray, Mapping[float, np.ndarray] | None]],
        interval_sigma: float = 1.96,
    ) -> dict[str, float]:
        if not regression_entries:
            return {}
        y_true = np.concatenate([entry[0] for entry in regression_entries])
        y_pred = np.concatenate([entry[1] for entry in regression_entries])
        residuals = y_true - y_pred
        std = np.std(residuals)

        coverage_values: list[float] = []
        for _, truth, quantiles in regression_entries:
            lower = upper = None
            if isinstance(quantiles, Mapping):
                lower = quantiles.get(0.1) or quantiles.get(0.05)
                upper = quantiles.get(0.9) or quantiles.get(0.95)
            if lower is not None and upper is not None:
                mask = (truth >= lower) & (truth <= upper)
                coverage_values.append(float(np.mean(mask)))

        coverage = float(np.mean(coverage_values)) if coverage_values else (
            float(np.mean(np.abs(residuals) <= interval_sigma * std)) if std > 0 else float("nan")
        )
        return {"residual_std": float(std), "interval_coverage": coverage}

    def _metric_confidence_intervals(
        self,
        splits: Sequence[Mapping[str, Any]],
        confidence: float = 0.95,
        n_boot: int = 200,
        *,
        seed: int | None = None,
    ) -> dict[str, tuple[float, float]]:
        if not splits:
            return {}
        metrics: dict[str, list[float]] = {}
        for entry in splits:
            for key, value in entry.items():
                if key in {"y_true", "y_pred", "proba_positive", "split", "run", "target", "horizon"}:
                    continue
                if isinstance(value, (int, float, np.number)) and np.isfinite(value):
                    metrics.setdefault(key, []).append(float(value))
        rng = np.random.default_rng(seed)
        intervals: dict[str, tuple[float, float]] = {}
        for key, values in metrics.items():
            if len(values) < 2:
                continue
            samples = []
            arr = np.asarray(values)
            for _ in range(n_boot):
                boot = rng.choice(arr, size=arr.size, replace=True)
                samples.append(float(np.mean(boot)))
            lower = float(np.percentile(samples, (1 - confidence) / 2 * 100))
            upper = float(np.percentile(samples, (1 + confidence) / 2 * 100))
            intervals[key] = (lower, upper)
        return intervals

    def _clone_cached(self, cache_key: tuple[str, int, int, int]) -> tuple[Any, Any]:
        cached = self._trained_window_cache.get(cache_key)
        if cached is None:
            return None, None
        model, pipeline = cached
        return copy.deepcopy(model), copy.deepcopy(pipeline) if pipeline is not None else None

    def _slice_frame(self, frame: pd.DataFrame, cfg: BacktestConfig) -> pd.DataFrame:
        index = pd.DatetimeIndex(frame.index).tz_localize(None)
        frame = frame.copy()
        frame.index = index
        if cfg.start_date is not None:
            start = pd.to_datetime(cfg.start_date).tz_localize(None)
            frame = frame.loc[frame.index >= start]
        if cfg.end_date is not None:
            end = pd.to_datetime(cfg.end_date).tz_localize(None)
            frame = frame.loc[frame.index <= end]
        business_index = pd.bdate_range(frame.index.min(), frame.index.max())
        frame = frame.reindex(frame.index.intersection(business_index))
        return frame

    def _align_target(self, series: pd.Series, feature_index: pd.Index) -> pd.Series:
        idx = pd.DatetimeIndex(series.index).tz_localize(None)
        series = series.copy()
        series.index = idx
        aligned = series.reindex(feature_index)
        return aligned.dropna()

    def _persist_outputs(self, result: BacktestResult, cfg: BacktestConfig) -> None:
        output_dir = cfg.output_dir or (self.config.data_dir / "reports" / "backtests")
        versioned_dir = output_dir / cfg.schema_version.replace("/", "_")
        versioned_dir.mkdir(parents=True, exist_ok=True)

        splits_frame = pd.DataFrame(result.splits)
        json_path = versioned_dir / "backtest.json"
        parquet_path = versioned_dir / "backtest.parquet"
        csv_path = versioned_dir / "backtest.csv"

        with json_path.open("w", encoding="utf-8") as handle:
            json.dump(result.as_dict(), handle, indent=2, default=str)

        try:
            splits_frame.to_parquet(parquet_path, index=False)
        except Exception:
            LOGGER.debug("Parquet export failed; continuing with CSV only.")
        splits_frame.to_csv(csv_path, index=False)

        result.outputs.update(
            {
                "json": str(json_path),
                "parquet": str(parquet_path),
                "csv": str(csv_path),
            }
        )

    def _build_warnings(
        self,
        splits: Sequence[Mapping[str, Any]],
        reliability: Mapping[str, Any],
        interval_coverage: Mapping[str, Any],
    ) -> list[str]:
        warnings: list[str] = []
        total_predictions = sum(int(entry.get("test_size", 0) or 0) for entry in splits)
        if total_predictions < 50:
            warnings.append(
                "Backtest covers fewer than 50 out-of-sample predictions; interpret metrics cautiously."
            )

        ece = reliability.get("expected_calibration_error")
        if isinstance(ece, (int, float)) and ece > 0.1:
            warnings.append(
                "Calibration drift detected (ECE > 0.10). Consider refreshing models or adjusting thresholds."
            )

        coverage = interval_coverage.get("interval_coverage")
        if isinstance(coverage, (int, float)) and coverage < 0.6:
            warnings.append(
                "Prediction intervals cover fewer than 60% of observations; distribution shift may be present."
            )
        return warnings


__all__ = ["Backtester", "BacktestConfig", "BacktestResult", "SCHEMA_VERSION"]
