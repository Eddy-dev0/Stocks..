"""High level orchestration for feature engineering, training and inference."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from numbers import Real
from typing import Any, Dict, Iterable, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import TimeSeriesSplit, train_test_split

from .backtesting import Backtester
from .config import PredictorConfig
from .data_fetcher import DataFetcher
from .database import ExperimentTracker
from .features import FeatureAssembler
from .models import (
    ModelFactory,
    classification_metrics,
    extract_feature_importance,
    model_supports_proba,
    regression_metrics,
)

LOGGER = logging.getLogger(__name__)


class ModelNotFoundError(FileNotFoundError):
    """Raised when a persisted model for a target/horizon cannot be located."""

    def __init__(self, target: str, horizon: int, path: Path) -> None:
        message = (
            f"Saved model for target '{target}' with horizon {horizon} not found at {path}."
        )
        super().__init__(message)
        self.target = target
        self.horizon = horizon
        self.path = path


class StockPredictorAI:
    """Pipeline that assembles features, trains models, and produces forecasts."""

    def __init__(self, config: PredictorConfig, *, horizon: Optional[int] = None) -> None:
        self.config = config
        self.horizon = self.config.resolve_horizon(horizon)
        self.fetcher = DataFetcher(config)
        self.feature_assembler = FeatureAssembler(
            config.feature_toggles, config.prediction_horizons
        )
        self.tracker = ExperimentTracker(config)
        self.models: dict[Tuple[str, int], Any] = {}
        self.metadata: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Data acquisition
    # ------------------------------------------------------------------
    def download_data(self, force: bool = False) -> Dict[str, Any]:
        """Refresh all datasets and return a summary of the ETL run."""

        LOGGER.info("Starting data refresh for %s", self.config.ticker)
        summary = self.fetcher.refresh_all(force=force)
        LOGGER.info("Data refresh completed for %s", self.config.ticker)
        return summary

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------
    def prepare_features(
        self,
        price_df: Optional[pd.DataFrame] = None,
        news_df: Optional[pd.DataFrame] = None,
    ) -> tuple[pd.DataFrame, dict[int, Dict[str, pd.Series]]]:
        if price_df is None:
            price_df = self.fetcher.fetch_price_data()
        if news_df is None and self.config.sentiment:
            news_df = self.fetcher.fetch_news_data()
        elif news_df is None:
            news_df = pd.DataFrame()

        feature_result = self.feature_assembler.build(price_df, news_df, self.config.sentiment)
        metadata = dict(feature_result.metadata)
        metadata.setdefault("sentiment_daily", pd.DataFrame(columns=["Date", "sentiment"]))
        metadata.setdefault("feature_groups", {})
        metadata["data_sources"] = self.fetcher.get_data_sources()
        metadata.setdefault("target_dates", {})
        metadata.setdefault("horizons", tuple(self.config.prediction_horizons))
        metadata["active_horizon"] = self.horizon
        self.metadata = metadata
        return feature_result.features, feature_result.targets

    # ------------------------------------------------------------------
    # Model persistence helpers
    # ------------------------------------------------------------------
    def _resolve_horizon(self, horizon: Optional[int]) -> int:
        if horizon is None:
            return self.horizon
        return self.config.resolve_horizon(horizon)

    def _get_model(self, target: str, horizon: Optional[int] = None) -> Any:
        resolved_horizon = self._resolve_horizon(horizon)
        key = (target, resolved_horizon)
        if key not in self.models:
            raise RuntimeError(
                f"Model for target '{target}' and horizon {resolved_horizon} is not loaded. Train or load it first."
            )
        return self.models[key]

    def load_model(self, target: str = "close", horizon: Optional[int] = None) -> Any:
        resolved_horizon = self._resolve_horizon(horizon)
        path = self.config.model_path_for(target, resolved_horizon)
        if not path.exists():
            raise ModelNotFoundError(target, resolved_horizon, path)
        LOGGER.info(
            "Loading %s model for target '%s' at horizon %s",
            self.config.model_type,
            target,
            resolved_horizon,
        )
        model = joblib.load(path)
        self.models[(target, resolved_horizon)] = model
        metadata_path = self.config.metrics_path_for(target, resolved_horizon)
        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as handle:
                stored = json.load(handle)
                feature_columns = stored.get("feature_columns")
                if feature_columns:
                    self.metadata["feature_columns"] = feature_columns
                indicator_columns = stored.get("indicator_columns")
                if indicator_columns:
                    self.metadata["indicator_columns"] = indicator_columns
                target_dates = stored.get("target_dates")
                if isinstance(target_dates, dict):
                    self.metadata.setdefault("target_dates", {}).update(target_dates)
                metrics_payload = {
                    key: value
                    for key, value in stored.items()
                    if key
                    not in {
                        "feature_columns",
                        "indicator_columns",
                        "target_dates",
                        "horizon",
                    }
                }
                if metrics_payload:
                    metrics_store = self.metadata.setdefault("metrics", {})
                    horizon_metrics = metrics_store.setdefault(target, {})
                    horizon_metrics[resolved_horizon] = metrics_payload
        self.metadata["active_horizon"] = resolved_horizon
        return model

    def save_state(self, target: str, horizon: int, metrics: Dict[str, Any]) -> None:
        payload = {
            **metrics,
            "feature_columns": self.metadata.get("feature_columns", []),
            "indicator_columns": self.metadata.get("indicator_columns", []),
            "horizon": horizon,
            "target_dates": self.metadata.get("target_dates", {}),
        }
        path = self.config.metrics_path_for(target, horizon)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, default=str)
        LOGGER.info("Saved metrics for target '%s' (horizon %s) to %s", target, horizon, path)
        if target == "close" and horizon == self.config.default_horizon:
            with open(self.config.metrics_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, default=str)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train_model(
        self,
        targets: Optional[Iterable[str]] = None,
        horizon: Optional[int] = None,
    ) -> Dict[str, Any]:
        resolved_horizon = self._resolve_horizon(horizon)
        self.horizon = resolved_horizon
        X, targets_by_horizon = self.prepare_features()
        feature_columns = self.metadata.get("feature_columns", list(X.columns))

        requested_targets = list(targets) if targets else list(self.config.prediction_targets)
        horizon_targets = targets_by_horizon.get(resolved_horizon)
        if not horizon_targets:
            raise RuntimeError(
                f"No targets available for horizon {resolved_horizon}."
            )
        available_targets = {
            name: series for name, series in horizon_targets.items() if name in requested_targets
        }
        if not available_targets:
            raise RuntimeError("No matching targets available for training.")

        metrics_by_target: dict[str, Dict[str, float]] = {}
        summary_metrics: dict[str, float] = {}

        for target, y in available_targets.items():
            LOGGER.info("Training target '%s' with model type %s", target, self.config.model_type)
            y_clean = y.dropna()
            if y_clean.empty:
                LOGGER.warning(
                    "Target '%s' has no usable samples after dropping NaNs. Skipping horizon %s.",
                    target,
                    resolved_horizon,
                )
                continue
            aligned_X = X.loc[y_clean.index]
            self._log_target_distribution(target, resolved_horizon, y_clean)
            self._validate_no_nans(target, resolved_horizon, aligned_X, y_clean)

            task = "classification" if target == "direction" else "regression"
            model_params = self.config.model_params.get(target, {})
            global_params = self.config.model_params.get("global", {})
            factory = ModelFactory(
                self.config.model_type,
                {**global_params, **model_params},
            )

            calibrate_override = model_params.get("calibrate")
            if calibrate_override is None:
                calibrate_override = global_params.get("calibrate")
            calibrate_flag = (target == "direction") if calibrate_override is None else bool(calibrate_override)

            evaluation = self._evaluate_model(
                factory,
                aligned_X,
                y_clean,
                task,
                target,
                calibrate_flag,
            )
            LOGGER.info(
                "Evaluation summary for target '%s' (horizon %s, strategy %s): %s",
                target,
                resolved_horizon,
                evaluation["strategy"],
                evaluation["aggregate"],
            )

            final_model = factory.create(task, calibrate=calibrate_flag)
            final_model.fit(aligned_X, y_clean)

            metrics = dict(evaluation["aggregate"])
            metrics["training_rows"] = int(len(aligned_X))
            metrics["test_rows"] = int(evaluation.get("evaluation_rows", 0))
            metrics["horizon"] = resolved_horizon
            metrics["evaluation_strategy"] = evaluation["strategy"]
            metrics["evaluation"] = {
                "strategy": evaluation["strategy"],
                "splits": evaluation["splits"],
                "aggregate": evaluation["aggregate"],
                "parameters": evaluation.get("parameters", {}),
                "samples": int(evaluation.get("evaluation_rows", 0)),
            }
            metrics_by_target[target] = metrics

            metrics_store = self.metadata.setdefault("metrics", {})
            horizon_metrics = metrics_store.setdefault(target, {})
            horizon_metrics[resolved_horizon] = metrics

            self.models[(target, resolved_horizon)] = final_model
            joblib.dump(final_model, self.config.model_path_for(target, resolved_horizon))
            self.save_state(target, resolved_horizon, metrics)
            self.tracker.log_run(
                target=target,
                run_type="training",
                parameters={"model_type": self.config.model_type, **model_params},
                metrics=metrics,
                context={"feature_columns": feature_columns, "horizon": resolved_horizon},
            )

            if target == "close":
                summary_metrics.update({k: v for k, v in metrics.items() if isinstance(v, (int, float))})

        LOGGER.info(
            "Training complete for targets %s at horizon %s",
            ", ".join(metrics_by_target),
            resolved_horizon,
        )
        self.metadata["active_horizon"] = resolved_horizon
        return {"horizon": resolved_horizon, "targets": metrics_by_target, **summary_metrics}

    def _log_target_distribution(
        self,
        target: str,
        horizon: int,
        series: pd.Series,
    ) -> None:
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.isna().any() or not np.isfinite(numeric.to_numpy()).all():
            raise ValueError(
                f"Non-finite values detected in target '{target}' for horizon {horizon}."
            )

        stats = {
            "count": int(numeric.count()),
            "mean": float(numeric.mean()),
            "std": float(numeric.std(ddof=0)),
            "min": float(numeric.min()),
            "max": float(numeric.max()),
        }
        quantiles = numeric.quantile([0.1, 0.5, 0.9]).to_dict()
        quantiles = {f"q{int(q * 100)}": float(value) for q, value in quantiles.items()}
        stats.update(quantiles)

        if target == "direction":
            counts = series.value_counts().to_dict()
            ratios = (
                series.value_counts(normalize=True)
                .mul(100)
                .round(2)
                .to_dict()
            )
            LOGGER.info(
                "Target distribution for '%s' (horizon %s): counts=%s ratios=%s stats=%s",
                target,
                horizon,
                counts,
                {key: f"{value:.2f}%" for key, value in ratios.items()},
                stats,
            )
        else:
            LOGGER.info(
                "Target distribution for '%s' (horizon %s): %s",
                target,
                horizon,
                stats,
            )

    def _validate_no_nans(
        self,
        target: str,
        horizon: int,
        features: pd.DataFrame,
        series: pd.Series,
    ) -> None:
        feature_check = features.replace([np.inf, -np.inf], np.nan)
        if feature_check.isna().any().any():
            columns = feature_check.columns[feature_check.isna().any()].tolist()
            raise ValueError(
                "Non-finite feature values detected for target '%s' at horizon %s in columns %s"
                % (target, horizon, ", ".join(columns))
            )

        numeric_series = pd.to_numeric(series, errors="coerce")
        if numeric_series.isna().any() or not np.isfinite(numeric_series.to_numpy()).all():
            raise ValueError(
                f"Non-finite values detected in target '{target}' for horizon {horizon}."
            )

    def _evaluate_model(
        self,
        factory: ModelFactory,
        features: pd.DataFrame,
        target_series: pd.Series,
        task: str,
        target_name: str,
        calibrate_flag: bool,
    ) -> Dict[str, Any]:
        strategy = self.config.evaluation_strategy
        evaluation_rows = 0
        parameters: Dict[str, Any] = {}
        splits: List[Dict[str, Any]] = []

        if strategy == "holdout":
            X_train, X_test, y_train, y_test = train_test_split(
                features,
                target_series,
                test_size=self.config.test_size,
                shuffle=self.config.shuffle_training,
                random_state=42,
            )
            model = factory.create(task, calibrate=calibrate_flag)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            metrics = self._compute_evaluation_metrics(task, y_test, y_pred, X_test)
            metrics.update(
                {
                    "split": 1,
                    "train_size": int(len(X_train)),
                    "test_size": int(len(X_test)),
                }
            )
            splits.append(metrics)
            aggregate = {
                key: float(value)
                for key, value in metrics.items()
                if key not in {"split", "train_size", "test_size"} and isinstance(value, Real)
            }
            evaluation_rows = int(len(X_test))
            parameters = {
                "test_size": float(self.config.test_size),
                "shuffle": bool(self.config.shuffle_training),
            }
        elif strategy == "time_series":
            splitter = TimeSeriesSplit(n_splits=self.config.evaluation_folds)
            for fold, (train_idx, test_idx) in enumerate(splitter.split(features, target_series), start=1):
                if len(test_idx) == 0:
                    continue
                X_train, X_test = features.iloc[train_idx], features.iloc[test_idx]
                y_train, y_test = target_series.iloc[train_idx], target_series.iloc[test_idx]
                model = factory.create(task, calibrate=calibrate_flag)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                metrics = self._compute_evaluation_metrics(task, y_test, y_pred, X_test)
                metrics.update(
                    {
                        "fold": fold,
                        "train_size": int(len(train_idx)),
                        "test_size": int(len(test_idx)),
                    }
                )
                splits.append(metrics)
            if not splits:
                raise RuntimeError(
                    "Time series cross-validation produced no evaluation splits."
                )
            aggregate = self._aggregate_evaluation_metrics(splits)
            evaluation_rows = int(sum(item.get("test_size", 0) for item in splits))
            parameters = {"folds": int(self.config.evaluation_folds)}
        elif strategy == "rolling":
            backtester = Backtester(
                model_factory=factory,
                strategy=self.config.backtest_strategy,
                window=self.config.evaluation_window,
                step=self.config.evaluation_step,
            )
            result = backtester.run(features, target_series, target_name)
            splits = result.splits
            aggregate = result.aggregate
            evaluation_rows = int(aggregate.get("test_rows", 0))
            parameters = {
                "window": int(self.config.evaluation_window),
                "step": int(self.config.evaluation_step),
                "mode": self.config.backtest_strategy,
            }
        else:  # pragma: no cover - configuration guard
            raise ValueError(f"Unknown evaluation strategy: {strategy}")

        return {
            "strategy": strategy,
            "splits": splits,
            "aggregate": aggregate,
            "evaluation_rows": evaluation_rows,
            "parameters": parameters,
        }

    def _compute_evaluation_metrics(
        self,
        task: str,
        y_true: pd.Series,
        y_pred: np.ndarray,
        X_test: pd.DataFrame,
    ) -> Dict[str, float]:
        if task == "classification":
            metrics = classification_metrics(y_true.to_numpy(), y_pred)
            metrics["directional_accuracy"] = metrics.get("accuracy", 0.0)
            return metrics

        metrics = regression_metrics(y_true.to_numpy(), y_pred)
        try:
            metrics["r2"] = float(r2_score(y_true, y_pred))
        except ValueError:
            metrics["r2"] = float("nan")
        baseline = (
            X_test["Close_Current"].to_numpy()
            if "Close_Current" in X_test
            else np.zeros_like(y_pred)
        )
        predicted_direction = np.sign(y_pred - baseline)
        actual_direction = np.sign(y_true.to_numpy() - baseline)
        if len(actual_direction) > 0:
            metrics["directional_accuracy"] = float(
                np.mean((predicted_direction >= 0) == (actual_direction >= 0))
            )
        else:
            metrics["directional_accuracy"] = 0.0
        metrics["signed_error"] = float(np.mean(y_pred - y_true.to_numpy()))
        return metrics

    def _aggregate_evaluation_metrics(
        self, splits: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        if not splits:
            return {}
        aggregate: Dict[str, float] = {}
        keys = {
            key
            for entry in splits
            for key in entry.keys()
            if key not in {"split", "fold", "train_size", "test_size"}
        }
        for key in keys:
            values: List[float] = []
            for entry in splits:
                value = entry.get(key)
                if isinstance(value, Real) and np.isfinite(value):
                    values.append(float(value))
            if values:
                aggregate[key] = float(np.mean(values))
        aggregate["folds"] = int(len(splits))
        return aggregate

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def predict(
        self,
        refresh_data: bool = False,
        targets: Optional[Iterable[str]] = None,
        horizon: Optional[int] = None,
    ) -> Dict[str, Any]:
        resolved_horizon = self._resolve_horizon(horizon)
        self.horizon = resolved_horizon
        if refresh_data:
            LOGGER.info("Refreshing data prior to prediction.")
            self.download_data(force=True)

        if not self.metadata or refresh_data:
            LOGGER.info("Preparing features before prediction.")
            self.prepare_features()

        feature_columns = self.metadata.get("feature_columns")
        latest_features = self.metadata.get("latest_features")
        if feature_columns is None or latest_features is None:
            raise RuntimeError("No feature metadata available. Train the model first.")
        self.metadata["active_horizon"] = resolved_horizon

        requested_targets = list(targets) if targets else list(self.config.prediction_targets)
        predictions: dict[str, Any] = {}
        confidences: dict[str, float] = {}
        probabilities: dict[str, Dict[str, float]] = {}
        uncertainties: dict[str, Dict[str, float]] = {}
        prediction_warnings: List[str] = []
        training_report: dict[str, Any] = {}

        for target in requested_targets:
            try:
                model = self.models.get((target, resolved_horizon)) or self.load_model(
                    target, resolved_horizon
                )
            except ModelNotFoundError:
                LOGGER.warning("Model for target '%s' missing. Triggering training.", target)
                report = self.train_model(targets=[target], horizon=resolved_horizon)
                training_report[target] = {
                    "horizon": resolved_horizon,
                    **report.get("targets", {}).get(target, {}),
                }
                model = self._get_model(target, resolved_horizon)

            current_features = latest_features[feature_columns]
            pred_value = model.predict(current_features)[0]
            predictions[target] = float(pred_value)

            uncertainty = self._estimate_prediction_uncertainty(target, model, current_features)
            if uncertainty:
                uncertainties[target] = uncertainty

            if model_supports_proba(model) and target == "direction":
                proba = model.predict_proba(current_features)[0]
                estimator = model.named_steps.get("estimator")
                classes = getattr(estimator, "classes_", None)
                class_prob_map: Dict[Any, float] = {}
                if classes is not None:
                    class_prob_map = {
                        cls: float(prob)
                        for cls, prob in zip(classes, proba)
                    }
                up_prob = float(
                    class_prob_map.get(1)
                    or class_prob_map.get(1.0)
                    or class_prob_map.get("1")
                    or class_prob_map.get(True)
                    or (float(proba[1]) if len(proba) > 1 else 0.0)
                )
                down_prob = float(
                    class_prob_map.get(0)
                    or class_prob_map.get(0.0)
                    or class_prob_map.get("0")
                    or class_prob_map.get(False)
                    or (float(proba[0]) if len(proba) > 0 else 0.0)
                )
                probabilities[target] = {"up": up_prob, "down": down_prob}
                confidence_value = float(max(up_prob, down_prob))
                confidences[target] = confidence_value
                threshold = float(self.config.direction_confidence_threshold)
                if confidence_value < threshold:
                    warning_msg = (
                        f"Direction model confidence {confidence_value:.3f} below threshold {threshold:.2f}."
                    )
                    LOGGER.warning(warning_msg)
                    prediction_warnings.append(warning_msg)

        close_prediction = predictions.get("close")
        latest_close = float(self.metadata.get("latest_close", np.nan))
        expected_change = None
        pct_change = None
        if close_prediction is not None and np.isfinite(latest_close):
            expected_change = close_prediction - latest_close
            pct_change = expected_change / latest_close if latest_close else 0.0

        prediction_timestamp = datetime.now()

        latest_date = self.metadata.get("latest_date")
        if isinstance(latest_date, pd.Timestamp):
            latest_date = latest_date.to_pydatetime()

        target_dates = self.metadata.get("target_dates", {})
        target_date = None
        if isinstance(target_dates, dict):
            target_date = target_dates.get(resolved_horizon)

        predicted_return = self._safe_float(predictions.get("return"))
        predicted_volatility = self._safe_float(predictions.get("volatility"))
        dir_prob = probabilities.get("direction") if isinstance(probabilities, dict) else None
        direction_probability_up = None
        direction_probability_down = None
        if isinstance(dir_prob, dict):
            direction_probability_up = self._safe_float(dir_prob.get("up"))
            direction_probability_down = self._safe_float(dir_prob.get("down"))

        uncertainty_clean: dict[str, Dict[str, float]] = {}
        for tgt, values in uncertainties.items():
            numeric_values = {
                key: float(value)
                for key, value in values.items()
                if value is not None and np.isfinite(value)
            }
            if numeric_values:
                uncertainty_clean[tgt] = numeric_values

        def _to_iso(value: Any) -> Optional[str]:
            if value is None:
                return None
            if isinstance(value, datetime):
                return value.isoformat(timespec="seconds")
            if isinstance(value, str):
                return value
            try:
                timestamp = pd.to_datetime(value)
            except (TypeError, ValueError):
                return str(value)
            if pd.isna(timestamp):
                return None
            return timestamp.to_pydatetime().isoformat(timespec="seconds")

        result = {
            "ticker": self.config.ticker,
            "as_of": _to_iso(latest_date) or "",
            "market_data_as_of": _to_iso(latest_date) or "",
            "generated_at": _to_iso(prediction_timestamp) or "",
            "last_close": latest_close,
            "predicted_close": close_prediction,
            "expected_change": expected_change,
            "expected_change_pct": pct_change,
            "predicted_return": predicted_return,
            "predicted_volatility": predicted_volatility,
            "direction_probability_up": direction_probability_up,
            "direction_probability_down": direction_probability_down,
            "predictions": predictions,
            "horizon": resolved_horizon,
            "target_date": _to_iso(target_date) or "",
        }
        if confidences:
            result["confidence"] = confidences
        if probabilities:
            result["probabilities"] = probabilities
        if uncertainty_clean:
            result["prediction_uncertainty"] = uncertainty_clean
        if training_report:
            result["training_metrics"] = training_report
        if prediction_warnings:
            result["warnings"] = prediction_warnings
        explanation = self._build_prediction_explanation(result, predictions)
        if explanation:
            result["explanation"] = explanation
        return result

    def _estimate_prediction_uncertainty(
        self,
        target: str,
        model: Any,
        features: pd.DataFrame,
    ) -> Optional[Dict[str, float]]:
        if not hasattr(model, "named_steps"):
            return None
        estimator = model.named_steps.get("estimator")
        if estimator is None:
            return None

        transformed = features
        if hasattr(model, "steps") and len(model.steps) > 1:
            try:
                transformed = model[:-1].transform(features)
            except Exception:  # pragma: no cover - defensive
                transformed = features

        if isinstance(transformed, pd.DataFrame):
            transformed = transformed.to_numpy()

        samples: list[float] = []

        if hasattr(estimator, "estimators_"):
            estimators = getattr(estimator, "estimators_")
            for member in estimators:
                try:
                    if target == "direction" and hasattr(member, "predict_proba"):
                        proba = member.predict_proba(transformed)[0]
                        if len(proba) > 1:
                            samples.append(float(proba[1]))
                        elif len(proba) == 1:
                            samples.append(float(proba[0]))
                    elif hasattr(member, "predict"):
                        prediction = member.predict(transformed)[0]
                        samples.append(float(prediction))
                except Exception:  # pragma: no cover - estimator specific quirks
                    continue

        if not samples:
            return None

        values = np.asarray(samples, dtype=float)
        if values.size == 0 or np.isnan(values).all():
            return None

        std = float(np.nanstd(values, ddof=1) if values.size > 1 else 0.0)
        spread = float(np.nanmax(values) - np.nanmin(values))
        return {"std": std, "range": spread}

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------
    def _build_prediction_explanation(
        self,
        prediction: Dict[str, Any],
        raw_predictions: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        _ = raw_predictions
        latest_features = self.metadata.get("latest_features")
        if latest_features is None:
            return None
        if not isinstance(latest_features, pd.DataFrame) or latest_features.empty:
            return None

        try:
            feature_row = latest_features.iloc[0]
        except (KeyError, IndexError):
            LOGGER.debug("Latest feature snapshot is unavailable for explanation generation.")
            return None

        raw_horizon = prediction.get("horizon") or self.metadata.get("active_horizon")
        try:
            horizon_value: Optional[int] = int(raw_horizon) if raw_horizon is not None else None
        except (TypeError, ValueError):
            horizon_value = None

        reasons = {
            "technical_reasons": self._technical_reasons(feature_row),
            "fundamental_reasons": self._fundamental_reasons(feature_row),
            "sentiment_reasons": self._sentiment_reasons(),
            "macro_reasons": self._macro_reasons(feature_row),
        }

        feature_importance = self._collect_feature_importance(horizon_value)
        top_feature_drivers = self._top_feature_drivers(feature_importance)
        confidence_indicators = self._collect_confidence_indicators(prediction, horizon_value)
        summary = self._compose_summary(
            prediction,
            reasons,
            top_feature_drivers,
            confidence_indicators,
            horizon_value,
        )
        sources_raw = self.metadata.get("data_sources") or self.fetcher.get_data_sources()
        sources = list(sources_raw) if isinstance(sources_raw, (list, tuple, set)) else []
        if not sources:
            sources = [f"Database cache: price history for {self.config.ticker}."]

        explanation: Dict[str, Any] = {
            "summary": summary,
            **reasons,
            "feature_importance": feature_importance,
            "top_feature_drivers": top_feature_drivers,
            "confidence_indicators": confidence_indicators,
            "horizon": horizon_value,
            "target_date": prediction.get("target_date"),
            "sources": sources,
        }
        return explanation

    def _technical_reasons(self, feature_row: pd.Series) -> list[str]:
        reasons: list[str] = []
        rsi = self._safe_float(feature_row.get("RSI_14"))
        if rsi is not None:
            if rsi >= 70:
                reasons.append(f"RSI(14) at {rsi:.1f} indicates overbought conditions.")
            elif rsi <= 30:
                reasons.append(f"RSI(14) at {rsi:.1f} signals potential oversold rebound.")

        macd = self._safe_float(feature_row.get("MACD"))
        signal = self._safe_float(feature_row.get("Signal"))
        if macd is not None and signal is not None:
            if macd > signal:
                reasons.append("MACD line above signal line, supporting bullish momentum.")
            elif macd < signal:
                reasons.append("MACD line below signal line, indicating bearish momentum.")

        close_price = self._safe_float(self.metadata.get("latest_close"))
        upper_band = self._safe_float(feature_row.get("Bollinger_Upper"))
        lower_band = self._safe_float(feature_row.get("Bollinger_Lower"))
        if close_price is not None and upper_band is not None and close_price > upper_band:
            reasons.append("Price recently closed above the Bollinger upper band, suggesting mean reversion risk.")
        if close_price is not None and lower_band is not None and close_price < lower_band:
            reasons.append("Price recently closed below the Bollinger lower band, signalling potential rebound.")

        sma5 = self._safe_float(feature_row.get("SMA_5"))
        sma20 = self._safe_float(feature_row.get("SMA_20"))
        if sma5 is not None and sma20 is not None:
            if sma5 > sma20:
                reasons.append("Short-term SMA(5) above SMA(20), highlighting positive short-term momentum.")
            elif sma5 < sma20:
                reasons.append("Short-term SMA(5) below SMA(20), highlighting weakening short-term momentum.")

        daily_return = self._safe_float(feature_row.get("Return_1d"))
        if daily_return is not None and abs(daily_return) >= 0.02:
            direction = "gain" if daily_return > 0 else "loss"
            reasons.append(
                f"Latest session showed a {abs(daily_return) * 100:.2f}% {direction}, influencing near-term trend."
            )

        volume_change = self._safe_float(feature_row.get("Volume_Change"))
        if volume_change is not None and abs(volume_change) >= 0.15:
            if volume_change > 0:
                reasons.append("Volume expanding versus prior day, confirming the latest move.")
            else:
                reasons.append("Volume contraction versus prior day, weakening conviction in the latest move.")

        return reasons

    def _fundamental_reasons(self, feature_row: pd.Series) -> list[str]:
        reasons: list[str] = []
        ratio20 = self._safe_float(feature_row.get("Price_to_SMA20"))
        if ratio20 is not None:
            if ratio20 > 1.05:
                reasons.append(
                    f"Price sits {((ratio20 - 1) * 100):.1f}% above the 20-day average, pointing to stretched valuation versus recent trend."
                )
            elif ratio20 < 0.95:
                reasons.append(
                    f"Price sits {((1 - ratio20) * 100):.1f}% below the 20-day average, suggesting discounted pricing versus recent trend."
                )

        ratio200 = self._safe_float(feature_row.get("Price_to_SMA200"))
        if ratio200 is not None:
            if ratio200 > 1.10:
                reasons.append("Price trading well above the 200-day average, reflecting optimistic longer-term positioning.")
            elif ratio200 < 0.90:
                reasons.append("Price trading well below the 200-day average, reflecting longer-term pessimism.")

        momentum12 = self._safe_float(feature_row.get("Momentum_12"))
        if momentum12 is not None and abs(momentum12) >= 0.1:
            if momentum12 > 0:
                reasons.append(f"Twelve-month momentum of {momentum12 * 100:.1f}% underscores longer-term strength.")
            else:
                reasons.append(f"Twelve-month momentum of {momentum12 * 100:.1f}% highlights longer-term weakness.")

        return reasons

    def _sentiment_reasons(self) -> list[str]:
        reasons: list[str] = []
        sentiment_df = self.metadata.get("sentiment_daily")
        if isinstance(sentiment_df, pd.DataFrame) and not sentiment_df.empty:
            latest = sentiment_df.iloc[-1]
            avg = self._safe_float(latest.get("Sentiment_Avg") or latest.get("sentiment"))
            change = self._safe_float(latest.get("Sentiment_Change"))
            if avg is not None:
                if avg >= 0.15:
                    reasons.append("News sentiment has been positive over the last week.")
                elif avg <= -0.15:
                    reasons.append("News sentiment has been negative over the last week.")
            if change is not None and abs(change) >= 0.1:
                if change > 0:
                    reasons.append("Sentiment momentum improving versus the prior period.")
                else:
                    reasons.append("Sentiment momentum deteriorating versus the prior period.")
        return reasons

    def _macro_reasons(self, feature_row: pd.Series) -> list[str]:
        reasons: list[str] = []
        vol21 = self._safe_float(feature_row.get("Volatility_21"))
        if vol21 is not None and vol21 >= 0.03:
            reasons.append("Short-term realised volatility elevated, indicating choppy market conditions.")

        trend_slope = self._safe_float(feature_row.get("Trend_Slope"))
        if trend_slope is not None:
            if trend_slope > 0:
                reasons.append("Medium-term trend slope positive, supporting upward bias.")
            elif trend_slope < 0:
                reasons.append("Medium-term trend slope negative, signalling downward bias.")

        trend_curvature = self._safe_float(feature_row.get("Trend_Curvature"))
        if trend_curvature is not None and trend_curvature < 0:
            reasons.append("Trend curvature turning lower, hinting at deceleration in momentum.")

        return reasons

    def _compose_summary(
        self,
        prediction: Dict[str, Any],
        reasons: Dict[str, list[str]],
        top_feature_drivers: Dict[str, list[str]],
        confidence_indicators: Dict[str, Any],
        horizon: Optional[int],
    ) -> str:
        change = self._safe_float(prediction.get("expected_change"))
        pct_change = self._safe_float(prediction.get("expected_change_pct"))
        forecast_return = self._safe_float(prediction.get("predicted_return"))
        volatility = self._safe_float(prediction.get("predicted_volatility"))

        if change is None or not np.isfinite(change):
            direction = "Neutral"
        elif change > 0:
            direction = "Bullish"
        elif change < 0:
            direction = "Bearish"
        else:
            direction = "Neutral"

        target_date_display = None
        target_date_raw = prediction.get("target_date")
        if target_date_raw:
            try:
                ts_value = pd.to_datetime(target_date_raw)
                if not pd.isna(ts_value):
                    target_date_display = ts_value.strftime("%Y-%m-%d")
            except Exception:  # pragma: no cover - parsing guard
                target_date_display = str(target_date_raw)

        if horizon and horizon > 0:
            horizon_phrase = f"{horizon}-day outlook"
        else:
            horizon_phrase = "outlook"
        target_fragment = f" into {target_date_display}" if target_date_display else ""

        highlight = next(
            (
                reason
                for key in (
                    "technical_reasons",
                    "sentiment_reasons",
                    "macro_reasons",
                    "fundamental_reasons",
                )
                for reason in reasons.get(key, [])
                if reason
            ),
            "",
        )

        sentences: list[str] = []
        base_sentence = f"{direction} {horizon_phrase}{target_fragment}".strip()
        if highlight:
            clause = highlight.rstrip(".")
            if clause:
                base_sentence += f" driven by {clause}"
        sentences.append(base_sentence + ".")

        if pct_change is not None and np.isfinite(pct_change) and pct_change != 0:
            sentences.append(f"Expected move of {pct_change * 100:.2f}% versus the last close.")

        if forecast_return is not None and np.isfinite(forecast_return):
            sentences.append(f"Projected horizon return of {forecast_return * 100:.2f}%.")

        if volatility is not None and np.isfinite(volatility) and volatility > 0:
            sentences.append(f"Anticipated volatility around {volatility * 100:.2f}%.")

        driver_sections: list[str] = []
        for category, names in sorted(top_feature_drivers.items()):
            if not names:
                continue
            display_category = category.replace("_", " ").title()
            joined = ", ".join(names)
            driver_sections.append(f"{display_category}: {joined}")
        if driver_sections:
            sentences.append("Key drivers by category – " + "; ".join(driver_sections) + ".")

        confidence_sections: list[str] = []
        direction_prob = confidence_indicators.get("direction_probability")
        if isinstance(direction_prob, dict):
            up_prob = self._safe_float(direction_prob.get("up"))
            down_prob = self._safe_float(direction_prob.get("down"))
            if up_prob is not None and down_prob is not None:
                confidence_sections.append(
                    f"Upside probability {up_prob * 100:.1f}% vs. downside {down_prob * 100:.1f}%"
                )
            elif up_prob is not None:
                confidence_sections.append(f"Upside probability {up_prob * 100:.1f}%")
            elif down_prob is not None:
                confidence_sections.append(f"Downside probability {down_prob * 100:.1f}%")

        validation_scores = confidence_indicators.get("validation_scores")
        if isinstance(validation_scores, dict) and validation_scores:
            metrics_parts: list[str] = []
            for label, key, formatter in (
                ("RMSE", "rmse", "{:.3f}"),
                ("MAE", "mae", "{:.3f}"),
                ("Directional accuracy", "directional_accuracy", "{:.1%}"),
                ("R²", "r2", "{:.2f}"),
            ):
                value = self._safe_float(validation_scores.get(key))
                if value is None:
                    continue
                try:
                    formatted = formatter.format(value)
                except (ValueError, TypeError):
                    continue
                metrics_parts.append(f"{label} {formatted}")
            if metrics_parts:
                confidence_sections.append("Validation " + ", ".join(metrics_parts))

        uncertainty_scores = confidence_indicators.get("uncertainty")
        if isinstance(uncertainty_scores, dict) and uncertainty_scores:
            first_target = next(iter(uncertainty_scores))
            target_uncertainty = uncertainty_scores.get(first_target) or {}
            std_value = self._safe_float(target_uncertainty.get("std"))
            if std_value is not None and std_value > 0:
                confidence_sections.append(
                    f"{first_target} prediction std ±{std_value:.3f}"
                )

        if confidence_sections:
            sentences.append("Confidence cues: " + "; ".join(confidence_sections) + ".")

        summary = " ".join(sentence.strip() for sentence in sentences if sentence)
        return summary.strip()

    def _collect_feature_importance(
        self, horizon: Optional[int] = None, top_n: int = 10
    ) -> list[Dict[str, Any]]:
        try:
            importance_map = self.feature_importance("close", horizon)
        except Exception as exc:  # pylint: disable=broad-except
            LOGGER.debug("Feature importance unavailable: %s", exc)
            return []
        if not importance_map:
            return []

        ordered = sorted(importance_map.items(), key=lambda item: abs(item[1]), reverse=True)[:top_n]
        results: list[Dict[str, Any]] = []
        for name, value in ordered:
            category = self._categorize_feature_name(name)
            results.append({
                "name": name,
                "importance": float(value),
                "category": category,
            })
        return results

    @staticmethod
    def _top_feature_drivers(
        feature_importance: list[Dict[str, Any]], per_category: int = 2
    ) -> Dict[str, list[str]]:
        drivers: dict[str, list[str]] = {}
        for entry in feature_importance:
            category = str(entry.get("category") or "other")
            name = entry.get("name")
            if not name:
                continue
            drivers.setdefault(category, []).append(str(name))
        return {
            category: values[:per_category]
            for category, values in drivers.items()
            if values
        }

    def _collect_confidence_indicators(
        self, prediction: Dict[str, Any], horizon: Optional[int]
    ) -> Dict[str, Any]:
        indicators: Dict[str, Any] = {}

        up_prob = self._safe_float(prediction.get("direction_probability_up"))
        down_prob = self._safe_float(prediction.get("direction_probability_down"))
        if up_prob is not None or down_prob is not None:
            probability_block: Dict[str, float] = {}
            if up_prob is not None:
                probability_block["up"] = up_prob
            if down_prob is not None:
                probability_block["down"] = down_prob
            indicators["direction_probability"] = probability_block

        uncertainties_raw = prediction.get("prediction_uncertainty")
        if isinstance(uncertainties_raw, dict):
            uncertainty_block: Dict[str, Dict[str, float]] = {}
            for target, values in uncertainties_raw.items():
                if not isinstance(values, dict):
                    continue
                numeric = {
                    key: float(value)
                    for key, value in values.items()
                    if self._safe_float(value) is not None
                }
                if numeric:
                    uncertainty_block[str(target)] = numeric
            if uncertainty_block:
                indicators["uncertainty"] = uncertainty_block

        validation_scores = self._validation_metrics("close", horizon)
        if validation_scores:
            indicators["validation_scores"] = validation_scores

        return indicators

    def _validation_metrics(
        self, target: str, horizon: Optional[int]
    ) -> Dict[str, float]:
        metrics_store = self.metadata.get("metrics")
        if not isinstance(metrics_store, dict):
            return {}
        target_metrics = metrics_store.get(target)
        if not isinstance(target_metrics, dict) or not target_metrics:
            return {}

        entry: Optional[Dict[str, Any]] = None
        if horizon is not None:
            entry = target_metrics.get(horizon)

        if entry is None:
            default_horizon = self.metadata.get("active_horizon")
            try:
                default_horizon = int(default_horizon)
            except (TypeError, ValueError):
                default_horizon = None
            if default_horizon is not None:
                entry = target_metrics.get(default_horizon)

        if entry is None:
            entry = next(iter(target_metrics.values()), None)

        if not isinstance(entry, dict):
            return {}

        metrics: Dict[str, float] = {}
        evaluation = entry.get("evaluation")
        if isinstance(evaluation, dict):
            aggregate = evaluation.get("aggregate")
            if isinstance(aggregate, dict):
                for key, value in aggregate.items():
                    numeric = self._safe_float(value)
                    if numeric is not None:
                        metrics[key] = numeric

        for key in ("rmse", "mae", "mape", "directional_accuracy", "r2"):
            if key in metrics:
                continue
            numeric = self._safe_float(entry.get(key))
            if numeric is not None:
                metrics[key] = numeric

        return metrics

    @staticmethod
    def _categorize_feature_name(name: str) -> str:
        token = name.lower()
        if any(keyword in token for keyword in ("rsi", "macd", "bollinger", "sma", "ema", "atr", "return")):
            return "technical"
        if "sentiment" in token:
            return "sentiment"
        if any(keyword in token for keyword in ("volatility", "trend", "correlation")):
            return "macro"
        if any(keyword in token for keyword in ("price_to", "momentum", "liquidity")):
            return "fundamental"
        if "volume" in token or "obv" in token:
            return "price"
        return "other"

    @staticmethod
    def _safe_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            result = float(value)
        except (TypeError, ValueError):
            return None
        if np.isnan(result):
            return None
        return result

    def feature_importance(
        self, target: str = "close", horizon: Optional[int] = None
    ) -> Dict[str, float]:
        resolved_horizon = self._resolve_horizon(horizon)
        model = self.models.get((target, resolved_horizon)) or self.load_model(
            target, resolved_horizon
        )
        feature_columns = self.metadata.get("feature_columns")
        if not feature_columns:
            raise RuntimeError("Feature columns unknown; train the model first.")
        return extract_feature_importance(model, list(feature_columns))

    def list_available_models(self) -> Dict[str, str]:
        entries: Dict[str, str] = {}
        for horizon in self.config.prediction_horizons:
            for target in self.config.prediction_targets:
                path = self.config.model_path_for(target, horizon)
                if path.exists():
                    entries[f"{target}_h{horizon}"] = str(path)
        return entries

    # ------------------------------------------------------------------
    # Backtesting
    # ------------------------------------------------------------------
    def run_backtest(
        self, targets: Optional[Iterable[str]] = None, horizon: Optional[int] = None
    ) -> Dict[str, Any]:
        resolved_horizon = self._resolve_horizon(horizon)
        self.horizon = resolved_horizon
        X, targets_by_horizon = self.prepare_features()
        horizon_targets = targets_by_horizon.get(resolved_horizon)
        if not horizon_targets:
            raise RuntimeError(f"No targets available for horizon {resolved_horizon}.")
        requested_targets = list(targets) if targets else list(self.config.prediction_targets)
        results: dict[str, Any] = {}

        for target in requested_targets:
            if target not in horizon_targets:
                LOGGER.warning("Skipping backtest for target '%s' (no data available).", target)
                continue
            factory = ModelFactory(
                self.config.model_type,
                {**self.config.model_params.get("global", {}), **self.config.model_params.get(target, {})},
            )
            y_clean = horizon_targets[target].dropna()
            aligned_X = X.loc[y_clean.index]

            backtester = Backtester(
                model_factory=factory,
                strategy=self.config.backtest_strategy,
                window=self.config.backtest_window,
                step=self.config.backtest_step,
            )
            result = backtester.run(aligned_X, y_clean, target)
            results[target] = {
                "aggregate": result.aggregate,
                "splits": result.splits,
            }
            self.tracker.log_run(
                target=target,
                run_type="backtest",
                parameters={
                    "model_type": self.config.model_type,
                    "strategy": self.config.backtest_strategy,
                    "window": self.config.backtest_window,
                    "step": self.config.backtest_step,
                },
                metrics=result.aggregate,
                context={"splits": result.splits, "horizon": resolved_horizon},
            )
        self.metadata["active_horizon"] = resolved_horizon
        return results

