"""High level orchestration for feature engineering, training and inference."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

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


class StockPredictorAI:
    """Pipeline that assembles features, trains models, and produces forecasts."""

    def __init__(self, config: PredictorConfig) -> None:
        self.config = config
        self.fetcher = DataFetcher(config)
        self.feature_assembler = FeatureAssembler(config.feature_sets)
        self.tracker = ExperimentTracker(config)
        self.models: dict[str, Any] = {}
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
    ) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
        if price_df is None:
            price_df = self.fetcher.fetch_price_data()
        if news_df is None and self.config.sentiment:
            news_df = self.fetcher.fetch_news_data()
        elif news_df is None:
            news_df = pd.DataFrame()

        feature_result = self.feature_assembler.build(price_df, news_df, self.config.sentiment)
        self.metadata = feature_result.metadata
        return feature_result.features, feature_result.targets

    # ------------------------------------------------------------------
    # Model persistence helpers
    # ------------------------------------------------------------------
    def _get_model(self, target: str) -> Any:
        if target not in self.models:
            raise RuntimeError(f"Model for target '{target}' is not loaded. Train or load it first.")
        return self.models[target]

    def load_model(self, target: str = "close") -> Any:
        path = self.config.model_path_for(target)
        if not path.exists():
            raise FileNotFoundError(f"Model file {path} not found. Train the model first.")
        LOGGER.info("Loading %s model for target '%s'", self.config.model_type, target)
        model = joblib.load(path)
        self.models[target] = model
        metadata_path = self.config.metrics_path_for(target)
        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as handle:
                stored = json.load(handle)
                feature_columns = stored.get("feature_columns")
                if feature_columns:
                    self.metadata["feature_columns"] = feature_columns
                indicator_columns = stored.get("indicator_columns")
                if indicator_columns:
                    self.metadata["indicator_columns"] = indicator_columns
        return self.model

    def save_state(self, metrics: Dict[str, Any]) -> None:
        """Persist metrics and feature information to disk."""

    def save_state(self, target: str, metrics: Dict[str, Any]) -> None:
        payload = {
            **metrics,
            "feature_columns": self.metadata.get("feature_columns", []),
            "indicator_columns": self.metadata.get("indicator_columns", []),
        }
        path = self.config.metrics_path_for(target)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, default=str)
        LOGGER.info("Saved metrics for target '%s' to %s", target, path)
        if target == "close":
            with open(self.config.metrics_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, default=str)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train_model(self, targets: Optional[Iterable[str]] = None) -> Dict[str, Any]:
        X, y_targets = self.prepare_features()
        feature_columns = self.metadata.get("feature_columns", list(X.columns))

        requested_targets = list(targets) if targets else list(self.config.prediction_targets)
        available_targets = {name: y for name, y in y_targets.items() if name in requested_targets}
        if not available_targets:
            raise RuntimeError("No matching targets available for training.")

        metrics_by_target: dict[str, Dict[str, float]] = {}
        summary_metrics: dict[str, float] = {}

        for target, y in available_targets.items():
            LOGGER.info("Training target '%s' with model type %s", target, self.config.model_type)
            y_clean = y.dropna()
            aligned_X = X.loc[y_clean.index]
            task = "classification" if target == "direction" else "regression"
            model_params = self.config.model_params.get(target, {})
            factory = ModelFactory(self.config.model_type, {**self.config.model_params.get("global", {}), **model_params})
            model = factory.create(task)

            X_train, X_test, y_train, y_test = train_test_split(
                aligned_X,
                y_clean,
                test_size=self.config.test_size,
                shuffle=self.config.shuffle_training,
                random_state=42,
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            if task == "classification":
                metrics = classification_metrics(y_test.to_numpy(), y_pred)
                metrics["directional_accuracy"] = metrics.get("accuracy", 0.0)
            else:
                metrics = regression_metrics(y_test.to_numpy(), y_pred)
                metrics["r2"] = float(r2_score(y_test, y_pred))
                metrics["directional_accuracy"] = float(
                    np.mean(np.sign(y_pred - X_test["Close_Current"]) == np.sign(y_test - X_test["Close_Current"]))
                )

            metrics["training_rows"] = int(len(X_train))
            metrics["test_rows"] = int(len(X_test))
            metrics_by_target[target] = metrics

            self.models[target] = model
            joblib.dump(model, self.config.model_path_for(target))
            self.save_state(target, metrics)
            self.tracker.log_run(
                target=target,
                run_type="training",
                parameters={"model_type": self.config.model_type, **model_params},
                metrics=metrics,
                context={"feature_columns": feature_columns},
            )

            if target == "close":
                summary_metrics.update({k: v for k, v in metrics.items() if isinstance(v, (int, float))})

        LOGGER.info("Training complete for targets: %s", ", ".join(metrics_by_target))
        return {"targets": metrics_by_target, **summary_metrics}

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def predict(self, refresh_data: bool = False, targets: Optional[Iterable[str]] = None) -> Dict[str, Any]:
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

        requested_targets = list(targets) if targets else list(self.config.prediction_targets)
        predictions: dict[str, Any] = {}
        confidences: dict[str, float] = {}
        training_report: dict[str, Any] = {}

        for target in requested_targets:
            try:
                model = self.models.get(target) or self.load_model(target)
            except FileNotFoundError:
                LOGGER.warning("Model for target '%s' missing. Triggering training.", target)
                report = self.train_model(targets=[target])
                training_report[target] = report.get("targets", {}).get(target, {})
                model = self._get_model(target)

            current_features = latest_features[feature_columns]
            pred_value = model.predict(current_features)[0]
            predictions[target] = float(pred_value)

            if model_supports_proba(model) and target == "direction":
                proba = model.predict_proba(current_features)[0]
                confidences[target] = float(max(proba))

        close_prediction = predictions.get("close")
        latest_close = float(self.metadata.get("latest_close", np.nan))
        expected_change = None
        pct_change = None
        if close_prediction is not None and np.isfinite(latest_close):
            expected_change = close_prediction - latest_close
            pct_change = expected_change / latest_close if latest_close else 0.0

        result = {
            "ticker": self.config.ticker,
            "as_of": str(self.metadata.get("latest_date")),
            "last_close": latest_close,
            "predicted_close": close_prediction,
            "expected_change": expected_change,
            "expected_change_pct": pct_change,
            "predictions": predictions,
        }
        if confidences:
            result["confidence"] = confidences
        if training_report:
            result["training_metrics"] = training_report
        return result

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------
    def feature_importance(self, target: str = "close") -> Dict[str, float]:
        model = self.models.get(target) or self.load_model(target)
        feature_columns = self.metadata.get("feature_columns")
        if not feature_columns:
            raise RuntimeError("Feature columns unknown; train the model first.")
        return extract_feature_importance(model, list(feature_columns))

    def list_available_models(self) -> Dict[str, str]:
        entries = {}
        for target in self.config.prediction_targets:
            path = self.config.model_path_for(target)
            if path.exists():
                entries[target] = str(path)
        return entries

    # ------------------------------------------------------------------
    # Backtesting
    # ------------------------------------------------------------------
    def run_backtest(self, targets: Optional[Iterable[str]] = None) -> Dict[str, Any]:
        X, y_targets = self.prepare_features()
        requested_targets = list(targets) if targets else list(self.config.prediction_targets)
        results: dict[str, Any] = {}

        for target in requested_targets:
            if target not in y_targets:
                LOGGER.warning("Skipping backtest for target '%s' (no data available).", target)
                continue
            factory = ModelFactory(
                self.config.model_type,
                {**self.config.model_params.get("global", {}), **self.config.model_params.get(target, {})},
            )
            y_clean = y_targets[target].dropna()
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
                context={"splits": result.splits},
            )
        return results

