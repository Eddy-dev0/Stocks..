"""Machine learning models for stock prediction."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import PredictorConfig
from .data_fetcher import DataFetcher
from .preprocessing import build_supervised_dataset

LOGGER = logging.getLogger(__name__)


class StockPredictorAI:
    """High level interface for training and using stock prediction models."""

    def __init__(self, config: PredictorConfig) -> None:
        self.config = config
        self.fetcher = DataFetcher(config)
        self.model: Optional[Pipeline] = None
        self.metadata: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Data acquisition
    # ------------------------------------------------------------------
    def download_data(self, force: bool = False) -> Dict[str, Path]:
        """Download raw data and return the cache file paths."""

        LOGGER.info("Starting data download for %s", self.config.ticker)
        _prices, news = self.fetcher.download_all(force=force)
        result = {"prices": self.config.price_cache_path}
        if not news.empty:
            result["news"] = self.config.news_cache_path
        LOGGER.info("Data download completed for %s", self.config.ticker)
        return result

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------
    def prepare_features(
        self,
        price_df: Optional[pd.DataFrame] = None,
        news_df: Optional[pd.DataFrame] = None,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Prepare the feature matrix and target vector."""

        if price_df is None:
            price_df = self.fetcher.fetch_price_data()
        if news_df is None and self.config.sentiment:
            news_df = self.fetcher.fetch_news_data()
        elif news_df is None:
            news_df = pd.DataFrame()

        X, y, metadata = build_supervised_dataset(price_df, news_df)
        self.metadata = metadata
        return X, y

    # ------------------------------------------------------------------
    # Model persistence helpers
    # ------------------------------------------------------------------
    def _get_model(self) -> Pipeline:
        if self.model is None:
            raise RuntimeError("Model is not loaded. Call load_model() or train_model() first.")
        return self.model

    def load_model(self) -> Pipeline:
        """Load a previously saved model from disk."""

        model_path = self.config.model_path
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file {model_path} not found. Train the model first."
            )
        LOGGER.info("Loading model from %s", model_path)
        self.model = joblib.load(model_path)
        metadata_path = self.config.metrics_path
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

        path = self.config.metrics_path
        payload = {
            **metrics,
            "feature_columns": self.metadata.get("feature_columns", []),
            "indicator_columns": self.metadata.get("indicator_columns", []),
        }
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, default=str)
        LOGGER.info("Saved metrics to %s", path)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train_model(self) -> Dict[str, Any]:
        """Train the configured machine learning model and persist it."""

        X, y = self.prepare_features()
        feature_columns = X.columns.tolist()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )

        if self.config.model_type != "random_forest":
            LOGGER.warning(
                "Model type '%s' not recognised; falling back to RandomForestRegressor.",
                self.config.model_type,
            )

        model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "regressor",
                    RandomForestRegressor(
                        n_estimators=200,
                        random_state=42,
                        max_depth=10,
                    ),
                ),
            ]
        )
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        metrics = {
            "mae": float(mean_absolute_error(y_test, predictions)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
            "r2": float(r2_score(y_test, predictions)),
            "training_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
        }
        LOGGER.info(
            "Training complete for %s - MAE: %.4f, RMSE: %.4f, R2: %.4f",
            self.config.ticker,
            metrics["mae"],
            metrics["rmse"],
            metrics["r2"],
        )

        self.model = model
        self.metadata["feature_columns"] = feature_columns
        joblib.dump(model, self.config.model_path)
        LOGGER.info("Saved model to %s", self.config.model_path)

        self.save_state(metrics)
        return metrics

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def predict(self, refresh_data: bool = False) -> Dict[str, Any]:
        """Generate a prediction for the next trading day."""

        if refresh_data:
            LOGGER.info("Refreshing data before prediction.")
            self.download_data(force=True)
        if not self.metadata or refresh_data:
            LOGGER.info("Preparing features prior to prediction.")
            self.prepare_features()

        training_metrics: Dict[str, Any] | None = None

        try:
            model = self.model or self.load_model()
        except FileNotFoundError as exc:
            LOGGER.warning("%s. Triggering automatic training.", exc)
            if refresh_data:
                LOGGER.info("Refreshing data before automatic training.")
                self.download_data(force=True)
            metrics = self.train_model()
            training_metrics = metrics
            LOGGER.info(
                "Automatic training complete for %s (MAE %.4f, RMSE %.4f, R2 %.4f)",
                self.config.ticker,
                metrics.get("mae", float("nan")),
                metrics.get("rmse", float("nan")),
                metrics.get("r2", float("nan")),
            )
            model = self._get_model()
        feature_columns = self.metadata.get("feature_columns")
        latest_features = self.metadata.get("latest_features")
        if latest_features is None or feature_columns is None:
            raise RuntimeError("No feature metadata available. Train the model first.")

        latest_features = latest_features[feature_columns]
        prediction = float(model.predict(latest_features)[0])
        latest_close = float(self.metadata.get("latest_close"))
        expected_change = prediction - latest_close
        pct_change = expected_change / latest_close if latest_close else 0.0

        result = {
            "ticker": self.config.ticker,
            "predicted_close": prediction,
            "last_close": latest_close,
            "expected_change": expected_change,
            "expected_change_pct": pct_change,
            "as_of": str(self.metadata.get("latest_date")),
        }
        if training_metrics is not None:
            result["training_metrics"] = training_metrics
        LOGGER.info(
            "Prediction for %s: %.2f (change %.2f / %.2f%%)",
            self.config.ticker,
            prediction,
            expected_change,
            pct_change * 100,
        )
        return result
