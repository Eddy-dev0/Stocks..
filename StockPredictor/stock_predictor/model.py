"""High level orchestration for feature engineering, training and inference."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

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


class ModelNotFoundError(FileNotFoundError):
    """Raised when a persisted model for a target cannot be located on disk."""

    def __init__(self, target: str, path: Path) -> None:
        super().__init__(f"Saved model for target '{target}' not found at {path}.")
        self.target = target
        self.path = path


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
        metadata = dict(feature_result.metadata)
        metadata.setdefault("sentiment_daily", pd.DataFrame(columns=["Date", "sentiment"]))
        metadata["data_sources"] = self.fetcher.get_data_sources()
        self.metadata = metadata
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
            raise ModelNotFoundError(target, path)
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
        return model

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
            except ModelNotFoundError:
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

        prediction_timestamp = datetime.now()

        latest_date = self.metadata.get("latest_date")
        if isinstance(latest_date, pd.Timestamp):
            latest_date = latest_date.to_pydatetime()

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
            "predictions": predictions,
        }
        if confidences:
            result["confidence"] = confidences
        if training_report:
            result["training_metrics"] = training_report
        explanation = self._build_prediction_explanation(result, predictions)
        if explanation:
            result["explanation"] = explanation
        return result

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

        reasons = {
            "technical_reasons": self._technical_reasons(feature_row),
            "fundamental_reasons": self._fundamental_reasons(feature_row),
            "sentiment_reasons": self._sentiment_reasons(),
            "macro_reasons": self._macro_reasons(feature_row),
        }

        summary = self._compose_summary(prediction, reasons)
        feature_importance = self._collect_feature_importance()
        sources_raw = self.metadata.get("data_sources") or self.fetcher.get_data_sources()
        sources = list(sources_raw) if isinstance(sources_raw, (list, tuple, set)) else []
        if not sources:
            sources = [f"Database cache: price history for {self.config.ticker}."]

        explanation: Dict[str, Any] = {
            "summary": summary,
            **reasons,
            "feature_importance": feature_importance,
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

    def _compose_summary(self, prediction: Dict[str, Any], reasons: Dict[str, list[str]]) -> str:
        change = self._safe_float(prediction.get("expected_change"))
        pct_change = self._safe_float(prediction.get("expected_change_pct"))
        if change is None or not np.isfinite(change):
            direction = "Neutral"
        elif change > 0:
            direction = "Bullish"
        elif change < 0:
            direction = "Bearish"
        else:
            direction = "Neutral"

        highlight = next(
            (reason for key in ("technical_reasons", "sentiment_reasons", "macro_reasons", "fundamental_reasons") for reason in reasons.get(key, []) if reason),
            "",
        )
        sentences: list[str] = []
        if highlight and direction != "Neutral":
            clause = highlight.rstrip(".")
            if clause:
                sentences.append(f"{direction} outlook driven by {clause}.")
            else:
                sentences.append(f"{direction} outlook.")
        else:
            sentences.append(f"{direction} outlook.")
            if highlight:
                sentences.append(f"Key driver: {highlight}")

        if pct_change is not None and np.isfinite(pct_change) and pct_change != 0:
            sentences.append(f"Expected move of {pct_change * 100:.2f}% versus the last close.")

        summary = " ".join(sentence.strip() for sentence in sentences if sentence)
        return summary.strip()

    def _collect_feature_importance(self, top_n: int = 10) -> list[Dict[str, Any]]:
        try:
            importance_map = self.feature_importance("close")
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

