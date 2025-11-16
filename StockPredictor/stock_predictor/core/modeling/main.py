"""High level orchestration for feature engineering, training and inference."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from numbers import Real
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline

from ..backtesting import Backtester
from ..config import PredictorConfig
from ..data_fetcher import DataFetcher
from ..database import ExperimentTracker
from ..features import FeatureAssembler
from ..ml_preprocessing import (
    PreprocessingBuilder,
    get_feature_names_from_pipeline,
)
from ..models import (
    ModelFactory,
    classification_metrics,
    extract_feature_importance,
    model_supports_proba,
    regression_metrics,
)

LOGGER = logging.getLogger(__name__)


LabelFunction = Callable[[pd.DataFrame, int, int], pd.Series]


@dataclass(frozen=True)
class TargetSpec:
    """Description of a supported prediction target."""

    name: str
    task: str
    default_model_type: Optional[str] = None
    label_fn: Optional[LabelFunction] = None


def make_volatility_label(
    df: pd.DataFrame, horizon: int, window: int = 20
) -> pd.Series:
    """Create a realised volatility label using rolling standard deviation."""

    if window <= 0:
        raise ValueError("window must be a positive integer")

    working = df.copy()
    lower_columns = {column.lower(): column for column in working.columns}

    if "return" in working.columns:
        returns = pd.to_numeric(working["return"], errors="coerce")
    else:
        close_column = lower_columns.get("close")
        if close_column is None:
            raise KeyError("Input dataframe must contain a 'close' column for volatility labels.")
        close_series = pd.to_numeric(working[close_column], errors="coerce")
        returns = close_series.pct_change()

    volatility = returns.rolling(window=window, min_periods=window).std()
    shift = int(horizon) if horizon and int(horizon) > 0 else 1
    return volatility.shift(-shift)


TARGET_SPECS: dict[str, TargetSpec] = {
    "close": TargetSpec("close", "regression"),
    "direction": TargetSpec("direction", "classification"),
    "return": TargetSpec("return", "regression"),
    "volatility": TargetSpec(
        "volatility", "regression", default_model_type="random_forest", label_fn=make_volatility_label
    ),
}

SUPPORTED_TARGETS = frozenset(TARGET_SPECS)


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
        self.preprocessors: dict[Tuple[str, int], Pipeline] = {}
        self.preprocessor_templates: dict[int, Pipeline] = {}
        self.preprocess_options: dict[str, Any] = {}
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
    ) -> tuple[pd.DataFrame, dict[int, Dict[str, pd.Series]], dict[int, Pipeline]]:
        if price_df is None:
            price_df = self.fetcher.fetch_price_data()
        if news_df is None and self.config.sentiment:
            news_df = self.fetcher.fetch_news_data()
        elif news_df is None:
            news_df = pd.DataFrame()

        macro_frame = self._load_macro_indicators()
        merged_price_df = self._merge_macro_columns(price_df, macro_frame)
        fundamentals_df: pd.DataFrame | None
        fetch_fundamentals = getattr(self.fetcher, "fetch_fundamentals", None)
        if callable(fetch_fundamentals):
            fundamentals_df = fetch_fundamentals()
        else:
            fundamentals_df = pd.DataFrame()

        feature_result = self.feature_assembler.build(
            merged_price_df,
            news_df,
            self.config.sentiment,
            fundamentals_df=fundamentals_df,
            macro_df=macro_frame if not macro_frame.empty else None,
        )
        metadata = dict(feature_result.metadata)
        raw_feature_columns = list(feature_result.features.columns)
        metadata.setdefault("feature_columns", raw_feature_columns)
        metadata["raw_feature_columns"] = raw_feature_columns
        metadata.setdefault("sentiment_daily", pd.DataFrame(columns=["Date", "sentiment"]))
        metadata.setdefault("feature_groups", {})
        metadata["data_sources"] = self.fetcher.get_data_sources()
        metadata.setdefault("target_dates", {})
        horizons = tuple(metadata.get("horizons", tuple(self.config.prediction_horizons)))
        metadata["horizons"] = horizons
        metadata["active_horizon"] = self.horizon

        volatility_spec = TARGET_SPECS.get("volatility")
        if volatility_spec and volatility_spec.label_fn:
            window = getattr(self.config, "volatility_window", 20)
            price_history = merged_price_df.copy()
            if "Date" in price_history.columns:
                price_history["Date"] = pd.to_datetime(price_history["Date"], errors="coerce")
                price_history = price_history.dropna(subset=["Date"])
                price_history = price_history.sort_values("Date").reset_index(drop=True)
            else:
                price_history = price_history.sort_index().reset_index(drop=True)

            lower_map = {column.lower(): column for column in price_history.columns}
            close_column = lower_map.get("close")
            if close_column is not None:
                price_history["return"] = pd.to_numeric(
                    price_history[close_column], errors="coerce"
                ).pct_change()

            for horizon_value in horizons:
                try:
                    label_series = volatility_spec.label_fn(
                        price_history, int(horizon_value), int(window)
                    )
                except Exception as exc:  # pragma: no cover - defensive
                    LOGGER.debug(
                        "Failed to compute volatility labels for horizon %s: %s",
                        horizon_value,
                        exc,
                    )
                    continue
                if label_series is None:
                    continue
                label_series = pd.Series(label_series, name="volatility")
                if label_series.dropna().empty:
                    continue
                label_series = label_series.reset_index(drop=True)
                label_series.index = feature_result.features.index
                horizon_targets = feature_result.targets.setdefault(int(horizon_value), {})
                horizon_targets["volatility"] = label_series
            metadata["volatility_window"] = int(window)

        preprocess_section = self.config.model_params.get("preprocessing", {})
        if isinstance(preprocess_section, dict):
            self.preprocess_options = dict(preprocess_section)
        else:
            self.preprocess_options = {}
        builder = PreprocessingBuilder(**self.preprocess_options)

        templates: dict[int, Pipeline] = {}
        template_feature_names: dict[int, list[str]] = {}
        features = feature_result.features
        for horizon_value in horizons:
            pipeline = builder.create_pipeline()
            X_source = features
            y_source: Optional[pd.Series] = None
            horizon_targets = feature_result.targets.get(horizon_value)
            if horizon_targets:
                candidate = horizon_targets.get("close")
                if candidate is None:
                    candidate = next(iter(horizon_targets.values()), None)
                if candidate is not None:
                    y_clean = candidate.dropna()
                    if not y_clean.empty:
                        aligned_index = y_clean.index.intersection(features.index)
                        if not aligned_index.empty:
                            X_source = features.loc[aligned_index]
                            y_source = y_clean.loc[aligned_index]
                        else:
                            y_source = y_clean
            pipeline.fit(X_source, y_source)
            templates[horizon_value] = pipeline
            template_feature_names[horizon_value] = get_feature_names_from_pipeline(pipeline)

        metadata["preprocessed_feature_columns"] = template_feature_names
        self.metadata = metadata
        self.preprocessor_templates = templates
        return features, feature_result.targets, templates

    def _load_macro_indicators(self) -> pd.DataFrame:
        """Load cached macro indicator close series and pivot them by date."""

        fetch_macro = getattr(self.fetcher, "fetch_indicator_data", None)
        if not callable(fetch_macro):
            return pd.DataFrame()

        try:
            macro_rows = fetch_macro(category="macro")
        except Exception as exc:  # pragma: no cover - guard around IO
            LOGGER.debug("Failed to fetch macro indicators: %s", exc)
            return pd.DataFrame()

        if macro_rows is None or macro_rows.empty:
            return pd.DataFrame()

        required_columns = {"Date", "Indicator", "Value"}
        if not required_columns.issubset(macro_rows.columns):
            return pd.DataFrame()

        working = macro_rows[["Date", "Indicator", "Value"]].copy()
        working["Date"] = pd.to_datetime(working["Date"], errors="coerce")
        working = working.dropna(subset=["Date"])
        working["Indicator"] = working["Indicator"].astype(str)
        working["Symbol"] = working["Indicator"].str.split(":", n=1).str[-1]
        working = working[working["Symbol"].astype(bool)]
        working["Column"] = working["Symbol"].apply(lambda sym: f"Close_{sym}")
        working["Value"] = pd.to_numeric(working["Value"], errors="coerce")
        working = working.dropna(subset=["Value"])
        if working.empty:
            return pd.DataFrame()

        pivot = (
            working.pivot_table(index="Date", columns="Column", values="Value", aggfunc="last")
            .sort_index()
            .reset_index()
        )
        if pivot.empty:
            return pd.DataFrame()
        pivot.columns = [str(col) for col in pivot.columns]
        return pivot

    def _merge_macro_columns(
        self, price_df: pd.DataFrame, macro_frame: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge macro benchmark columns onto the base price frame."""

        if macro_frame is None or macro_frame.empty:
            return price_df.copy()

        price = price_df.copy()
        macro = macro_frame.copy()

        if "Date" not in price.columns:
            if isinstance(price.index, pd.DatetimeIndex):
                price = price.reset_index().rename(columns={"index": "Date"})
            elif price.index.name and price.index.name.lower() == "date":
                price = price.reset_index().rename(columns={price.index.name: "Date"})
            else:
                return price

        price["Date"] = pd.to_datetime(price["Date"], errors="coerce")
        macro["Date"] = pd.to_datetime(macro["Date"], errors="coerce")
        price = price.dropna(subset=["Date"])
        macro = macro.dropna(subset=["Date"])

        merged = pd.merge(price, macro, on="Date", how="left")
        return merged

    # ------------------------------------------------------------------
    # Model persistence helpers
    # ------------------------------------------------------------------
    def _resolve_horizon(self, horizon: Optional[int]) -> int:
        if horizon is None:
            return self.horizon
        return self.config.resolve_horizon(horizon)

    def _ensure_models_dir(self) -> None:
        """Ensure that the models directory exists."""

        Path(self.config.models_dir).mkdir(parents=True, exist_ok=True)

    def _get_model(self, target: str, horizon: Optional[int] = None) -> Any:
        resolved_horizon = self._resolve_horizon(horizon)
        key = (target, resolved_horizon)
        if key not in self.models:
            raise RuntimeError(
                f"Model for target '{target}' and horizon {resolved_horizon} is not loaded. Train or load it first."
            )
        return self.models[key]

    def _load_preprocessor(self, target: str, horizon: int) -> Optional[Pipeline]:
        key = (target, horizon)
        if key in self.preprocessors:
            return self.preprocessors[key]
        path = self.config.preprocessor_path_for(target, horizon)
        if not path.exists():
            return None
        try:
            pipeline: Pipeline = joblib.load(path)
        except Exception as exc:  # pragma: no cover - defensive guard around disk IO
            LOGGER.warning("Failed to load preprocessor for %s horizon %s: %s", target, horizon, exc)
            return None
        self.preprocessors[key] = pipeline
        feature_names = get_feature_names_from_pipeline(pipeline)
        if feature_names:
            feature_map = self.metadata.setdefault("feature_columns_by_target", {})
            feature_map[key] = feature_names
            self.metadata["feature_columns"] = feature_names
        return pipeline

    def load_model(self, target: str = "close", horizon: Optional[int] = None) -> Any:
        resolved_horizon = self._resolve_horizon(horizon)
        path = self.config.model_path_for(target, resolved_horizon)
        self._ensure_models_dir()
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
        self._load_preprocessor(target, resolved_horizon)
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
        path.parent.mkdir(parents=True, exist_ok=True)
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
        *,
        force: bool = False,
    ) -> Dict[str, Any]:
        resolved_horizon = self._resolve_horizon(horizon)
        self.horizon = resolved_horizon
        X, targets_by_horizon, _ = self.prepare_features()
        raw_feature_columns = self.metadata.get("raw_feature_columns", list(X.columns))

        requested_targets = list(targets) if targets else list(self.config.prediction_targets)
        supported_targets: list[str] = []
        for target in requested_targets:
            if target not in SUPPORTED_TARGETS:
                LOGGER.warning("Skipping unsupported target '%s'.", target)
                continue
            if target not in supported_targets:
                supported_targets.append(target)

        if not supported_targets:
            LOGGER.error("No supported targets requested for training: %s", requested_targets)
            return {"horizon": resolved_horizon, "targets": {}}

        horizon_targets = targets_by_horizon.get(resolved_horizon)
        if not horizon_targets:
            LOGGER.error("No targets available for horizon %s.", resolved_horizon)
            return {"horizon": resolved_horizon, "targets": {}}

        available_targets: dict[str, pd.Series] = {}
        for target in supported_targets:
            series = horizon_targets.get(target)
            if series is None:
                LOGGER.warning(
                    "Skipping target '%s' at horizon %s: no training data available.",
                    target,
                    resolved_horizon,
                )
                continue
            available_targets[target] = series

        if not available_targets:
            LOGGER.error(
                "No matching targets available for training after filtering unsupported or missing data."
            )
            return {"horizon": resolved_horizon, "targets": {}}

        seed = int(self.config.model_params.get("global", {}).get("random_state", 42))
        np.random.seed(seed)

        metrics_by_target: dict[str, Dict[str, Any]] = {}
        summary_metrics: dict[str, float] = {}

        self._ensure_models_dir()

        for target, y in available_targets.items():
            if not self._should_retrain(target, resolved_horizon, force=force):
                LOGGER.info(
                    "Skipping training for target '%s' horizon %s; existing model is up-to-date.",
                    target,
                    resolved_horizon,
                )
                try:
                    self.load_model(target, resolved_horizon)
                except ModelNotFoundError:
                    LOGGER.debug(
                        "Cached model for %s horizon %s missing despite retrain guard; forcing rebuild.",
                        target,
                        resolved_horizon,
                    )
                else:
                    continue

            spec = TARGET_SPECS.get(target)
            task = spec.task if spec else ("classification" if target == "direction" else "regression")
            model_type = (
                spec.default_model_type if spec and spec.default_model_type else self.config.model_type
            )
            LOGGER.info("Training target '%s' with model type %s", target, model_type)
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

            model_params = self.config.model_params.get(target, {})
            global_params = self.config.model_params.get("global", {})
            factory = ModelFactory(model_type, {**global_params, **model_params})

            calibrate_override = model_params.get("calibrate")
            if calibrate_override is None:
                calibrate_override = global_params.get("calibrate")
            if calibrate_override is None:
                calibrate_flag = task == "classification" and target == "direction"
            else:
                calibrate_flag = bool(calibrate_override)

            template = self.preprocessor_templates.get(resolved_horizon)
            evaluation = self._evaluate_model(
                factory,
                aligned_X,
                y_clean,
                task,
                target,
                calibrate_flag,
                template,
            )
            LOGGER.info(
                "Evaluation summary for target '%s' (horizon %s, strategy %s): %s",
                target,
                resolved_horizon,
                evaluation["strategy"],
                evaluation["aggregate"],
            )

            if template is not None:
                final_pipeline = clone(template)
            else:
                builder = PreprocessingBuilder(**self.preprocess_options)
                final_pipeline = builder.create_pipeline()
            # Intentionally fit with a named DataFrame so downstream estimators retain
            # feature metadata, preventing scikit-learn from emitting feature name
            # mismatch warnings when predicting.
            final_pipeline.fit(aligned_X, y_clean)
            transformed_X = final_pipeline.transform(aligned_X)
            feature_names = get_feature_names_from_pipeline(final_pipeline)
            if not feature_names:
                feature_names = list(transformed_X.columns)
            feature_map = self.metadata.setdefault("feature_columns_by_target", {})
            feature_map[(target, resolved_horizon)] = feature_names
            self.metadata.setdefault("feature_columns_by_horizon", {})[resolved_horizon] = feature_names
            preprocessed_map = self.metadata.setdefault("preprocessed_feature_columns", {})
            preprocessed_map[resolved_horizon] = feature_names
            self.metadata["feature_columns"] = feature_names

            final_model = factory.create(task, calibrate=calibrate_flag)
            final_model.fit(transformed_X, y_clean)
            distribution_summary = self._estimate_prediction_uncertainty(
                target, final_model, transformed_X
            )

            metrics: Dict[str, Any] = dict(evaluation["aggregate"])
            metrics["training_rows"] = int(len(transformed_X))
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
            if distribution_summary:
                metrics["forecast_distribution"] = distribution_summary
                self.metadata.setdefault("forecast_distribution", {})[
                    (target, resolved_horizon)
                ] = distribution_summary

            in_sample_proba = None
            estimator = final_model.named_steps.get("estimator")
            if task == "classification" and model_supports_proba(final_model):
                try:
                    in_sample_proba = final_model.predict_proba(transformed_X)
                except Exception as exc:  # pragma: no cover - defensive
                    LOGGER.debug("Failed to compute in-sample probabilities: %s", exc)
            in_sample_pred = final_model.predict(transformed_X)
            metrics["in_sample"] = self._compute_evaluation_metrics(
                task,
                y_clean,
                in_sample_pred,
                transformed_X,
                aligned_X,
                in_sample_proba,
                getattr(estimator, "classes_", None),
            )
            metrics["out_of_sample"] = evaluation["aggregate"]

            metrics_by_target[target] = metrics

            metrics_store = self.metadata.setdefault("metrics", {})
            horizon_metrics = metrics_store.setdefault(target, {})
            horizon_metrics[resolved_horizon] = metrics

            self.models[(target, resolved_horizon)] = final_model
            self.preprocessors[(target, resolved_horizon)] = final_pipeline
            model_path = self.config.model_path_for(target, resolved_horizon)
            preprocessor_path = self.config.preprocessor_path_for(target, resolved_horizon)
            model_path.parent.mkdir(parents=True, exist_ok=True)
            preprocessor_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(final_model, model_path)
            joblib.dump(final_pipeline, preprocessor_path)
            self.save_state(target, resolved_horizon, metrics)
            self.tracker.log_run(
                target=target,
                run_type="training",
                parameters={"model_type": model_type, **model_params},
                metrics=metrics,
                context={"feature_columns": feature_names, "horizon": resolved_horizon},
            )

            if target == "close":
                summary_metrics.update({k: v for k, v in metrics.items() if isinstance(v, (int, float))})

        LOGGER.info(
            "Training complete for targets %s at horizon %s",
            ", ".join(metrics_by_target),
            resolved_horizon,
        )
        self.metadata["feature_columns"] = (
            self.metadata.get("feature_columns_by_horizon", {}).get(resolved_horizon)
            or self.metadata.get("feature_columns", raw_feature_columns)
        )
        self.metadata["active_horizon"] = resolved_horizon
        return {"horizon": resolved_horizon, "targets": metrics_by_target, **summary_metrics}

    def _should_retrain(self, target: str, horizon: int, *, force: bool) -> bool:
        if force:
            return True
        model_path = self.config.model_path_for(target, horizon)
        if not model_path.exists():
            return True
        metrics_path = self.config.metrics_path_for(target, horizon)
        if not metrics_path.exists():
            return True

        try:
            data_timestamp = self.fetcher.get_last_updated("prices")
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.debug("Unable to determine last data refresh: %s", exc)
            return False

        cooldown_hours = float(getattr(self.config, "retrain_cooldown_hours", 0) or 0)
        model_mtime = datetime.fromtimestamp(model_path.stat().st_mtime)

        if cooldown_hours > 0:
            if datetime.utcnow() - model_mtime < timedelta(hours=cooldown_hours):
                return False

        if data_timestamp is None:
            return False

        if isinstance(data_timestamp, datetime):
            data_moment = data_timestamp
        else:
            data_moment = datetime.combine(data_timestamp, datetime.min.time())
        data_moment = data_moment.replace(tzinfo=None)

        return data_moment > model_mtime

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
        preprocessor: Optional[Pipeline] = None,
    ) -> Dict[str, Any]:
        strategy = self.config.evaluation_strategy
        evaluation_rows = 0
        parameters: Dict[str, Any] = {}
        splits: List[Dict[str, Any]] = []

        if strategy == "holdout":
            split_idx = max(
                1, int(len(features) * (1 - self.config.test_size))
            )
            split_idx = min(split_idx, len(features) - 1)
            X_train, X_test = features.iloc[:split_idx], features.iloc[split_idx:]
            y_train, y_test = target_series.iloc[:split_idx], target_series.iloc[split_idx:]
            pipeline = clone(preprocessor) if preprocessor is not None else None
            if pipeline is not None:
                pipeline.fit(X_train, y_train)
                X_train_transformed = pipeline.transform(X_train)
                X_test_transformed = pipeline.transform(X_test)
            else:
                X_train_transformed = X_train
                X_test_transformed = X_test
            model = factory.create(task, calibrate=calibrate_flag)
            model.fit(X_train_transformed, y_train)
            y_pred = model.predict(X_test_transformed)
            proba = None
            estimator = model.named_steps.get("estimator")
            if task == "classification" and model_supports_proba(model):
                try:
                    proba = model.predict_proba(X_test_transformed)
                except Exception as exc:  # pragma: no cover - defensive
                    LOGGER.debug("Failed to compute holdout probabilities: %s", exc)
            metrics = self._compute_evaluation_metrics(
                task,
                y_test,
                y_pred,
                X_test_transformed,
                X_test,
                proba,
                getattr(estimator, "classes_", None),
            )
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
                "shuffle": False,
            }
        elif strategy == "time_series":
            splitter = TimeSeriesSplit(n_splits=self.config.evaluation_folds)
            for fold, (train_idx, test_idx) in enumerate(splitter.split(features, target_series), start=1):
                if len(test_idx) == 0:
                    continue
                X_train, X_test = features.iloc[train_idx], features.iloc[test_idx]
                y_train, y_test = target_series.iloc[train_idx], target_series.iloc[test_idx]
                pipeline = clone(preprocessor) if preprocessor is not None else None
                if pipeline is not None:
                    pipeline.fit(X_train, y_train)
                    X_train_transformed = pipeline.transform(X_train)
                    X_test_transformed = pipeline.transform(X_test)
                else:
                    X_train_transformed = X_train
                    X_test_transformed = X_test
                model = factory.create(task, calibrate=calibrate_flag)
                model.fit(X_train_transformed, y_train)
                y_pred = model.predict(X_test_transformed)
                proba = None
                estimator = model.named_steps.get("estimator")
                if task == "classification" and model_supports_proba(model):
                    try:
                        proba = model.predict_proba(X_test_transformed)
                    except Exception as exc:  # pragma: no cover - defensive
                        LOGGER.debug("Failed to compute CV probabilities: %s", exc)
                metrics = self._compute_evaluation_metrics(
                    task,
                    y_test,
                    y_pred,
                    X_test_transformed,
                    X_test,
                    proba,
                    getattr(estimator, "classes_", None),
                )
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
            result = backtester.run(
                features,
                target_series,
                target_name,
                preprocessor_template=preprocessor,
            )
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
        raw_X_test: pd.DataFrame,
        proba: np.ndarray | None = None,
        classes: Sequence[Any] | None = None,
    ) -> Dict[str, Any]:
        if task == "classification":
            metrics: Dict[str, Any] = classification_metrics(y_true.to_numpy(), y_pred)
            metrics["directional_accuracy"] = metrics.get("accuracy", 0.0)
            if proba is not None:
                calibration_metrics = self._calibration_report(
                    y_true.to_numpy(), proba, classes
                )
                if calibration_metrics:
                    metrics.update(calibration_metrics)
            return metrics

        metrics = regression_metrics(y_true.to_numpy(), y_pred)
        try:
            metrics["r2"] = float(r2_score(y_true, y_pred))
        except ValueError:
            metrics["r2"] = float("nan")
        baseline = (
            raw_X_test["Close_Current"].to_numpy()
            if "Close_Current" in raw_X_test
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

    def _calibration_report(
        self,
        y_true: np.ndarray,
        proba: np.ndarray,
        classes: Sequence[Any] | None,
    ) -> Dict[str, Any]:
        try:
            if classes is not None:
                classes_array = np.asarray(classes)
                positive_index = None
                for idx, cls in enumerate(classes_array):
                    if cls in {1, True, "1", "up"}:
                        positive_index = idx
                        break
                if positive_index is None:
                    positive_index = int(np.argmax(classes_array))
            else:
                positive_index = 1 if proba.shape[1] > 1 else 0

            if proba.ndim == 1:
                positive_proba = proba
            else:
                positive_proba = proba[:, positive_index]

            if classes is not None:
                mapping = {cls: idx for idx, cls in enumerate(classes)}
                y_binary = np.array(
                    [
                        1
                        if mapping.get(value, value)
                        == mapping.get(classes[positive_index], positive_index)
                        else 0
                    ]
                    for value in y_true
                )
            else:
                y_binary = np.array([1 if value in {1, True, "1"} else 0 for value in y_true])

            brier = brier_score_loss(y_binary, positive_proba)
            frac_pos, mean_pred = calibration_curve(
                y_binary, positive_proba, n_bins=10, strategy="uniform"
            )
            ece = float(np.abs(frac_pos - mean_pred).mean())
            mce = float(np.abs(frac_pos - mean_pred).max())

            return {
                "brier_score": float(brier),
                "expected_calibration_error": ece,
                "max_calibration_error": mce,
                "calibration_curve": {
                    "fraction_positives": [float(x) for x in frac_pos],
                    "mean_predicted_value": [float(x) for x in mean_pred],
                },
            }
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.debug("Failed to compute calibration metrics: %s", exc)
            return {}

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

        needs_feature_refresh = refresh_data or not self.metadata
        price_df: Optional[pd.DataFrame] = None
        latest_price_timestamp: Optional[pd.Timestamp] = None

        try:
            price_df = self.fetcher.fetch_price_data()
        except Exception as exc:  # pragma: no cover - provider level failures are optional
            LOGGER.debug("Unable to fetch price data for staleness check: %s", exc)
        else:
            if not price_df.empty and "Date" in price_df.columns:
                candidate_dates = pd.to_datetime(price_df["Date"], errors="coerce").dropna()
                if not candidate_dates.empty:
                    latest_price_timestamp = candidate_dates.iloc[-1]

        metadata_latest_timestamp: Optional[pd.Timestamp] = None
        if self.metadata:
            raw_latest = self.metadata.get("latest_date")
            if raw_latest is not None:
                try:
                    metadata_latest_timestamp = pd.to_datetime(raw_latest)
                except (TypeError, ValueError):
                    metadata_latest_timestamp = None
                else:
                    if pd.isna(metadata_latest_timestamp):
                        metadata_latest_timestamp = None

        if latest_price_timestamp is not None:
            if metadata_latest_timestamp is None or latest_price_timestamp > metadata_latest_timestamp:
                if not needs_feature_refresh:
                    LOGGER.info(
                        "Detected new market data available through %s; rebuilding features.",
                        latest_price_timestamp,
                    )
                needs_feature_refresh = True

        if needs_feature_refresh:
            LOGGER.info("Preparing features before prediction.")
            if price_df is not None:
                self.prepare_features(price_df=price_df)
            else:
                self.prepare_features()

        latest_features = self.metadata.get("latest_features")
        if latest_features is None:
            raise RuntimeError("No feature metadata available. Train the model first.")
        raw_feature_columns = self.metadata.get("raw_feature_columns")
        if raw_feature_columns is None:
            raw_feature_columns = list(latest_features.columns)
            self.metadata["raw_feature_columns"] = raw_feature_columns
        self.metadata["active_horizon"] = resolved_horizon

        requested_targets = list(targets) if targets else list(self.config.prediction_targets)
        predictions: dict[str, Any] = {}
        confidences: dict[str, float] = {}
        probabilities: dict[str, Dict[str, float]] = {}
        uncertainties: dict[str, Dict[str, float]] = {}
        quantile_forecasts: dict[str, Dict[str, float]] = {}
        prediction_intervals: dict[str, Dict[str, float]] = {}
        prediction_warnings: List[str] = []
        training_report: dict[str, Any] = {}
        event_probabilities: dict[str, Dict[str, float]] = {}

        filtered_targets: list[str] = []
        for target in requested_targets:
            if target not in SUPPORTED_TARGETS:
                LOGGER.warning("Skipping target '%s': not supported.", target)
                continue
            if target not in filtered_targets:
                filtered_targets.append(target)

        for target in filtered_targets:
            model = self.models.get((target, resolved_horizon))
            if model is None:
                try:
                    model = self.load_model(target, resolved_horizon)
                except ModelNotFoundError:
                    LOGGER.warning(
                        "Model for target '%s' missing. Attempting on-demand training.",
                        target,
                    )
                    report = self.train_model(
                        targets=[target], horizon=resolved_horizon, force=True
                    )
                    target_metrics = report.get("targets", {}).get(target)
                    training_report[target] = {
                        "horizon": resolved_horizon,
                        **(target_metrics or {}),
                    }
                    model = self.models.get((target, resolved_horizon))
                    if model is None:
                        try:
                            model = self.load_model(target, resolved_horizon)
                        except ModelNotFoundError:
                            LOGGER.warning(
                                "Skipping target '%s' at horizon %s: not supported or training unavailable.",
                                target,
                                resolved_horizon,
                            )
                            continue

            pipeline = self.preprocessors.get((target, resolved_horizon))
            if pipeline is None:
                pipeline = self._load_preprocessor(target, resolved_horizon)
            current_raw = latest_features[raw_feature_columns]
            if pipeline is not None:
                transformed_features = pipeline.transform(current_raw)
                feature_names = get_feature_names_from_pipeline(pipeline)
                if feature_names:
                    self.metadata.setdefault("feature_columns_by_target", {})[(target, resolved_horizon)] = feature_names
                    self.metadata["feature_columns"] = feature_names
                else:
                    self.metadata["feature_columns"] = list(transformed_features.columns)
            else:
                transformed_features = current_raw
                self.metadata["feature_columns"] = list(transformed_features.columns)
            self.metadata["latest_transformed_features"] = transformed_features

            model_input = self._prepare_features_for_model(model, transformed_features)
            pred_value = model.predict(model_input)[0]
            predictions[target] = float(pred_value)

            uncertainty = self._estimate_prediction_uncertainty(
                target, model, transformed_features
            )
            if uncertainty:
                metrics_block = uncertainty.get("metrics") if isinstance(uncertainty, dict) else None
                quantiles_block = uncertainty.get("quantiles") if isinstance(uncertainty, dict) else None
                interval_block = uncertainty.get("interval") if isinstance(uncertainty, dict) else None
                if isinstance(metrics_block, dict) and metrics_block:
                    uncertainties[target] = metrics_block
                if isinstance(quantiles_block, dict) and quantiles_block:
                    quantile_forecasts[target] = {
                        str(key): float(value)
                        for key, value in quantiles_block.items()
                        if self._safe_float(value) is not None
                    }
                if isinstance(interval_block, dict) and interval_block:
                    prediction_intervals[target] = {
                        str(key): float(value)
                        for key, value in interval_block.items()
                        if self._safe_float(value) is not None
                    }

            if model_supports_proba(model) and target == "direction":
                proba = model.predict_proba(model_input)[0]
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
                        f"Direction model confidence {confidence_value:.3f} below threshold "
                        f"{threshold:.2f}. Consider tuning 'direction_confidence_threshold'."
                    )
                    LOGGER.warning(warning_msg)
                    prediction_warnings.append(warning_msg)

            event_threshold = self._event_threshold(target)
            event_prob = self._estimate_event_probability(
                model, model_input, threshold=event_threshold
            )
            if event_prob is not None:
                if target == "return":
                    label = f"return>{event_threshold:+.2%}"
                elif target == "volatility":
                    label = f"volatility>{event_threshold:.4f}"
                else:
                    label = f"{target}>{event_threshold}"
                event_probabilities.setdefault(target, {})[label] = float(event_prob)

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

        latest_timestamp: Optional[pd.Timestamp] = None
        if latest_date is not None:
            try:
                latest_timestamp = pd.to_datetime(latest_date)
            except (TypeError, ValueError):
                latest_timestamp = None
            else:
                if pd.isna(latest_timestamp):
                    latest_timestamp = None

        target_timestamp: Optional[pd.Timestamp] = None
        if target_date is not None:
            try:
                target_timestamp = pd.to_datetime(target_date)
            except (TypeError, ValueError):
                target_timestamp = None
            else:
                if pd.isna(target_timestamp):
                    target_timestamp = None

        if latest_timestamp is not None:
            if target_timestamp is None or target_timestamp <= latest_timestamp:
                try:
                    offset = pd.tseries.offsets.BDay(int(resolved_horizon))
                except Exception:  # pragma: no cover - guard invalid horizon types
                    offset = pd.Timedelta(days=int(resolved_horizon))
                target_timestamp = latest_timestamp + offset
                self.metadata.setdefault("target_dates", {})[resolved_horizon] = target_timestamp
            target_date = target_timestamp
        else:
            target_date = target_timestamp

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
        if event_probabilities:
            result["event_probabilities"] = event_probabilities
        if uncertainty_clean:
            result["prediction_uncertainty"] = uncertainty_clean
        if quantile_forecasts:
            result["quantile_forecasts"] = quantile_forecasts
        if prediction_intervals:
            result["prediction_intervals"] = prediction_intervals
        if training_report:
            result["training_metrics"] = training_report
        if prediction_warnings:
            result["warnings"] = prediction_warnings
        explanation = self._build_prediction_explanation(result, predictions)
        if explanation:
            result["explanation"] = explanation
        recommendation = self._generate_recommendation(result)
        if recommendation:
            result["recommendation"] = recommendation
        return result

    def _prepare_features_for_model(
        self, model: Any, features: pd.DataFrame | np.ndarray
    ) -> pd.DataFrame | np.ndarray:
        """Return feature matrix in a format compatible with *model*."""

        if not isinstance(features, pd.DataFrame):
            return features

        def _expects_named_features(candidate: Any) -> bool:
            return bool(candidate is not None and hasattr(candidate, "feature_names_in_"))

        estimator = None
        if hasattr(model, "named_steps") and isinstance(model.named_steps, Mapping):
            estimator = model.named_steps.get("estimator")

        if _expects_named_features(model) or _expects_named_features(estimator):
            return features

        return features.to_numpy()

    def _estimate_prediction_uncertainty(
        self,
        target: str,
        model: Any,
        features: pd.DataFrame,
    ) -> Optional[Dict[str, Any]]:
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

        metrics: Dict[str, float] = {}
        quantiles_payload: Optional[Dict[str, float]] = None
        interval_payload: Optional[Dict[str, float]] = None

        if hasattr(estimator, "get_uncertainty_summary"):
            try:
                summary = estimator.get_uncertainty_summary(transformed)
            except Exception:  # pragma: no cover - estimator specific quirks
                summary = {}
            if summary:
                quantiles = summary.pop("quantiles", None)
                interval = summary.pop("interval", None)
                metrics.update({
                    key: float(value)
                    for key, value in summary.items()
                    if self._safe_float(value) is not None
                })
                if isinstance(quantiles, dict):
                    quantiles_payload = {
                        str(key): float(self._safe_float(value) or 0.0)
                        for key, value in quantiles.items()
                    }
                if isinstance(interval, dict):
                    interval_payload = {
                        str(key): float(self._safe_float(value) or 0.0)
                        for key, value in interval.items()
                    }

        if not metrics:
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

            if samples:
                values = np.asarray(samples, dtype=float)
                if values.size and not np.isnan(values).all():
                    metrics["std"] = float(np.nanstd(values, ddof=1) if values.size > 1 else 0.0)
                    metrics["range"] = float(np.nanmax(values) - np.nanmin(values))

        if hasattr(estimator, "predict_quantiles") and quantiles_payload is None:
            try:
                quantile_values = estimator.predict_quantiles(transformed)
            except Exception:  # pragma: no cover - estimator quirks
                quantile_values = {}
            if isinstance(quantile_values, dict):
                quantiles_payload = {
                    str(key): float(np.nanmean(value))
                    for key, value in quantile_values.items()
                    if isinstance(value, (list, tuple, np.ndarray)) and len(value) > 0
                }

        if hasattr(estimator, "prediction_interval") and interval_payload is None:
            try:
                interval_values = estimator.prediction_interval(transformed)
            except Exception:  # pragma: no cover - estimator quirks
                interval_values = {}
            if isinstance(interval_values, dict):
                interval_payload = {
                    str(key): float(np.nanmean(value))
                    for key, value in interval_values.items()
                    if isinstance(value, (list, tuple, np.ndarray)) and len(value) > 0
                }

        if not metrics and quantiles_payload is None and interval_payload is None:
            return None

        result: Dict[str, Any] = {}
        if metrics:
            result["metrics"] = metrics
        if quantiles_payload:
            result["quantiles"] = quantiles_payload
        if interval_payload:
            result["interval"] = interval_payload
        return result or None

    def _estimate_event_probability(
        self,
        model: Pipeline,
        features: pd.DataFrame | np.ndarray,
        *,
        threshold: float,
    ) -> float | None:
        estimator = model.named_steps.get("estimator")
        if estimator is None:
            return None
        try:
            if hasattr(estimator, "estimators_"):
                if isinstance(features, pd.DataFrame):
                    feature_matrix = features.to_numpy()
                else:
                    feature_matrix = np.asarray(features)
                member_preds = np.vstack(
                    [tree.predict(feature_matrix) for tree in estimator.estimators_]
                )
                if member_preds.size == 0:
                    return None
                return float(np.mean(member_preds[:, 0] > threshold))
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.debug("Failed to estimate event probability: %s", exc)
            return None
        return None

    def _event_threshold(self, target: str) -> float:
        params = self.config.model_params.get(target, {})
        if isinstance(params, Mapping) and "event_threshold" in params:
            try:
                return float(params["event_threshold"])
            except (TypeError, ValueError):
                LOGGER.debug(
                    "Ignoring invalid event_threshold for %s: %s",
                    target,
                    params.get("event_threshold"),
                )
        if target == "return":
            return 0.0
        if target == "volatility":
            latest_vol = self.metadata.get("latest_realised_volatility")
            if latest_vol is not None:
                return float(latest_vol)
        return 0.0

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
        sources = self._render_source_descriptions(sources_raw)
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

    def _generate_recommendation(self, prediction: Dict[str, Any]) -> Dict[str, Any]:
        expected_return = self._safe_float(prediction.get("predicted_return"))
        if expected_return is None:
            expected_return = self._safe_float(prediction.get("expected_change_pct"))
        threshold = max(0.0, float(getattr(self.config, "backtest_neutral_threshold", 0.001)))
        action = "hold"
        if expected_return is not None:
            if expected_return > threshold:
                action = "long"
            elif expected_return < -threshold:
                action = "short"

        probabilities = prediction.get("probabilities")
        direction_probs = probabilities.get("direction") if isinstance(probabilities, dict) else None
        prob_score = None
        if isinstance(direction_probs, dict):
            candidates = [self._safe_float(direction_probs.get(key)) for key in ("up", "down")]
            candidates = [value for value in candidates if value is not None]
            if candidates:
                prob_score = max(candidates)

        uncertainty_block = prediction.get("prediction_uncertainty")
        uncertainty_metrics = None
        if isinstance(uncertainty_block, dict):
            if "close" in uncertainty_block:
                uncertainty_metrics = uncertainty_block.get("close")
            else:
                uncertainty_metrics = next(iter(uncertainty_block.values()), None)
        std_uncertainty = None
        if isinstance(uncertainty_metrics, dict):
            std_uncertainty = self._safe_float(
                uncertainty_metrics.get("std")
                or uncertainty_metrics.get("median_std")
                or uncertainty_metrics.get("range")
            )

        volatility = self._safe_float(prediction.get("predicted_volatility"))
        quantiles = None
        if isinstance(prediction.get("quantile_forecasts"), dict):
            quantiles = prediction["quantile_forecasts"].get("close")
        interval = None
        if isinstance(prediction.get("prediction_intervals"), dict):
            interval = prediction["prediction_intervals"].get("close")

        confidence = 0.5
        if prob_score is not None:
            confidence = prob_score
        elif expected_return is not None:
            confidence = min(0.95, 0.5 + min(0.4, abs(expected_return)))
        if std_uncertainty is not None and std_uncertainty > 0:
            confidence *= max(0.25, 1.0 / (1.0 + std_uncertainty))
        confidence = float(np.clip(confidence, 0.0, 1.0))

        allocation = 0.0
        if expected_return is not None:
            if std_uncertainty is not None and std_uncertainty > 0:
                allocation = float(np.clip(abs(expected_return) / (std_uncertainty * 2.0), 0.0, 1.0))
            else:
                allocation = float(np.clip(abs(expected_return), 0.0, 1.0))
        if action == "hold":
            allocation = 0.0

        key_drivers: List[str] = []
        explanation = prediction.get("explanation") or {}
        if isinstance(explanation, dict):
            top_drivers = explanation.get("top_feature_drivers") or {}
            if isinstance(top_drivers, dict):
                for category, names in top_drivers.items():
                    if not names:
                        continue
                    label = str(category).replace("_", " ").title()
                    joined = ", ".join(names)
                    key_drivers.append(f"{label}: {joined}")

        risk_guidance: Dict[str, Any] = {"suggested_allocation": allocation}
        if volatility is not None:
            risk_guidance["volatility"] = volatility
        if std_uncertainty is not None:
            risk_guidance["uncertainty_std"] = std_uncertainty
        if interval:
            risk_guidance["interval"] = interval
        if quantiles:
            risk_guidance["quantiles"] = quantiles

        expected_pct_value = (
            float(expected_return * 100) if expected_return is not None else None
        )

        return {
            "action": action,
            "confidence": confidence,
            "expected_return_pct": expected_pct_value,
            "key_drivers": key_drivers,
            "risk_guidance": risk_guidance,
        }

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
        pct_change_cols = [
            column
            for column in feature_row.index
            if column.startswith("Fundamental_") and column.endswith("_PctChange_63")
        ]
        for column in sorted(pct_change_cols):
            change = self._safe_float(feature_row.get(column))
            if change is None or abs(change) < 0.05:
                continue
            metric_label = self._format_fundamental_label(column, suffix="_PctChange_63")
            if change > 0:
                reasons.append(
                    f"{metric_label} improved {change * 100:.1f}% versus the prior quarter, signalling healthier fundamentals."
                )
            else:
                reasons.append(
                    f"{metric_label} contracted {abs(change) * 100:.1f}% versus the prior quarter, pointing to emerging pressure."
                )
            if len(reasons) >= 3:
                break

        if len(reasons) < 3:
            zscore_cols = [
                column
                for column in feature_row.index
                if column.startswith("Fundamental_") and column.endswith("_ZScore_252")
            ]
            for column in sorted(zscore_cols):
                zscore = self._safe_float(feature_row.get(column))
                if zscore is None or abs(zscore) < 1.0:
                    continue
                metric_label = self._format_fundamental_label(column, suffix="_ZScore_252")
                if zscore > 0:
                    reasons.append(
                        f"{metric_label} sits {zscore:.1f} standard deviations above its multi-year average, highlighting strength."
                    )
                else:
                    reasons.append(
                        f"{metric_label} sits {abs(zscore):.1f} standard deviations below its multi-year average, highlighting weakness."
                    )
                if len(reasons) >= 3:
                    break

        if not reasons:
            latest_cols = [
                column
                for column in feature_row.index
                if column.startswith("Fundamental_") and column.endswith("_Latest")
            ]
            for column in sorted(latest_cols):
                value = self._safe_float(feature_row.get(column))
                if value is None:
                    continue
                metric_label = self._format_fundamental_label(column, suffix="_Latest")
                reasons.append(f"{metric_label} latest reading registered at {value:.2f}.")
                if len(reasons) >= 2:
                    break

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

        quantiles_raw = prediction.get("quantile_forecasts")
        if isinstance(quantiles_raw, dict):
            quantile_block = quantiles_raw.get("close") or next(
                (values for values in quantiles_raw.values() if isinstance(values, dict)),
                None,
            )
            if isinstance(quantile_block, dict):
                filtered = {
                    str(key): float(value)
                    for key, value in quantile_block.items()
                    if self._safe_float(value) is not None
                }
                if filtered:
                    indicators["quantiles"] = filtered

        intervals_raw = prediction.get("prediction_intervals")
        if isinstance(intervals_raw, dict):
            interval_block = intervals_raw.get("close") or next(
                (values for values in intervals_raw.values() if isinstance(values, dict)),
                None,
            )
            if isinstance(interval_block, dict):
                filtered_interval = {
                    str(key): float(value)
                    for key, value in interval_block.items()
                    if self._safe_float(value) is not None
                }
                if filtered_interval:
                    indicators["interval"] = filtered_interval

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
    def _render_source_descriptions(sources: Any) -> list[str]:
        if sources is None:
            return []
        if isinstance(sources, Mapping):
            iterable = sources.values()
        elif isinstance(sources, Iterable) and not isinstance(sources, (str, bytes)):
            iterable = sources
        else:
            iterable = [sources]

        descriptions: list[str] = []
        seen: set[str] = set()
        for entry in iterable:
            label: Any = None
            provider_id: Any = None
            if isinstance(entry, Mapping):
                provider_id = entry.get("id") or entry.get("provider") or entry.get("provider_id")
                label = entry.get("description") or entry.get("label")
            else:
                provider_id = getattr(entry, "provider_id", None) or getattr(entry, "id", None)
                label = getattr(entry, "description", None)
            if label is None and isinstance(entry, str):
                label = entry
            if label is None and provider_id is not None:
                label = StockPredictorAI._humanize_source_id(provider_id)
            if label is None and entry is not None:
                label = str(entry)
            label_str = str(label).strip() if label is not None else ""
            if not label_str or label_str in seen:
                continue
            seen.add(label_str)
            descriptions.append(label_str)
        return descriptions

    @staticmethod
    def _humanize_source_id(provider_id: Any) -> str:
        text = str(provider_id or "").replace("_", " ").strip()
        if not text:
            return "Unknown source"
        parts = [part for part in text.split(" ") if part]
        if not parts:
            return "Unknown source"
        return " ".join(part.capitalize() for part in parts)

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
    def _format_fundamental_label(column: str, *, suffix: str) -> str:
        base = column[len("Fundamental_") :]
        if base.endswith(suffix):
            base = base[: -len(suffix)]
        label = base.replace("_", " ").strip()
        if not label:
            return "Fundamental metric"
        return " ".join(part.capitalize() for part in label.split())

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
        pipeline = self.preprocessors.get((target, resolved_horizon))
        if pipeline is None:
            pipeline = self._load_preprocessor(target, resolved_horizon)
        feature_columns: Optional[List[str]] = None
        if pipeline is not None:
            feature_columns = get_feature_names_from_pipeline(pipeline)
        if not feature_columns:
            mapped = self.metadata.get("feature_columns_by_target", {})
            if isinstance(mapped, dict):
                feature_columns = mapped.get((target, resolved_horizon))
        if not feature_columns:
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
        X, targets_by_horizon, _ = self.prepare_features()
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

            auxiliary_columns: Dict[str, pd.Series] = {}
            for aux_name, series in horizon_targets.items():
                if aux_name == target:
                    continue
                auxiliary_columns[aux_name] = series.reindex(aligned_X.index)
            auxiliary_df = None
            if auxiliary_columns:
                auxiliary_df = pd.DataFrame(auxiliary_columns, index=aligned_X.index).fillna(0.0)

            backtester = Backtester(
                model_factory=factory,
                strategy=self.config.backtest_strategy,
                window=self.config.backtest_window,
                step=self.config.backtest_step,
                slippage_bps=self.config.backtest_slippage_bps,
                fee_bps=self.config.backtest_fee_bps,
                neutral_threshold=self.config.backtest_neutral_threshold,
                risk_free_rate=self.config.risk_free_rate,
            )
            template = self.preprocessor_templates.get(resolved_horizon)
            result = backtester.run(
                aligned_X,
                y_clean,
                target,
                preprocessor_template=template,
                auxiliary_targets=auxiliary_df,
            )
            results[target] = {
                "aggregate": result.aggregate,
                "splits": result.splits,
                "feature_importance": result.feature_importance,
            }
            self.tracker.log_run(
                target=target,
                run_type="backtest",
                parameters={
                    "model_type": self.config.model_type,
                    "strategy": self.config.backtest_strategy,
                    "window": self.config.backtest_window,
                    "step": self.config.backtest_step,
                    "slippage_bps": self.config.backtest_slippage_bps,
                    "fee_bps": self.config.backtest_fee_bps,
                },
                metrics=result.aggregate,
                context={
                    "splits": result.splits,
                    "horizon": resolved_horizon,
                    "feature_importance": result.feature_importance,
                },
            )
        self.metadata["active_horizon"] = resolved_horizon
        return results

