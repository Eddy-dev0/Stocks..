"""Modern multi-horizon modelling backend.

This module rebuilds the training and inference pipeline so we can
construct large supervised datasets across multiple horizons and
persist per-horizon artefacts.  It relies on the feature assembler and
providers already available in the project but enforces clearer
structure around preprocessing and model persistence.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from ..config import DEFAULT_MIN_SAMPLES_PER_HORIZON, PredictorConfig, interval_is_intraday
from ..features import FeatureAssembler
from ..training_data import TrainingDatasetBuilder
from ..indicator_bundle import evaluate_signal_confluence
from ..sentiment import aggregate_daily_sentiment
from ..ml_preprocessing import get_feature_names_from_pipeline
from ..database import Database
from .exceptions import InsufficientSamplesError
from .prediction_result import PredictionOutcome

LOGGER = logging.getLogger(__name__)


@dataclass
class HorizonArtifacts:
    """Persisted artefacts for a single horizon."""

    horizon: int
    preprocessor: Pipeline
    models: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    sample_counts: dict[str, int] = field(default_factory=dict)


@dataclass
class MultiHorizonDataset:
    """Container holding features and multi-target labels."""

    features: pd.DataFrame
    targets: dict[int, dict[str, pd.Series]]
    metadata: dict[str, Any]

    def count_targets(self) -> dict[int, dict[str, int]]:
        counts: dict[int, dict[str, int]] = {}
        for horizon, label_map in self.targets.items():
            counts[horizon] = {name: int(series.dropna().shape[0]) for name, series in label_map.items()}
        return counts


class MultiHorizonModelingEngine:
    """New training and inference orchestrator."""

    def __init__(
        self,
        config: PredictorConfig,
        *,
        fetcher: Any,
        feature_assembler: FeatureAssembler,
    ) -> None:
        self.config = config
        self.fetcher = fetcher
        self.feature_assembler = feature_assembler
        self.database: Database = getattr(fetcher, "database", Database(config.database_url))
        self.dataset_builder = TrainingDatasetBuilder(config, database=self.database)
        self.models_dir = Path(config.models_dir) / "nextgen"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self._untrainable_until: dict[int, dict[str, Any]] = {}
        self._insufficient_log_state: dict[int, pd.Timestamp | None] = {}

    # ------------------------------------------------------------------
    # Dataset construction
    # ------------------------------------------------------------------
    def _interval_is_intraday(self) -> bool:
        return interval_is_intraday(self.config.interval)

    @staticmethod
    def _aggregate_daily_prices(price_df: pd.DataFrame) -> pd.DataFrame:
        frame = price_df.copy()
        if "Date" in frame.columns:
            frame = frame.sort_values("Date")
            index = pd.to_datetime(frame["Date"], errors="coerce")
            frame = frame.set_index(index)
        else:
            frame = frame.sort_index()
            frame.index = pd.to_datetime(frame.index, errors="coerce")
        frame = frame[~frame.index.isna()]
        if frame.empty:
            return frame

        lower_columns = {column.lower(): column for column in frame.columns}
        agg_map: dict[str, str] = {}
        for name, agg in (
            ("open", "first"),
            ("high", "max"),
            ("low", "min"),
            ("close", "last"),
            ("adj close", "last"),
            ("adj_close", "last"),
            ("volume", "sum"),
        ):
            column = lower_columns.get(name)
            if column:
                agg_map[column] = agg

        if not agg_map:
            return frame

        daily = frame.resample("1D").agg(agg_map)
        daily = daily.dropna(how="all")
        daily.index = pd.to_datetime(daily.index).normalize()
        return daily

    @staticmethod
    def _align_daily_targets(series: pd.Series, target_index: pd.Index) -> pd.Series:
        if target_index.empty:
            return series
        normalized = pd.to_datetime(target_index, errors="coerce").normalize()
        aligned = series.reindex(normalized)
        aligned.index = target_index
        return aligned

    def _compute_targets(self, price_df: pd.DataFrame, horizons: Sequence[int]) -> dict[int, dict[str, pd.Series]]:
        working = price_df.copy()
        use_daily_targets = self._interval_is_intraday()
        target_frame = self._aggregate_daily_prices(working) if use_daily_targets else working
        lower_columns = {column.lower(): column for column in target_frame.columns}
        basis = getattr(self.config, "target_price_basis", "adj_close")
        basis = str(basis or "adj_close").strip().lower()
        if basis == "adj_close":
            close_col = (
                lower_columns.get("adj close")
                or lower_columns.get("adj_close")
                or lower_columns.get("close")
            )
        else:
            close_col = (
                lower_columns.get("close")
                or lower_columns.get("adj close")
                or lower_columns.get("adj_close")
            )
        if close_col is None:
            raise KeyError("Price dataframe must contain a close column for target generation.")

        closes = pd.to_numeric(target_frame[close_col], errors="coerce")
        neutral_threshold = float(getattr(self.config, "direction_neutral_threshold", 0.0))
        targets: dict[int, dict[str, pd.Series]] = {}
        for horizon in horizons:
            future_close = closes.shift(-int(horizon))
            returns = (future_close - closes) / closes
            if neutral_threshold > 0:
                direction = np.where(
                    returns > neutral_threshold,
                    1,
                    np.where(returns < -neutral_threshold, -1, 0),
                )
            else:
                direction = np.where(future_close > closes, 1, -1)
            direction_series = pd.Series(direction, index=target_frame.index, dtype=float)
            if use_daily_targets:
                targets[int(horizon)] = {
                    "close_h": self._align_daily_targets(future_close, working.index),
                    "direction_h": self._align_daily_targets(direction_series, working.index),
                    "return_h": self._align_daily_targets(returns, working.index),
                }
            else:
                targets[int(horizon)] = {
                    "close_h": future_close,
                    "direction_h": direction_series,
                    "return_h": returns,
                }
        return targets

    def build_dataset(
        self,
        *,
        horizons: Sequence[int] | None = None,
        tickers: Iterable[str] | None = None,
        targets: Iterable[str] | None = None,
        force: bool = False,
    ) -> MultiHorizonDataset:
        horizons = tuple(horizons) if horizons is not None else tuple(self.config.prediction_horizons)
        ticker_universe = list(tickers) if tickers else [self.config.ticker]
        requested_targets = set(targets) if targets else {"close_h", "direction_h", "return_h"}
        minimum_viable_rows = max(1, int(getattr(self.config, "minimum_viable_rows", 1)))
        if self._interval_is_intraday() and 1 in horizons:
            LOGGER.warning(
                "Interval %s is intraday; treating 'Tomorrow' as 1 trading day and using daily bars for targets.",
                self.config.interval,
            )

        def _frame_stats(frame: pd.DataFrame) -> dict[str, Any]:
            if frame is None:
                return {"row_count": 0, "date_min": None, "date_max": None, "columns": []}
            date_series = pd.Series(dtype="datetime64[ns]")
            if "Date" in frame.columns:
                date_series = pd.to_datetime(frame["Date"], errors="coerce")
            elif isinstance(frame.index, pd.DatetimeIndex):
                date_series = pd.to_datetime(frame.index, errors="coerce")
            return {
                "row_count": int(frame.shape[0]),
                "date_min": date_series.min(),
                "date_max": date_series.max(),
                "columns": sorted(list(frame.columns)),
            }

        def _log_stage(stage: str, frame: pd.DataFrame, *, ticker: str | None = None, extra: Mapping[str, Any] | None = None) -> None:
            payload: dict[str, Any] = {"stage": stage, "ticker": ticker}
            payload.update(_frame_stats(frame))
            if extra:
                payload.update(dict(extra))
            LOGGER.info("dataset_build_stage", extra={"dataset_stage": payload})

        def _raise_stage_error(stage: str, reason: str, *, details: Mapping[str, Any] | None = None) -> None:
            detail_payload = {"stage": stage, "reason": reason}
            if details:
                detail_payload.update(dict(details))
            message = (
                f"Dataset build failed after {stage}: {reason}. "
                f"Details: {json.dumps(detail_payload, default=str)}"
            )
            LOGGER.error(message)
            raise InsufficientSamplesError(
                message,
                horizons=horizons,
                targets=tuple(sorted(requested_targets)),
                sample_counts={},
                missing_targets={},
            )

        feature_frames: list[pd.DataFrame] = []
        targets: dict[int, dict[str, list[pd.Series]]] = {h: {"close_h": [], "direction_h": [], "return_h": []} for h in horizons}
        metadata: dict[str, Any] = {
            "horizons": horizons,
            "tickers": ticker_universe,
        }

        sentiment_enabled = bool(self.config.sentiment and self.config.feature_toggles.sentiment)
        for ticker in ticker_universe:
            price_df = self.database.get_prices(
                ticker=ticker,
                interval=self.config.interval,
                start=self.config.start_date,
                end=self.config.end_date,
            )
            if price_df.empty:
                LOGGER.warning("No price data available for %s; skipping", ticker)
                continue

            _log_stage("price_fetch", price_df, ticker=ticker)
            if price_df.shape[0] < minimum_viable_rows:
                _raise_stage_error(
                    "price_fetch",
                    "insufficient rows returned from price fetch",
                    details={"ticker": ticker, "row_count": price_df.shape[0], "minimum_viable_rows": minimum_viable_rows},
                )

            canonical_price_df = price_df.copy()
            if "Date" in canonical_price_df.columns:
                canonical_price_df = canonical_price_df.sort_values("Date").drop_duplicates("Date", keep="last")
                canonical_index = pd.to_datetime(canonical_price_df["Date"])
            else:
                canonical_price_df = canonical_price_df.sort_index()
                canonical_index = pd.to_datetime(canonical_price_df.index)
            canonical_price_df.index = canonical_index

            _log_stage(
                "canonicalized_prices",
                canonical_price_df,
                ticker=ticker,
                extra={
                    "deduplicated_rows": int(price_df.shape[0] - canonical_price_df.shape[0]),
                },
            )
            if canonical_price_df.shape[0] < minimum_viable_rows:
                _raise_stage_error(
                    "canonicalized_prices",
                    "insufficient rows after sorting/deduplication",
                    details={
                        "ticker": ticker,
                        "row_count": canonical_price_df.shape[0],
                        "deduplicated": int(price_df.shape[0] - canonical_price_df.shape[0]),
                        "minimum_viable_rows": minimum_viable_rows,
                    },
                )

            news_df = self.database.get_news(ticker) if sentiment_enabled else pd.DataFrame()
            macro_df = pd.DataFrame()
            if self.config.feature_toggles.macro:
                macro_df = self.database.get_indicators(
                    ticker=ticker, interval=self.config.interval, category="macro"
                )
            feature_result = self.feature_assembler.build(
                canonical_price_df,
                news_df,
                sentiment_enabled,
                macro_df=macro_df if not macro_df.empty else None,
            )
            features = feature_result.features.copy()
            features["ticker"] = ticker
            features.index = canonical_index
            feature_frames.append(features)

            _log_stage(
                "feature_merge",
                features,
                ticker=ticker,
                extra={
                    "dropped_rows": int(canonical_price_df.shape[0] - features.shape[0]),
                    "source_rows": int(canonical_price_df.shape[0]),
                },
            )
            if features.shape[0] < minimum_viable_rows:
                _raise_stage_error(
                    "feature_merge",
                    "feature assembly produced insufficient rows (likely warmup trimming)",
                    details={
                        "ticker": ticker,
                        "row_count": features.shape[0],
                        "dropped_rows": int(canonical_price_df.shape[0] - features.shape[0]),
                        "minimum_viable_rows": minimum_viable_rows,
                    },
                )

            ticker_targets = self._compute_targets(canonical_price_df, horizons)
            for horizon, label_map in ticker_targets.items():
                for name, series in label_map.items():
                    aligned = series.reindex(canonical_index)
                    targets[horizon][name].append(aligned)

            target_counts = {
                int(h): {name: int(series.dropna().shape[0]) for name, series in label_map.items()}
                for h, label_map in ticker_targets.items()
            }
            _log_stage(
                "target_generation",
                canonical_price_df,
                ticker=ticker,
                extra={"target_counts": target_counts},
            )
            if any(count == 0 for horizon_counts in target_counts.values() for count in horizon_counts.values()):
                _raise_stage_error(
                    "target_generation",
                    "target generation produced empty columns",
                    details={"ticker": ticker, "target_counts": target_counts},
                )

            if news_df is not None and not news_df.empty:
                metadata.setdefault("sentiment_daily", {})[ticker] = aggregate_daily_sentiment(news_df)
            try:
                metadata.setdefault("signal_confluence", {})[ticker] = evaluate_signal_confluence(price_df)
            except Exception:  # pragma: no cover - defensive
                metadata.setdefault("signal_confluence", {})[ticker] = {}

        if not feature_frames:
            raise InsufficientSamplesError(
                "Unable to build dataset: no price history available for the requested tickers.",
                horizons=horizons,
                targets=tuple(sorted(requested_targets)),
                sample_counts={h: {t: 0 for t in sorted(requested_targets)} for h in horizons},
                missing_targets={h: {t: 0 for t in sorted(requested_targets)} for h in horizons},
            )

        features_df = pd.concat(feature_frames, axis=0).sort_index()
        _log_stage("feature_concat", features_df, extra={"tickers": ticker_universe})
        target_map: dict[int, dict[str, pd.Series]] = {}
        for horizon, label_map in targets.items():
            target_map[horizon] = {
                name: pd.concat(series_list, axis=0).sort_index() if series_list else pd.Series(dtype=float)
                for name, series_list in label_map.items()
            }

        # Align all targets to the available feature dates and drop rows where no target exists.
        alignment_candidates: dict[str, pd.Series] = {}
        for horizon_value, label_map in target_map.items():
            for name, series in label_map.items():
                if name not in requested_targets:
                    continue
                alignment_candidates[f"{horizon_value}:{name}"] = series

        if alignment_candidates:
            concatenated_targets = pd.DataFrame(alignment_candidates).reindex(features_df.index)
            presence_mask = concatenated_targets.notna().any(axis=1)
            aligned_features_df = features_df.loc[presence_mask]
            drop_count = int(features_df.shape[0] - aligned_features_df.shape[0])
            for horizon_value, label_map in target_map.items():
                for name, series in label_map.items():
                    target_map[horizon_value][name] = series.reindex(aligned_features_df.index)

            _log_stage(
                "final_alignment",
                aligned_features_df,
                extra={
                    "dropped_for_alignment": drop_count,
                    "initial_rows": int(features_df.shape[0]),
                    "target_columns": sorted(list(alignment_candidates.keys())),
                },
            )
            if aligned_features_df.shape[0] < minimum_viable_rows:
                _raise_stage_error(
                    "final_alignment",
                    "insufficient rows after aligning targets to features",
                    details={
                        "row_count": aligned_features_df.shape[0],
                        "dropped_for_alignment": drop_count,
                        "minimum_viable_rows": minimum_viable_rows,
                    },
                )
            features_df = aligned_features_df

        horizon_one_targets = target_map.get(1, {})
        horizon_one_counts = {name: int(series.dropna().shape[0]) for name, series in horizon_one_targets.items() if name in requested_targets}
        if any(count == 0 for count in horizon_one_counts.values()) or not horizon_one_counts:
            _raise_stage_error(
                "target_alignment",
                "horizon 1 targets are empty after alignment",
                details={"horizon_one_counts": horizon_one_counts},
            )

        dataset = MultiHorizonDataset(features=features_df, targets=target_map, metadata=metadata)
        counts = dataset.count_targets()
        metadata["target_counts"] = counts
        if 1 in counts and len(features_df) > 1:
            missing_alignment = [
                name for name in requested_targets if counts[1].get(name, 0) == 0
            ]
            if missing_alignment:
                message = (
                    "Target alignment failed for horizon 1: no aligned values for "
                    f"{sorted(missing_alignment)}"
                )
                LOGGER.error(message)
                raise ValueError(message)
        insufficient: dict[int, dict[str, int]] = {}
        threshold = int(
            getattr(
                self.config, "min_samples_per_horizon", DEFAULT_MIN_SAMPLES_PER_HORIZON
            )
        )
        for horizon, horizon_counts in counts.items():
            below_threshold = {
                name: int(count)
                for name, count in horizon_counts.items()
                if name in requested_targets and count < threshold
            }
            if below_threshold:
                insufficient[horizon] = below_threshold
        if insufficient:
            latest_ts = None
            if not dataset.features.empty:
                latest_ts = pd.to_datetime(dataset.features.index.max())
            for horizon, missing_targets in insufficient.items():
                previous_ts = self._insufficient_log_state.get(int(horizon))
                repeated = previous_ts == latest_ts
                log_fn = LOGGER.warning if not repeated else LOGGER.debug
                log_fn(
                    "Targets below min_samples_per_horizon=%s for horizon %s: %s",
                    threshold,
                    horizon,
                    json.dumps(missing_targets),
                )
                self._insufficient_log_state[int(horizon)] = latest_ts
            metadata["insufficient_samples"] = insufficient
        metadata["min_samples_per_horizon"] = threshold
        return dataset

    # ------------------------------------------------------------------
    # Training and persistence
    # ------------------------------------------------------------------
    def _build_preprocessor(self, features: pd.DataFrame) -> Pipeline:
        numeric_cols = [col for col in features.columns if pd.api.types.is_numeric_dtype(features[col])]
        categorical_cols = [col for col in features.columns if col not in numeric_cols]
        transformers = []
        if numeric_cols:
            transformers.append(("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_cols))
        if categorical_cols:
            transformers.append(("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical_cols))
        preprocessor = ColumnTransformer(transformers)
        return Pipeline([("pre", preprocessor)])

    def _train_single_model(self, X_train: pd.DataFrame, y_train: pd.Series, task: str, random_state: int) -> Any:
        if task == "classification":
            model = HistGradientBoostingClassifier(random_state=random_state)
        else:
            model = HistGradientBoostingRegressor(random_state=random_state)
        model.fit(X_train, y_train)
        return model

    def train(
        self,
        *,
        targets: Iterable[str] | None = None,
        horizon: int | None = None,
        force: bool = False,
    ) -> dict[str, Any]:
        horizons = (int(horizon),) if horizon else tuple(self.config.prediction_horizons)
        requested_targets = set(targets) if targets else {"close_h", "direction_h", "return_h"}
        dataset = self.build_dataset(horizons=horizons, targets=requested_targets)

        sample_counts = dataset.metadata.get("target_counts") or dataset.count_targets()
        requested_counts = {
            int(h): {t: int(sample_counts.get(int(h), {}).get(t, 0)) for t in sorted(requested_targets)}
            for h in horizons
        }
        min_samples = int(
            getattr(
                self.config, "min_samples_per_horizon", DEFAULT_MIN_SAMPLES_PER_HORIZON
            )
        )
        missing_targets = {}
        for horizon_value, horizon_counts in requested_counts.items():
            missing = {t: count for t, count in horizon_counts.items() if count < min_samples}
            if missing:
                missing_targets[int(horizon_value)] = missing

        artefacts: dict[int, HorizonArtifacts] = {}
        random_state = int(self.config.model_params.get("global", {}).get("random_state", 42))

        # Train per horizon
        for horizon_value in horizons:
            horizon_targets = dataset.targets.get(int(horizon_value), {})
            horizon_counts = requested_counts.get(int(horizon_value), {})
            if not any(count >= min_samples for count in horizon_counts.values()):
                raise InsufficientSamplesError(
                    f"Horizon {horizon_value} does not meet minimum sample requirements.",
                    horizons=horizons,
                    targets=tuple(sorted(requested_targets)),
                    sample_counts=requested_counts,
                    missing_targets=missing_targets,
                )

            below_threshold = {
                target: count
                for target, count in horizon_counts.items()
                if 0 < count < min_samples
            }
            for target_name, count in below_threshold.items():
                LOGGER.warning(
                    "Target %s for horizon %s has %s samples, below min_samples_per_horizon=%s; skipping training for this target.",
                    target_name,
                    horizon_value,
                    count,
                    min_samples,
                )

            aligned_features = dataset.features.copy()
            aligned_features = aligned_features.loc[horizon_targets.get("close_h", pd.Series(index=aligned_features.index)).index]
            aligned_features = aligned_features.dropna(axis=1, how="all")
            preprocessor = self._build_preprocessor(aligned_features)

            target_metrics: dict[str, Any] = {}
            target_sample_counts: dict[str, int] = {}
            trained_models: dict[str, Any] = {}
            fitted_pre: Pipeline | None = None

            for target_name, series in horizon_targets.items():
                if target_name not in requested_targets:
                    continue
                target_count = horizon_counts.get(target_name, 0)
                if target_count < min_samples:
                    continue
                y = series.dropna()
                if y.empty:
                    continue
                X = aligned_features.loc[y.index]
                # chronological split
                X_train, X_temp, y_train, y_temp = train_test_split(
                    X, y, test_size=0.3, shuffle=False
                )
                X_val, X_test, y_val, y_test = train_test_split(
                    X_temp, y_temp, test_size=0.5, shuffle=False
                )
                fitted_pre = preprocessor.fit(X_train)
                X_train_t = fitted_pre.transform(X_train)
                X_val_t = fitted_pre.transform(X_val)
                X_test_t = fitted_pre.transform(X_test)

                task = "classification" if "direction" in target_name else "regression"
                model = self._train_single_model(X_train_t, y_train, task, random_state)
                y_hat_val = model.predict(X_val_t)
                y_hat_test = model.predict(X_test_t)

                if task == "classification":
                    metric_value = {
                        "val_accuracy": float(accuracy_score(y_val, y_hat_val)),
                        "test_accuracy": float(accuracy_score(y_test, y_hat_test)),
                    }
                else:
                    mse_val = mean_squared_error(y_val, y_hat_val)
                    mse_test = mean_squared_error(y_test, y_hat_test)
                    metric_value = {
                        "val_mae": float(mean_absolute_error(y_val, y_hat_val)),
                        "val_rmse": float(math.sqrt(mse_val)),
                        "test_mae": float(mean_absolute_error(y_test, y_hat_test)),
                        "test_rmse": float(math.sqrt(mse_test)),
                    }

                target_metrics[target_name] = metric_value
                target_sample_counts[target_name] = int(len(y_train))
                trained_models[target_name] = model

            if fitted_pre is None or not trained_models:
                LOGGER.warning(
                    "Skipping persistence for horizon %s; no models were trained (insufficient samples or targets).",
                    horizon_value,
                )
                continue

            artefacts[horizon_value] = HorizonArtifacts(
                horizon=horizon_value,
                preprocessor=fitted_pre,
                models=trained_models,
                metrics=target_metrics,
                sample_counts=target_sample_counts,
            )

            # Persist artefacts
            horizon_dir = self.models_dir / f"horizon_{horizon_value}"
            horizon_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(fitted_pre, horizon_dir / "preprocessor.joblib")
            with open(horizon_dir / "feature_names.json", "w", encoding="utf-8") as handle:
                json.dump(get_feature_names_from_pipeline(fitted_pre), handle, indent=2)
            for target_name, model in trained_models.items():
                joblib.dump(model, horizon_dir / f"{target_name}_model.joblib")
            with open(horizon_dir / "metrics.json", "w", encoding="utf-8") as handle:
                json.dump(target_metrics, handle, indent=2)
            with open(horizon_dir / "samples.json", "w", encoding="utf-8") as handle:
                json.dump(target_sample_counts, handle, indent=2)

        if not artefacts:
            raise InsufficientSamplesError(
                "Insufficient samples to train any requested horizons.",
                horizons=horizons,
                targets=tuple(sorted(requested_targets)),
                sample_counts=requested_counts,
                missing_targets=missing_targets,
            )

        metadata = {
            **dataset.metadata,
            "artefacts": {h: art.metrics for h, art in artefacts.items()},
        }
        return {
            "status": "trained",
            "horizons": horizons,
            "targets": requested_targets,
            "metadata": metadata,
        }

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def _load_horizon_artefacts(self, horizon: int) -> HorizonArtifacts:
        horizon_dir = self.models_dir / f"horizon_{horizon}"
        preprocessor_path = horizon_dir / "preprocessor.joblib"
        if not preprocessor_path.exists():
            raise FileNotFoundError(f"Missing preprocessor for horizon {horizon}")
        preprocessor = joblib.load(preprocessor_path)
        models: dict[str, Any] = {}
        for model_path in horizon_dir.glob("*_model.joblib"):
            target_name = model_path.name.replace("_model.joblib", "")
            models[target_name] = joblib.load(model_path)
        metrics = {}
        samples = {}
        metrics_path = horizon_dir / "metrics.json"
        samples_path = horizon_dir / "samples.json"
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text())
        if samples_path.exists():
            samples = json.loads(samples_path.read_text())
        return HorizonArtifacts(horizon=horizon, preprocessor=preprocessor, models=models, metrics=metrics, sample_counts=samples)

    def predict_latest(
        self,
        *,
        targets: Iterable[str] | None = None,
        horizon: int | None = None,
    ) -> PredictionOutcome:
        resolved_horizon = int(horizon or min(self.config.prediction_horizons))
        
        def _to_utc(timestamp: pd.Timestamp | None) -> pd.Timestamp | None:
            if timestamp is None:
                return None
            ts = pd.to_datetime(timestamp)
            if ts.tzinfo is None or ts.tzinfo.utcoffset(ts) is None:
                return ts.tz_localize("UTC")
            return ts.tz_convert("UTC")

        def _latest_timestamp(frame: pd.DataFrame | None) -> pd.Timestamp | None:
            if frame is None or frame.empty:
                return None
            if "Date" in frame.columns:
                normalized = pd.to_datetime(frame["Date"], errors="coerce")
                if not normalized.empty:
                    return _to_utc(normalized.max())
            if isinstance(frame.index, pd.DatetimeIndex) and not frame.index.empty:
                return _to_utc(pd.to_datetime(frame.index.max()))
            return None

        def _build_payload(
            *,
            status: Literal["ok", "no_data", "error"],
            reason: str | None,
            predictions: Mapping[str, Any],
            probabilities: Mapping[str, Any],
            quantile_forecasts: Mapping[str, Any],
            sample_counts: Mapping[int, Any],
            missing_targets: Mapping[int, Any],
            feature_columns: Iterable[str] | None,
            metrics: Mapping[str, Any],
            checked_at: pd.Timestamp | None,
            message: str | None = None,
        ) -> PredictionOutcome:
            return PredictionOutcome(
                status=status,
                reason=reason,
                horizon=resolved_horizon,
                symbol=self.config.ticker,
                predictions=dict(predictions),
                probabilities=dict(probabilities),
                quantile_forecasts=dict(quantile_forecasts),
                feature_columns=list(feature_columns or []),
                sample_counts=dict(sample_counts),
                missing_targets=dict(missing_targets),
                metrics=dict(metrics),
                checked_at=checked_at,
                message=message,
            )

        def _log_unavailability(horizon_value: int, *, reason: str, message: str | None = None) -> None:
            cache = self._untrainable_until.get(int(horizon_value)) or {}
            first = not cache.get("warning_emitted")
            log_fn = LOGGER.warning if first else LOGGER.debug
            log_fn(
                "Prediction unavailable for horizon %s (%s)%s",
                horizon_value,
                reason,
                f": {message}" if message else "",
            )
            cache["warning_emitted"] = True
            cache.setdefault("reason", reason)
            self._untrainable_until[int(horizon_value)] = cache

        def _unavailable(
            reason: str,
            *,
            sample_counts: Mapping[int, Any] | None = None,
            missing_targets: Mapping[int, Any] | None = None,
            message: str | None = None,
            checked_at: pd.Timestamp | None = None,
            last_training_attempt: pd.Timestamp | None = None,
        ) -> PredictionOutcome:
            cache = self._untrainable_until.get(resolved_horizon) or {}
            checked_at = _to_utc(checked_at)
            if checked_at is not None:
                cache["data_timestamp"] = checked_at
                cache.setdefault("checked_at", checked_at)
            cache["not_trainable"] = True
            last_training_attempt = _to_utc(last_training_attempt)
            if last_training_attempt is not None:
                cache["last_training_attempt"] = last_training_attempt
            cache.setdefault("sample_counts", sample_counts or {})
            cache.setdefault("missing_targets", missing_targets or {})
            cache.setdefault("warning_emitted", False)
            cache.setdefault("reason", reason)
            self._untrainable_until[resolved_horizon] = cache
            _log_unavailability(resolved_horizon, reason=reason, message=message)
            return _build_payload(
                status="no_data",
                reason=reason,
                predictions={},
                probabilities={},
                quantile_forecasts={},
                sample_counts=sample_counts or {},
                missing_targets=missing_targets or {},
                feature_columns=[],
                metrics={},
                checked_at=checked_at,
                message=message,
            )

        def _handle_insufficient(exc: InsufficientSamplesError, *, latest_ts: pd.Timestamp | None) -> PredictionOutcome:
            current_attempt = pd.Timestamp.now(tz="UTC")
            self._untrainable_until[resolved_horizon] = {
                "data_timestamp": latest_ts,
                "sample_counts": getattr(exc, "sample_counts", None) or {},
                "missing_targets": getattr(exc, "missing_targets", None) or {},
                "reason": "insufficient_samples",
                "checked_at": latest_ts,
                "warning_emitted": False,
                "not_trainable": True,
                "last_training_attempt": current_attempt,
            }
            return _unavailable(
                "insufficient_samples",
                sample_counts=getattr(exc, "sample_counts", None),
                missing_targets=getattr(exc, "missing_targets", None),
                checked_at=latest_ts,
                message=str(exc),
                last_training_attempt=current_attempt,
            )

        try:
            price_df = self.fetcher.fetch_price_data()
            latest_timestamp = _latest_timestamp(price_df)
            cache = self._untrainable_until.get(resolved_horizon)
            cache_ts = _to_utc(cache.get("data_timestamp")) if cache else None
            new_data_available = (
                cache_ts is not None and latest_timestamp is not None and latest_timestamp > cache_ts
            )
            if cache and new_data_available:
                cache["warning_emitted"] = False
                cache["data_timestamp"] = latest_timestamp
                self._untrainable_until[resolved_horizon] = cache
            if cache and cache.get("not_trainable") and not new_data_available:
                if cache_ts is None or (latest_timestamp is None or latest_timestamp <= cache_ts):
                    msg = f"Horizon {resolved_horizon} still below min samples; waiting for more data"
                    LOGGER.info(msg)
                    LOGGER.debug("%s (latest_ts=%s, cache_ts=%s)", msg, latest_timestamp, cache_ts)
                    return _unavailable(
                        cache.get("reason", "insufficient_samples") or "insufficient_samples",
                        sample_counts=cache.get("sample_counts"),
                        missing_targets=cache.get("missing_targets"),
                        checked_at=cache_ts,
                        message=msg,
                    )

            try:
                artefacts = self._load_horizon_artefacts(resolved_horizon)
            except FileNotFoundError:
                LOGGER.info(
                    "No persisted artefacts found for horizon %s; triggering training run.",
                    resolved_horizon,
                )
                self.train(horizon=resolved_horizon, force=True)
                artefacts = self._load_horizon_artefacts(resolved_horizon)

            news_df = self.fetcher.fetch_news_data() if self.config.sentiment else pd.DataFrame()
            macro_df = pd.DataFrame()
            if self.config.feature_toggles.macro:
                macro_df = self.database.get_indicators(
                    ticker=self.config.ticker, interval=self.config.interval, category="macro"
                )
            feature_result = self.feature_assembler.build(
                price_df,
                news_df,
                bool(self.config.sentiment),
                macro_df=macro_df if not macro_df.empty else None,
            )
            latest_features = feature_result.features.copy()
            latest_features["ticker"] = self.config.ticker
            latest_row = latest_features.iloc[[-1]]
            try:
                transformed = artefacts.preprocessor.transform(latest_row)
            except NotFittedError:
                cached = self._untrainable_until.get(resolved_horizon)
                if cached and cached.get("not_trainable") and not new_data_available:
                    cache_ts = _to_utc(cached.get("data_timestamp"))
                    if cache_ts is not None and (latest_timestamp is None or latest_timestamp <= cache_ts):
                        return _unavailable(
                            cached.get("reason", "insufficient_samples") or "insufficient_samples",
                            sample_counts=cached.get("sample_counts"),
                            missing_targets=cached.get("missing_targets"),
                            checked_at=cache_ts,
                            message="Preprocessor not fitted; waiting for more data before retraining.",
                        )
                LOGGER.warning(
                    "Preprocessor for horizon %s is not fitted; retraining before inference.",
                    resolved_horizon,
                )
                self.train(horizon=resolved_horizon, force=True)
                self._untrainable_until.pop(resolved_horizon, None)
                artefacts = self._load_horizon_artefacts(resolved_horizon)
                transformed = artefacts.preprocessor.transform(latest_row)

            if not artefacts.models:
                current_attempt = pd.Timestamp.now(tz="UTC")
                self._untrainable_until[resolved_horizon] = {
                    "data_timestamp": latest_timestamp,
                    "sample_counts": artefacts.sample_counts,
                    "missing_targets": {},
                    "reason": "insufficient_samples",
                    "checked_at": latest_timestamp,
                    "warning_emitted": False,
                    "not_trainable": True,
                    "last_training_attempt": current_attempt,
                }
                return _unavailable(
                    "insufficient_samples",
                    sample_counts=artefacts.sample_counts,
                    missing_targets={},
                    checked_at=latest_timestamp,
                    last_training_attempt=current_attempt,
                )

            requested_targets = set(targets) if targets else set(artefacts.models.keys())
            predictions: dict[str, Any] = {}
            probabilities: dict[str, dict[str, float]] = {}
            quantile_forecasts: dict[str, dict[str, float]] = {}
            for target_name, model in artefacts.models.items():
                if target_name not in requested_targets:
                    continue
                y_hat = model.predict(transformed)
                predictions[target_name] = float(y_hat[0])

                estimator = model.named_steps.get("estimator") if hasattr(model, "named_steps") else model

                if hasattr(estimator, "predict_proba"):
                    try:
                        proba = estimator.predict_proba(transformed)[0]
                        classes = getattr(estimator, "classes_", range(len(proba)))
                        probabilities[target_name] = {
                            str(label): float(prob)
                            for label, prob in zip(classes, proba)
                        }
                    except Exception:  # pragma: no cover - defensive against model quirks
                        probabilities[target_name] = {}

                if hasattr(estimator, "predict_quantiles"):
                    try:
                        quantiles = estimator.predict_quantiles(transformed)
                    except Exception:  # pragma: no cover - estimator specific
                        quantiles = {}
                    if isinstance(quantiles, dict):
                        quantile_forecasts[target_name] = {
                            str(key): float(np.nanmean(value)) if hasattr(value, "__iter__") else float(value)
                            for key, value in quantiles.items()
                            if value is not None and not (isinstance(value, float) and math.isnan(value))
                        }

            self._untrainable_until.pop(resolved_horizon, None)
            return _build_payload(
                status="ok",
                reason=None,
                predictions=predictions,
                probabilities=probabilities,
                quantile_forecasts=quantile_forecasts,
                feature_columns=get_feature_names_from_pipeline(artefacts.preprocessor),
                sample_counts=artefacts.sample_counts,
                missing_targets={},
                metrics=artefacts.metrics,
                checked_at=latest_timestamp,
            )
        except InsufficientSamplesError as exc:
            return _handle_insufficient(exc, latest_ts=locals().get("latest_timestamp"))


__all__ = ["MultiHorizonModelingEngine", "MultiHorizonDataset", "HorizonArtifacts"]
