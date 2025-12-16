"""Batch training dataset builder for deterministic model inputs."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
from sklearn.pipeline import Pipeline

from .config import DEFAULT_MIN_SAMPLES_PER_HORIZON, PredictorConfig
from .features import FeatureAssembler
from .ml_preprocessing import PreprocessingBuilder, get_feature_names_from_pipeline
from .sentiment import aggregate_daily_sentiment
from .indicator_bundle import evaluate_signal_confluence
from stock_predictor.providers.database import Database

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class TrainingDataset:
    """Container holding cached training artefacts."""

    features: pd.DataFrame
    targets: dict[int, dict[str, pd.Series]]
    preprocessors: dict[int, Pipeline]
    metadata: dict[str, Any]


class TrainingDatasetBuilder:
    """Build and cache supervised datasets from persisted market data."""

    def __init__(self, config: PredictorConfig, *, database: Database | None = None) -> None:
        self.config = config
        self.database = database or Database(config.database_url)
        self.feature_assembler = FeatureAssembler(
            config.feature_toggles, config.prediction_horizons
        )

    def build(self, *, force: bool = False) -> TrainingDataset:
        if not force and self._cache_available():
            return self._load_cached()

        price_df = self.database.get_prices(
            ticker=self.config.ticker,
            interval=self.config.interval,
            start=self.config.start_date,
            end=self.config.end_date,
        )
        if price_df.empty:
            raise ValueError("No price history available to build training dataset.")

        news_df = self.database.get_news(self.config.ticker)
        sentiment_enabled = bool(self.config.sentiment and self.config.feature_toggles.sentiment)
        macro_df = pd.DataFrame()
        if self.config.feature_toggles.macro:
            macro_df = self.database.get_indicators(
                ticker=self.config.ticker, interval=self.config.interval, category="macro"
            )

        feature_result = self.feature_assembler.build(
            price_df,
            news_df,
            sentiment_enabled,
            macro_df=macro_df if not macro_df.empty else None,
        )

        metadata = dict(feature_result.metadata)
        metadata.setdefault("feature_columns", list(feature_result.features.columns))
        metadata.setdefault("raw_feature_columns", list(feature_result.features.columns))
        metadata.setdefault("horizons", tuple(self.config.prediction_horizons))
        metadata["feature_toggles"] = self.config.feature_toggles.asdict()
        metadata.setdefault("data_sources", [])
        metadata.setdefault("target_dates", {})

        # Aggregate helpers used downstream in the UI for diagnostics.
        metadata.setdefault(
            "sentiment_daily",
            aggregate_daily_sentiment(news_df) if not news_df.empty else pd.DataFrame(),
        )
        try:
            metadata.setdefault(
                "signal_confluence",
                evaluate_signal_confluence(price_df)
                if "Date" in price_df.columns and not price_df.empty
                else {},
            )
        except Exception as exc:  # pragma: no cover - defensive guard
            LOGGER.debug("Failed to compute signal confluence: %s", exc)

        target_counts: dict[int, dict[str, int]] = {}
        for horizon, target_map in feature_result.targets.items():
            horizon_counts: dict[str, int] = {}
            for name, series in target_map.items():
                horizon_counts[name] = int(series.dropna().shape[0])
            target_counts[horizon] = horizon_counts
        metadata["target_counts"] = target_counts

        insufficient: dict[int, list[str]] = {}
        threshold = getattr(
            self.config, "min_samples_per_horizon", DEFAULT_MIN_SAMPLES_PER_HORIZON
        )
        for horizon, horizon_counts in target_counts.items():
            missing = [
                name for name, count in horizon_counts.items() if count < int(threshold)
            ]
            if missing:
                insufficient[horizon] = missing
        if insufficient:
            LOGGER.warning(
                "Insufficient samples for some targets: %s", json.dumps(insufficient)
            )
            metadata["insufficient_samples"] = insufficient

        preprocessors, template_feature_names = self._build_preprocessors(feature_result)
        metadata["preprocessed_feature_columns"] = template_feature_names

        dataset = TrainingDataset(
            features=feature_result.features,
            targets=feature_result.targets,
            preprocessors=preprocessors,
            metadata=metadata,
        )
        self._persist_cache(dataset)
        return dataset

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------
    def _cache_available(self) -> bool:
        cache_path = self.config.training_cache_path
        target_path = cache_path.with_name(f"{cache_path.stem}_targets.parquet")
        pickle_cache_path = cache_path.with_suffix(".pkl")
        pickle_target_path = target_path.with_suffix(".pkl")

        metadata_exists = self.config.training_metadata_path.exists()
        parquet_available = cache_path.exists() and target_path.exists()
        pickle_available = pickle_cache_path.exists() and pickle_target_path.exists()
        return metadata_exists and (parquet_available or pickle_available)

    def _load_cached(self) -> TrainingDataset:
        features: pd.DataFrame | None = None
        parquet_path = self.config.training_cache_path
        pickle_path = parquet_path.with_suffix(".pkl")

        if parquet_path.exists():
            try:
                features = pd.read_parquet(parquet_path)
            except ImportError:
                LOGGER.info(
                    "Parquet engine unavailable; attempting to load cached features from pickle at %s",
                    pickle_path,
                )

        if features is None and pickle_path.exists():
            features = pd.read_pickle(pickle_path)

        if features is None:
            raise FileNotFoundError(
                f"No cached features found at {parquet_path} or {pickle_path}"
            )

        targets = self._read_cached_targets(parquet_path=parquet_path)
        with open(self.config.training_metadata_path, "r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        preprocessors = self._build_preprocessors_from_metadata(features, targets, metadata)
        return TrainingDataset(features=features, targets=targets, preprocessors=preprocessors, metadata=metadata)

    def _persist_cache(self, dataset: TrainingDataset) -> None:
        path = self.config.training_cache_path
        metadata_path = self.config.training_metadata_path
        path.parent.mkdir(parents=True, exist_ok=True)

        target_frame = self._serialise_targets(dataset.targets)
        target_parquet_path = path.with_name(f"{path.stem}_targets.parquet")
        target_pickle_path = target_parquet_path.with_suffix(".pkl")

        try:
            dataset.features.to_parquet(path, index=True)
            target_frame.to_parquet(target_parquet_path, index=False)
        except ImportError:
            LOGGER.info(
                "Parquet engine unavailable; caching training data as pickle at %s",
                path.with_suffix(".pkl"),
            )
            dataset.features.to_pickle(path.with_suffix(".pkl"))
            target_frame.to_pickle(target_pickle_path)

        with open(metadata_path, "w", encoding="utf-8") as handle:
            json.dump(dataset.metadata, handle, indent=2, default=str)

    def _read_cached_targets(self, *, parquet_path: Path | None = None) -> dict[int, dict[str, pd.Series]]:
        cache_path = parquet_path or self.config.training_cache_path
        path = cache_path.with_name(f"{cache_path.stem}_targets.parquet")
        pickle_path = path.with_suffix(".pkl")

        frame: pd.DataFrame | None = None
        if path.exists():
            try:
                frame = pd.read_parquet(path)
            except ImportError:
                LOGGER.info(
                    "Parquet engine unavailable; attempting to load cached targets from pickle at %s",
                    pickle_path,
                )

        if frame is None and pickle_path.exists():
            frame = pd.read_pickle(pickle_path)

        if frame is None:
            return {}
        targets: dict[int, dict[str, pd.Series]] = {}
        for (horizon, name), group in frame.groupby(["horizon", "target"], sort=False):
            series = pd.Series(group["value"].values, index=pd.Index(group["index"]))
            targets.setdefault(int(horizon), {})[str(name)] = series
        return targets

    @staticmethod
    def _serialise_targets(targets: Mapping[int, Mapping[str, pd.Series]]) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for horizon, target_map in targets.items():
            for name, series in target_map.items():
                frame = series.reset_index()
                frame.columns = ["index", "value"]
                frame["horizon"] = int(horizon)
                frame["target"] = str(name)
                rows.append(frame)
        if not rows:
            return pd.DataFrame(columns=["index", "value", "horizon", "target"])
        return pd.concat(rows, ignore_index=True)

    # ------------------------------------------------------------------
    # Preprocessing helpers
    # ------------------------------------------------------------------
    def _build_preprocessors(
        self, feature_result: Any
    ) -> tuple[dict[int, Pipeline], dict[int, list[str]]]:
        preprocess_section = self.config.model_params.get("preprocessing", {})
        preprocess_options = dict(preprocess_section) if isinstance(preprocess_section, Mapping) else {}
        builder = PreprocessingBuilder(**preprocess_options)
        templates: dict[int, Pipeline] = {}
        template_feature_names: dict[int, list[str]] = {}
        features = feature_result.features
        horizons = tuple(feature_result.metadata.get("horizons", self.config.prediction_horizons))
        for horizon_value in horizons:
            pipeline = builder.create_pipeline()
            X_source = features
            y_source = None
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
        return templates, template_feature_names

    def _build_preprocessors_from_metadata(
        self,
        features: pd.DataFrame,
        targets: Mapping[int, Mapping[str, pd.Series]],
        metadata: Mapping[str, Any],
    ) -> dict[int, Pipeline]:
        preprocess_section = self.config.model_params.get("preprocessing", {})
        preprocess_options = dict(preprocess_section) if isinstance(preprocess_section, Mapping) else {}
        builder = PreprocessingBuilder(**preprocess_options)
        horizons = tuple(metadata.get("horizons", self.config.prediction_horizons))
        templates: dict[int, Pipeline] = {}
        for horizon_value in horizons:
            pipeline = builder.create_pipeline()
            X_source = features
            y_source = None
            horizon_targets = targets.get(int(horizon_value))
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
            templates[int(horizon_value)] = pipeline
        metadata_preprocessed = metadata.get("preprocessed_feature_columns")
        if isinstance(metadata_preprocessed, Mapping):
            for horizon_value, columns in metadata_preprocessed.items():
                if int(horizon_value) in templates and isinstance(columns, list):
                    continue
        return templates


__all__ = ["TrainingDataset", "TrainingDatasetBuilder"]
