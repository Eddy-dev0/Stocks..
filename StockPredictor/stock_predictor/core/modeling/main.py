"""High level orchestration for feature engineering, training and inference."""

from __future__ import annotations

import copy
import json
import logging
import math
import multiprocessing as mp
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, time as dt_time
from itertools import repeat
from pathlib import Path
from numbers import Real
from typing import (
    Any,
    Callable,
    DefaultDict,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LinearRegression
from sklearn.metrics import brier_score_loss, r2_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline

from ..backtesting import Backtester as LegacyBacktester
from ..config import PredictorConfig
from ..data_fetcher import DataFetcher
from ..database import ExperimentTracker
from ..directional import DirectionalStats, evaluate_directional_predictions
from ..features import FeatureAssembler, FeatureToggles
from ..indicator_bundle import evaluate_signal_confluence
from ..training_data import TrainingDatasetBuilder
from ...evaluation.backtester import BacktestConfig, Backtester
from ...evaluation.simulation_backtester import SimulationBacktestConfig, SimulationBacktester
from .multi_horizon_engine import MultiHorizonModelingEngine
from ..ml_preprocessing import (
    DataFrameSimpleImputer,
    PreprocessingBuilder,
    get_feature_names_from_pipeline,
)
from ..pipeline import (
    DEFAULT_MARKET_TIMEZONE,
    US_MARKET_CLOSE,
    US_MARKET_OPEN,
    resolve_market_timezone,
)
from .prediction_result import FeatureUsageSummary, PredictionResult
from ..support_levels import indicator_support_floor
from .ensembles import EnsembleMember
from ..models import (
    LGBMClassifier,
    LGBMRegressor,
    ModelFactory,
    XGBClassifier,
    XGBRegressor,
    classification_metrics,
    extract_feature_importance,
    model_supports_proba,
    regression_metrics,
)
from ..sentiment import aggregate_daily_sentiment, attach_sentiment
from ..time_series import evaluate_time_series_baselines
from .simulation import run_monte_carlo, run_monte_carlo_adaptive


def _simulate_direction_bootstrap(
    seed: int, size: int, mean_return: float, predicted_volatility: float
) -> tuple[int, int]:
    """Simulate bootstrapped returns for direction refinement."""

    if size <= 0:
        return 0, 0

    rng = np.random.default_rng(seed)
    returns = rng.normal(loc=mean_return, scale=predicted_volatility, size=size)
    up = int(np.count_nonzero(returns > 0))
    down = int(size - up)
    return up, down

LOGGER = logging.getLogger(__name__)

DEFAULT_EXPECTED_LOW_SIGMA = 1.0
DEFAULT_STOP_LOSS_MULTIPLIER = 1.0
DEFAULT_EXPECTED_LOW_MAX_VOLATILITY = 1.0
DEFAULT_EXPECTED_LOW_FLOOR_WINDOW = 20
DEFAULT_EXPECTED_LOW_ATR_MULTIPLIER = 1.0
DEFAULT_EXPECTED_LOW_IV_MULTIPLIER = 1.0
DEFAULT_STOP_LOSS_ATR_MULTIPLIER = 1.2
DEFAULT_STOP_LOSS_IV_MULTIPLIER = 1.0
DEFAULT_IMPLIED_VOLATILITY_RULE_OF_16 = 16.0
DEFAULT_EXPECTED_LOW_ATR_PERIOD = 14
DEFAULT_PLAUSIBILITY_SIGMA_MULTIPLIER = 3.0
DEFAULT_LOG_RETURN_WINDOW = 252
DEFAULT_VOLATILITY_FLOOR = 0.02
DEFAULT_MAX_EXPECTED_CHANGE_PCT = 0.1
EU_MARKET_OPEN = dt_time(9, 0)
EU_MARKET_CLOSE = dt_time(17, 30)


LabelFunction = Callable[[pd.DataFrame, int, Any], pd.Series]


def _normalize_datetime_series(
    values: pd.Series, *, target_timezone: ZoneInfo | None
) -> pd.Series:
    """Return a timezone-naive datetime series aligned to ``target_timezone``.

    The input is coerced to datetimes and, if timezone-aware, converted to the
    requested timezone before the timezone information is stripped. Naive values
    are returned unchanged apart from the ``to_datetime`` coercion.
    """

    datetimes = pd.to_datetime(values, errors="coerce")
    tzinfo = getattr(datetimes.dt, "tz", None)
    if tzinfo is not None:
        normalized_zone = target_timezone or tzinfo
        datetimes = datetimes.dt.tz_convert(normalized_zone).dt.tz_localize(None)
    return datetimes


def _normalize_timestamp(
    value: Any, *, target_timezone: ZoneInfo | None
) -> Optional[pd.Timestamp]:
    """Normalize a single datetime-like value for staleness comparison."""

    if value is None:
        return None

    try:
        normalized = _normalize_datetime_series(
            pd.Series([value]), target_timezone=target_timezone
        )
    except (TypeError, ValueError):
        return None

    timestamp = normalized.iloc[0]
    if pd.isna(timestamp):
        return None
    return timestamp


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


def make_target_hit_label(
    df: pd.DataFrame, horizon: int, target_gain_pct: float = 0.05
) -> pd.Series:
    """Label rows where the price reaches a target within the horizon."""

    if horizon <= 0:
        raise ValueError("horizon must be a positive integer")

    working = df.copy()
    lower_columns = {column.lower(): column for column in working.columns}
    close_column = lower_columns.get("close")
    if close_column is None:
        raise KeyError("Input dataframe must contain a 'close' column for target-hit labels.")

    closes = pd.to_numeric(working[close_column], errors="coerce")
    gain = float(target_gain_pct)
    target_prices = closes * (1 + gain)
    window = int(horizon)
    future_max = closes.shift(-1).rolling(window=window, min_periods=1).max()

    hits = (future_max >= target_prices).astype(float)
    hits[target_prices.isna() | future_max.isna()] = np.nan
    return hits


TARGET_SPECS: dict[str, TargetSpec] = {
    "close": TargetSpec("close", "regression"),
    "direction": TargetSpec("direction", "classification"),
    "return": TargetSpec("return", "regression"),
    "volatility": TargetSpec(
        "volatility", "regression", default_model_type="random_forest", label_fn=make_volatility_label
    ),
    "target_hit": TargetSpec("target_hit", "classification", default_model_type="logistic"),
}

SUPPORTED_TARGETS = frozenset(TARGET_SPECS)


def _historical_drift_volatility(
    price_df: pd.DataFrame, window: int = 90
) -> tuple[float | None, float | None]:
    """Return mean log-return drift (per step) and volatility from price history."""

    if price_df is None or price_df.empty:
        return None, None

    lower_columns = {column.lower(): column for column in price_df.columns}
    close_column = lower_columns.get("close") or lower_columns.get("adj close")
    if close_column is None:
        return None, None

    closes = pd.to_numeric(price_df[close_column], errors="coerce").dropna()
    if closes.empty:
        return None, None

    log_returns = np.log(closes / closes.shift(1)).dropna()
    if window and len(log_returns) > window:
        log_returns = log_returns.tail(window)

    if log_returns.empty:
        return None, None

    return float(log_returns.mean()), float(log_returns.std())


def _recalibrated_log_drift(
    expected_close: float | None, latest_close: float | None, horizon: int | float | None
) -> float | None:
    """Return per-step log drift for Monte Carlo recalibration."""

    if expected_close is None or latest_close is None or horizon is None:
        return None

    horizon_value = float(horizon)
    if horizon_value <= 0:
        return None

    close_now = float(latest_close)
    target_close = float(expected_close)

    if not np.isfinite(close_now) or not np.isfinite(target_close):
        return None
    if close_now <= 0 or target_close <= 0:
        return None

    price_ratio = target_close / close_now
    if price_ratio <= 0:
        return None

    return float(math.log(price_ratio) / horizon_value)


def _max_drawdown_fraction(price_df: pd.DataFrame | None) -> float | None:
    """Return the magnitude of the worst historical drawdown as a fraction."""

    if price_df is None or price_df.empty:
        return None

    lower_columns = {column.lower(): column for column in price_df.columns}
    close_column = lower_columns.get("close") or lower_columns.get("adj close")
    if close_column is None:
        return None

    closes = pd.to_numeric(price_df[close_column], errors="coerce").dropna()
    if closes.empty:
        return None

    running_max = closes.cummax()
    drawdowns = closes / running_max - 1.0
    if drawdowns.empty:
        return None

    worst_drawdown = float(drawdowns.min())
    return abs(worst_drawdown)


def _trailing_low(price_df: pd.DataFrame | None, window: int) -> float | None:
    """Return the lowest recent price over the trailing *window* period."""

    if price_df is None or price_df.empty or window <= 0:
        return None

    lower_columns = {column.lower(): column for column in price_df.columns}
    low_column = (
        lower_columns.get("low")
        or lower_columns.get("l")
        or lower_columns.get("close")
        or lower_columns.get("adj close")
    )
    if low_column is None:
        return None

    lows = pd.to_numeric(price_df[low_column], errors="coerce").dropna()
    if lows.empty:
        return None

    if len(lows) > window:
        lows = lows.tail(window)
    return float(lows.min())


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
            config.feature_toggles,
            config.prediction_horizons,
            regime_params={
                "trend_window": config.regime_trend_window,
                "mean_reversion_window": config.regime_mean_reversion_window,
                "vol_short_window": config.regime_vol_short_window,
                "vol_long_window": config.regime_vol_long_window,
                "trend_threshold": config.regime_trend_threshold,
                "mean_reversion_threshold": config.regime_mean_reversion_threshold,
                "vol_high_threshold": config.regime_vol_high_threshold,
                "vol_low_threshold": config.regime_vol_low_threshold,
            },
            event_params={
                "event_window": config.event_feature_window,
                "keywords": config.event_feature_keywords,
            },
            direction_params={"neutral_threshold": config.direction_neutral_threshold},
        )
        self.training_builder = TrainingDatasetBuilder(
            config, database=getattr(self.fetcher, "database", None)
        )
        self.market_timezone: ZoneInfo | None = resolve_market_timezone(self.config)
        self.tracker = ExperimentTracker(config)
        self.models: dict[Tuple[str, int], Any] = {}
        self.preprocessors: dict[Tuple[str, int], Pipeline] = {}
        self.regime_models: dict[Tuple[str, int, str], Any] = {}
        self.regime_preprocessors: dict[Tuple[str, int, str], Pipeline] = {}
        self.preprocessor_templates: dict[int, Pipeline] = {}
        self.preprocess_options: dict[str, Any] = {}
        self.metadata: Dict[str, Any] = {}
        self.nextgen_engine = MultiHorizonModelingEngine(
            config, fetcher=self.fetcher, feature_assembler=self.feature_assembler
        )

    def _select_boosting_model_type(self) -> str:
        preferred = (self.config.model_type or "").strip().lower()
        if preferred and preferred not in {"random_forest", "auto", "boosting"}:
            return preferred
        if LGBMRegressor is not None and LGBMClassifier is not None:
            return "lightgbm"
        if XGBRegressor is not None and XGBClassifier is not None:
            return "xgboost"
        return "hist_gb"

    def _refresh_feature_assembler(self) -> None:
        """Rebuild the feature assembler if the toggle state has changed."""

        refreshed = False
        if self.feature_assembler is None:
            self.feature_assembler = FeatureAssembler(
                self.config.feature_toggles,
                self.config.prediction_horizons,
                regime_params={
                    "trend_window": self.config.regime_trend_window,
                    "mean_reversion_window": self.config.regime_mean_reversion_window,
                    "vol_short_window": self.config.regime_vol_short_window,
                    "vol_long_window": self.config.regime_vol_long_window,
                    "trend_threshold": self.config.regime_trend_threshold,
                    "mean_reversion_threshold": self.config.regime_mean_reversion_threshold,
                    "vol_high_threshold": self.config.regime_vol_high_threshold,
                    "vol_low_threshold": self.config.regime_vol_low_threshold,
                },
                event_params={
                    "event_window": self.config.event_feature_window,
                    "keywords": self.config.event_feature_keywords,
                },
                direction_params={
                    "neutral_threshold": self.config.direction_neutral_threshold,
                },
            )
            refreshed = True
        elif self.feature_assembler.feature_toggles != self.config.feature_toggles:
            self.feature_assembler = FeatureAssembler(
                self.config.feature_toggles,
                self.config.prediction_horizons,
                regime_params={
                    "trend_window": self.config.regime_trend_window,
                    "mean_reversion_window": self.config.regime_mean_reversion_window,
                    "vol_short_window": self.config.regime_vol_short_window,
                    "vol_long_window": self.config.regime_vol_long_window,
                    "trend_threshold": self.config.regime_trend_threshold,
                    "mean_reversion_threshold": self.config.regime_mean_reversion_threshold,
                    "vol_high_threshold": self.config.regime_vol_high_threshold,
                    "vol_low_threshold": self.config.regime_vol_low_threshold,
                },
                event_params={
                    "event_window": self.config.event_feature_window,
                    "keywords": self.config.event_feature_keywords,
                },
                direction_params={
                    "neutral_threshold": self.config.direction_neutral_threshold,
                },
            )
            refreshed = True

        if refreshed and self.nextgen_engine is not None:
            self.nextgen_engine.feature_assembler = self.feature_assembler

    # ------------------------------------------------------------------
    # Data acquisition
    # ------------------------------------------------------------------
    def download_data(self, force: bool = False) -> Dict[str, Any]:
        """Refresh all datasets and return a summary of the ETL run."""

        LOGGER.info("Starting data refresh for %s", self.config.ticker)
        summary = self.fetcher.refresh_all(force=force)
        LOGGER.info("Data refresh completed for %s", self.config.ticker)
        return summary

    def refresh_indicators(
        self, price_frame: pd.DataFrame | None = None, *, force: bool = False
    ) -> int:
        """Refresh technical indicators through the ETL layer.

        The UI rebuilds the pipeline when feature toggles change and expects a
        ``refresh_indicators`` method to be available. Delegate to the
        underlying ETL implementation when reachable, returning the number of
        indicators inserted.
        """

        etl = getattr(self.fetcher, "etl", None)
        if etl is None:
            raise AttributeError("Indicator refresh not available on this fetcher")
        return etl.refresh_indicators(price_frame, force=force)

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------
    def prepare_features(
        self,
        price_df: Optional[pd.DataFrame] = None,
        news_df: Optional[pd.DataFrame] = None,
        *,
        force_live_price: bool = False,
        use_cached_dataset: bool | None = None,
    ) -> tuple[pd.DataFrame, dict[int, Dict[str, pd.Series]], dict[int, Pipeline]]:
        self._refresh_feature_assembler()

        dataset_flag = (
            self.config.use_cached_training_data if use_cached_dataset is None else use_cached_dataset
        )
        if dataset_flag:
            self.training_builder = TrainingDatasetBuilder(
                self.config, database=getattr(self.fetcher, "database", None)
            )
            dataset = self.training_builder.build(force=force_live_price)
            self.metadata = dataset.metadata
            self.preprocessor_templates = dataset.preprocessors
            self.metadata.setdefault("active_horizon", self.horizon)
            return dataset.features, dataset.targets, dataset.preprocessors

        feature_toggles = self.config.feature_toggles
        sentiment_enabled = bool(
            self.config.sentiment and feature_toggles.get("sentiment", False)
        )
        macro_enabled = bool(feature_toggles.get("macro", False))

        if price_df is None:
            price_df = self.fetcher.fetch_price_data()
        if news_df is None and sentiment_enabled:
            news_df = self.fetcher.fetch_news_data()
        elif news_df is None:
            news_df = pd.DataFrame()

        macro_frame = pd.DataFrame()
        if macro_enabled:
            macro_frame = self._load_macro_indicators()
        merged_price_df = (
            self._merge_macro_columns(price_df, macro_frame)
            if macro_enabled and not macro_frame.empty
            else price_df
        )

        feature_result = self.feature_assembler.build(
            merged_price_df,
            news_df,
            sentiment_enabled,
            macro_df=macro_frame if macro_enabled and not macro_frame.empty else None,
        )
        metadata = dict(feature_result.metadata)
        data_availability = {
            "sentiment": sentiment_enabled,
            "macro": macro_enabled,
        }
        raw_feature_columns = list(feature_result.features.columns)
        metadata.setdefault("feature_columns", raw_feature_columns)
        metadata["raw_feature_columns"] = raw_feature_columns
        metadata.setdefault("sentiment_daily", pd.DataFrame(columns=["Date", "sentiment"]))
        metadata.setdefault("feature_groups", {})
        feature_groups = metadata.get("feature_groups")
        if isinstance(feature_groups, dict):
            for name, available in data_availability.items():
                if name in feature_groups and isinstance(feature_groups[name], Mapping):
                    feature_groups[name]["available"] = available
        metadata["available_feature_groups"] = dict(data_availability)
        metadata["data_sources"] = self.fetcher.get_data_sources()
        metadata.setdefault("target_dates", {})
        horizons = tuple(metadata.get("horizons", tuple(self.config.prediction_horizons)))
        metadata["horizons"] = horizons
        metadata["active_horizon"] = self.horizon
        metadata.setdefault("feature_toggles", self.config.feature_toggles.asdict())
        metadata.setdefault(
            "executed_feature_groups",
            [
                name
                for name, summary in feature_groups.items()
                if summary.get("executed")
            ]
            if isinstance(feature_groups, Mapping)
            else [],
        )

        latest_close, latest_date = self._latest_settled_close(price_df)
        if latest_close is not None:
            metadata["latest_close"] = latest_close
            if latest_date is not None:
                metadata["latest_date"] = latest_date
                metadata.setdefault("market_data_as_of", latest_date)

        try:
            live_price, live_timestamp = self.fetcher.fetch_live_price(
                force=force_live_price
            )
        except Exception as exc:  # pragma: no cover - optional live quote path
            LOGGER.debug("Live price fetch failed: %s", exc)
        else:
            if live_price is not None:
                metadata["latest_price"] = float(live_price)
                if live_timestamp is not None:
                    metadata["market_data_as_of"] = live_timestamp
                    metadata["latest_price_timestamp"] = live_timestamp

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

            trailing_window = getattr(
                self.config, "expected_low_floor_window", DEFAULT_EXPECTED_LOW_FLOOR_WINDOW
            )
            trailing_low = _trailing_low(price_history, int(trailing_window))
            if trailing_low is not None:
                metadata["trailing_low"] = float(trailing_low)
                metadata["trailing_low_window"] = int(trailing_window)

            max_drawdown_fraction = _max_drawdown_fraction(price_history)
            if max_drawdown_fraction is not None:
                metadata["max_drawdown_fraction"] = float(max_drawdown_fraction)

            lower_map = {column.lower(): column for column in price_history.columns}
            close_column = lower_map.get("close")
            if close_column is not None:
                price_history["return"] = pd.to_numeric(
                    price_history[close_column], errors="coerce"
                ).pct_change()

                recent_returns = price_history["return"].dropna()
                if not recent_returns.empty:
                    recent_window = recent_returns.tail(int(window))
                    return_std = float(recent_window.std())
                    metadata["recent_return_std"] = return_std
                    metadata["recent_return_mean"] = float(recent_window.mean())

                    expected_low_std_window = getattr(
                        self.config, "expected_low_std_window", int(window)
                    )
                    try:
                        expected_low_std_window = int(expected_low_std_window)
                    except (TypeError, ValueError):
                        expected_low_std_window = int(window)
                    if expected_low_std_window > 0:
                        expected_low_window = recent_returns.tail(expected_low_std_window)
                        if not expected_low_window.empty:
                            expected_low_return_std = float(expected_low_window.std())
                            metadata["expected_low_return_std"] = expected_low_return_std
                            metadata["expected_low_return_window"] = int(expected_low_std_window)

                log_return_window = getattr(
                    self.config, "log_return_window", DEFAULT_LOG_RETURN_WINDOW
                )
                log_returns = np.log(pd.to_numeric(price_history[close_column], errors="coerce"))
                log_returns = log_returns.diff().dropna()
                if not log_returns.empty:
                    trailing_log_returns = log_returns.tail(int(log_return_window))
                    log_return_std = float(trailing_log_returns.std())
                    if np.isfinite(log_return_std):
                        metadata["daily_log_return_std"] = log_return_std
                        metadata["daily_log_return_window"] = int(log_return_window)

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

        target_hit_spec = TARGET_SPECS.get("target_hit")
        if target_hit_spec:
            gain_pct = getattr(self.config, "target_gain_pct", 0.0)
            price_history = merged_price_df.copy()
            if "Date" in price_history.columns:
                price_history["Date"] = pd.to_datetime(price_history["Date"], errors="coerce")
                price_history = price_history.dropna(subset=["Date"])
                price_history = price_history.sort_values("Date").reset_index(drop=True)
            else:
                price_history = price_history.sort_index().reset_index(drop=True)

            for horizon_value in horizons:
                try:
                    label_series = make_target_hit_label(
                        price_history, int(horizon_value), float(gain_pct)
                    )
                except Exception as exc:  # pragma: no cover - defensive guard
                    LOGGER.debug(
                        "Failed to compute target-hit labels for horizon %s: %s",
                        horizon_value,
                        exc,
                    )
                    continue
                if label_series is None:
                    continue
                label_series = pd.Series(label_series, name="target_hit")
                if label_series.dropna().empty:
                    continue
                label_series = label_series.reset_index(drop=True)
                label_series.index = feature_result.features.index
                horizon_targets = feature_result.targets.setdefault(int(horizon_value), {})
                horizon_targets["target_hit"] = label_series
            metadata["target_gain_pct"] = float(gain_pct)
            latest_close = metadata.get("latest_close")
            if latest_close is not None:
                try:
                    metadata["target_price"] = float(latest_close) * (1 + float(gain_pct))
                except Exception:
                    pass

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

    def get_used_feature_groups(self) -> list[str]:
        """Return the feature groups that were executed in the last build."""

        groups = self.metadata.get("feature_groups") if isinstance(self.metadata, Mapping) else {}
        if not isinstance(groups, Mapping):
            return []
        return [
            name
            for name, summary in groups.items()
            if isinstance(summary, Mapping) and summary.get("executed")
        ]

    def _latest_settled_close(
        self, price_df: Optional[pd.DataFrame]
    ) -> tuple[float | None, pd.Timestamp | None]:
        """Return the last completed session's close and timestamp.

        Intraday intervals can surface prices for the current trading session,
        which are not representative of the previous official close. This
        helper walks backward from the current (or most recent) completed
        session to find the latest settled close value.
        """

        if price_df is None or price_df.empty:
            return None, None
        if "Date" not in price_df.columns:
            return None, None
        lower_columns = {column.lower(): column for column in price_df.columns}
        basis = str(getattr(self.config, "target_price_basis", "adj_close") or "adj_close").strip().lower()
        if basis == "adj_close":
            close_column = (
                lower_columns.get("adj close")
                or lower_columns.get("adj_close")
                or lower_columns.get("close")
            )
        else:
            close_column = (
                lower_columns.get("close")
                or lower_columns.get("adj close")
                or lower_columns.get("adj_close")
            )
        if close_column is None:
            return None, None

        market_tz = self.market_timezone or DEFAULT_MARKET_TIMEZONE
        frame = price_df.copy()
        frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
        frame[close_column] = pd.to_numeric(frame[close_column], errors="coerce")
        frame = frame.dropna(subset=["Date", close_column])
        if frame.empty:
            return None, None

        if getattr(frame["Date"].dt, "tz", None) is None:
            frame["Date"] = frame["Date"].dt.tz_localize(market_tz)
        else:
            frame["Date"] = frame["Date"].dt.tz_convert(market_tz)

        frame = frame.sort_values("Date")
        available_dates = frame["Date"].dt.date
        earliest_date = available_dates.min()
        now = datetime.now(market_tz)
        session_date = now.date()
        if (now.hour, now.minute, now.second) < (
            US_MARKET_CLOSE.hour,
            US_MARKET_CLOSE.minute,
            US_MARKET_CLOSE.second,
        ):
            session_date = session_date - timedelta(days=1)

        while session_date not in set(available_dates):
            session_date = session_date - timedelta(days=1)
            if earliest_date is not None and session_date < earliest_date:
                return None, None

        settled_rows = frame[frame["Date"].dt.date == session_date]
        if settled_rows.empty:
            return None, None

        latest_row = settled_rows.iloc[-1]
        return float(latest_row[close_column]), latest_row["Date"]

    def _refresh_live_price_metadata(self, *, force: bool = False) -> None:
        """Update cached live price metadata in-place when available."""

        if not isinstance(self.metadata, dict):
            return
        try:
            live_price, live_timestamp = self.fetcher.fetch_live_price(force=force)
        except Exception as exc:  # pragma: no cover - optional dependency path
            LOGGER.debug("Skipping live price refresh: %s", exc)
            return
        if live_price is None:
            return
        self.metadata["latest_price"] = float(live_price)
        if live_timestamp is not None:
            self.metadata["market_data_as_of"] = live_timestamp
            self.metadata["latest_price_timestamp"] = live_timestamp

    def _resolve_market_session(self, timestamp: datetime) -> str:
        """Classify the trading session for a given timestamp."""

        market_tz = self.market_timezone or DEFAULT_MARKET_TIMEZONE
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=market_tz)
        us_timestamp = timestamp.astimezone(market_tz)
        us_time = us_timestamp.time()

        if US_MARKET_OPEN <= us_time < US_MARKET_CLOSE:
            return "US-RTH"
        if us_time >= US_MARKET_CLOSE:
            return "US-Afterhours"

        try:
            eu_timezone = ZoneInfo("Europe/Berlin")
        except Exception:  # pragma: no cover - defensive fallback
            eu_timezone = None

        if eu_timezone is not None:
            eu_time = us_timestamp.astimezone(eu_timezone).time()
            if EU_MARKET_OPEN <= eu_time < EU_MARKET_CLOSE:
                return "EU-Open"

        return "US-Premarket"

    def _session_open_price(
        self, price_df: pd.DataFrame | None, *, timestamp: datetime
    ) -> float | None:
        if price_df is None or price_df.empty or "Date" not in price_df.columns:
            return None

        frame = price_df.copy()
        frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
        frame = frame.dropna(subset=["Date"])
        if frame.empty:
            return None

        lower_columns = {column.lower(): column for column in frame.columns}
        open_column = lower_columns.get("open") or lower_columns.get("o")
        if open_column is None:
            return None

        market_tz = self.market_timezone or DEFAULT_MARKET_TIMEZONE
        if getattr(frame["Date"].dt, "tz", None) is None:
            frame["Date"] = frame["Date"].dt.tz_localize(market_tz)
        else:
            frame["Date"] = frame["Date"].dt.tz_convert(market_tz)

        session_date = timestamp.astimezone(market_tz).date()
        has_intraday = (frame["Date"].dt.time != dt_time(0, 0)).any()
        if has_intraday:
            open_time = datetime.combine(session_date, US_MARKET_OPEN, tzinfo=market_tz)
            session_rows = frame[
                (frame["Date"] >= open_time) & (frame["Date"].dt.date == session_date)
            ]
        else:
            session_rows = frame[frame["Date"].dt.date == session_date]

        if session_rows.empty:
            return None

        return self._safe_float(session_rows.iloc[0][open_column])

    def _session_open_vwap(
        self, price_df: pd.DataFrame | None, *, timestamp: datetime
    ) -> float | None:
        if price_df is None or price_df.empty or "Date" not in price_df.columns:
            return None

        frame = price_df.copy()
        frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
        frame = frame.dropna(subset=["Date"])
        if frame.empty:
            return None

        lower_columns = {column.lower(): column for column in frame.columns}
        volume_column = lower_columns.get("volume") or lower_columns.get("vol")
        price_column = (
            lower_columns.get("close")
            or lower_columns.get("adj close")
            or lower_columns.get("adj_close")
        )
        if volume_column is None or price_column is None:
            return None

        market_tz = self.market_timezone or DEFAULT_MARKET_TIMEZONE
        if getattr(frame["Date"].dt, "tz", None) is None:
            frame["Date"] = frame["Date"].dt.tz_localize(market_tz)
        else:
            frame["Date"] = frame["Date"].dt.tz_convert(market_tz)

        has_intraday = (frame["Date"].dt.time != dt_time(0, 0)).any()
        if not has_intraday:
            return None

        session_date = timestamp.astimezone(market_tz).date()
        open_time = datetime.combine(session_date, US_MARKET_OPEN, tzinfo=market_tz)
        end_time = open_time + timedelta(minutes=5)
        session_rows = frame[(frame["Date"] >= open_time) & (frame["Date"] < end_time)]
        if session_rows.empty:
            return None

        prices = pd.to_numeric(session_rows[price_column], errors="coerce")
        volumes = pd.to_numeric(session_rows[volume_column], errors="coerce")
        valid = prices.notna() & volumes.notna() & (volumes > 0)
        if not valid.any():
            return None

        weighted = (prices[valid] * volumes[valid]).sum()
        total_volume = volumes[valid].sum()
        if total_volume <= 0:
            return None

        return float(weighted / total_volume)

    def _resolve_anchor_price(
        self,
        price_df: pd.DataFrame | None,
        *,
        latest_close: float | None,
        last_price_value: float | None,
        prediction_timestamp: datetime,
        market_session: str,
        prediction_warnings: list[str],
    ) -> tuple[float | None, str]:
        requested_source = str(
            getattr(self.config, "anchor_price_source", "close") or "close"
        ).strip().lower()
        anchor_price_source = requested_source

        if market_session != "US-RTH" and requested_source in {"live", "open", "vwap_5m"}:
            anchor_price_source = "close"
            prediction_warnings.append(
                "Anchor price locked to last close outside US RTH to avoid unstable session gaps."
            )

        anchor_price = self._safe_float(latest_close)
        if anchor_price_source == "live":
            anchor_price = self._safe_float(last_price_value)
        elif anchor_price_source == "open":
            anchor_price = self._session_open_price(price_df, timestamp=prediction_timestamp)
        elif anchor_price_source == "vwap_5m":
            anchor_price = self._session_open_vwap(price_df, timestamp=prediction_timestamp)
            if anchor_price is None:
                anchor_price = self._session_open_price(
                    price_df, timestamp=prediction_timestamp
                )
                if anchor_price is not None:
                    anchor_price_source = "open"

        if anchor_price is None and anchor_price_source != "close":
            anchor_price_source = "close"
            anchor_price = self._safe_float(latest_close)
            if anchor_price is not None:
                prediction_warnings.append(
                    "Anchor price source unavailable; reverted to last close."
                )

        if anchor_price is None and anchor_price_source != "live":
            candidate = self._safe_float(last_price_value)
            if candidate is not None:
                anchor_price = candidate
                anchor_price_source = "live"

        return anchor_price, anchor_price_source

    def _collect_live_sentiment(
        self, *, force: bool = False
    ) -> tuple[float | None, float | None, pd.DataFrame | None, str | None]:
        """Fetch recent sentiment and compute headline averages.

        News or pre-aggregated sentiment signals are fetched via the provider layer.
        When only raw headlines are available we run ``attach_sentiment`` locally so
        the UI always surfaces a computed score and short-term trend.
        """

        aggregated: pd.DataFrame | None = None
        error_message: str | None = None

        sentiment_df: pd.DataFrame | None = None
        try:
            # Prefer the dedicated sentiment endpoint which already aggregates
            # multiple providers (database or live API depending on ``force``).
            sentiment_df = self.fetcher.fetch_sentiment_signals(force=force)
        except Exception as exc:  # pragma: no cover - optional dependency path
            LOGGER.debug("Live sentiment signal fetch failed: %s", exc)
            error_message = str(exc)

        if isinstance(sentiment_df, pd.DataFrame) and not sentiment_df.empty:
            working = sentiment_df.copy()
            date_col = None
            for candidate in ("AsOf", "Date"):
                if candidate in working.columns:
                    date_col = candidate
                    break

            if date_col:
                working[date_col] = pd.to_datetime(working[date_col], errors="coerce")
                working = working.dropna(subset=[date_col])
                working["Date"] = working[date_col].dt.date

            score_col = None
            for candidate in (
                "Score",
                "sentiment",
                "Sentiment_Avg",
                "sentiment_score",
            ):
                if candidate in working.columns:
                    score_col = candidate
                    break

            if date_col and score_col:
                # Aggregate by calendar day to stabilise noisy intraday signals.
                aggregated = (
                    working.groupby("Date")[score_col]
                    .mean()
                    .rename("Sentiment_Avg")
                    .reset_index()
                )

        if (aggregated is None or aggregated.empty) and getattr(
            self.config, "sentiment", False
        ):
            try:
                # Fall back to headline text, annotating it with VADER/FinBERT
                # scores locally so we still produce a numeric sentiment feed.
                news_df = self.fetcher.fetch_news_data(force=force)
                if isinstance(news_df, pd.DataFrame) and not news_df.empty:
                    scored = attach_sentiment(news_df)
                    aggregated = aggregate_daily_sentiment(scored)
            except Exception as exc:  # pragma: no cover - optional dependency path
                LOGGER.debug("Headline sentiment scoring failed: %s", exc)
                if error_message is None:
                    error_message = str(exc)

        if aggregated is None or aggregated.empty:
            return None, None, aggregated, error_message

        aggregated["Date"] = pd.to_datetime(aggregated["Date"], errors="coerce")
        aggregated = aggregated.dropna(subset=["Date"]).sort_values("Date")
        if not aggregated.empty:
            error_message = None
        sentiment_series = None
        if "Sentiment_Avg" in aggregated.columns:
            sentiment_series = aggregated["Sentiment_Avg"]
        elif "sentiment" in aggregated.columns:
            sentiment_series = aggregated["sentiment"]
        if sentiment_series is not None:
            aggregated["sentiment"] = sentiment_series

        latest_avg = self._safe_float(aggregated.iloc[-1].get("sentiment"))
        trailing = pd.to_numeric(aggregated["sentiment"], errors="coerce").dropna()
        trend_avg: float | None = None
        if not trailing.empty:
            trend_avg = float(trailing.tail(7).mean()) if trailing.tail(7).size else None

        return latest_avg, trend_avg, aggregated, error_message

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

        price["Date"] = _normalize_datetime_series(
            price["Date"], target_timezone=self.market_timezone
        )
        macro["Date"] = _normalize_datetime_series(
            macro["Date"], target_timezone=self.market_timezone
        )
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

    def _latest_raw_features(self, raw_feature_columns: Optional[Iterable[str]] = None) -> Optional[pd.DataFrame]:
        """Return the most recent raw feature frame, respecting stored column order."""

        latest_features = self.metadata.get("latest_features")
        if not isinstance(latest_features, pd.DataFrame) or latest_features.empty:
            return None

        columns = list(raw_feature_columns) if raw_feature_columns is not None else list(
            self.metadata.get("raw_feature_columns") or []
        )
        if columns:
            missing = [col for col in columns if col not in latest_features.columns]
            if not missing:
                return latest_features[columns].copy()
        return latest_features.copy()

    @staticmethod
    def _extract_imputer_from_pipeline(pipeline: Optional[Pipeline]) -> Optional[DataFrameSimpleImputer]:
        if pipeline is None:
            return None
        for _, step in pipeline.steps:
            if isinstance(step, DataFrameSimpleImputer):
                return step
        imputer = pipeline.named_steps.get("imputer") if hasattr(pipeline, "named_steps") else None
        if isinstance(imputer, DataFrameSimpleImputer):
            return imputer
        return None

    @staticmethod
    def _imputer_is_fitted(imputer: DataFrameSimpleImputer) -> bool:
        required_attrs = ("feature_names_", "valid_columns_", "valid_output_columns_")
        return all(hasattr(imputer, attr) for attr in required_attrs)

    def _ensure_preprocessor_fitted(
        self,
        pipeline: Optional[Pipeline],
        target: str,
        horizon: int,
        *,
        raw_features: Optional[pd.DataFrame] = None,
        target_series: Optional[pd.Series] = None,
    ) -> Optional[Pipeline]:
        """Verify pipelines include a fitted imputer, refitting when necessary."""

        if pipeline is None:
            return None

        imputer = self._extract_imputer_from_pipeline(pipeline)
        if imputer is not None and self._imputer_is_fitted(imputer):
            return pipeline

        LOGGER.warning(
            "Preprocessor for target '%s' horizon %s is missing fitted imputer state; attempting refit.",
            target,
            horizon,
        )

        raw_features = raw_features if raw_features is not None else self._latest_raw_features()
        if raw_features is None or raw_features.empty:
            raise RuntimeError(
                "Preprocessor for target '%s' at horizon %s is not fitted and no feature data is available to refit. "
                "Please retrain the model so a fitted preprocessor is saved."
                % (target, horizon)
            )

        candidate = pipeline
        try:
            candidate.fit(raw_features, target_series)
        except Exception:
            template = self.preprocessor_templates.get(horizon)
            if template is not None:
                candidate = copy.deepcopy(template)
            else:
                builder = PreprocessingBuilder(**self.preprocess_options)
                candidate = builder.create_pipeline()
            candidate.fit(raw_features, target_series)

        self.preprocessors[(target, horizon)] = candidate
        feature_names = get_feature_names_from_pipeline(candidate)
        if feature_names:
            feature_map = self.metadata.setdefault("feature_columns_by_target", {})
            feature_map[(target, horizon)] = feature_names
            self.metadata["feature_columns"] = feature_names
        return candidate

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
        pipeline = self._ensure_preprocessor_fitted(
            pipeline,
            target,
            horizon,
            raw_features=self._latest_raw_features(),
        )
        feature_names = get_feature_names_from_pipeline(pipeline) if pipeline is not None else []
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
                target_kind = stored.get("target_kind")
                if target_kind:
                    self.metadata["target_kind"] = target_kind
                target_variants = stored.get("target_variants")
                if target_variants:
                    self.metadata["target_variants"] = target_variants
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
            "target_kind": self.metadata.get("target_kind"),
            "target_variants": self.metadata.get("target_variants"),
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
        selected_model_type = self._select_boosting_model_type()
        if selected_model_type != self.config.model_type:
            LOGGER.info(
                "Selected model type '%s' for next-gen training (previously '%s').",
                selected_model_type,
                self.config.model_type,
            )
            self.config.model_type = selected_model_type
        resolved_horizon = self._resolve_horizon(horizon)
        self.horizon = resolved_horizon
        modern_report = self.nextgen_engine.train(
            targets=list(targets) if targets else None,
            horizon=resolved_horizon,
            force=force,
        )
        self.metadata = modern_report.get("metadata", {})
        return modern_report
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

            model_params = dict(self.config.model_params.get(target, {}))
            global_params = dict(self.config.model_params.get("global", {}))
            model_params = self._apply_cost_sensitive_params(task, y_clean, model_params)
            ensemble_config = self._resolve_ensemble_config(
                target, task, params=self.config.model_params
            )
            calibration_params = self._build_calibration_params(task)
            for scope in (global_params.get("calibration"), model_params.get("calibration")):
                if isinstance(scope, Mapping):
                    calibration_params.update(scope)
            calibration_enabled = bool(
                calibration_params.get("enabled", False)
                or calibration_params.get("residuals", False)
                or calibration_params.get("residual_calibration", False)
            )

            calibrate_override = model_params.get("calibrate")
            if calibrate_override is None:
                calibrate_override = global_params.get("calibrate")
            if calibrate_override is None:
                calibrate_flag = task == "classification"
            else:
                calibrate_flag = bool(calibrate_override)
            if task == "classification" and not self.config.probability_calibration_enabled:
                calibrate_flag = False

            if ensemble_config.get("enabled"):
                members = self._build_ensemble_members(
                    ensemble_config["members"],
                    task,
                    global_params=global_params,
                    model_params=model_params,
                    calibrate_flag=calibrate_flag,
                    calibration_params=calibration_params,
                    weights=ensemble_config.get("weights", {}),
                )
                if members:
                    model_params["members"] = members
                if ensemble_config.get("quantiles"):
                    model_params["quantiles"] = ensemble_config["quantiles"]
                model_params["interval_alpha"] = ensemble_config.get("interval_alpha", 0.2)
                model_type = "ensemble"

            template = self.preprocessor_templates.get(resolved_horizon)
            factory = ModelFactory(model_type, {**global_params, **model_params})
            tuning_summary = self._tune_model(
                factory,
                aligned_X,
                y_clean,
                task,
                preprocessor=template,
                calibrate_flag=calibrate_flag,
                calibration_params=calibration_params,
            )
            tuned_params = tuning_summary.get("best_params") if isinstance(tuning_summary, Mapping) else None
            if tuned_params:
                model_params.update(tuned_params)
                factory = ModelFactory(model_type, {**global_params, **model_params})
                tuning_store = self.metadata.setdefault("tuning", {})
                tuning_store[(target, resolved_horizon)] = {
                    "best_params": tuned_params,
                    "best_score": tuning_summary.get("best_score"),
                    "folds": int(self.config.tuning_folds or self.config.evaluation_folds),
                }

            evaluation = self._evaluate_model(
                factory,
                aligned_X,
                y_clean,
                task,
                target,
                calibrate_flag,
                calibration_params,
                template,
                collect_predictions=calibration_enabled,
                horizon=resolved_horizon,
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

            final_model = factory.create(
                task,
                calibrate=calibrate_flag,
                calibration_params=dict(calibration_params or {}),
            )
            final_model.fit(transformed_X, y_clean)
            regime_summary = self._train_regime_models(
                target,
                resolved_horizon,
                task,
                aligned_X,
                y_clean,
                factory=factory,
                template=template,
                calibrate_flag=calibrate_flag,
                calibration_params=calibration_params,
            )
            calibrator = None
            calibration_stats = None
            validation_block = evaluation.get("validation") if calibration_enabled else None
            if (
                calibration_enabled
                and isinstance(validation_block, Mapping)
                and validation_block.get("y_true")
                and validation_block.get("y_pred")
            ):
                calibrator, calibration_stats = self._fit_residual_calibrator(
                    validation_block.get("y_true", []), validation_block.get("y_pred", [])
                )
                if calibrator is not None:
                    setattr(final_model, "residual_calibrator_", calibrator)
                    if calibration_stats:
                        setattr(final_model, "residual_calibration_", calibration_stats)
            distribution_summary = self._estimate_prediction_uncertainty(
                target, final_model, transformed_X
            )

            metrics: Dict[str, Any] = dict(evaluation["aggregate"])
            metrics["training_rows"] = int(len(transformed_X))
            metrics["test_rows"] = int(evaluation.get("evaluation_rows", 0))
            metrics["horizon"] = resolved_horizon
            metrics["evaluation_strategy"] = evaluation["strategy"]
            if regime_summary:
                metrics["regime_models"] = regime_summary
            if calibration_stats:
                metrics["residual_calibration"] = calibration_stats
            if tuning_summary:
                metrics["tuning"] = {
                    "enabled": bool(self.config.tuning_enabled),
                    "best_params": tuning_summary.get("best_params"),
                    "best_score": tuning_summary.get("best_score"),
                }
            metrics["evaluation"] = {
                "strategy": evaluation["strategy"],
                "splits": evaluation["splits"],
                "aggregate": evaluation["aggregate"],
                "parameters": evaluation.get("parameters", {}),
                "samples": int(evaluation.get("evaluation_rows", 0)),
            }
            directional_block = evaluation.get("directional")
            if isinstance(directional_block, Mapping):
                metrics["directional"] = directional_block
                metrics["evaluation"]["directional"] = directional_block
                self.metadata.setdefault("directional_metrics", {})[
                    (target, resolved_horizon)
                ] = directional_block
            metrics["cv_metrics"] = evaluation["aggregate"]
            if tuning_summary:
                metrics["evaluation"]["tuning"] = tuning_summary
            if evaluation.get("baseline_aggregate"):
                metrics["baselines"] = evaluation["baseline_aggregate"]
                metrics["evaluation"]["baselines"] = evaluation["baseline_aggregate"]
                metrics["evaluation"]["baseline_splits"] = evaluation.get(
                    "baseline_splits", []
                )
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
                target,
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

    def _resolve_ensemble_config(
        self, target: str, task: str, *, params: Mapping[str, Any]
    ) -> dict[str, Any]:
        target_params = params.get(target, {}) if isinstance(params, Mapping) else {}
        global_params = params.get("global", {}) if isinstance(params, Mapping) else {}
        resolved: dict[str, Any] = {}
        ensemble_blocks = [
            block
            for block in (global_params.get("ensemble"), target_params.get("ensemble"))
            if isinstance(block, Mapping)
        ]
        for block in ensemble_blocks:
            resolved.update(block)

        enabled = bool(resolved.get("enabled", False)) and task == "regression"
        if not enabled:
            return {"enabled": False}

        members = resolved.get("members") or resolved.get("models")
        member_list = [
            str(name).strip().lower()
            for name in (members or [])
            if str(name).strip()
        ]
        if not member_list:
            # Default to a diverse blend around the configured base model
            member_list = [
                self.config.model_type,
                "random_forest",
                "hist_gb",
                "mlp",
                "transformer",
            ]

        weights = resolved.get("weights", {}) if isinstance(resolved.get("weights"), Mapping) else {}
        return {
            "enabled": True,
            "members": member_list,
            "weights": {str(k).strip().lower(): float(v) for k, v in weights.items()},
            "quantiles": resolved.get("quantiles"),
            "interval_alpha": resolved.get("interval_alpha", 0.2),
            "blender": resolved.get("blender", "ridge"),
        }

    def _build_calibration_params(self, task: str) -> dict[str, Any]:
        if task != "classification" or not self.config.probability_calibration_enabled:
            return {}
        folds = max(2, int(self.config.probability_calibration_folds))
        return {
            "method": self.config.probability_calibration_method,
            "cv": TimeSeriesSplit(n_splits=folds),
        }

    def _apply_cost_sensitive_params(
        self, task: str, y_values: pd.Series, model_params: dict[str, Any]
    ) -> dict[str, Any]:
        if task != "classification" or not self.config.cost_sensitive_enabled:
            return model_params

        counts = y_values.value_counts(dropna=True).to_dict()
        if not counts:
            return model_params

        total = float(sum(counts.values()))
        class_weights: dict[float, float] = {}
        for label, count in counts.items():
            if count <= 0:
                continue
            base_weight = total / (len(counts) * float(count))
            if label in {1, 1.0, True}:
                base_weight *= float(self.config.cost_sensitive_up_weight)
            elif label in {-1, -1.0}:
                base_weight *= float(self.config.cost_sensitive_down_weight)
            elif label in {0, 0.0}:
                base_weight *= float(self.config.cost_sensitive_neutral_weight)
            class_weights[float(label)] = float(base_weight)

        if class_weights:
            model_params = dict(model_params)
            model_params["class_weight"] = class_weights
        return model_params

    def _build_ensemble_members(
        self,
        member_types: Sequence[str],
        task: str,
        *,
        global_params: Mapping[str, Any],
        model_params: Mapping[str, Any],
        calibrate_flag: bool,
        calibration_params: Mapping[str, Any] | None,
        weights: Mapping[str, float],
    ) -> list[EnsembleMember]:
        members: list[EnsembleMember] = []
        for name in member_types:
            if not name:
                continue
            overrides = {**global_params, **model_params}
            weight = float(weights.get(name, 1.0))
            factory = ModelFactory(name, overrides)
            estimator = factory.create(
                task,
                calibrate=calibrate_flag,
                calibration_params=dict(calibration_params or {}),
            )
            members.append(EnsembleMember(name=name, estimator=estimator, weight=weight))
        return members

    def _fit_residual_calibrator(
        self, y_true: Sequence[float] | np.ndarray, y_pred: Sequence[float] | np.ndarray
    ) -> tuple[LinearRegression, dict[str, float]] | tuple[None, None]:
        if y_true is None or y_pred is None:
            return None, None

        y_true_arr = np.asarray(y_true, dtype=float)
        y_pred_arr = np.asarray(y_pred, dtype=float)
        if y_true_arr.size == 0 or y_true_arr.shape != y_pred_arr.shape:
            return None, None

        residuals = y_pred_arr - y_true_arr
        if residuals.size < 5 or not np.isfinite(residuals).any():
            return None, None

        calibrator = LinearRegression()
        calibrator.fit(y_pred_arr.reshape(-1, 1), residuals.reshape(-1, 1))
        stats = {
            "residual_std": float(np.nanstd(residuals)),
            "residual_bias": float(np.nanmean(residuals)),
        }
        return calibrator, stats

    def _resolve_regime_label(self, features: pd.DataFrame) -> str | None:
        if features is None or features.empty:
            return None
        label_series = None
        for col in ("Regime_Label", "Regime_TrendFlag"):
            if col in features.columns:
                label_series = pd.to_numeric(features[col], errors="coerce")
                break
        if label_series is None or label_series.empty:
            return None
        latest_value = float(label_series.iloc[-1])
        if np.isnan(latest_value):
            return None
        return "trend" if latest_value >= 0.5 else "range"

    def _load_regime_preprocessor(
        self, target: str, horizon: int, regime: str
    ) -> Pipeline | None:
        key = (target, horizon, regime)
        path = self.config.preprocessor_path_for_regime(target, horizon, regime)
        if not path.exists():
            return None
        try:
            pipeline = joblib.load(path)
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.warning(
                "Failed to load regime preprocessor for %s horizon %s (%s): %s",
                target,
                horizon,
                regime,
                exc,
            )
            return None
        self.regime_preprocessors[key] = pipeline
        return pipeline

    def _load_regime_model(self, target: str, horizon: int, regime: str) -> Any | None:
        key = (target, horizon, regime)
        path = self.config.model_path_for_regime(target, horizon, regime)
        if not path.exists():
            return None
        try:
            model = joblib.load(path)
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.warning(
                "Failed to load regime model for %s horizon %s (%s): %s",
                target,
                horizon,
                regime,
                exc,
            )
            return None
        self.regime_models[key] = model
        return model

    def _train_regime_models(
        self,
        target: str,
        horizon: int,
        task: str,
        features: pd.DataFrame,
        target_series: pd.Series,
        *,
        factory: ModelFactory,
        template: Pipeline | None,
        calibrate_flag: bool,
        calibration_params: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        if task != "classification" or not self.config.regime_models_enabled:
            return {}
        if "Regime_Label" not in features.columns:
            return {}

        regime_series = pd.to_numeric(features["Regime_Label"], errors="coerce")
        regime_series = regime_series.reindex(target_series.index)
        if regime_series.isna().all():
            return {}

        results: dict[str, Any] = {}
        for regime, mask in {
            "trend": regime_series >= 0.5,
            "range": regime_series < 0.5,
        }.items():
            subset_idx = target_series.index[mask.fillna(False)]
            if subset_idx.size < max(10, int(self.config.min_samples_per_horizon * 0.25)):
                continue
            X_subset = features.loc[subset_idx]
            y_subset = target_series.loc[subset_idx]

            if template is not None:
                regime_pipeline = clone(template)
            else:
                builder = PreprocessingBuilder(**self.preprocess_options)
                regime_pipeline = builder.create_pipeline()
            regime_pipeline.fit(X_subset, y_subset)
            transformed = regime_pipeline.transform(X_subset)

            regime_model = factory.create(
                task,
                calibrate=calibrate_flag,
                calibration_params=dict(calibration_params or {}),
            )
            regime_model.fit(transformed, y_subset)

            key = (target, horizon, regime)
            self.regime_models[key] = regime_model
            self.regime_preprocessors[key] = regime_pipeline

            joblib.dump(regime_model, self.config.model_path_for_regime(target, horizon, regime))
            joblib.dump(
                regime_pipeline,
                self.config.preprocessor_path_for_regime(target, horizon, regime),
            )
            results[regime] = {
                "rows": int(len(X_subset)),
                "label": regime,
            }

        return results

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

    def _evaluate_time_series_baselines(
        self, train_series: pd.Series, test_series: pd.Series
    ) -> Dict[str, Any]:
        if (
            not self.config.time_series_baselines
            or train_series.empty
            or test_series.empty
        ):
            return {}

        params = self.config.time_series_params or {}
        global_params = params.get("global", {})
        merged_params: Dict[str, Dict[str, Any]] = {}
        for name in self.config.time_series_baselines:
            lower = name.lower()
            merged = dict(global_params)
            merged.update(params.get(lower, {}))
            merged_params[lower] = merged

        seasonal_periods = global_params.get("seasonal_periods")
        return evaluate_time_series_baselines(
            train_series,
            test_series,
            self.config.time_series_baselines,
            seasonal_periods=seasonal_periods,
            model_params=merged_params,
        )

    def _aggregate_baseline_metrics(
        self, metrics: Mapping[str, List[Dict[str, Any]]]
    ) -> Dict[str, Dict[str, float]]:
        aggregate: Dict[str, Dict[str, float]] = {}
        for model_name, entries in metrics.items():
            keys = {
                key
                for entry in entries
                for key in entry.keys()
                if key not in {"model", "split", "train_size", "test_size"}
            }
            summary: Dict[str, float] = {"splits": len(entries)}
            for key in keys:
                values = [
                    float(value)
                    for value in (entry.get(key) for entry in entries)
                    if isinstance(value, Real) and np.isfinite(value)
                ]
                if values:
                    summary[key] = float(np.mean(values))
            aggregate[model_name] = summary
        return aggregate

    def _combine_pipelines(
        self, preprocessor: Pipeline | None, model: Pipeline
    ) -> Pipeline:
        steps: list[tuple[str, Any]] = []
        if preprocessor is not None:
            steps.extend(clone(preprocessor).steps)
        steps.extend(clone(model).steps)
        return Pipeline(steps=steps)

    def _hyperparameter_search_space(self, model_type: str, task: str) -> Dict[str, List[Any]]:
        search_spaces: Dict[str, Dict[str, List[Any]]] = {
            "random_forest": {
                "estimator__n_estimators": [100, 200, 400, 800],
                "estimator__max_depth": [None, 5, 10, 20],
                "estimator__min_samples_split": [2, 5, 10],
                "estimator__min_samples_leaf": [1, 2, 4],
                "estimator__max_features": ["auto", "sqrt", "log2", 0.5],
            },
            "logistic": {
                "estimator__C": [0.1, 0.5, 1.0, 2.0, 5.0],
                "estimator__penalty": ["l2", "none"],
                "estimator__solver": ["lbfgs", "saga"],
            },
            "mlp": {
                "estimator__hidden_layer_sizes": [(64,), (128,), (128, 64)],
                "estimator__activation": ["relu", "tanh"],
                "estimator__alpha": [1e-5, 1e-4, 1e-3],
                "estimator__learning_rate_init": [1e-3, 5e-4, 1e-4],
            },
            "hist_gb": {
                "estimator__max_depth": [3, 5, 8],
                "estimator__learning_rate": [0.01, 0.05, 0.1],
                "estimator__max_iter": [200, 400, 800],
                "estimator__min_samples_leaf": [20, 50, 100],
            },
        }
        if task == "classification":
            search_spaces["random_forest"]["estimator__class_weight"] = [None, "balanced"]
        return search_spaces.get(model_type, {})

    def _tune_model(
        self,
        factory: ModelFactory,
        features: pd.DataFrame,
        target_series: pd.Series,
        task: str,
        *,
        preprocessor: Pipeline | None,
        calibrate_flag: bool,
        calibration_params: Mapping[str, Any] | None = None,
    ) -> Dict[str, Any]:
        if not self.config.tuning_enabled or self.config.tuning_method == "none":
            return {}
        search_space = self._hyperparameter_search_space(factory.model_type, task)
        if not search_space:
            return {}

        try:
            base_model = factory.create(
                task,
                calibrate=calibrate_flag,
                calibration_params=dict(calibration_params or {}),
            )
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.warning("Skipping tuning for %s due to model construction failure: %s", factory.model_type, exc)
            return {}

        pipeline = self._combine_pipelines(preprocessor, base_model)
        folds = self.config.tuning_folds or self.config.evaluation_folds
        if folds <= 1:
            return {}
        if len(features) <= folds:
            LOGGER.warning(
                "Insufficient samples (%s) for %s-fold tuning; skipping.",
                len(features),
                folds,
            )
            return {}

        splitter = TimeSeriesSplit(n_splits=folds)
        scoring = "f1_weighted" if task == "classification" else "neg_mean_absolute_error"
        search = RandomizedSearchCV(
            pipeline,
            search_space,
            n_iter=int(self.config.tuning_iterations),
            cv=splitter,
            random_state=42,
            n_jobs=self.config.tuning_n_jobs,
            scoring=scoring,
            refit=False,
        )
        try:
            search.fit(features, target_series)
        except ValueError as exc:  # pragma: no cover - defensive
            LOGGER.warning("Hyperparameter search failed; continuing with defaults: %s", exc)
            return {}

        best_params = search.best_params_ or {}
        trimmed_params = {
            key.split("estimator__", 1)[-1]: value
            for key, value in best_params.items()
            if key.startswith("estimator__")
        }

        return {
            "best_params": trimmed_params,
            "cv_results": search.cv_results_,
            "best_score": float(search.best_score_),
        }

    def _evaluate_model(
        self,
        factory: ModelFactory,
        features: pd.DataFrame,
        target_series: pd.Series,
        task: str,
        target_name: str,
        calibrate_flag: bool,
        calibration_params: Mapping[str, Any] | None,
        preprocessor: Optional[Pipeline] = None,
        *,
        collect_predictions: bool = False,
        horizon: Optional[int] = None,
    ) -> Dict[str, Any]:
        strategy = self.config.evaluation_strategy
        evaluation_rows = 0
        parameters: Dict[str, Any] = {}
        splits: List[Dict[str, Any]] = []
        baseline_records: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)
        baseline_splits: List[Dict[str, Any]] = []
        validation_true: list[float] = []
        validation_pred: list[float] = []
        directional_totals = DirectionalStats()
        directional_summary: Dict[str, Any] = {}

        def _purged_splits() -> Iterable[tuple[np.ndarray, np.ndarray]]:
            n_samples = len(features)
            purge_window = max(
                0,
                int(self.config.purged_cv_purge_window or 0),
            )
            embargo = max(0, int(self.config.purged_cv_embargo or 0))
            splitter = TimeSeriesSplit(n_splits=self.config.evaluation_folds)
            for train_idx, test_idx in splitter.split(np.arange(n_samples)):
                if test_idx.size == 0:
                    continue
                test_start = int(test_idx[0])
                test_end = int(test_idx[-1])
                purge_start = max(0, test_start - purge_window)
                purge_end = min(n_samples - 1, test_end + embargo)
                mask = (train_idx < purge_start) | (train_idx > purge_end)
                yield train_idx[mask], test_idx

        def _record_baselines(
            split_label: int, train_series: pd.Series, test_series: pd.Series
        ) -> None:
            if task != "regression":
                return
            results = self._evaluate_time_series_baselines(train_series, test_series)
            for name, result in results.items():
                entry = {
                    "model": name,
                    "split": split_label,
                    "train_size": int(len(train_series)),
                    "test_size": int(len(test_series)),
                    **result.metrics,
                }
                baseline_records[name].append(entry)
                baseline_splits.append(entry)

        if strategy == "holdout":
            if getattr(self.config, "shuffle_training", False):
                raise ValueError(
                    "shuffle_training must be False when using holdout evaluation to "
                    "preserve chronological splits."
                )
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
            model = factory.create(
                task,
                calibrate=calibrate_flag,
                calibration_params=dict(calibration_params or {}),
            )
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
                target_name,
                proba,
                getattr(estimator, "classes_", None),
            )
            directional_totals.merge(
                evaluate_directional_predictions(
                    task, target_name, y_test, y_pred, raw_features=X_test
                )
            )
            if collect_predictions:
                validation_true.extend([float(y) for y in y_test.to_numpy()])
                validation_pred.extend([float(p) for p in np.ravel(y_pred)])
            metrics.update(
                {
                    "split": 1,
                    "train_size": int(len(X_train)),
                    "test_size": int(len(X_test)),
                }
            )
            splits.append(metrics)
            _record_baselines(1, y_train, y_test)
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
                model = factory.create(
                    task,
                    calibrate=calibrate_flag,
                    calibration_params=dict(calibration_params or {}),
                )
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
                    target_name,
                    proba,
                    getattr(estimator, "classes_", None),
                )
                directional_totals.merge(
                    evaluate_directional_predictions(
                        task, target_name, y_test, y_pred, raw_features=X_test
                    )
                )
                if collect_predictions:
                    validation_true.extend([float(y) for y in y_test.to_numpy()])
                    validation_pred.extend([float(p) for p in np.ravel(y_pred)])
                metrics.update(
                    {
                        "fold": fold,
                        "train_size": int(len(train_idx)),
                    "test_size": int(len(test_idx)),
                }
            )
            splits.append(metrics)
            _record_baselines(fold, y_train, y_test)
            if not splits:
                raise RuntimeError(
                    "Time series cross-validation produced no evaluation splits."
                )
            aggregate = self._aggregate_evaluation_metrics(splits)
            evaluation_rows = int(sum(item.get("test_size", 0) for item in splits))
            parameters = {"folds": int(self.config.evaluation_folds)}
        elif strategy == "purged":
            for fold, (train_idx, test_idx) in enumerate(_purged_splits(), start=1):
                if len(test_idx) == 0 or len(train_idx) == 0:
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
                model = factory.create(
                    task,
                    calibrate=calibrate_flag,
                    calibration_params=dict(calibration_params or {}),
                )
                model.fit(X_train_transformed, y_train)
                y_pred = model.predict(X_test_transformed)
                proba = None
                estimator = model.named_steps.get("estimator")
                if task == "classification" and model_supports_proba(model):
                    try:
                        proba = model.predict_proba(X_test_transformed)
                    except Exception as exc:  # pragma: no cover - defensive
                        LOGGER.debug("Failed to compute purged probabilities: %s", exc)
                metrics = self._compute_evaluation_metrics(
                    task,
                    y_test,
                    y_pred,
                    X_test_transformed,
                    X_test,
                    target_name,
                    proba,
                    getattr(estimator, "classes_", None),
                )
                directional_totals.merge(
                    evaluate_directional_predictions(
                        task, target_name, y_test, y_pred, raw_features=X_test
                    )
                )
                if collect_predictions:
                    validation_true.extend([float(y) for y in y_test.to_numpy()])
                    validation_pred.extend([float(p) for p in np.ravel(y_pred)])
                metrics.update(
                    {
                        "fold": fold,
                        "train_size": int(len(train_idx)),
                        "test_size": int(len(test_idx)),
                    }
                )
                splits.append(metrics)
                _record_baselines(fold, y_train, y_test)
            if not splits:
                raise RuntimeError("Purged cross-validation produced no evaluation splits.")
            aggregate = self._aggregate_evaluation_metrics(splits)
            evaluation_rows = int(sum(item.get("test_size", 0) for item in splits))
            parameters = {
                "folds": int(self.config.evaluation_folds),
                "purge_window": int(self.config.purged_cv_purge_window),
                "embargo": int(self.config.purged_cv_embargo),
            }
        elif strategy == "rolling":
            backtester = LegacyBacktester(
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
                target_kind=self.metadata.get("target_kind"),
                horizon=horizon,
            )
            splits = result.splits
            aggregate = result.aggregate
            evaluation_rows = int(aggregate.get("test_rows", 0))
            directional_summary = dict(result.directional)
            parameters = {
                "window": int(self.config.evaluation_window),
                "step": int(self.config.evaluation_step),
                "mode": self.config.backtest_strategy,
            }
            if task == "regression" and self.config.time_series_baselines:
                for split_idx, (train_slice, test_slice) in enumerate(
                    backtester._generate_splits(len(target_series)), start=1
                ):
                    y_train_split = target_series.iloc[train_slice]
                    y_test_split = target_series.iloc[test_slice]
                    if len(y_test_split) == 0:
                        continue
                    _record_baselines(split_idx, y_train_split, y_test_split)
        else:  # pragma: no cover - configuration guard
            raise ValueError(f"Unknown evaluation strategy: {strategy}")

        baseline_aggregate = (
            self._aggregate_baseline_metrics(baseline_records)
            if baseline_records
            else {}
        )

        payload: Dict[str, Any] = {
            "strategy": strategy,
            "splits": splits,
            "aggregate": aggregate,
            "evaluation_rows": evaluation_rows,
            "parameters": parameters,
            "baseline_splits": baseline_splits,
            "baseline_aggregate": baseline_aggregate,
        }
        if strategy != "rolling":
            directional_summary = directional_totals.as_summary(horizon=horizon)
            aggregate.update(directional_summary)
        if directional_summary.get("n_predictions_h", 0) or directional_summary.get("acc_h"):
            payload["directional"] = directional_summary
        if collect_predictions and validation_true and validation_pred:
            payload["validation"] = {
                "y_true": validation_true,
                "y_pred": validation_pred,
            }
        return payload

    def _compute_evaluation_metrics(
        self,
        task: str,
        y_true: pd.Series,
        y_pred: np.ndarray,
        X_test: pd.DataFrame,
        raw_X_test: pd.DataFrame,
        target_name: str,
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
            trading_metrics = self._simple_trading_metrics(
                target_name,
                y_true,
                y_pred,
                raw_X_test,
                proba=proba,
                classes=classes,
            )
            if trading_metrics:
                metrics.update(trading_metrics)
            return metrics

        metrics = regression_metrics(y_true.to_numpy(), y_pred)
        try:
            metrics["r2"] = float(r2_score(y_true, y_pred))
        except ValueError:
            metrics["r2"] = float("nan")
        baseline = np.zeros_like(y_pred)
        if target_name == "close" and "Close_Current" in raw_X_test:
            baseline = raw_X_test["Close_Current"].to_numpy()

        predicted_direction = np.sign(y_pred - baseline)
        actual_direction = np.sign(y_true.to_numpy() - baseline)
        if len(actual_direction) > 0:
            metrics["directional_accuracy"] = float(
                np.mean((predicted_direction >= 0) == (actual_direction >= 0))
            )
        else:
            metrics["directional_accuracy"] = 0.0
        metrics["signed_error"] = float(np.mean(y_pred - y_true.to_numpy()))
        trading_metrics = self._simple_trading_metrics(
            target_name, y_true, y_pred, raw_X_test
        )
        if trading_metrics:
            metrics.update(trading_metrics)
        return metrics

    def _simple_trading_metrics(
        self,
        target_name: str,
        y_true: pd.Series,
        y_pred: np.ndarray,
        raw_X_test: pd.DataFrame,
        *,
        proba: np.ndarray | None = None,
        classes: Sequence[Any] | None = None,
        slippage_bps: float | None = None,
        fee_bps: float | None = None,
        fixed_cost: float | None = None,
    ) -> Dict[str, float]:
        signals = self._long_flat_signals(
            target_name, y_pred, raw_X_test, proba=proba, classes=classes
        )
        actual_returns = self._actual_returns_for_metrics(target_name, y_true, raw_X_test)
        if signals.size == 0 or actual_returns.size == 0:
            return {
                "pnl_sharpe": 0.0,
                "pnl_max_drawdown": 0.0,
                "pnl_net_return": 0.0,
                "pnl_hit_rate": 0.0,
            }

        n = min(signals.size, actual_returns.size)
        trimmed_signals = signals[:n]
        trimmed_returns = actual_returns[:n]
        per_trade_slippage_bps = (
            float(slippage_bps)
            if slippage_bps is not None
            else float(getattr(self.config, "evaluation_slippage_bps", 0.0))
        )
        per_trade_fee_bps = (
            float(fee_bps)
            if fee_bps is not None
            else float(getattr(self.config, "evaluation_fee_bps", 0.0))
        )
        per_trade_fixed_cost = (
            float(fixed_cost)
            if fixed_cost is not None
            else float(getattr(self.config, "evaluation_fixed_cost", 0.0))
        )
        trade_change_mask = np.abs(np.diff(np.insert(trimmed_signals, 0, 0.0)))
        per_trade_cost_rate = max(0.0, per_trade_slippage_bps + per_trade_fee_bps) / 10000.0
        per_trade_cost_rate += max(0.0, per_trade_fixed_cost)
        trade_costs = trade_change_mask * per_trade_cost_rate

        net_returns = trimmed_signals * trimmed_returns - trade_costs
        equity_curve = np.cumprod(1 + net_returns)

        sharpe = 0.0
        if net_returns.size > 1:
            excess = net_returns - 0.0
            std = float(np.std(excess, ddof=1))
            if std > 0:
                sharpe = float(np.mean(excess) / std * np.sqrt(net_returns.size))

        max_drawdown = 0.0
        if equity_curve.size:
            running_max = np.maximum.accumulate(equity_curve)
            safe_running = np.where(running_max == 0, 1.0, running_max)
            drawdowns = equity_curve / safe_running - 1.0
            max_drawdown = float(np.min(drawdowns))

        gross_profit = float(np.sum(net_returns[net_returns > 0])) if net_returns.size else 0.0
        gross_loss = float(-np.sum(net_returns[net_returns < 0])) if net_returns.size else 0.0
        if gross_loss > 0:
            profit_factor = gross_profit / gross_loss
        else:
            profit_factor = float("inf") if gross_profit > 0 else 0.0
        expectancy = float(np.mean(net_returns)) if net_returns.size else 0.0

        pnl_metrics = {
            "pnl_sharpe": float(sharpe),
            "pnl_max_drawdown": max_drawdown,
            "pnl_net_return": float(equity_curve[-1] - 1) if equity_curve.size else 0.0,
            "pnl_hit_rate": float(np.mean(net_returns > 0)) if net_returns.size else 0.0,
            "pnl_profit_factor": float(profit_factor),
            "pnl_expectancy": expectancy,
        }
        return pnl_metrics

    def _long_flat_signals(
        self,
        target_name: str,
        y_pred: np.ndarray,
        raw_X_test: pd.DataFrame,
        *,
        proba: np.ndarray | None = None,
        classes: Sequence[Any] | None = None,
    ) -> np.ndarray:
        forecast = np.asarray(y_pred, dtype=float)
        if forecast.size == 0:
            return np.array([])

        if target_name == "close" and "Close_Current" in raw_X_test:
            baseline = raw_X_test["Close_Current"].to_numpy(dtype=float)
            denominator = np.clip(np.abs(baseline), 1e-6, None)
            forecast = (forecast - baseline) / denominator

        if proba is not None and proba.size:
            probability_array = np.asarray(proba, dtype=float)
            positive_index = 1 if probability_array.ndim > 1 else 0
            neutral_index = None
            if probability_array.ndim > 1 and classes is not None:
                for idx, cls in enumerate(classes):
                    if cls in {1, True, "1", "up"}:
                        positive_index = idx
                        break
                for idx, cls in enumerate(classes):
                    if cls in {0, 0.0, "0", "neutral"}:
                        neutral_index = idx
                        break
            positive_proba = (
                probability_array
                if probability_array.ndim == 1
                else probability_array[:, min(positive_index, probability_array.shape[1] - 1)]
            )
            threshold = float(self.config.direction_trade_threshold)
            if neutral_index is not None and probability_array.ndim > 1:
                neutral_proba = probability_array[:, min(neutral_index, probability_array.shape[1] - 1)]
                mask = (positive_proba >= threshold) & (neutral_proba <= (1 - threshold))
                return np.where(mask, 1.0, 0.0)
            return np.where(positive_proba >= threshold, 1.0, 0.0)

        return np.where(forecast > 0.0, 1.0, 0.0)

    def _actual_returns_for_metrics(
        self, target_name: str, y_true: pd.Series, raw_X_test: pd.DataFrame
    ) -> np.ndarray:
        if target_name == "close" and "Close_Current" in raw_X_test:
            baseline = raw_X_test["Close_Current"].to_numpy(dtype=float)
            denominator = np.clip(np.abs(baseline), 1e-6, None)
            return (y_true.to_numpy(dtype=float) - baseline) / denominator

        if target_name == "log_return":
            return np.expm1(y_true.to_numpy(dtype=float))

        if target_name == "direction":
            values = y_true.to_numpy(dtype=float)
            return np.where(values > 0, 0.01, np.where(values < 0, -0.01, 0.0))

        return y_true.to_numpy(dtype=float)

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
                        for value in y_true
                    ]
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

    def _extract_backtest_success_rates(
        self, payload: Mapping[str, Any]
    ) -> Dict[str, float]:
        success_rates: Dict[str, float] = {}
        target_payload = payload.get("targets", {})
        if isinstance(target_payload, Mapping):
            for target, entry in target_payload.items():
                if not isinstance(entry, Mapping):
                    continue
                aggregate = entry.get("aggregate", {})
                if not isinstance(aggregate, Mapping):
                    continue
                accuracy = aggregate.get("directional_accuracy")
                if accuracy is None:
                    accuracy = aggregate.get("accuracy")
                if isinstance(accuracy, Real) and math.isfinite(float(accuracy)):
                    success_rates[str(target)] = float(accuracy)

        summary = payload.get("summary", {})
        if isinstance(summary, Mapping):
            classification = summary.get("classification", {})
            if isinstance(classification, Mapping):
                overall = classification.get("accuracy")
                if isinstance(overall, Real) and math.isfinite(float(overall)):
                    success_rates.setdefault("overall_accuracy", float(overall))
        return success_rates

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def predict(
        self,
        refresh_data: bool = False,
        targets: Optional[Iterable[str]] = None,
        horizon: Optional[int] = None,
    ) -> PredictionResult:
        resolved_horizon = self._resolve_horizon(horizon)
        modern_predictions = self.nextgen_engine.predict_latest(
            targets=list(targets) if targets else None,
            horizon=resolved_horizon,
        )
        modern_payload = (
            modern_predictions.to_dict()
            if hasattr(modern_predictions, "to_dict")
            else dict(modern_predictions or {})
        )
        payload_horizon = modern_payload.get("horizon")
        if isinstance(payload_horizon, int):
            resolved_horizon = payload_horizon
        self.horizon = resolved_horizon
        status = modern_payload.get("status")
        reason = modern_payload.get("reason")
        message = modern_payload.get("message")
        modern_preds = modern_payload.get("predictions") or {}
        modern_probabilities = modern_payload.get("probabilities") or {}
        modern_quantiles = modern_payload.get("quantile_forecasts") or {}
        modern_sample_counts = modern_payload.get("sample_counts") or {}
        modern_missing_targets = modern_payload.get("missing_targets") or {}
        modern_metrics = modern_payload.get("metrics") or {}
        modern_feature_columns = modern_payload.get("feature_columns") or []
        modern_checked_at = modern_payload.get("checked_at")

        if modern_feature_columns:
            self.metadata.setdefault("feature_columns", modern_feature_columns)
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
                candidate_dates = _normalize_datetime_series(
                    price_df["Date"], target_timezone=self.market_timezone
                ).dropna()
                if not candidate_dates.empty:
                    latest_price_timestamp = candidate_dates.iloc[-1]

        metadata_latest_timestamp: Optional[pd.Timestamp] = None
        if self.metadata:
            raw_latest = self.metadata.get("latest_date")
            if raw_latest is not None:
                metadata_latest_timestamp = _normalize_timestamp(
                    raw_latest, target_timezone=self.market_timezone
                )

        historical_drift: float | None = None
        historical_volatility: float | None = None
        if price_df is not None and not price_df.empty:
            historical_drift, historical_volatility = _historical_drift_volatility(price_df)
            if historical_drift is not None:
                # Drift is stored as a per-step log-return so downstream consumers can
                # convert to cumulative growth consistently.
                self.metadata["historical_drift"] = historical_drift
            if historical_volatility is not None:
                self.metadata["historical_volatility"] = historical_volatility

        drift_per_step = float(historical_drift) if historical_drift is not None else None
        drift_growth_factor: float | None = None
        drift_horizon = float(resolved_horizon) if resolved_horizon else None

        if latest_price_timestamp is not None:
            if metadata_latest_timestamp is None or latest_price_timestamp > metadata_latest_timestamp:
                if not needs_feature_refresh:
                    LOGGER.info(
                        "Detected new market data available through %s; rebuilding features.",
                        latest_price_timestamp,
                    )
                needs_feature_refresh = True

        if not needs_feature_refresh:
            self._refresh_live_price_metadata(force=refresh_data)

        if needs_feature_refresh:
            LOGGER.info("Preparing features before prediction.")
            if price_df is not None:
                try:
                    self.prepare_features(price_df=price_df, force_live_price=refresh_data)
                except TypeError:
                    self.prepare_features(price_df=price_df)
            else:
                try:
                    self.prepare_features(force_live_price=refresh_data)
                except TypeError:
                    self.prepare_features()

        latest_features = self.metadata.get("latest_features")
        if latest_features is None:
            raise RuntimeError("No feature metadata available. Train the model first.")
        raw_feature_columns = self.metadata.get("raw_feature_columns")
        if raw_feature_columns is None:
            raw_feature_columns = list(latest_features.columns)
            self.metadata["raw_feature_columns"] = raw_feature_columns
        self.metadata["active_horizon"] = resolved_horizon
        regime_label = self._resolve_regime_label(latest_features)
        if regime_label:
            self.metadata["regime_label"] = regime_label

        confluence_assessment = evaluate_signal_confluence(latest_features)
        self.metadata["signal_confluence"] = confluence_assessment

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

        def _normalize_target(name: str) -> str:
            return name[:-2] if name.endswith("_h") else name

        for target_name, value in modern_preds.items():
            predictions[_normalize_target(str(target_name))] = value
        for target_name, value in modern_probabilities.items():
            probabilities[_normalize_target(str(target_name))] = value
        for target_name, value in modern_quantiles.items():
            quantile_forecasts[_normalize_target(str(target_name))] = value

        filtered_targets: list[str] = []
        for target in requested_targets:
            if target not in SUPPORTED_TARGETS:
                LOGGER.warning("Skipping target '%s': not supported.", target)
                continue
            if target not in filtered_targets:
                filtered_targets.append(target)

        for target in filtered_targets:
            if target in predictions:
                continue
            spec = TARGET_SPECS.get(target)
            task = spec.task if spec else ("classification" if target == "direction" else "regression")
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
            if regime_label and self.config.regime_models_enabled:
                regime_key = (target, resolved_horizon, regime_label)
                regime_model = self.regime_models.get(regime_key) or self._load_regime_model(
                    target, resolved_horizon, regime_label
                )
                regime_pipeline = self.regime_preprocessors.get(
                    regime_key
                ) or self._load_regime_preprocessor(target, resolved_horizon, regime_label)
                if regime_model is not None and regime_pipeline is not None:
                    model = regime_model
                    pipeline = regime_pipeline
                    training_report.setdefault(target, {})["regime"] = regime_label
            current_raw = latest_features[raw_feature_columns]
            if pipeline is not None:
                pipeline = self._ensure_preprocessor_fitted(
                    pipeline,
                    target,
                    resolved_horizon,
                    raw_features=current_raw,
                )
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
            pred_value = self._apply_residual_correction(model, float(pred_value))
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

            if model_supports_proba(model) and task == "classification":
                proba = model.predict_proba(model_input)[0]
                estimator = model.named_steps.get("estimator")
                classes = getattr(estimator, "classes_", None)
                class_prob_map: Dict[Any, float] = {}
                if classes is not None:
                    class_prob_map = {
                        cls: float(prob)
                        for cls, prob in zip(classes, proba)
                    }
                else:
                    class_prob_map = {idx: float(prob) for idx, prob in enumerate(proba)}

                probabilities[target] = class_prob_map
                confidence_value = max(class_prob_map.values()) if class_prob_map else 0.0
                confidences[target] = float(confidence_value)
                if target == "direction":
                    up_prob = float(
                        class_prob_map.get(1)
                        or class_prob_map.get(1.0)
                        or class_prob_map.get("1")
                        or class_prob_map.get(True)
                        or (float(proba[1]) if len(proba) > 1 else 0.0)
                    )
                    down_prob = float(
                        class_prob_map.get(-1)
                        or class_prob_map.get(-1.0)
                        or class_prob_map.get("-1")
                        or class_prob_map.get(0)
                        or class_prob_map.get(0.0)
                        or class_prob_map.get("0")
                        or class_prob_map.get(False)
                        or (float(proba[0]) if len(proba) > 0 else 0.0)
                    )
                    has_negative = any(
                        key in class_prob_map for key in (-1, -1.0, "-1")
                    )
                    neutral_prob = None
                    if has_negative:
                        neutral_prob = float(
                            class_prob_map.get(0)
                            or class_prob_map.get(0.0)
                            or class_prob_map.get("0")
                            or (float(proba[1]) if len(proba) > 2 else 0.0)
                        )
                    probabilities[target] = {"up": up_prob, "down": down_prob}
                    if neutral_prob is not None:
                        probabilities[target]["neutral"] = neutral_prob
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

        confluence_block = None
        combined_confidence = None
        confluence_score = None
        confluence_passed = False
        sentiment_factor = None
        trend_alignment_note = None
        confidence_notes: list[str] = []

        predicted_return = self._safe_float(predictions.get("return"))
        predicted_intraday_move = self._safe_float(predictions.get("return_oc"))
        use_intraday_as_return = resolved_horizon == 1 and predicted_intraday_move is not None
        if use_intraday_as_return:
            predicted_return = predicted_intraday_move
        return_quantiles = quantile_forecasts.get("return_oc" if use_intraday_as_return else "return")
        predicted_return = self._clamp_with_uncertainty(
            predicted_return,
            return_quantiles,
            prediction_intervals.get("return"),
        )
        if predicted_return is None and drift_per_step is not None and drift_horizon is not None:
            predicted_return, drift_growth_factor = self._compound_log_drift(
                drift_per_step, drift_horizon
            )
            prediction_warnings.append(
                "Return model unavailable or produced NaN; falling back to historical drift."
            )

        predicted_volatility = self._safe_float(predictions.get("volatility"))
        if predicted_volatility is None and historical_volatility is not None:
            predicted_volatility = float(abs(historical_volatility))
            prediction_warnings.append(
                "Volatility model unavailable or produced NaN; using realised historical volatility."
            )

        latest_close = float(self.metadata.get("latest_close", np.nan))
        last_price_value = self._safe_float(self.metadata.get("latest_price"))
        if last_price_value is None:
            last_price_value = latest_close

        market_tz = self.market_timezone or DEFAULT_MARKET_TIMEZONE
        prediction_timestamp = datetime.now(market_tz)
        market_session = self._resolve_market_session(prediction_timestamp)
        anchor_price, anchor_price_source = self._resolve_anchor_price(
            price_df,
            latest_close=latest_close,
            last_price_value=last_price_value,
            prediction_timestamp=prediction_timestamp,
            market_session=market_session,
            prediction_warnings=prediction_warnings,
        )

        close_prediction_raw = predictions.get("close")
        close_prediction = self._safe_float(close_prediction_raw)
        close_from_return = None
        if predicted_return is not None and anchor_price is not None:
            close_from_return = float(anchor_price * (1 + predicted_return))
        if close_prediction is None:
            close_prediction = close_from_return
        elif close_from_return is not None:
            close_prediction = float(np.nanmean([close_prediction, close_from_return]))
        if close_prediction is None and anchor_price is not None and drift_per_step is not None:
            _, drift_growth_factor = self._compound_log_drift(
                drift_per_step, drift_horizon if drift_horizon is not None else 1.0
            )
            close_prediction = float(anchor_price * drift_growth_factor)
            prediction_warnings.append(
                "Close model unavailable or produced NaN; reverting to drift-adjusted anchor price."
            )
        close_prediction = self._clamp_with_uncertainty(
            close_prediction,
            quantile_forecasts.get("close"),
            prediction_intervals.get("close"),
        )

        expected_change = None
        pct_change = None
        if close_prediction is not None and anchor_price is not None:
            expected_change = close_prediction - anchor_price
            pct_change = expected_change / anchor_price if anchor_price else 0.0
        elif predicted_return is not None:
            pct_change = predicted_return
            if anchor_price is not None:
                expected_change = anchor_price * predicted_return
        expected_intraday_move = predicted_intraday_move

        historical_confidence, historical_note = self._historical_confidence_score(
            expected_change=expected_change, horizon=resolved_horizon
        )
        if historical_confidence is not None:
            self.metadata["historical_confidence"] = historical_confidence
            if combined_confidence is None:
                combined_confidence = historical_confidence
            else:
                combined_confidence = float(
                    np.clip(
                        0.7 * combined_confidence + 0.3 * historical_confidence,
                        0.0,
                        1.0,
                    )
                )
        if historical_note:
            confidence_notes.append(historical_note)

        latest_date = self.metadata.get("latest_date")
        if isinstance(latest_date, pd.Timestamp):
            latest_date = latest_date.to_pydatetime()

        target_dates = self.metadata.get("target_dates", {})
        target_date = None
        if isinstance(target_dates, dict):
            target_date = target_dates.get(resolved_horizon)

        latest_timestamp = _normalize_timestamp(
            latest_date, target_timezone=self.market_timezone
        )

        target_timestamp = _normalize_timestamp(
            target_date, target_timezone=self.market_timezone
        )

        market_data_as_of = self.metadata.get("market_data_as_of") or latest_date

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

        dir_prob = probabilities.get("direction") if isinstance(probabilities, dict) else None
        direction_probability_up = None
        direction_probability_down = None
        direction_probability_neutral = None
        if isinstance(dir_prob, dict):
            direction_probability_up = self._safe_float(dir_prob.get("up"))
            direction_probability_down = self._safe_float(dir_prob.get("down"))
            direction_probability_neutral = self._safe_float(dir_prob.get("neutral"))
        target_hit_prob = probabilities.get("target_hit") if isinstance(probabilities, dict) else None
        target_hit_probability = None
        if isinstance(target_hit_prob, dict):
            hit_candidates = [
                target_hit_prob.get(key)
                for key in (1, 1.0, "1", True, "hit")
            ]
            hit_candidates = [self._safe_float(value) for value in hit_candidates]
            hit_candidates = [value for value in hit_candidates if value is not None]
            if hit_candidates:
                target_hit_probability = max(hit_candidates)

        sentiment_avg = None
        sentiment_trend = None
        sentiment_error = None
        if getattr(self.config, "sentiment", False):
            sentiment_avg, sentiment_trend, sentiment_frame, sentiment_error = (
                self._collect_live_sentiment(force=refresh_data)
            )
            if isinstance(sentiment_frame, pd.DataFrame) and not sentiment_frame.empty:
                # Keep the short-run series so downstream UI and explanations have
                # access to the same live sentiment baseline used for display.
                self.metadata["sentiment_daily"] = sentiment_frame
            snapshot_meta = self.metadata.setdefault("sentiment_snapshot", {})
            if sentiment_avg is not None:
                snapshot_meta["average"] = sentiment_avg
            if sentiment_trend is not None:
                snapshot_meta["trend_7d"] = sentiment_trend
            if sentiment_error:
                snapshot_meta["error"] = sentiment_error

        indicator_floor, indicator_components = self._indicator_support_floor()
        residual_error = self._residual_error_scale(resolved_horizon)
        expected_low = self._compute_expected_low(
            close_prediction,
            predicted_volatility,
            quantile_forecasts=quantile_forecasts,
            prediction_intervals=prediction_intervals,
            indicator_floor=indicator_floor,
            residual_error=residual_error,
            anchor_price=anchor_price,
        )

        base_tolerance_band = self.config.resolve_tolerance_band(resolved_horizon)
        intraday_volatility = None
        for candidate in (
            expected_intraday_move,
            predicted_volatility,
            historical_volatility,
        ):
            numeric = self._safe_float(candidate)
            if numeric is None:
                continue
            if np.isfinite(numeric) and numeric != 0:
                intraday_volatility = abs(float(numeric))
                break

        tolerance_band_value = None
        if base_tolerance_band is not None and intraday_volatility is not None:
            scaled_band = float(base_tolerance_band) * float(intraday_volatility)
            if np.isfinite(scaled_band) and scaled_band > 0:
                tolerance_band_value = float(scaled_band)
        tolerance_probability = None
        monte_carlo_target_probability = None
        monte_carlo_iterations = None
        target_price_value = self.metadata.get("target_price")
        if (
            target_price_value is not None
            and price_df is not None
            and resolved_horizon is not None
            and latest_close is not None
            and np.isfinite(latest_close)
        ):
            hist_drift, hist_vol = _historical_drift_volatility(price_df)
            drift_value = hist_drift
            volatility_value = hist_vol
            if drift_value is None and predicted_return is not None and resolved_horizon > 0:
                drift_value = float(predicted_return) / float(resolved_horizon)
            if volatility_value is None and predicted_volatility is not None:
                volatility_value = float(predicted_volatility)

            if drift_value is not None and volatility_value is not None:
                paths = int(self.config.monte_carlo_paths)
                precision_target = getattr(self.config, "monte_carlo_precision", None)
                precision_value = float(precision_target) if precision_target is not None else 0.001
                max_paths_config = getattr(self.config, "monte_carlo_max_paths", None)
                max_paths_value = int(max_paths_config) if max_paths_config is not None else 2_000_000
                min_paths_required = max(
                    1_000_000, int(getattr(self.config, "monte_carlo_min_paths", 0) or 0)
                )
                max_paths_value = max(max_paths_value, min_paths_required)

                recalibration_threshold = self._safe_float(
                    getattr(self.config, "monte_carlo_recalibration_threshold", 0.01)
                )
                if recalibration_threshold is None or recalibration_threshold <= 0:
                    recalibration_threshold = 0.01
                max_iterations_config = getattr(self.config, "monte_carlo_max_iterations", None)
                max_iterations = (
                    int(max_iterations_config)
                    if max_iterations_config is not None
                    else None
                )

                cumulative_paths = 0
                cumulative_hits = 0.0
                cumulative_tolerance_hits: float | None = (
                    0.0
                    if tolerance_band_value is not None
                    and tolerance_band_value > 0
                    and close_prediction is not None
                    else None
                )
                standard_error = None
                expected_close_value = self._safe_float(close_prediction)

                def _material_shift(previous: float | None, updated: float | None) -> bool:
                    if updated is None:
                        return False
                    if previous is None:
                        return True
                    baseline = expected_close_value or latest_close or 1.0
                    baseline = abs(baseline) if baseline else 1.0
                    return abs(updated - previous) / baseline >= recalibration_threshold

                def _recalibrate_volatility(
                    base_volatility: float,
                    close_value: float | None,
                    low_value: float | None,
                    historical_vol: float | None,
                ) -> float:
                    candidates: list[float] = []
                    if base_volatility is not None:
                        candidates.append(float(base_volatility))
                    if historical_vol is not None:
                        candidates.append(float(historical_vol))
                    if close_value is not None and low_value is not None and close_value != 0:
                        implied_volatility = abs(close_value - low_value) / abs(close_value)
                        candidates.append(float(implied_volatility))
                    finite_candidates = [value for value in candidates if np.isfinite(value)]
                    if not finite_candidates:
                        return base_volatility
                    return max(finite_candidates)

                expected_low_value = expected_low
                monte_carlo_iterations = 0

                while cumulative_paths < max_paths_value and (
                    max_iterations is None or monte_carlo_iterations < max_iterations
                ):
                    monte_carlo_iterations += 1
                    remaining_budget = max_paths_value - cumulative_paths
                    initial_paths = min(paths, remaining_budget)
                    (
                        probability,
                        used_paths,
                        iteration_standard_error,
                        tolerance_probability_iteration,
                    ) = run_monte_carlo_adaptive(
                        current_price=float(latest_close),
                        target_price=float(target_price_value),
                        drift=float(drift_value),
                        volatility=float(volatility_value),
                        horizon=int(resolved_horizon),
                        initial_paths=initial_paths,
                        max_paths=remaining_budget,
                        precision_target=float(precision_value),
                        tolerance_center=float(close_prediction)
                        if close_prediction is not None
                        else None,
                        tolerance_fraction=float(tolerance_band_value)
                        if tolerance_band_value is not None
                        else None,
                    )

                    cumulative_hits += probability * used_paths
                    cumulative_paths += used_paths
                    if (
                        cumulative_tolerance_hits is not None
                        and tolerance_probability_iteration is not None
                    ):
                        cumulative_tolerance_hits += tolerance_probability_iteration * used_paths
                    if cumulative_paths <= 0:
                        break

                    monte_carlo_target_probability = float(cumulative_hits) / float(
                        cumulative_paths
                    )
                    standard_error = float(
                        np.sqrt(
                            monte_carlo_target_probability
                            * (1 - monte_carlo_target_probability)
                            / cumulative_paths
                        )
                    )

                    updated_expected_low = self._compute_expected_low(
                        close_prediction,
                        volatility_value,
                        quantile_forecasts=quantile_forecasts,
                        prediction_intervals=prediction_intervals,
                        indicator_floor=indicator_floor,
                        residual_error=residual_error,
                        anchor_price=anchor_price,
                    )
                    updated_expected_close = self._safe_float(predictions.get("close"))
                    if updated_expected_close is not None:
                        expected_close_value = updated_expected_close

                    shift_detected = _material_shift(expected_low_value, updated_expected_low) or _material_shift(
                        expected_close_value, updated_expected_close
                    )
                    if updated_expected_low is not None:
                        expected_low_value = updated_expected_low

                    if shift_detected:
                        updated_drift = _recalibrated_log_drift(
                            expected_close_value, latest_close, resolved_horizon
                        )
                        if updated_drift is not None:
                            drift_value = updated_drift
                        volatility_value = _recalibrate_volatility(
                            float(volatility_value),
                            expected_close_value,
                            expected_low_value,
                            hist_vol,
                        )

                    precision_met = bool(
                        standard_error is not None and standard_error <= float(precision_value)
                    )
                    min_paths_met = cumulative_paths >= min_paths_required
                    if precision_met and min_paths_met and not shift_detected:
                        break

                used_paths = cumulative_paths
                expected_low = expected_low_value
                if cumulative_tolerance_hits is not None and cumulative_paths > 0:
                    tolerance_probability = float(
                        cumulative_tolerance_hits / float(cumulative_paths)
                    )

                event_probabilities["monte_carlo_target_hit"] = {
                    "probability": monte_carlo_target_probability,
                    "paths": used_paths,
                    "drift": float(drift_value),
                    "volatility": float(volatility_value),
                    "method": "geometric_brownian_motion",
                    "iterations": monte_carlo_iterations,
                }
                event_probabilities["monte_carlo_target_hit"]["precision_target"] = float(
                    precision_value
                )
                if standard_error is not None:
                    event_probabilities["monte_carlo_target_hit"]["standard_error"] = float(
                        standard_error
                    )
                event_probabilities["monte_carlo_target_hit"]["min_paths"] = int(
                    min_paths_required
                )
                event_probabilities["monte_carlo_target_hit"][
                    "stabilized_probability"
                ] = monte_carlo_target_probability
                if tolerance_probability is not None:
                    event_probabilities["monte_carlo_within_tolerance"] = {
                        "probability": tolerance_probability,
                        "paths": used_paths,
                        "tolerance_band": float(tolerance_band_value),
                        "method": "geometric_brownian_motion",
                    }

        stop_loss = self._compute_stop_loss(
            close_prediction,
            predicted_volatility,
            expected_low=expected_low,
            residual_error=residual_error,
            anchor_price=anchor_price,
        )

        if expected_low is not None and close_prediction is not None:
            expected_low = min(float(expected_low), float(close_prediction))
        if stop_loss is not None and close_prediction is not None:
            stop_loss = min(float(stop_loss), float(close_prediction))

        (
            predicted_return,
            pct_change,
            close_prediction,
            expected_low,
            stop_loss,
        ) = self._validate_prediction_ranges(
            predicted_return=predicted_return,
            expected_change_pct=pct_change,
            predicted_close=close_prediction,
            expected_low=expected_low,
            stop_loss=stop_loss,
            anchor_price=anchor_price,
            horizon=resolved_horizon,
            residual_error=residual_error,
            prediction_warnings=prediction_warnings,
        )

        beta_metrics, beta_notes = self._beta_context()

        uncertainty_clean: dict[str, Dict[str, float]] = {}
        for tgt, values in uncertainties.items():
            numeric_values = {
                key: float(value)
                for key, value in values.items()
                if value is not None and np.isfinite(value)
            }
            if numeric_values:
                uncertainty_clean[tgt] = numeric_values

        def _parse_quantile_key(key: Any) -> Optional[float]:
            if isinstance(key, (int, float)):
                return float(key) if float(key) <= 1 else float(key) / 100.0
            if isinstance(key, str):
                cleaned = key.strip().lower()
                if cleaned in {"low", "lower", "p10", "q10"}:
                    return 0.1
                if cleaned in {"mid", "median", "p50", "q50"}:
                    return 0.5
                if cleaned in {"high", "upper", "p90", "q90"}:
                    return 0.9
                cleaned = cleaned.removeprefix("quantile_")
                cleaned = cleaned.removeprefix("q").removeprefix("p")
                if cleaned.endswith("%"):
                    cleaned = cleaned[:-1]
                    numeric = self._safe_float(cleaned)
                    return numeric / 100.0 if numeric is not None else None
                numeric = self._safe_float(cleaned)
                if numeric is not None and numeric > 1:
                    numeric = numeric / 100.0
                return numeric
            return None

        def _extract_quantiles(block: Mapping[str, Any] | None) -> dict[float, float]:
            if not isinstance(block, Mapping):
                return {}
            parsed: dict[float, float] = {}
            for key, raw in block.items():
                numeric = _parse_quantile_key(key)
                if numeric is None:
                    continue
                value = self._safe_float(raw)
                if value is not None:
                    parsed[float(numeric)] = float(value)
            return parsed

        close_quantiles = _extract_quantiles(quantile_forecasts.get("close"))
        close_q10 = close_quantiles.get(0.1)
        close_q50 = close_quantiles.get(0.5)
        close_q90 = close_quantiles.get(0.9)
        if close_q50 is None and close_prediction is not None:
            close_q50 = float(close_prediction)

        std_value = None
        std_candidate = uncertainty_clean.get("close", {}).get("std")
        if std_candidate is not None and np.isfinite(std_candidate):
            std_value = float(std_candidate)

        z10 = -1.281551565545
        z90 = 1.281551565545
        if std_value is None and close_q10 is not None and close_q90 is not None:
            diff = close_q90 - close_q10
            if np.isfinite(diff) and diff != 0:
                std_value = float(diff) / (z90 - z10)

        if std_value is not None and close_q50 is not None:
            if close_q10 is None:
                close_q10 = float(close_q50 + z10 * std_value)
            if close_q90 is None:
                close_q90 = float(close_q50 + z90 * std_value)

        pred_low_q10 = close_q10
        if expected_low is not None:
            if pred_low_q10 is None or expected_low < pred_low_q10:
                pred_low_q10 = float(expected_low)
        pred_high_q90 = close_q90

        sigma_price = None
        if std_value is not None and std_value > 0:
            sigma_price = float(abs(std_value))
        elif predicted_volatility is not None and anchor_price is not None:
            sigma_price = float(abs(predicted_volatility)) * float(anchor_price)

        range_1sigma = None
        if sigma_price is not None and anchor_price is not None:
            range_1sigma = {
                "low": float(anchor_price) - sigma_price,
                "high": float(anchor_price) + sigma_price,
            }

        def _normal_cdf(value: float) -> float:
            return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))

        hit_levels = getattr(self.config, "hit_probability_levels", None)
        if not hit_levels:
            hit_levels = [0.01, 0.02, 0.05]
        p_hit_up: dict[str, float] = {}
        p_hit_down: dict[str, float] = {}
        if (
            sigma_price is not None
            and sigma_price > 0
            and anchor_price is not None
            and close_q50 is not None
        ):
            for level in hit_levels:
                try:
                    level_value = float(level)
                except (TypeError, ValueError):
                    continue
                if level_value <= 0:
                    continue
                up_target = float(anchor_price) * (1 + level_value)
                down_target = float(anchor_price) * (1 - level_value)
                z_up = (up_target - float(close_q50)) / sigma_price
                z_down = (down_target - float(close_q50)) / sigma_price
                p_hit_up[str(level_value)] = float(1.0 - _normal_cdf(z_up))
                p_hit_down[str(level_value)] = float(_normal_cdf(z_down))

        calibration_block = None
        if isinstance(getattr(self, "metadata", None), Mapping):
            metrics_store = self.metadata.get("metrics")
            if isinstance(metrics_store, Mapping):
                target_metrics = metrics_store.get("close")
                if isinstance(target_metrics, Mapping):
                    calibration_entry = None
                    if resolved_horizon is not None:
                        calibration_entry = target_metrics.get(resolved_horizon)
                    if calibration_entry is None:
                        calibration_entry = next(iter(target_metrics.values()), None)
                    if isinstance(calibration_entry, Mapping):
                        calibration_block = calibration_entry.get("residual_calibration")

        uncertainty_calibrated = isinstance(calibration_block, Mapping)
        calibration_version = "residual-linear-v1" if uncertainty_calibrated else None

        def _to_iso(value: Any) -> Optional[str]:
            return self._format_timestamp(value)

        confluence_scaled = False

        if confluence_assessment is not None:
            confluence_score = float(confluence_assessment.score)
            confluence_passed = bool(confluence_assessment.passed)
            confluence_block = {
                "passed": confluence_passed,
                "score": confluence_score,
                "components": dict(confluence_assessment.components),
            }

        regime_shift_risk = False
        if confluence_score is not None and confluence_passed is False:
            if confluence_score < 0.4:
                regime_shift_risk = True

        base_confidence = None
        if confidences:
            numeric_conf = [
                self._safe_float(value)
                for value in confidences.values()
                if self._safe_float(value) is not None
            ]
            if numeric_conf:
                base_confidence = max(numeric_conf)
        if base_confidence is None and direction_probability_up is not None:
            base_confidence = max(
                value
                for value in (direction_probability_up, direction_probability_down)
                if value is not None
            )
        if base_confidence is not None:
            combined_confidence = float(base_confidence)
            if confluence_score is not None:
                combined_confidence *= max(0.0, min(1.0, confluence_score))
            if confluence_passed is False:
                combined_confidence *= 0.5
                confluence_scaled = True
                confidence_notes.append(
                    "Confluence checks failed; probability scaled instead of hidden."
                )
            combined_confidence = float(combined_confidence)

        combined_confidence, trend_alignment_note = self._apply_trend_alignment_adjustment(
            combined_confidence
        )

        (
            combined_confidence,
            direction_probability_up,
            direction_probability_down,
            sentiment_factor,
        ) = self._apply_sentiment_adjustment(
            combined_confidence,
            direction_probability_up,
            direction_probability_down,
        )

        (
            direction_probability_up,
            direction_probability_down,
        ) = self._refine_direction_probability(
            direction_probability_up,
            direction_probability_down,
            predicted_return,
            predicted_volatility,
            resolved_horizon,
        )

        if sentiment_factor is not None and isinstance(probabilities.get("direction"), dict):
            probabilities["direction"] = {
                "up": direction_probability_up,
                "down": direction_probability_down,
            }

        feature_groups_meta: Mapping[str, Any] | None = None
        executed_feature_groups: list[str] | None = None
        feature_toggle_flags: Mapping[str, bool] | None = None
        feature_vectors_by_group: dict[str, list[str]] = {}
        feature_usage_summary: list[FeatureUsageSummary] = []
        indicator_columns: list[str] = []
        if isinstance(self.metadata, Mapping):
            feature_groups_meta = self.metadata.get("feature_groups")
            executed_feature_groups = self.metadata.get("executed_feature_groups")
            indicator_source = self.metadata.get("indicator_columns")
            if isinstance(indicator_source, (list, tuple, set)):
                indicator_columns = [str(name) for name in indicator_source]

        feature_toggle_source = getattr(self.config, "feature_toggles", None)
        if isinstance(feature_toggle_source, FeatureToggles):
            feature_toggle_flags = feature_toggle_source.asdict()
        elif isinstance(feature_toggle_source, Mapping):
            feature_toggle_flags = {
                str(name): bool(value)
                for name, value in feature_toggle_source.items()
                if str(name).strip()
            }

        if not executed_feature_groups:
            executed_feature_groups = self.get_used_feature_groups()

        if isinstance(feature_groups_meta, Mapping):
            for name, summary in feature_groups_meta.items():
                if not isinstance(summary, Mapping):
                    continue
                columns = summary.get("columns")
                if isinstance(columns, (list, tuple, set)):
                    feature_vectors_by_group[name] = [str(col) for col in columns]

            if not indicator_columns:
                technical_summary = feature_groups_meta.get("technical")
                if isinstance(technical_summary, Mapping):
                    columns = technical_summary.get("indicator_columns")
                    if isinstance(columns, (list, tuple, set)):
                        indicator_columns = [str(col) for col in columns]

        if executed_feature_groups:
            for name in executed_feature_groups:
                columns = feature_vectors_by_group.get(name) or []
                signal_count = len(columns)
                if not signal_count and name == "technical" and indicator_columns:
                    signal_count = len(indicator_columns)
                feature_usage_summary.append(
                    FeatureUsageSummary(name=str(name), count=int(signal_count))
                )

        executed_set = {
            str(name).strip() for name in executed_feature_groups or [] if str(name).strip()
        }
        if feature_toggle_flags is not None:
            for name, enabled in feature_toggle_flags.items():
                label = str(name).strip()
                if not label:
                    continue
                if enabled and label not in executed_set:
                    prediction_warnings.append(
                        f"Feature group '{label}' was enabled but not reported in executed_feature_groups."
                    )
                if not enabled and label in executed_set:
                    prediction_warnings.append(
                        f"Feature group '{label}' executed despite being disabled via feature_toggles."
                    )

        latest_features_snapshot: list[dict[str, Any]] | None = None
        if isinstance(getattr(self, "metadata", None), Mapping):
            raw_snapshot = self.metadata.get("latest_features")
            if isinstance(raw_snapshot, pd.DataFrame) and not raw_snapshot.empty:
                latest_features_snapshot = raw_snapshot.head(1).to_dict(orient="records")

        source_descriptions = self._render_source_descriptions(
            getattr(self, "metadata", {}).get("data_sources")
        )

        training_accuracy: dict[str, Any] | None = None
        try:
            training_accuracy = self.accuracy_summary(horizon=resolved_horizon)
        except Exception as exc:  # pragma: no cover - defensive guard
            LOGGER.debug("Failed to compute accuracy summary: %s", exc)

        trade_gate = self._build_trade_gate(
            direction_probability_up,
            direction_probability_down,
            direction_probability_neutral,
            monte_carlo_target_probability,
        )

        result = {
            "ticker": self.config.ticker,
            "as_of": _to_iso(market_data_as_of) or "",
            "market_data_as_of": _to_iso(market_data_as_of) or "",
            "generated_at": _to_iso(prediction_timestamp) or "",
            "prediction_timestamp": _to_iso(prediction_timestamp) or "",
            "market_session": market_session,
            "last_close": latest_close,
            "last_price": last_price_value,
            "anchor_price": anchor_price,
            "anchor_price_source": anchor_price_source,
            "predicted_close": close_prediction,
            "pred_close_q10": close_q10,
            "pred_close_q50": close_q50,
            "pred_close_q90": close_q90,
            "pred_low_q10": pred_low_q10,
            "pred_high_q90": pred_high_q90,
            "expected_change": expected_change,
            "expected_change_pct": pct_change,
            "expected_intraday_move": expected_intraday_move,
            "predicted_return": predicted_return,
            "predicted_volatility": predicted_volatility,
            "expected_low": expected_low,
            "stop_loss": stop_loss,
            "range_1sigma": range_1sigma,
            "probability_within_tolerance": tolerance_probability,
            "tolerance_band": tolerance_band_value,
            "direction_probability_up": direction_probability_up,
            "direction_probability_down": direction_probability_down,
            "direction_probability_neutral": direction_probability_neutral,
            "target_hit_probability": target_hit_probability,
            "monte_carlo_target_hit_probability": monte_carlo_target_probability,
            "monte_carlo_stabilized_probability": monte_carlo_target_probability,
            "monte_carlo_iterations": monte_carlo_iterations,
            "p_hit_up": p_hit_up or None,
            "p_hit_down": p_hit_down or None,
            "hit_probabilities": {"up": p_hit_up, "down": p_hit_down} if p_hit_up or p_hit_down else None,
            "close_quantiles": {
                "q10": close_q10,
                "q50": close_q50,
                "q90": close_q90,
            }
            if close_q10 is not None or close_q50 is not None or close_q90 is not None
            else None,
            "uncertainty_calibrated": uncertainty_calibrated,
            "calibration_version": calibration_version,
            "regime_shift_risk": regime_shift_risk,
            "trade_gate": trade_gate,
            "training_accuracy": training_accuracy,
            "target_price": self.metadata.get("target_price"),
            "feature_toggles": dict(getattr(self.config, "feature_toggles", {})),
            "predictions": predictions,
            "horizon": resolved_horizon,
            "target_date": _to_iso(target_date) or "",
            "status": status,
            "reason": reason,
            "message": message,
            "sample_counts": modern_sample_counts,
            "missing_targets": modern_missing_targets,
            "metrics": modern_metrics,
            "checked_at": _to_iso(modern_checked_at) if modern_checked_at else None,
        }
        if latest_features_snapshot:
            result["latest_features_snapshot"] = latest_features_snapshot
        if source_descriptions:
            result["data_sources"] = source_descriptions
        if isinstance(getattr(self, "metadata", None), Mapping):
            validation_block = self.metadata.get("target_validation")
            if validation_block:
                result["target_validation"] = validation_block
        trailing_low_value = None
        drawdown_fraction = None
        if feature_usage_summary:
            result["feature_usage_summary"] = [
                summary.to_dict() for summary in feature_usage_summary
            ]
        if isinstance(getattr(self, "metadata", None), Mapping):
            trailing_low_value = self._safe_float(self.metadata.get("trailing_low"))
            drawdown_fraction = self._safe_float(self.metadata.get("max_drawdown_fraction"))
        if trailing_low_value is not None:
            result["trailing_low"] = trailing_low_value
            trailing_window = getattr(self, "metadata", {}).get("trailing_low_window")
            if trailing_window is not None:
                result["trailing_low_window"] = trailing_window
        if drawdown_fraction is not None:
            result["max_drawdown_fraction"] = drawdown_fraction
        if isinstance(getattr(self, "metadata", None), Mapping):
            expected_low_return_std = self._safe_float(
                self.metadata.get("expected_low_return_std")
            )
            expected_low_return_window = self.metadata.get("expected_low_return_window")
            if expected_low_return_std is not None:
                result["expected_low_return_std"] = expected_low_return_std
            if expected_low_return_window is not None:
                result["expected_low_return_window"] = expected_low_return_window
        if confluence_block:
            result["signal_confluence"] = confluence_block
        if combined_confidence is not None:
            result["confluence_confidence"] = combined_confidence
        if confluence_scaled:
            result["confluence_scaled"] = True
        if historical_confidence is not None:
            result["historical_confidence"] = historical_confidence
        if trend_alignment_note:
            result["trend_alignment_note"] = trend_alignment_note
        if sentiment_factor is not None:
            result["sentiment_factor"] = sentiment_factor
        if indicator_floor is not None:
            result["indicator_expected_low"] = indicator_floor
        if indicator_components:
            result["indicator_support_components"] = indicator_components
        if confidences:
            result["confidence"] = confidences
        if feature_groups_meta:
            result["feature_groups"] = feature_groups_meta
        if feature_vectors_by_group:
            result["feature_vectors_by_group"] = feature_vectors_by_group
        if executed_feature_groups:
            result["executed_feature_groups"] = list(executed_feature_groups)
        if indicator_columns:
            result["indicator_columns"] = indicator_columns
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
        if beta_metrics:
            result["beta_metrics"] = beta_metrics
        if beta_notes:
            result["beta_warnings"] = beta_notes
        if training_report:
            result["training_metrics"] = training_report
        if prediction_warnings:
            result["warnings"] = prediction_warnings
        if confidence_notes:
            result["confidence_note"] = "; ".join(confidence_notes)
        if sentiment_avg is not None:
            result["Sentiment_Avg"] = sentiment_avg
            result["sentiment_score"] = sentiment_avg
        if sentiment_trend is not None:
            result["Sentiment_7d"] = sentiment_trend
        if sentiment_error:
            result["sentiment_error"] = sentiment_error
        explanation = self._build_prediction_explanation(result, predictions)
        if explanation:
            result["explanation"] = explanation
        recommendation = self._generate_recommendation(result)
        if recommendation:
            result["recommendation"] = recommendation
        meta = dict(result)
        meta.pop("used_feature_groups", None)
        return PredictionResult(
            predicted_close=close_prediction,
            expected_low=expected_low,
            stop_loss=stop_loss,
            status=status,
            reason=reason,
            message=message,
            sample_counts=modern_sample_counts,
            missing_targets=modern_missing_targets,
            feature_groups_used=list(executed_feature_groups or []),
            indicators_used=indicator_columns,
            feature_usage_summary=feature_usage_summary,
            meta=meta,
        )

    def _prepare_features_for_model(
        self, model: Any, features: pd.DataFrame | np.ndarray
    ) -> pd.DataFrame | np.ndarray:
        """Return feature matrix in a format compatible with *model*."""

        if not isinstance(features, pd.DataFrame):
            return features

        def _feature_names(candidate: Any) -> Sequence[str] | None:
            if candidate is None:
                return None
            return getattr(candidate, "feature_names_in_", None)

        estimator = None
        if hasattr(model, "named_steps") and isinstance(model.named_steps, Mapping):
            estimator = model.named_steps.get("estimator")

        model_features = _feature_names(model)
        estimator_features = None if model_features is not None else _feature_names(estimator)
        expected_features = model_features if model_features is not None else estimator_features
        if expected_features is not None:
            return features.reindex(columns=expected_features, fill_value=0)

        return features.to_numpy()

    def _residual_error_scale(self, horizon: Optional[int]) -> Optional[float]:
        """Return a typical absolute error scale derived from backtests.

        The method prefers RMSE, falling back to MAE, and ignores non-positive
        values so downstream consumers can bail out gracefully when no
        historical evaluation is available.
        """

        metrics = self._validation_metrics("close", horizon)
        if not metrics:
            return None

        calibration_block = metrics.get("residual_calibration")
        if isinstance(calibration_block, Mapping):
            band = self._safe_float(calibration_block.get("residual_std"))
            if band is not None and band > 0:
                return band

        for key in ("rmse", "mae"):
            value = self._safe_float(metrics.get(key))
            if value is not None and value > 0:
                return float(value)

        return None

    def _apply_residual_correction(self, model: Any, prediction: float) -> float:
        calibrator = getattr(model, "residual_calibrator_", None)
        if calibrator is None:
            return prediction

        try:
            correction = calibrator.predict(np.asarray([[prediction]], dtype=float))
        except Exception:  # pragma: no cover - defensive
            return prediction

        try:
            correction_value = float(np.ravel(correction)[0])
        except Exception:
            correction_value = 0.0
        return float(prediction - correction_value)

    @staticmethod
    def _compound_log_drift(drift_per_step: float, horizon: float) -> tuple[float, float]:
        """Convert per-step log drift into compounded return and growth factor."""

        drift_exponent = drift_per_step * horizon
        compounded_return = math.expm1(drift_exponent)
        growth_factor = math.exp(drift_exponent)
        return compounded_return, growth_factor

    @staticmethod
    def _clamp_with_uncertainty(
        value: Optional[float],
        quantiles: Mapping[str, float] | None,
        intervals: Mapping[str, float] | None,
    ) -> Optional[float]:
        if value is None:
            return None

        lower_bounds: list[float] = []
        upper_bounds: list[float] = []

        if isinstance(quantiles, Mapping):
            for key, raw in quantiles.items():
                numeric = None
                try:
                    numeric = float(key)
                except (TypeError, ValueError):
                    if isinstance(key, str) and key.lower().startswith("q"):
                        try:
                            numeric = float(key[1:]) / 100 if "%" not in key else None
                        except (TypeError, ValueError):
                            numeric = None
                value_float = None
                try:
                    value_float = float(raw)
                except (TypeError, ValueError):
                    continue
                if numeric is None:
                    continue
                if numeric <= 0.5:
                    lower_bounds.append(value_float)
                if numeric >= 0.5:
                    upper_bounds.append(value_float)

        if isinstance(intervals, Mapping):
            for key, raw in intervals.items():
                try:
                    numeric = float(raw)
                except (TypeError, ValueError):
                    continue
                key_lower = str(key).lower()
                if any(token in key_lower for token in ["lower", "low", "min"]):
                    lower_bounds.append(numeric)
                if any(token in key_lower for token in ["upper", "high", "max"]):
                    upper_bounds.append(numeric)

        lower = max(lower_bounds) if lower_bounds else None
        upper = min(upper_bounds) if upper_bounds else None

        clamped = value
        if lower is not None:
            clamped = max(clamped, lower)
        if upper is not None:
            clamped = min(clamped, upper)
        return clamped

    def _resolve_range_anchors(
        self,
        anchor_price: float | None,
    ) -> dict[str, float | None]:
        atr_value: float | None = None
        implied_volatility: float | None = None
        latest_features = None
        if isinstance(getattr(self, "metadata", None), Mapping):
            atr_value = self._safe_float(
                self.metadata.get("atr")
                or self.metadata.get("atr_14")
                or self.metadata.get("average_true_range")
            )
            implied_volatility = self._safe_float(
                self.metadata.get("implied_volatility")
                or self.metadata.get("iv")
                or self.metadata.get("options_implied_volatility")
            )
            latest_features = self.metadata.get("latest_features")

        if atr_value is None and isinstance(latest_features, pd.DataFrame) and not latest_features.empty:
            target_period = getattr(self.config, "expected_low_atr_period", DEFAULT_EXPECTED_LOW_ATR_PERIOD)
            try:
                target_period = int(target_period)
            except (TypeError, ValueError):
                target_period = DEFAULT_EXPECTED_LOW_ATR_PERIOD
            atr_columns = [col for col in latest_features.columns if "atr" in str(col).lower()]
            if atr_columns:
                scored: list[tuple[int, str]] = []
                for column in atr_columns:
                    match = re.search(r"atr[_\s-]?(\d+)", str(column).lower())
                    if match:
                        period = int(match.group(1))
                    else:
                        period = target_period
                    scored.append((abs(period - target_period), str(column)))
                selected_column = sorted(scored, key=lambda item: item[0])[0][1]
                atr_value = self._safe_float(latest_features.iloc[0][selected_column])

        if implied_volatility is not None:
            implied_volatility = abs(float(implied_volatility))
            if implied_volatility > 1.0 and implied_volatility <= 100:
                implied_volatility /= 100.0

        atr_fraction = None
        iv_fraction = None
        iv_price_sigma = None
        if atr_value is not None and anchor_price is not None and anchor_price > 0:
            atr_fraction = abs(float(atr_value)) / float(anchor_price)
        iv_rule = getattr(
            self.config, "implied_volatility_rule_of_16", DEFAULT_IMPLIED_VOLATILITY_RULE_OF_16
        )
        try:
            iv_rule_value = float(iv_rule)
        except (TypeError, ValueError):
            iv_rule_value = DEFAULT_IMPLIED_VOLATILITY_RULE_OF_16
        if not np.isfinite(iv_rule_value) or iv_rule_value <= 0:
            iv_rule_value = DEFAULT_IMPLIED_VOLATILITY_RULE_OF_16

        if implied_volatility is not None:
            iv_fraction = implied_volatility / iv_rule_value
            if anchor_price is not None and anchor_price > 0:
                iv_price_sigma = float(anchor_price) * iv_fraction

        return {
            "atr_value": atr_value,
            "atr_fraction": atr_fraction,
            "iv_fraction": iv_fraction,
            "iv_price_sigma": iv_price_sigma,
        }

    def _compute_expected_low(
        self,
        predicted_close: Any,
        predicted_volatility: Any,
        *,
        quantile_forecasts: Mapping[str, Dict[str, float]] | None,
        prediction_intervals: Mapping[str, Dict[str, float]] | None,
        indicator_floor: float | None = None,
        residual_error: float | None = None,
        anchor_price: float | None = None,
        pct_hint: float | None = None,
    ) -> Optional[float]:
        """Estimate a conservative lower bound for the forecasted close.

        ``predicted_volatility`` is expected as a fractional value (e.g., ``0.02``
        for 2%). Values outside ``[0, 1]`` will be interpreted defensively and
        clipped to guard against percentage-scale inputs.
        """

        def _extract_lower_bound(block: Mapping[str, Any] | None) -> Optional[float]:
            if not isinstance(block, Mapping):
                return None
            preferred_keys = (
                "lower",
                "lower_bound",
                "low",
                "p10",
                "10%",
                "10",
                "0.1",
                "quantile_0.1",
            )
            for key in preferred_keys:
                if key in block:
                    candidate = self._safe_float(block.get(key))
                    if candidate is not None:
                        return candidate
            numeric_values: list[float] = []
            for value in block.values():
                numeric = self._safe_float(value)
                if numeric is not None:
                    numeric_values.append(numeric)
            if numeric_values:
                return float(min(numeric_values))
            return None

        if isinstance(prediction_intervals, Mapping) and prediction_intervals:
            close_interval = prediction_intervals.get("close")
            interval_value = _extract_lower_bound(close_interval)
            if interval_value is None:
                for candidate in prediction_intervals.values():
                    interval_value = _extract_lower_bound(candidate)
                    if interval_value is not None:
                        break
            if interval_value is not None:
                return interval_value

        if isinstance(quantile_forecasts, Mapping) and quantile_forecasts:
            close_quantiles = quantile_forecasts.get("close")
            quantile_value = _extract_lower_bound(close_quantiles)
            if quantile_value is None:
                for candidate in quantile_forecasts.values():
                    quantile_value = _extract_lower_bound(candidate)
                    if quantile_value is not None:
                        break
            if quantile_value is not None:
                return quantile_value

        floor_candidates: list[float] = []
        indicator_value = self._safe_float(indicator_floor)
        if indicator_value is not None and indicator_value > 0:
            floor_candidates.append(indicator_value)
        trailing_low = None
        if isinstance(getattr(self, "metadata", None), Mapping):
            trailing_low = self.metadata.get("trailing_low")
        trailing_low_value = self._safe_float(trailing_low)
        if trailing_low_value is not None and trailing_low_value > 0:
            floor_candidates.append(trailing_low_value)
        floor_value = max(floor_candidates) if floor_candidates else None

        anchor_value = self._safe_float(anchor_price)
        numeric_close = self._safe_float(predicted_close)
        if numeric_close is None:
            numeric_close = anchor_value
        volatility_value = self._safe_float(predicted_volatility)
        range_anchors = self._resolve_range_anchors(anchor_value)
        atr_fraction = range_anchors.get("atr_fraction")
        iv_fraction = range_anchors.get("iv_fraction")
        atr_multiplier = getattr(
            self.config, "expected_low_atr_multiplier", DEFAULT_EXPECTED_LOW_ATR_MULTIPLIER
        )
        iv_multiplier = getattr(
            self.config, "expected_low_iv_multiplier", DEFAULT_EXPECTED_LOW_IV_MULTIPLIER
        )
        range_floor_candidates: list[float] = []
        if atr_fraction is not None:
            try:
                atr_multiplier_value = float(atr_multiplier)
            except (TypeError, ValueError):
                atr_multiplier_value = DEFAULT_EXPECTED_LOW_ATR_MULTIPLIER
            if np.isfinite(atr_multiplier_value) and atr_multiplier_value > 0:
                range_floor_candidates.append(atr_fraction * atr_multiplier_value)
        if iv_fraction is not None:
            try:
                iv_multiplier_value = float(iv_multiplier)
            except (TypeError, ValueError):
                iv_multiplier_value = DEFAULT_EXPECTED_LOW_IV_MULTIPLIER
            if np.isfinite(iv_multiplier_value) and iv_multiplier_value > 0:
                range_floor_candidates.append(iv_fraction * iv_multiplier_value)
        range_floor = max(range_floor_candidates) if range_floor_candidates else None

        if numeric_close is None:
            return floor_value

        if residual_error is not None:
            error_mag = abs(float(residual_error))
            if anchor_value is not None and anchor_value > 0:
                error_mag = min(error_mag, float(anchor_value) * 0.3)
            candidate = max(0.0, float(numeric_close) - error_mag)
            if floor_value is not None:
                candidate = max(candidate, floor_value)
            return float(min(candidate, numeric_close))

        pct_hint_value = self._safe_float(pct_hint)
        if pct_hint_value is not None and anchor_value is not None:
            drop_fraction = abs(float(pct_hint_value))
            candidate = float(anchor_value * (1 - drop_fraction))
            candidate = max(0.0, candidate)
            if floor_value is not None:
                candidate = max(candidate, floor_value)
            return float(min(candidate, numeric_close))

        if volatility_value is None:
            if range_floor is None:
                base_low = float(numeric_close)
                if floor_value is not None:
                    base_low = max(base_low, floor_value)
                return float(min(base_low, numeric_close))
            volatility_value = range_floor

        volatility_pct = abs(float(volatility_value))
        if volatility_pct > 1.0 and volatility_pct <= 100:
            LOGGER.warning(
                "Predicted volatility %.4f interpreted as percentage; expected fractional (e.g., 0.02).",
                volatility_pct,
            )
            volatility_pct /= 100.0

        max_volatility = self._safe_float(
            getattr(self.config, "expected_low_max_volatility", DEFAULT_EXPECTED_LOW_MAX_VOLATILITY)
        )
        if max_volatility is None or max_volatility <= 0:
            max_volatility = DEFAULT_EXPECTED_LOW_MAX_VOLATILITY

        historical_cap = None
        if isinstance(getattr(self, "metadata", None), Mapping):
            historical_cap = self._safe_float(self.metadata.get("max_drawdown_fraction"))
        cap_candidates = [value for value in (max_volatility, historical_cap) if value and value > 0]
        volatility_cap = min(cap_candidates) if cap_candidates else max_volatility

        expected_low_std_window = getattr(self.config, "expected_low_std_window", 0)
        expected_low_std_cap = getattr(self.config, "expected_low_std_cap", 0.0)
        try:
            expected_low_std_window = int(expected_low_std_window)
        except (TypeError, ValueError):
            expected_low_std_window = 0
        try:
            expected_low_std_cap = float(expected_low_std_cap)
        except (TypeError, ValueError):
            expected_low_std_cap = 0.0
        std_cap_value = None
        if expected_low_std_window > 0 and expected_low_std_cap > 0:
            std_value = None
            if isinstance(getattr(self, "metadata", None), Mapping):
                std_value = self._safe_float(self.metadata.get("expected_low_return_std"))
                if std_value is None:
                    std_value = self._safe_float(self.metadata.get("recent_return_std"))
                if std_value is None:
                    std_value = self._safe_float(self.metadata.get("daily_log_return_std"))
            if std_value is not None and std_value > 0:
                std_cap_value = abs(float(std_value)) * expected_low_std_cap
                if std_cap_value > 0:
                    volatility_cap = min(volatility_cap, std_cap_value)

        if volatility_pct > volatility_cap:
            LOGGER.warning(
                "Predicted volatility %.4f exceeds cap %.4f; clipping.",
                volatility_pct,
                volatility_cap,
            )
            volatility_pct = volatility_cap
        if range_floor is not None and volatility_pct < range_floor:
            LOGGER.warning(
                "Predicted volatility %.4f below range floor %.4f; using floor.",
                volatility_pct,
                range_floor,
            )
            volatility_pct = range_floor

        multiplier = getattr(self.config, "expected_low_sigma", DEFAULT_EXPECTED_LOW_SIGMA)
        try:
            multiplier_value = float(multiplier)
        except (TypeError, ValueError):
            multiplier_value = DEFAULT_EXPECTED_LOW_SIGMA
        if not np.isfinite(multiplier_value) or multiplier_value <= 0:
            multiplier_value = DEFAULT_EXPECTED_LOW_SIGMA
        delta = float(numeric_close) * volatility_pct * multiplier_value
        if not np.isfinite(delta):
            delta = 0.0
        expected_low = float(numeric_close - delta)
        expected_low = max(0.0, expected_low)
        if floor_value is not None:
            expected_low = max(expected_low, floor_value)
        return float(min(expected_low, numeric_close))

    def _compute_stop_loss(
        self,
        predicted_close: Any,
        predicted_volatility: Any,
        *,
        expected_low: Any = None,
        residual_error: float | None = None,
        anchor_price: float | None = None,
        pct_hint: float | None = None,
    ) -> Optional[float]:
        """Derive a stop-loss level using the forecasted volatility."""

        anchor_value = self._safe_float(anchor_price)
        predicted_close_value = self._safe_float(predicted_close)
        numeric_close = predicted_close_value
        if numeric_close is None:
            numeric_close = anchor_value
        fallback_low = self._safe_float(expected_low)
        if numeric_close is None:
            return fallback_low

        if fallback_low is not None:
            volatility_sigma = self._safe_float(predicted_volatility)
            multiplier = getattr(self.config, "k_stop", DEFAULT_STOP_LOSS_MULTIPLIER)
            try:
                multiplier_value = float(multiplier)
            except (TypeError, ValueError):
                multiplier_value = DEFAULT_STOP_LOSS_MULTIPLIER
            if not np.isfinite(multiplier_value) or multiplier_value <= 0:
                multiplier_value = DEFAULT_STOP_LOSS_MULTIPLIER

            range_anchor_price = anchor_value if anchor_value is not None else numeric_close
            range_anchors = self._resolve_range_anchors(range_anchor_price)
            atr_value = range_anchors.get("atr_value")
            iv_price_sigma = range_anchors.get("iv_price_sigma")
            atr_multiplier = getattr(
                self.config, "stop_loss_atr_multiplier", DEFAULT_STOP_LOSS_ATR_MULTIPLIER
            )
            iv_multiplier = getattr(
                self.config, "stop_loss_iv_multiplier", DEFAULT_STOP_LOSS_IV_MULTIPLIER
            )
            buffer_candidates: list[float] = []
            if volatility_sigma is not None:
                sigma = abs(float(volatility_sigma))
                if np.isfinite(sigma) and sigma > 0:
                    buffer_candidates.append(sigma * multiplier_value)
            if atr_value is not None:
                try:
                    atr_multiplier_value = float(atr_multiplier)
                except (TypeError, ValueError):
                    atr_multiplier_value = DEFAULT_STOP_LOSS_ATR_MULTIPLIER
                if np.isfinite(atr_multiplier_value) and atr_multiplier_value > 0:
                    buffer_candidates.append(abs(float(atr_value)) * atr_multiplier_value)
            if iv_price_sigma is not None:
                try:
                    iv_multiplier_value = float(iv_multiplier)
                except (TypeError, ValueError):
                    iv_multiplier_value = DEFAULT_STOP_LOSS_IV_MULTIPLIER
                if np.isfinite(iv_multiplier_value) and iv_multiplier_value > 0:
                    buffer_candidates.append(float(iv_price_sigma) * iv_multiplier_value)
            buffer = max(buffer_candidates) if buffer_candidates else 0.0
            stop_loss = float(fallback_low) - buffer

            stop_loss = max(0.0, stop_loss)
            if predicted_close_value is not None:
                stop_loss = min(stop_loss, float(predicted_close_value))
            return float(stop_loss)

        if residual_error is not None:
            delta = abs(float(residual_error)) * 1.5
            if anchor_value is not None and anchor_value > 0:
                delta = min(delta, float(anchor_value) * 0.45)
            stop_loss = float(numeric_close - delta)
            stop_loss = max(0.0, stop_loss)
            return float(min(stop_loss, numeric_close))

        pct_hint_value = self._safe_float(pct_hint)
        if pct_hint_value is not None and anchor_value is not None:
            stop_loss = max(0.0, float(anchor_value * (1 - abs(float(pct_hint_value)))))
            return float(min(stop_loss, numeric_close))

        volatility_pct = self._safe_float(predicted_volatility)
        range_anchor_price = anchor_value if anchor_value is not None else numeric_close
        range_anchors = self._resolve_range_anchors(range_anchor_price)
        atr_fraction = range_anchors.get("atr_fraction")
        iv_fraction = range_anchors.get("iv_fraction")
        atr_multiplier = getattr(
            self.config, "stop_loss_atr_multiplier", DEFAULT_STOP_LOSS_ATR_MULTIPLIER
        )
        iv_multiplier = getattr(
            self.config, "stop_loss_iv_multiplier", DEFAULT_STOP_LOSS_IV_MULTIPLIER
        )
        range_floor_candidates: list[float] = []
        if atr_fraction is not None:
            try:
                atr_multiplier_value = float(atr_multiplier)
            except (TypeError, ValueError):
                atr_multiplier_value = DEFAULT_STOP_LOSS_ATR_MULTIPLIER
            if np.isfinite(atr_multiplier_value) and atr_multiplier_value > 0:
                range_floor_candidates.append(atr_fraction * atr_multiplier_value)
        if iv_fraction is not None:
            try:
                iv_multiplier_value = float(iv_multiplier)
            except (TypeError, ValueError):
                iv_multiplier_value = DEFAULT_STOP_LOSS_IV_MULTIPLIER
            if np.isfinite(iv_multiplier_value) and iv_multiplier_value > 0:
                range_floor_candidates.append(iv_fraction * iv_multiplier_value)
        range_floor = max(range_floor_candidates) if range_floor_candidates else None
        if volatility_pct is None:
            volatility_pct = range_floor
            if volatility_pct is None:
                return None

        multiplier = getattr(self.config, "k_stop", DEFAULT_STOP_LOSS_MULTIPLIER)
        try:
            multiplier_value = float(multiplier)
        except (TypeError, ValueError):
            multiplier_value = DEFAULT_STOP_LOSS_MULTIPLIER
        if not np.isfinite(multiplier_value) or multiplier_value <= 0:
            multiplier_value = DEFAULT_STOP_LOSS_MULTIPLIER

        volatility_pct = abs(float(volatility_pct))
        if range_floor is not None and volatility_pct < range_floor:
            LOGGER.warning(
                "Predicted volatility %.4f below stop-loss range floor %.4f; using floor.",
                volatility_pct,
                range_floor,
            )
            volatility_pct = range_floor
        delta = float(numeric_close) * volatility_pct * multiplier_value
        if not np.isfinite(delta):
            delta = 0.0

        stop_loss = float(numeric_close - delta)
        stop_loss = max(0.0, min(float(numeric_close), stop_loss))
        return float(stop_loss)

    def _indicator_support_floor(self) -> tuple[Optional[float], Dict[str, float]]:
        latest_features = None
        if isinstance(getattr(self, "metadata", None), Mapping):
            latest_features = self.metadata.get("latest_features")
        if not isinstance(latest_features, pd.DataFrame):
            latest_features = None
        floor_value, components = indicator_support_floor(latest_features)
        cleaned_components: Dict[str, float] = {}
        for column, value in components.items():
            numeric = self._safe_float(value)
            if numeric is None:
                continue
            if numeric <= 0:
                continue
            cleaned_components[str(column)] = float(numeric)
        resolved_floor = self._safe_float(floor_value)
        if resolved_floor is None and cleaned_components:
            resolved_floor = min(cleaned_components.values())
        return resolved_floor, cleaned_components

    def _validate_prediction_ranges(
        self,
        *,
        predicted_return: float | None,
        expected_change_pct: float | None,
        predicted_close: float | None,
        expected_low: float | None,
        stop_loss: float | None,
        anchor_price: float | None,
        horizon: int,
        residual_error: float | None = None,
        prediction_warnings: Optional[list[str]] = None,
    ) -> tuple[float | None, float | None, float | None, float | None, float | None]:
        """Clamp predictions to a volatility-aware band and surface warnings.

        The maximum allowed move scales with the forecast horizon and is
        derived from recent realised volatility (``recent_return_std``) or
        backtest residual error. When a warning sink is provided, extreme
        values are clamped into the plausible band and a warning is recorded so
        the UI can flag discarded forecasts. Without a warning sink, a
        ``ValueError`` is raised to avoid silent distortion.
        """

        anchor_value = self._safe_float(anchor_price)
        horizon_days = max(1, int(horizon))

        sigma_candidates: list[float] = []
        if isinstance(getattr(self, "metadata", None), Mapping):
            sigma_candidates.append(
                self._safe_float(self.metadata.get("daily_log_return_std")) or 0.0
            )
            sigma_candidates.append(self._safe_float(self.metadata.get("recent_return_std")) or 0.0)

        residual_pct = None
        if residual_error is not None and anchor_value is not None and anchor_value > 0:
            residual_pct = abs(float(residual_error)) / float(anchor_value)
            sigma_candidates.append(residual_pct)

        sigma_values = [value for value in sigma_candidates if value and value > 0]
        sigma_multiplier = getattr(
            self.config, "plausibility_sigma_multiplier", DEFAULT_PLAUSIBILITY_SIGMA_MULTIPLIER
        )
        sigma_multiplier_value = self._safe_float(sigma_multiplier) or DEFAULT_PLAUSIBILITY_SIGMA_MULTIPLIER
        if sigma_multiplier_value <= 0:
            sigma_multiplier_value = DEFAULT_PLAUSIBILITY_SIGMA_MULTIPLIER

        fallback_sigma = getattr(self.config, "volatility_floor", DEFAULT_VOLATILITY_FLOOR)
        fallback_sigma_value = self._safe_float(fallback_sigma) or DEFAULT_VOLATILITY_FLOOR
        if fallback_sigma_value <= 0:
            fallback_sigma_value = DEFAULT_VOLATILITY_FLOOR

        resolved_sigma = max(sigma_values) if sigma_values else fallback_sigma_value
        max_move = resolved_sigma * sigma_multiplier_value * math.sqrt(horizon_days)
        max_expected_change_pct = self._safe_float(
            getattr(self.config, "max_expected_change_pct", DEFAULT_MAX_EXPECTED_CHANGE_PCT)
        )
        if max_expected_change_pct is not None and max_expected_change_pct > 0:
            max_move = min(max_move, max_expected_change_pct * math.sqrt(horizon_days))

        warning_sink = prediction_warnings if isinstance(prediction_warnings, list) else None
        metadata_warnings = None
        if isinstance(getattr(self, "metadata", None), Mapping):
            metadata_warnings = self.metadata.setdefault("warnings", [])

        def _add_warning(message: str) -> None:
            if warning_sink is None:
                return
            if message not in warning_sink:
                warning_sink.append(message)
            if isinstance(metadata_warnings, list) and message not in metadata_warnings:
                metadata_warnings.append(message)

        validated_return = predicted_return
        validated_change_pct = expected_change_pct

        for label, value in (
            ("predicted_return", predicted_return),
            ("expected_change_pct", expected_change_pct),
        ):
            numeric = self._safe_float(value)
            if numeric is None:
                continue
            if abs(numeric) > max_move:
                warning_message = (
                    f"{label}={numeric:+.3f} exceeds volatility band +/-{max_move:.3f} for {horizon_days}-day horizon"
                )
                if warning_sink is None:
                    raise ValueError(warning_message)
                _add_warning(warning_message)
                numeric = math.copysign(max_move, numeric)
                if label == "predicted_return":
                    validated_return = numeric
                else:
                    validated_change_pct = numeric

        validated_close = predicted_close
        if anchor_value is not None and validated_change_pct is not None:
            validated_close = float(anchor_value * (1 + validated_change_pct))
        elif anchor_value is not None and validated_close is None:
            validated_close = anchor_value

        if validated_close is not None and validated_close < 0:
            raise ValueError("Predicted close cannot be negative.")
        if anchor_value is not None and validated_close is not None:
            band_floor = anchor_value * 0.01
            if validated_close < band_floor:
                raise ValueError("Predicted close unrealistically small relative to anchor price.")

        validated_expected_low = expected_low
        validated_stop_loss = stop_loss
        for label, value in (("expected_low", expected_low), ("stop_loss", stop_loss)):
            numeric = self._safe_float(value)
            if numeric is None:
                continue
            if numeric < 0:
                raise ValueError(f"{label} cannot be negative.")
            if validated_close is not None and numeric - validated_close > 1e-6:
                warning_message = (
                    f"{label}={numeric:.3f} exceeds predicted close {validated_close:.3f}; clipping to close"
                )
                if warning_sink is None:
                    raise ValueError(f"{label} must not exceed the predicted close.")
                _add_warning(warning_message)
                numeric = validated_close
            if label == "expected_low":
                validated_expected_low = numeric
            else:
                validated_stop_loss = numeric

        if validated_expected_low is not None and validated_close is not None:
            validated_expected_low = min(validated_expected_low, validated_close)
        if validated_stop_loss is not None and validated_close is not None:
            validated_stop_loss = min(validated_stop_loss, validated_close)

        return (
            None if validated_return is None else float(validated_return),
            None if validated_change_pct is None else float(validated_change_pct),
            None if validated_close is None else float(validated_close),
            None if validated_expected_low is None else float(validated_expected_low),
            None if validated_stop_loss is None else float(validated_stop_loss),
        )

    def live_price_snapshot(self, *, horizon: int | None = None) -> Dict[str, Any]:
        """Compose a live snapshot using fresh model predictions."""

        resolved_horizon = self._resolve_horizon(horizon)
        self.horizon = resolved_horizon

        last_price, timestamp = self.fetcher.fetch_live_price(force=True)
        anchor_price = self._safe_float(last_price)

        prediction_output = self.nextgen_engine.predict_latest(
            horizon=resolved_horizon,
        )
        is_prediction_mapping = isinstance(prediction_output, Mapping)
        status = prediction_output.get("status") if is_prediction_mapping else None
        reason = prediction_output.get("reason") if is_prediction_mapping else None
        preds = prediction_output.get("predictions", {}) if is_prediction_mapping else {}
        quantiles = prediction_output.get("quantile_forecasts", {}) if is_prediction_mapping else {}
        prob_block = prediction_output.get("probabilities", {}) if is_prediction_mapping else {}

        predicted_close = self._safe_float(preds.get("close_h"))
        predicted_return = self._safe_float(preds.get("return_h"))
        predicted_intraday_move = self._safe_float(preds.get("return_oc_h"))
        if resolved_horizon == 1 and predicted_intraday_move is not None:
            predicted_return = predicted_intraday_move
        predicted_volatility = None

        def _extract_quantile(block: Mapping[str, Any] | None) -> tuple[Optional[float], Optional[float]]:
            if not isinstance(block, Mapping):
                return None, None
            numeric_block: dict[str, float] = {}
            for key, value in block.items():
                numeric = self._safe_float(value)
                if numeric is not None:
                    numeric_block[str(key)] = float(numeric)
            preferred_keys = ("0.1", "0.10", "p10", "10%", "10", "quantile_0.1", "lower")
            lower_quantile = None
            for key in preferred_keys:
                if key in numeric_block:
                    lower_quantile = numeric_block[key]
                    break
            if lower_quantile is None and numeric_block:
                lower_quantile = min(numeric_block.values())
            median_value = numeric_block.get("0.5") or numeric_block.get("0.50") or numeric_block.get("median")
            return lower_quantile, median_value

        return_quantiles = (
            quantiles.get("return_oc_h")
            if resolved_horizon == 1 and quantiles.get("return_oc_h") is not None
            else quantiles.get("return_h") or quantiles.get("return")
        )
        return_lower, return_median = _extract_quantile(return_quantiles)
        close_quantiles = quantiles.get("close_h") or quantiles.get("close")
        close_lower, close_median = _extract_quantile(close_quantiles)

        quantile_forecasts: dict[str, Dict[str, float]] = {}
        if isinstance(return_quantiles, Mapping):
            quantile_forecasts["return"] = {
                str(key): float(value)
                for key, value in return_quantiles.items()
                if self._safe_float(value) is not None
            }
        if isinstance(close_quantiles, Mapping):
            quantile_forecasts["close"] = {
                str(key): float(value)
                for key, value in close_quantiles.items()
                if self._safe_float(value) is not None
            }

        if predicted_close is None and anchor_price is not None and predicted_return is not None:
            predicted_close = float(anchor_price * (1 + predicted_return))
        if predicted_close is None and anchor_price is not None and close_median is not None:
            predicted_close = float(close_median)

        expected_change_pct = None
        if predicted_close is not None and anchor_price is not None and anchor_price != 0:
            expected_change_pct = (float(predicted_close) - float(anchor_price)) / float(anchor_price)
        elif predicted_return is not None:
            expected_change_pct = float(predicted_return)

        if predicted_return is not None and return_lower is not None:
            predicted_volatility = abs(float(predicted_return) - float(return_lower))
        elif predicted_close is not None and close_lower is not None and predicted_close != 0:
            predicted_volatility = abs(float(predicted_close) - float(close_lower)) / float(predicted_close)

        residual_error = self._residual_error_scale(resolved_horizon)
        expected_low = self._compute_expected_low(
            predicted_close,
            predicted_volatility,
            quantile_forecasts=quantile_forecasts or None,
            prediction_intervals=None,
            indicator_floor=None,
            residual_error=residual_error,
            anchor_price=anchor_price,
            pct_hint=None,
        )
        stop_loss = self._compute_stop_loss(
            predicted_close,
            predicted_volatility,
            expected_low=expected_low,
            residual_error=residual_error,
            anchor_price=anchor_price,
            pct_hint=None,
        )

        if expected_low is not None and predicted_close is not None:
            expected_low = min(expected_low, predicted_close)
        if stop_loss is not None and predicted_close is not None:
            stop_loss = min(stop_loss, predicted_close)

        snapshot_warnings: list[str] = []
        # Return-model output can diverge from close-model-derived change.
        (
            _,
            expected_change_pct,
            predicted_close,
            expected_low,
            stop_loss,
        ) = self._validate_prediction_ranges(
            predicted_return=predicted_return,
            expected_change_pct=expected_change_pct,
            predicted_close=predicted_close,
            expected_low=expected_low,
            stop_loss=stop_loss,
            anchor_price=anchor_price,
            horizon=resolved_horizon or 1,
            residual_error=residual_error,
            prediction_warnings=snapshot_warnings,
        )

        direction_probabilities = prob_block.get("direction_h") or prob_block.get("direction") or {}
        prob_up = None
        prob_down = None
        if isinstance(direction_probabilities, Mapping):
            up_candidates = [
                direction_probabilities.get(key)
                for key in (1, "1", True, "up", "UP")
            ]
            down_candidates = [
                direction_probabilities.get(key)
                for key in (-1, "-1", False, "down", "DOWN")
            ]
            up_candidates = [self._safe_float(value) for value in up_candidates if value is not None]
            down_candidates = [self._safe_float(value) for value in down_candidates if value is not None]
            if up_candidates:
                prob_up = float(max(up_candidates))
            if down_candidates:
                prob_down = float(max(down_candidates))
            if prob_up is None and prob_down is not None:
                prob_up = float(max(0.0, 1 - prob_down))
            if prob_down is None and prob_up is not None:
                prob_down = float(max(0.0, 1 - prob_up))

        return {
            "ticker": self.config.ticker,
            "market_time": self._format_timestamp(timestamp),
            "last_price": last_price,
            "status": status,
            "reason": reason,
            "predicted_close": predicted_close,
            "expected_change_pct": expected_change_pct,
            "expected_intraday_move": predicted_intraday_move,
            "expected_low": expected_low,
            "stop_loss": stop_loss,
            "probabilities": {"up": prob_up, "down": prob_down},
            "warnings": snapshot_warnings or None,
        }

    def _format_timestamp(self, value: Any) -> Optional[str]:
        if value is None:
            return None

        config_timezone = getattr(self.config, "timezone", None)
        target_timezone: ZoneInfo | None
        if isinstance(config_timezone, ZoneInfo):
            target_timezone = config_timezone
        elif config_timezone:
            try:
                target_timezone = ZoneInfo(str(config_timezone))
            except Exception:
                target_timezone = None
        else:
            target_timezone = None

        def _normalize(dt: datetime) -> datetime:
            if target_timezone is None:
                return dt
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=target_timezone)
            return dt.astimezone(target_timezone)

        def _format(dt: datetime) -> str:
            normalized = _normalize(dt)
            tz_label = normalized.strftime("%Z") if normalized.tzinfo else ""
            if tz_label:
                return f"{normalized.strftime('%Y-%m-%d %H:%M:%S')} {tz_label}".strip()
            return normalized.isoformat(timespec="seconds")

        if isinstance(value, datetime):
            return _format(value)

        try:
            timestamp = pd.to_datetime(value)
        except (TypeError, ValueError):
            return str(value)
        if pd.isna(timestamp):
            return None
        if isinstance(timestamp, pd.Timestamp):
            return _format(timestamp.to_pydatetime())
        if isinstance(timestamp, datetime):
            return _format(timestamp)
        return str(timestamp)

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
            if hasattr(estimator, "predict_proba"):
                probs = estimator.predict_proba(features)
                if isinstance(probs, list):  # pragma: no cover - defensive
                    probs = np.asarray(probs)
                if probs.ndim == 2:
                    return float(np.mean(probs[:, -1]))
                return float(np.mean(probs))
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

    def _build_trade_gate(
        self,
        direction_prob_up: float | None,
        direction_prob_down: float | None,
        direction_prob_neutral: float | None,
        monte_carlo_prob: float | None,
    ) -> dict[str, Any] | None:
        if direction_prob_up is None and direction_prob_down is None:
            return None

        threshold = float(self.config.direction_trade_threshold)
        up = float(direction_prob_up or 0.0)
        down = float(direction_prob_down or 0.0)
        neutral = float(direction_prob_neutral or 0.0)

        direction = "up" if up >= down else "down"
        confidence = max(up, down)
        allowed = confidence >= threshold
        reason = "ok" if allowed else "confidence_below_threshold"

        if (
            allowed
            and self.config.monte_carlo_confirmation_enabled
            and monte_carlo_prob is not None
        ):
            mc_threshold = float(self.config.monte_carlo_confirmation_threshold)
            if monte_carlo_prob < mc_threshold:
                allowed = False
                reason = "monte_carlo_below_threshold"

        return {
            "allowed": bool(allowed),
            "direction": direction,
            "confidence": confidence,
            "threshold": threshold,
            "neutral_probability": neutral,
            "monte_carlo_probability": monte_carlo_prob,
            "monte_carlo_threshold": float(
                self.config.monte_carlo_confirmation_threshold
            )
            if self.config.monte_carlo_confirmation_enabled
            else None,
            "reason": reason,
        }

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

        confidence, fundamentals = self._apply_fundamental_penalty(
            confidence,
            expected_return,
        )
        if fundamentals:
            risk_guidance["fundamentals"] = fundamentals
            self.metadata["fundamental_confidence"] = fundamentals

        beta_metrics, beta_rationales = self._beta_context()
        if beta_metrics:
            risk_guidance["beta"] = beta_metrics

        expected_pct_value = (
            float(expected_return * 100) if expected_return is not None else None
        )

        return {
            "action": action,
            "confidence": confidence,
            "expected_return_pct": expected_pct_value,
            "key_drivers": key_drivers,
            "risk_guidance": risk_guidance,
            "risk_rationale": beta_rationales,
        }

    def _apply_fundamental_penalty(
        self, confidence: float, expected_return: float | None
    ) -> tuple[float, dict[str, float]]:
        snapshot = self.metadata.get("fundamental_snapshot")
        fundamentals: dict[str, float] = {}
        if not isinstance(snapshot, dict) or not snapshot:
            return float(np.clip(confidence, 0.0, 1.0)), fundamentals

        latest_close = self._safe_float(self.metadata.get("latest_close"))
        pe_ratio = self._safe_float(snapshot.get("pe_ratio"))
        sector_pe = self._safe_float(snapshot.get("sector_pe"))
        debt_to_equity = self._safe_float(snapshot.get("debt_to_equity"))
        sector_debt = self._safe_float(snapshot.get("sector_debt_to_equity"))
        earnings_growth = self._safe_float(snapshot.get("earnings_growth"))
        sector_growth = self._safe_float(snapshot.get("sector_growth"))
        eps = self._safe_float(snapshot.get("eps"))

        implied_pe = None
        if eps is not None and latest_close is not None and expected_return is not None:
            implied_price = latest_close * (1 + expected_return)
            implied_pe = implied_price / eps if eps else None
        elif pe_ratio is not None and expected_return is not None:
            implied_pe = pe_ratio * (1 + expected_return)
        elif pe_ratio is not None:
            implied_pe = pe_ratio

        penalty = 1.0

        if implied_pe is not None:
            fundamentals["implied_pe"] = float(implied_pe)
        if sector_pe is not None:
            fundamentals["sector_pe"] = float(sector_pe)

        if implied_pe is not None and sector_pe is not None and sector_pe > 0:
            valuation_gap = implied_pe / sector_pe
            fundamentals["valuation_gap"] = float(valuation_gap)
            if valuation_gap >= 2.0:
                penalty *= 0.7
            elif valuation_gap >= 1.5:
                penalty *= 0.85
            elif valuation_gap >= 1.2:
                penalty *= 0.95
            elif valuation_gap <= 0.6:
                penalty *= 0.9

        if debt_to_equity is not None:
            fundamentals["debt_to_equity"] = float(debt_to_equity)
        if sector_debt is not None:
            fundamentals["sector_debt_to_equity"] = float(sector_debt)

        if (
            debt_to_equity is not None
            and sector_debt is not None
            and sector_debt > 0
            and debt_to_equity > sector_debt * 1.5
        ):
            penalty *= 0.9
            fundamentals["leverage_flag"] = 1.0

        if earnings_growth is not None:
            fundamentals["earnings_growth"] = float(earnings_growth)
        if sector_growth is not None:
            fundamentals["sector_growth"] = float(sector_growth)

        if (
            earnings_growth is not None
            and sector_growth is not None
            and earnings_growth < sector_growth * 0.5
        ):
            penalty *= 0.92
            fundamentals["growth_flag"] = 1.0

        adjusted = float(np.clip(confidence * penalty, 0.0, 1.0))
        return adjusted, fundamentals

    def _historical_confidence_score(
        self, *, expected_change: float | None, horizon: Optional[int]
    ) -> tuple[Optional[float], Optional[str]]:
        """Estimate reliability from backtests and historical error dispersion.

        The score combines two signals derived from historical performance:

        * Directional accuracy observed during backtesting.
        * Probability that a move of the forecasted magnitude would have landed
          within historical error bands assuming normally distributed residuals.

        Both components are clipped to [0, 1] and averaged so the result can be
        used as a probability-like discount factor for live predictions.
        """

        metrics = self._validation_metrics("close", horizon)
        if not metrics:
            return None, None

        directional_accuracy = self._safe_float(metrics.get("directional_accuracy"))
        error_scale = self._safe_float(metrics.get("rmse")) or self._safe_float(
            metrics.get("mae")
        )

        probability_within_band = None
        if expected_change is not None and error_scale is not None and error_scale > 0:
            magnitude = abs(expected_change)
            z_score = magnitude / max(error_scale, 1e-6)
            tail_cdf = 0.5 * (1.0 + math.erf(z_score / math.sqrt(2.0)))
            probability_within_band = float(np.clip(2 * tail_cdf - 1.0, 0.0, 1.0))

        components: list[float] = []
        if directional_accuracy is not None and np.isfinite(directional_accuracy):
            components.append(float(np.clip(directional_accuracy, 0.0, 1.0)))
        if probability_within_band is not None:
            components.append(probability_within_band)

        if not components:
            return None, None

        historical_score = float(np.clip(float(np.mean(components)), 0.0, 1.0))
        notes: list[str] = []
        if directional_accuracy is not None and np.isfinite(directional_accuracy):
            notes.append(f"Directional accuracy {directional_accuracy:.1%} from backtests")
        if probability_within_band is not None:
            notes.append(
                f"{probability_within_band:.1%} likelihood move fits historical error band (σ≈{error_scale:.4f})"
            )

        return historical_score, "; ".join(notes) if notes else None

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

        trend_note = None
        if isinstance(self.metadata, Mapping):
            trend_note = self.metadata.get("trend_alignment_note")
        if trend_note:
            reasons.append(str(trend_note))

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

    def _sentiment_confidence_factor(self) -> Optional[float]:
        if not getattr(self.config, "sentiment_confidence_adjustment", False):
            return None

        sentiment_df = self.metadata.get("sentiment_daily")
        if not isinstance(sentiment_df, pd.DataFrame) or sentiment_df.empty:
            return None

        df = sentiment_df
        if "Date" in df.columns:
            try:
                df = df.sort_values("Date")
            except Exception:  # pragma: no cover - defensive against malformed data
                df = sentiment_df

        value_column = None
        for candidate in ("Sentiment_Avg", "sentiment"):
            if candidate in df.columns:
                value_column = candidate
                break

        if value_column is None:
            return None

        window = getattr(self.config, "sentiment_confidence_window", 7)
        try:
            window = int(window)
        except (TypeError, ValueError):  # pragma: no cover - defensive defaulting
            window = 7
        window = max(1, window)

        recent_values = pd.to_numeric(df[value_column].tail(window), errors="coerce").dropna()
        if recent_values.empty:
            return None

        average = float(np.nanmean(recent_values))
        if not np.isfinite(average):
            return None
        return float(np.clip(average, -1.0, 1.0))

    def _apply_sentiment_adjustment(
        self,
        combined_confidence: Optional[float],
        direction_probability_up: Optional[float],
        direction_probability_down: Optional[float],
    ) -> tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
        if not getattr(self.config, "sentiment_confidence_adjustment", False):
            return combined_confidence, direction_probability_up, direction_probability_down, None

        sentiment_factor = self._sentiment_confidence_factor()
        if sentiment_factor is None:
            return combined_confidence, direction_probability_up, direction_probability_down, None

        try:
            weight = float(getattr(self.config, "sentiment_confidence_weight", 0.2))
        except (TypeError, ValueError):  # pragma: no cover - defensive defaulting
            weight = 0.2

        weight = max(0.0, min(weight, 2.0))
        multiplier = max(0.0, 1.0 + sentiment_factor * weight)

        adjusted_confidence = combined_confidence
        if adjusted_confidence is not None:
            adjusted_confidence = float(np.clip(adjusted_confidence * multiplier, 0.0, 1.0))

        adjusted_up = direction_probability_up
        adjusted_down = direction_probability_down
        if adjusted_up is not None and adjusted_down is not None:
            prob_delta = sentiment_factor * weight
            adjusted_up_raw = max(0.0, float(adjusted_up) * (1.0 + prob_delta))
            adjusted_down_raw = max(0.0, float(adjusted_down) * (1.0 - prob_delta))
            total = adjusted_up_raw + adjusted_down_raw
            if total > 0:
                adjusted_up = float(adjusted_up_raw / total)
                adjusted_down = float(adjusted_down_raw / total)

        return adjusted_confidence, adjusted_up, adjusted_down, sentiment_factor

    def _refine_direction_probability(
        self,
        direction_probability_up: Optional[float],
        direction_probability_down: Optional[float],
        predicted_return: Optional[float],
        predicted_volatility: Optional[float],
        horizon: Optional[int],
    ) -> tuple[Optional[float], Optional[float]]:
        """Boost direction probabilities with CPU-intensive bootstrap sampling."""

        if not getattr(self.config, "direction_bootstrap_enabled", True):
            return direction_probability_up, direction_probability_down

        try:
            total_paths = int(getattr(self.config, "direction_bootstrap_paths", 0))
        except (TypeError, ValueError):
            return direction_probability_up, direction_probability_down

        if total_paths <= 0:
            return direction_probability_up, direction_probability_down

        if predicted_volatility is None or not np.isfinite(predicted_volatility):
            return direction_probability_up, direction_probability_down

        cpu_count = os.cpu_count() or 1
        workers_value = getattr(self.config, "direction_bootstrap_workers", None)
        try:
            workers = int(workers_value) if workers_value is not None else cpu_count
        except (TypeError, ValueError):
            workers = cpu_count
        workers = max(1, min(workers, cpu_count))

        chunk_sizes = [
            total_paths // workers + (1 if idx < total_paths % workers else 0)
            for idx in range(workers)
        ]

        if not any(chunk_sizes):
            return direction_probability_up, direction_probability_down

        mean_return = 0.0
        if predicted_return is not None and np.isfinite(predicted_return):
            try:
                horizon_days = max(int(horizon or 1), 1)
            except (TypeError, ValueError):
                horizon_days = 1
            mean_return = float(predicted_return) / float(horizon_days)

        seed_seq = np.random.SeedSequence()
        seeds = seed_seq.generate_state(len(chunk_sizes)).tolist()

        with mp.Pool(processes=workers) as pool:
            results = pool.starmap(
                _simulate_direction_bootstrap,
                zip(seeds, chunk_sizes, repeat(mean_return), repeat(predicted_volatility)),
            )

        total_up = sum(up for up, _ in results)
        total_down = sum(down for _, down in results)
        total_samples = total_up + total_down
        if total_samples <= 0:
            return direction_probability_up, direction_probability_down

        refined_up = float(total_up / total_samples)
        refined_down = float(total_down / total_samples)

        try:
            blend_weight = float(getattr(self.config, "direction_bootstrap_blend", 0.65))
        except (TypeError, ValueError):
            blend_weight = 0.65
        blend_weight = max(0.0, min(1.0, blend_weight))

        base_up = direction_probability_up
        base_down = direction_probability_down

        if base_up is None and base_down is None:
            return refined_up, refined_down

        if base_up is None:
            return refined_up, refined_down

        if base_down is None:
            return refined_up, refined_down

        adjusted_up = blend_weight * refined_up + (1 - blend_weight) * float(base_up)
        adjusted_down = blend_weight * refined_down + (1 - blend_weight) * float(base_down)
        total = adjusted_up + adjusted_down
        if total > 0:
            adjusted_up /= total
            adjusted_down /= total

        return float(adjusted_up), float(adjusted_down)

    def _apply_trend_alignment_adjustment(
        self, combined_confidence: Optional[float]
    ) -> tuple[Optional[float], Optional[str]]:
        summary = self.metadata.get("trend_summary") if isinstance(self.metadata, Mapping) else None
        note: Optional[str] = None

        if not isinstance(summary, Mapping):
            return combined_confidence, note

        base_label = summary.get("base_timeframe") or "daily"
        timeframes = summary.get("timeframes") or {}
        base_snapshot = timeframes.get(base_label)
        if not isinstance(base_snapshot, Mapping):
            return combined_confidence, note

        base_bias = base_snapshot.get("bias")
        if not base_bias or base_bias == "neutral":
            return combined_confidence, note

        aligned: list[str] = []
        conflicts: list[str] = []
        conflict_biases: list[str] = []
        for label, snapshot in timeframes.items():
            if label == base_label or not isinstance(snapshot, Mapping):
                continue
            bias = snapshot.get("bias")
            if bias is None or bias == "neutral":
                continue
            if bias == base_bias:
                aligned.append(label)
            else:
                conflicts.append(label)
                conflict_biases.append(str(bias))

        factor: Optional[float] = None
        if aligned and not conflicts:
            factor = 1.1
            joined = ", ".join(label.title() for label in aligned)
            note = f"{joined} trend {base_bias} supports {base_label} view."
        elif conflicts and not aligned:
            factor = 0.85
            joined = ", ".join(label.title() for label in conflicts)
            conflict_bias = conflict_biases[0] if conflict_biases else "opposing"
            note = f"{joined} trend {conflict_bias} conflicts with {base_label} {base_bias}, tempering conviction."
        elif aligned and conflicts:
            factor = 0.95
            joined = ", ".join(label.title() for label in aligned + conflicts)
            note = f"Mixed higher-timeframe trends across {joined} versus {base_label} {base_bias}."

        adjusted_confidence = combined_confidence
        if factor is not None and adjusted_confidence is not None:
            adjusted_confidence = float(np.clip(adjusted_confidence * factor, 0.0, 1.0))

        if isinstance(self.metadata, dict):
            if note:
                self.metadata["trend_alignment_note"] = note
            else:
                self.metadata.pop("trend_alignment_note", None)

        return adjusted_confidence, note

    def _macro_reasons(self, feature_row: pd.Series) -> list[str]:
        reasons: list[str] = []
        vol21 = self._safe_float(feature_row.get("Volatility_21"))
        if vol21 is not None and vol21 >= 0.03:
            reasons.append("Short-term realised volatility elevated, indicating choppy market conditions.")

        slope_keys = ("Trend_Slope_21", "Trend_Slope")
        trend_slope = None
        for key in slope_keys:
            trend_slope = self._safe_float(feature_row.get(key))
            if trend_slope is not None:
                break
        if trend_slope is not None:
            if trend_slope > 0:
                reasons.append("Medium-term trend slope positive, supporting upward bias.")
            elif trend_slope < 0:
                reasons.append("Medium-term trend slope negative, signalling downward bias.")

        curvature_keys = ("Trend_Curvature_63", "Trend_Curvature")
        trend_curvature = None
        for key in curvature_keys:
            trend_curvature = self._safe_float(feature_row.get(key))
            if trend_curvature is not None:
                break
        if trend_curvature is not None and trend_curvature < 0:
            reasons.append("Trend curvature turning lower, hinting at deceleration in momentum.")

        _, beta_reasons = self._beta_context(feature_row)
        reasons.extend(beta_reasons)

        return reasons

    def _beta_context(
        self, feature_row: pd.Series | None = None
    ) -> tuple[Dict[str, Dict[str, float]], list[str]]:
        latest_features = None
        if feature_row is None:
            if isinstance(getattr(self, "metadata", None), Mapping):
                latest_features = self.metadata.get("latest_features")
            if isinstance(latest_features, pd.DataFrame) and not latest_features.empty:
                try:
                    feature_row = latest_features.iloc[0]
                except (KeyError, IndexError):
                    feature_row = None
        if feature_row is None:
            return {}, []

        beta_values: Dict[str, Dict[str, float]] = {}
        rationales: list[str] = []
        missing_labels: set[str] = set()

        for column, value in feature_row.items():
            name = str(column)
            if not name.startswith("Beta_"):
                continue

            parts = name.split("_")
            if len(parts) < 3:
                continue

            benchmark_key = parts[1].lower()
            label = self._format_beta_label(benchmark_key)
            numeric_value = self._safe_float(value)
            if numeric_value is None:
                missing_labels.add(label)
                continue
            try:
                window = int(parts[2])
            except (TypeError, ValueError):
                window = None

            current = beta_values.get(benchmark_key)
            if current is not None and window is not None and "window" in current:
                try:
                    current_window = int(current.get("window"))
                except (TypeError, ValueError):
                    current_window = None
                if current_window is not None and window >= current_window:
                    continue

            label = self._format_beta_label(benchmark_key)
            beta_payload: Dict[str, float] = {"label": label, "value": float(numeric_value)}
            if window is not None:
                beta_payload["window"] = int(window)

            risk_level = self._beta_risk_band(numeric_value)
            if risk_level:
                beta_payload["risk_level"] = risk_level

            beta_values[benchmark_key] = beta_payload

            if numeric_value >= 1.5:
                rationales.append(
                    f"{label} beta at {numeric_value:.2f} (>1.5) signals elevated sensitivity to market swings."
                )
            elif numeric_value <= 0.7:
                rationales.append(
                    f"{label} beta at {numeric_value:.2f} (<0.7) points to a more defensive risk profile."
                )

        for label in sorted(missing_labels):
            rationales.append(f"{label} beta unavailable due to missing benchmark data.")

        return beta_values, rationales

    @staticmethod
    def _format_beta_label(key: str) -> str:
        normalized = key.lower()
        if normalized in {"sp500", "gspc", "s&p500"}:
            return "S&P 500"
        if normalized == "vix":
            return "VIX"
        return key.upper()

    @staticmethod
    def _beta_risk_band(beta_value: float) -> str:
        if beta_value >= 1.5:
            return "high"
        if beta_value <= 0.7:
            return "defensive"
        return "moderate"

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

        trend_note = prediction.get("trend_alignment_note") or self.metadata.get(
            "trend_alignment_note"
        )
        if trend_note:
            sentences.append(str(trend_note))

        if pct_change is not None and np.isfinite(pct_change) and pct_change != 0:
            sentences.append(f"Expected move of {pct_change * 100:.2f}% versus the last close.")

        if forecast_return is not None and np.isfinite(forecast_return):
            sentences.append(f"Projected horizon return of {forecast_return * 100:.2f}%.")

        if volatility is not None and np.isfinite(volatility) and volatility > 0:
            sentences.append(f"Anticipated volatility around {volatility * 100:.2f}%.")

        fundamental_sentence = self._render_fundamental_headline()
        if fundamental_sentence:
            sentences.append(fundamental_sentence)

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

        target_hit_prob = confidence_indicators.get("target_hit_probability")
        if isinstance(target_hit_prob, dict) and target_hit_prob:
            hit_value = self._safe_float(target_hit_prob.get("hit"))
            if hit_value is not None:
                sentences.append(
                    f"Target price hit probability {hit_value * 100:.1f}% within the horizon."
                )

        summary = " ".join(sentence.strip() for sentence in sentences if sentence)
        return summary.strip()

    def _render_fundamental_headline(self) -> str | None:
        snapshot = self.metadata.get("fundamental_snapshot")
        if not isinstance(snapshot, dict) or not snapshot:
            return None

        sentences: list[str] = []
        pe_ratio = self._safe_float(snapshot.get("pe_ratio"))
        sector_pe = self._safe_float(snapshot.get("sector_pe"))
        if pe_ratio is not None and sector_pe is not None:
            sentences.append(f"Current P/E {pe_ratio:.1f} vs sector {sector_pe:.1f}")
        debt_to_equity = self._safe_float(snapshot.get("debt_to_equity"))
        sector_debt = self._safe_float(snapshot.get("sector_debt_to_equity"))
        if debt_to_equity is not None and sector_debt is not None:
            sentences.append(
                f"Debt/Equity {debt_to_equity:.1f} vs sector {sector_debt:.1f}"
            )
        earnings_growth = self._safe_float(snapshot.get("earnings_growth"))
        sector_growth = self._safe_float(snapshot.get("sector_growth"))
        if earnings_growth is not None and sector_growth is not None:
            sentences.append(
                f"Earnings growth {earnings_growth * 100:.1f}% vs sector {sector_growth * 100:.1f}%"
            )

        if not sentences:
            return None
        return "; ".join(sentences) + "."

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

        hit_prob = self._safe_float(prediction.get("target_hit_probability"))
        if hit_prob is not None:
            hit_block: Dict[str, float] = {"hit": hit_prob}
            prob_store = prediction.get("probabilities")
            if isinstance(prob_store, dict):
                hit_probs = prob_store.get("target_hit")
                if isinstance(hit_probs, dict):
                    miss_prob = self._safe_float(
                        hit_probs.get(0)
                        or hit_probs.get(0.0)
                        or hit_probs.get("0")
                        or hit_probs.get(False)
                    )
                    if miss_prob is not None:
                        hit_block["miss"] = miss_prob
            indicators["target_hit_probability"] = hit_block

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

        fundamentals = self.metadata.get("fundamental_confidence")
        if isinstance(fundamentals, dict) and fundamentals:
            indicators["fundamentals"] = fundamentals

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
            directional_block = evaluation.get("directional")
            if isinstance(directional_block, dict):
                for key, value in directional_block.items():
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
        self,
        targets: Optional[Iterable[str]] = None,
        horizon: Optional[int] = None,
        *,
        backtest_config: BacktestConfig | None = None,
    ) -> Dict[str, Any]:
        resolved_horizon = self._resolve_horizon(horizon)
        self.horizon = resolved_horizon
        requested_targets = tuple(targets) if targets else tuple(self.config.prediction_targets)

        default_config = BacktestConfig(
            n_runs=getattr(self.config, "backtest_runs", 1),
            start_date=self.config.start_date,
            end_date=self.config.end_date,
            horizon=resolved_horizon,
            targets=requested_targets,
            toggles=self.config.feature_toggles,
            training_window=self.config.backtest_window,
            step_size=self.config.backtest_step,
            embargo=getattr(self.config, "backtest_embargo", 1),
            random_seed=getattr(self.config, "random_seed", None),
            caching=True,
            cache_trained_windows=True,
        )
        cfg = default_config
        if backtest_config is not None:
            merged = {**default_config.__dict__, **{k: v for k, v in backtest_config.__dict__.items() if v is not None}}
            cfg = BacktestConfig(**merged)

        backtester = Backtester(self.config, trainer=self)
        result = backtester.run(targets=requested_targets, backtest_config=cfg)

        target_breakdown: dict[str, Any] = {}
        for target_name in requested_targets:
            target_splits = [
                split for split in result.splits if split.get("target") == target_name
            ]
            if not target_splits:
                continue
            target_breakdown[target_name] = {
                "splits": target_splits,
                "aggregate": self._aggregate_evaluation_metrics(target_splits),
            }

        payload = result.as_dict()
        payload["targets"] = target_breakdown
        payload["horizon"] = resolved_horizon
        self.metadata["active_horizon"] = resolved_horizon
        self.metadata["backtest_summary"] = payload.get("summary", {})
        success_rates = self._extract_backtest_success_rates(payload)
        if success_rates:
            self.metadata["backtest_success_rates"] = success_rates
        self.tracker.log_run(
            target=",".join(requested_targets),
            run_type="backtest",
            parameters={
                "model_type": self.config.model_type,
                "training_window": cfg.training_window,
                "step_size": cfg.step_size,
                "embargo": cfg.embargo,
                "runs": cfg.n_runs,
                "seed": cfg.random_seed,
                "start_date": str(cfg.start_date) if cfg.start_date is not None else None,
                "end_date": str(cfg.end_date) if cfg.end_date is not None else None,
            },
            metrics=result.aggregate,
            context={
                "splits": result.splits,
                "confidence_intervals": result.confidence_intervals,
                "calibration": result.calibration,
                "reliability": result.reliability,
                "interval_coverage": result.interval_coverage,
            },
        )
        return payload

    def run_simulation_backtest(
        self,
        targets: Optional[Iterable[str]] = None,
        horizon: Optional[int] = None,
        *,
        simulation_config: SimulationBacktestConfig | None = None,
    ) -> Dict[str, Any]:
        resolved_horizon = self._resolve_horizon(horizon)
        requested_targets = tuple(targets) if targets else tuple(self.config.prediction_targets)

        default_config = SimulationBacktestConfig(
            horizon=resolved_horizon,
            targets=requested_targets,
            random_seed=getattr(self.config, "random_seed", None),
            parallelism=max(1, int(getattr(self.config, "direction_bootstrap_workers", 1) or 1)),
        )
        cfg = default_config
        if simulation_config is not None:
            merged = {
                **default_config.__dict__,
                **{k: v for k, v in simulation_config.__dict__.items() if v is not None},
            }
            cfg = SimulationBacktestConfig(**merged)

        backtester = SimulationBacktester(self.config, trainer=self)
        result = backtester.run(targets=requested_targets, horizon=resolved_horizon, sim_config=cfg)

        payload = result.as_dict()
        payload["horizon"] = resolved_horizon
        payload["targets"] = requested_targets
        self.metadata["simulation_backtest"] = payload.get("summary", {})
        self.tracker.log_run(
            target=",".join(requested_targets),
            run_type="simulation_backtest",
            parameters={
                "simulations": cfg.simulations,
                "batch_size": cfg.batch_size,
                "window_size": cfg.window_size,
                "parallelism": cfg.parallelism,
                "seed": cfg.random_seed,
                "tolerance": cfg.tolerance,
            },
            metrics=payload.get("summary"),
            context={"results": payload.get("results"), "horizon": resolved_horizon},
        )
        return payload

    # ------------------------------------------------------------------
    # Evaluation summaries
    # ------------------------------------------------------------------
    def accuracy_summary(self, *, horizon: Optional[int] = None) -> Dict[str, Any]:
        """Aggregate directional accuracy and tolerance hit rates across backtests."""

        resolved_horizon = self._resolve_horizon(horizon)
        runs = self.tracker.load_runs(run_type="backtest")

        total_predictions = 0
        correct_predictions = 0.0
        tolerance_predictions = 0
        tolerance_hits = 0.0
        runs_considered = 0

        for run in runs:
            context = run.get("context") or {}
            run_horizon = context.get("horizon")
            if resolved_horizon is not None and run_horizon is not None:
                try:
                    if int(run_horizon) != int(resolved_horizon):
                        continue
                except (TypeError, ValueError):
                    LOGGER.debug("Skipping run with unparsable horizon: %s", run_horizon)
                    continue

            splits = [entry for entry in context.get("splits", []) if isinstance(entry, Mapping)]
            split_total, split_correct, split_tolerance_total, split_tolerance_hits = (
                self._summarise_splits(splits)
            )

            metrics = run.get("metrics") or {}
            if split_total == 0:
                (
                    aggregate_total,
                    aggregate_correct,
                    aggregate_tolerance_total,
                    aggregate_tolerance_hits,
                ) = self._summarise_aggregate(metrics)
                split_total = aggregate_total
                split_correct = aggregate_correct
                split_tolerance_total = aggregate_tolerance_total
                split_tolerance_hits = aggregate_tolerance_hits

            if split_total > 0:
                total_predictions += split_total
                correct_predictions += split_correct
                runs_considered += 1
            if split_tolerance_total > 0:
                tolerance_predictions += split_tolerance_total
                tolerance_hits += split_tolerance_hits

        correct_count = int(round(correct_predictions))
        correct_count = min(correct_count, total_predictions)
        incorrect_count = max(total_predictions - correct_count, 0)
        correct_pct = (correct_predictions / total_predictions) if total_predictions else 0.0
        incorrect_pct = (incorrect_count / total_predictions) if total_predictions else 0.0

        tolerance_hit_count = int(round(tolerance_hits))
        tolerance_hit_count = min(tolerance_hit_count, tolerance_predictions)
        tolerance_miss_count = max(tolerance_predictions - tolerance_hit_count, 0)
        tolerance_hit_rate = (
            (tolerance_hits / tolerance_predictions) if tolerance_predictions else 0.0
        )

        return {
            "horizon": resolved_horizon,
            "runs_considered": runs_considered,
            "total_predictions": total_predictions,
            "correct": correct_count,
            "incorrect": incorrect_count,
            "correct_pct": correct_pct,
            "incorrect_pct": incorrect_pct,
            "tolerance_band": self.config.resolve_tolerance_band(resolved_horizon),
            "tolerance_total": tolerance_predictions,
            "tolerance_hits": tolerance_hit_count,
            "tolerance_misses": tolerance_miss_count,
            "tolerance_hit_rate": tolerance_hit_rate,
        }

    @staticmethod
    def _coerce_accuracy(payload: Mapping[str, Any]) -> float | None:
        for key in ("directional_accuracy", "accuracy"):
            value = payload.get(key)
            if isinstance(value, Real) and math.isfinite(value):
                return float(value)
        return None

    def _summarise_splits(
        self, splits: Iterable[Mapping[str, Any]]
    ) -> tuple[int, float, int, float]:
        total = 0
        correct = 0.0
        tolerance_total = 0
        tolerance_hits = 0.0
        for split in splits:
            test_size = split.get("test_size")
            accuracy = self._coerce_accuracy(split)
            if not isinstance(test_size, (int, float)) or accuracy is None:
                continue
            if not math.isfinite(test_size):
                continue
            total += int(test_size)
            correct += float(accuracy) * int(test_size)
            tolerance_count = split.get("tolerance_total")
            tolerance_hit_count = split.get("tolerance_hits")
            if isinstance(tolerance_count, (int, float)) and isinstance(
                tolerance_hit_count, (int, float)
            ):
                if math.isfinite(tolerance_count) and math.isfinite(tolerance_hit_count):
                    tolerance_total += int(tolerance_count)
                    tolerance_hits += float(tolerance_hit_count)
        return total, correct, tolerance_total, tolerance_hits

    def _summarise_aggregate(
        self, metrics: Mapping[str, Any]
    ) -> tuple[int, float, int, float]:
        total = 0
        correct = 0.0
        tolerance_total = 0
        tolerance_hits = 0.0
        if not isinstance(metrics, Mapping):
            return total, correct, tolerance_total, tolerance_hits

        test_rows = metrics.get("test_rows")
        accuracy = self._coerce_accuracy(metrics)
        if isinstance(test_rows, (int, float)) and accuracy is not None and math.isfinite(test_rows):
            total = int(test_rows)
            correct = float(accuracy) * total
        tol_total_value = metrics.get("tolerance_total")
        tol_hit_value = metrics.get("tolerance_hits")
        if isinstance(tol_total_value, (int, float)) and math.isfinite(tol_total_value):
            tolerance_total = int(tol_total_value)
        if isinstance(tol_hit_value, (int, float)) and math.isfinite(tol_hit_value):
            tolerance_hits = float(tol_hit_value)
        return total, correct, tolerance_total, tolerance_hits
