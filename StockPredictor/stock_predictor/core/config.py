"""Configuration utilities for the stock predictor package."""

from __future__ import annotations

import json
import math
import os

from dataclasses import dataclass, field, fields
from datetime import date, timedelta
from pathlib import Path
import re

from typing import Any, Iterable, Mapping, Optional, Sequence

from dotenv import load_dotenv

from .features import FEATURE_REGISTRY, FeatureToggles, default_feature_toggles
from .preprocessing import default_price_feature_toggles, derive_price_feature_toggles

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_DATABASE_PATH = DEFAULT_DATA_DIR / "market_data.sqlite"


def _default_database_url() -> str:
    return f"sqlite:///{DEFAULT_DATABASE_PATH}"


DEFAULT_PREDICTION_TARGETS: tuple[str, ...] = (
    "close",
    "direction",
    "return",
    "volatility",
)

DEFAULT_PREDICTION_HORIZONS: tuple[int, ...] = (1, 5, 21, 63)

DEFAULT_TEST_SIZE = 0.2
DEFAULT_BACKTEST_STRATEGY = "rolling"
DEFAULT_BACKTEST_WINDOW = 252
DEFAULT_BACKTEST_STEP = 21
DEFAULT_VOLATILITY_WINDOW = 20
DEFAULT_MIN_SAMPLES_PER_HORIZON = 100
DEFAULT_TARGET_GAIN_PCT = 0.03
DEFAULT_MACRO_MERGE_SYMBOLS: tuple[str, ...] = ("^VIX", "DXY", "^TNX")


HORIZON_UNIT_TO_DAYS: dict[str, int] = {
    "d": 1,
    "day": 1,
    "days": 1,
    "bd": 1,
    "businessday": 1,
    "businessdays": 1,
    "w": 5,
    "wk": 5,
    "wks": 5,
    "week": 5,
    "weeks": 5,
    "m": 21,
    "mo": 21,
    "month": 21,
    "months": 21,
}


def _coerce_min_samples_per_horizon(value: Any | None) -> int:
    """Validate the minimum sample threshold for each horizon."""

    if value is None:
        return DEFAULT_MIN_SAMPLES_PER_HORIZON

    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("min_samples_per_horizon must be an integer.") from exc

    return max(1, parsed)


@dataclass
class BuyZoneConfirmationSettings:
    """Tunable thresholds for buy-zone confirmation checks."""

    enable_rsi: bool = True
    rsi_threshold: float = 40.0
    enable_macd: bool = True
    macd_hist_threshold: float = 0.0
    enable_bollinger: bool = True
    bollinger_proximity_pct: float = 0.02
    enable_volatility: bool = True
    max_atr_fraction_of_price: float = 0.04
    enable_volume: bool = True
    obv_lookback: int = 5
    mfi_threshold: float = 45.0

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> "BuyZoneConfirmationSettings":
        if not isinstance(payload, Mapping):
            return cls()

        def _coerce_bool(key: str, default: bool) -> bool:
            raw = payload.get(key, default)
            if isinstance(raw, str):
                return raw.strip().lower() in {"1", "true", "yes", "y", "on"}
            return bool(raw)

        def _coerce_float(key: str, default: float) -> float:
            try:
                return float(payload.get(key, default))
            except (TypeError, ValueError):
                return default

        def _coerce_int(key: str, default: int) -> int:
            try:
                return int(payload.get(key, default))
            except (TypeError, ValueError):
                return default

        return cls(
            enable_rsi=_coerce_bool("enable_rsi", True),
            rsi_threshold=_coerce_float("rsi_threshold", 40.0),
            enable_macd=_coerce_bool("enable_macd", True),
            macd_hist_threshold=_coerce_float("macd_hist_threshold", 0.0),
            enable_bollinger=_coerce_bool("enable_bollinger", True),
            bollinger_proximity_pct=_coerce_float("bollinger_proximity_pct", 0.02),
            enable_volatility=_coerce_bool("enable_volatility", True),
            max_atr_fraction_of_price=_coerce_float("max_atr_fraction_of_price", 0.04),
            enable_volume=_coerce_bool("enable_volume", True),
            obv_lookback=_coerce_int("obv_lookback", 5),
            mfi_threshold=_coerce_float("mfi_threshold", 45.0),
        )


@dataclass
class PredictorConfig:
    """Runtime configuration for :class:`StockPredictorAI`."""

    ticker: str
    start_date: date = field(default_factory=lambda: date.today() - timedelta(days=365))
    end_date: Optional[date] = None
    interval: str = "1d"
    model_type: str = "random_forest"
    data_dir: Path = DEFAULT_DATA_DIR
    models_dir: Path = DEFAULT_MODELS_DIR
    news_api_key: Optional[str] = None
    news_limit: int = 50
    sentiment: bool = True
    database_url: str = field(default_factory=_default_database_url)
    feature_toggles: FeatureToggles = field(default_factory=default_feature_toggles)
    price_feature_toggles: dict[str, bool] = field(
        default_factory=default_price_feature_toggles
    )
    prediction_targets: tuple[str, ...] = field(
        default_factory=lambda: DEFAULT_PREDICTION_TARGETS
    )
    prediction_horizons: tuple[int, ...] = field(
        default_factory=lambda: DEFAULT_PREDICTION_HORIZONS
    )
    model_params: dict[str, dict[str, Any]] = field(
        default_factory=lambda: {"global": {}}
    )
    test_size: float = DEFAULT_TEST_SIZE
    shuffle_training: bool = False
    backtest_strategy: str = DEFAULT_BACKTEST_STRATEGY
    backtest_window: int = DEFAULT_BACKTEST_WINDOW
    backtest_step: int = DEFAULT_BACKTEST_STEP
    backtest_slippage_bps: float = 1.0
    backtest_fee_bps: float = 1.0
    backtest_neutral_threshold: float = 0.001
    evaluation_strategy: str = "time_series"
    evaluation_folds: int = 5
    evaluation_window: int = DEFAULT_BACKTEST_WINDOW
    evaluation_step: int = DEFAULT_BACKTEST_STEP
    evaluation_slippage_bps: float = 0.0
    evaluation_fee_bps: float = 0.0
    evaluation_fixed_cost: float = 0.0
    tuning_enabled: bool = True
    tuning_method: str = "random"
    tuning_iterations: int = 10
    tuning_folds: int | None = None
    tuning_n_jobs: int | None = None
    direction_confidence_threshold: float = 0.5
    volatility_window: int = DEFAULT_VOLATILITY_WINDOW
    target_gain_pct: float = DEFAULT_TARGET_GAIN_PCT
    risk_free_rate: float = 0.0
    research_api_keys: tuple[str, ...] = field(default_factory=tuple)
    research_allow_list: tuple[str, ...] = field(default_factory=tuple)
    sentiment_confidence_adjustment: bool = False
    sentiment_confidence_window: int = 7
    sentiment_confidence_weight: float = 0.2
    monte_carlo_paths: int = 500_000
    monte_carlo_precision: float | None = None
    monte_carlo_max_paths: int | None = None
    direction_bootstrap_enabled: bool = True
    direction_bootstrap_paths: int = 2_000_000
    direction_bootstrap_workers: int | None = None
    direction_bootstrap_blend: float = 0.65
    price_backfill_page_days: int = 365
    price_backfill_page_size: int | None = None
    training_cache_dir: Path | None = None
    use_cached_training_data: bool = True
    min_samples_per_horizon: int = DEFAULT_MIN_SAMPLES_PER_HORIZON
    forecast_tolerance_bands: dict[int, float] = field(default_factory=dict)
    # Provide a local CSV file path to enable the CSVPriceLoader provider.
    csv_price_loader_path: Path | None = None
    # Provide a local Parquet file path to enable the ParquetPriceLoader provider.
    parquet_price_loader_path: Path | None = None
    yahoo_rate_limit_per_second: float | None = None
    yahoo_rate_limit_per_minute: float | None = None
    yahoo_cooldown_seconds: float | None = None
    price_provider_priority: tuple[str, ...] = field(default_factory=tuple)
    disabled_providers: tuple[str, ...] = field(default_factory=tuple)
    memory_cache_seconds: float | None = None
    market_timezone: str | None = None
    k_stop: float = 1.0
    expected_low_sigma: float = 1.0
    expected_low_max_volatility: float = 1.0
    expected_low_floor_window: int = 20
    time_series_baselines: tuple[str, ...] = field(default_factory=tuple)
    time_series_params: dict[str, dict[str, Any]] = field(default_factory=dict)
    buy_zone: BuyZoneConfirmationSettings = field(
        default_factory=BuyZoneConfirmationSettings
    )
    macro_merge_symbols: tuple[str, ...] = field(
        default_factory=lambda: DEFAULT_MACRO_MERGE_SYMBOLS
    )

    def __post_init__(self) -> None:
        self.ticker = self.ticker.upper()
        self.feature_toggles = _coerce_feature_toggles(self.feature_toggles)
        self.prediction_targets = self._normalise_collection(
            self.prediction_targets, DEFAULT_PREDICTION_TARGETS
        )
        self.prediction_horizons = self._normalise_horizons(self.prediction_horizons)
        self.model_params = self._normalise_model_params(self.model_params)
        self.shuffle_training = bool(self.shuffle_training)
        self.backtest_strategy = (
            str(self.backtest_strategy).strip().lower() or DEFAULT_BACKTEST_STRATEGY
        )
        self.evaluation_strategy = str(self.evaluation_strategy).strip().lower() or "time_series"
        if not self.sentiment:
            self.feature_toggles["sentiment"] = False
        self.price_feature_toggles = _coerce_price_feature_toggles(
            self.price_feature_toggles
        )
        self.macro_merge_symbols = _coerce_iterable(
            self.macro_merge_symbols, DEFAULT_MACRO_MERGE_SYMBOLS
        )
        self.forecast_tolerance_bands = _coerce_tolerance_bands(
            self.forecast_tolerance_bands
        )
        if not 0 < self.test_size < 1:
            raise ValueError("test_size must be between 0 and 1.")
        if self.backtest_window <= 0:
            raise ValueError("backtest_window must be positive.")
        if self.backtest_step <= 0:
            raise ValueError("backtest_step must be positive.")
        self.backtest_slippage_bps = max(0.0, float(self.backtest_slippage_bps))
        self.backtest_fee_bps = max(0.0, float(self.backtest_fee_bps))
        self.backtest_neutral_threshold = float(self.backtest_neutral_threshold)
        if self.backtest_neutral_threshold < 0:
            raise ValueError("backtest_neutral_threshold must be non-negative.")
        if self.evaluation_strategy not in {"holdout", "time_series", "rolling"}:
            raise ValueError(
                "evaluation_strategy must be one of 'holdout', 'time_series', or 'rolling'."
            )
        if self.evaluation_folds <= 1 and self.evaluation_strategy == "time_series":
            raise ValueError("evaluation_folds must be greater than 1 for time series cross-validation.")
        if self.evaluation_window <= 0:
            raise ValueError("evaluation_window must be positive.")
        if self.evaluation_step <= 0:
            raise ValueError("evaluation_step must be positive.")
        self.evaluation_slippage_bps = max(0.0, float(self.evaluation_slippage_bps))
        self.evaluation_fee_bps = max(0.0, float(self.evaluation_fee_bps))
        self.evaluation_fixed_cost = max(0.0, float(self.evaluation_fixed_cost))
        self.tuning_enabled = bool(self.tuning_enabled)
        self.tuning_method = str(self.tuning_method or "random").strip().lower()
        if self.tuning_method not in {"random", "optuna", "none"}:
            raise ValueError("tuning_method must be one of 'random', 'optuna', or 'none'.")
        try:
            self.tuning_iterations = int(self.tuning_iterations)
        except (TypeError, ValueError):
            self.tuning_iterations = 10
        self.tuning_iterations = max(1, self.tuning_iterations)
        if self.tuning_folds is not None:
            try:
                self.tuning_folds = int(self.tuning_folds)
            except (TypeError, ValueError):
                self.tuning_folds = None
            if self.tuning_folds is not None and self.tuning_folds <= 1:
                raise ValueError("tuning_folds must be greater than 1 when provided.")
        if self.tuning_n_jobs is not None:
            try:
                self.tuning_n_jobs = int(self.tuning_n_jobs)
            except (TypeError, ValueError):
                self.tuning_n_jobs = None
            if self.tuning_n_jobs == 0:
                self.tuning_n_jobs = None
        if not 0.5 <= float(self.direction_confidence_threshold) < 1:
            raise ValueError("direction_confidence_threshold must be between 0.5 and 1.0.")
        if int(self.volatility_window) <= 0:
            raise ValueError("volatility_window must be a positive integer.")
        self.volatility_window = int(self.volatility_window)
        try:
            self.target_gain_pct = float(self.target_gain_pct)
        except (TypeError, ValueError):
            self.target_gain_pct = DEFAULT_TARGET_GAIN_PCT
        if self.target_gain_pct <= -0.5:
            raise ValueError("target_gain_pct must be greater than -0.5.")
        self.risk_free_rate = float(self.risk_free_rate)
        self.research_api_keys = self._normalise_strings(self.research_api_keys)
        self.research_allow_list = self._normalise_strings(
            self.research_allow_list, lower=True
        )
        self.sentiment_confidence_adjustment = bool(self.sentiment_confidence_adjustment)
        try:
            self.sentiment_confidence_window = int(self.sentiment_confidence_window)
        except (TypeError, ValueError):  # pragma: no cover - defensive defaulting
            self.sentiment_confidence_window = 7
        if self.sentiment_confidence_window <= 0:
            raise ValueError("sentiment_confidence_window must be positive.")
        try:
            self.sentiment_confidence_weight = float(self.sentiment_confidence_weight)
        except (TypeError, ValueError):  # pragma: no cover - defensive defaulting
            self.sentiment_confidence_weight = 0.2
        if self.sentiment_confidence_weight < 0:
            raise ValueError("sentiment_confidence_weight must be non-negative.")
        self.sentiment_confidence_weight = float(min(self.sentiment_confidence_weight, 2.0))
        if self.csv_price_loader_path is not None:
            self.csv_price_loader_path = Path(self.csv_price_loader_path).expanduser()
        if self.parquet_price_loader_path is not None:
            self.parquet_price_loader_path = Path(
                self.parquet_price_loader_path
            ).expanduser()
        if self.yahoo_rate_limit_per_second is not None:
            self.yahoo_rate_limit_per_second = max(
                0.0, float(self.yahoo_rate_limit_per_second)
            )
        if self.yahoo_rate_limit_per_minute is not None:
            self.yahoo_rate_limit_per_minute = max(
                0.0, float(self.yahoo_rate_limit_per_minute)
            )
        if self.yahoo_cooldown_seconds is not None:
            self.yahoo_cooldown_seconds = max(
                0.0, float(self.yahoo_cooldown_seconds)
            )
        self.price_provider_priority = self._normalise_strings(
            self.price_provider_priority, lower=True
        )
        self.disabled_providers = self._normalise_strings(
            self.disabled_providers, lower=True
        )
        self.time_series_baselines = self._normalise_strings(
            self.time_series_baselines, lower=True
        )
        self.time_series_params = self._normalise_model_params(self.time_series_params)
        if self.memory_cache_seconds is not None:
            self.memory_cache_seconds = max(60.0, float(self.memory_cache_seconds))
        if self.market_timezone is not None:
            tz_str = str(self.market_timezone).strip()
            self.market_timezone = tz_str or None
        try:
            self.k_stop = float(self.k_stop)
        except (TypeError, ValueError):
            self.k_stop = 1.0
        if not math.isfinite(self.k_stop) or self.k_stop <= 0:
            self.k_stop = 1.0
        try:
            self.monte_carlo_paths = int(self.monte_carlo_paths)
        except (TypeError, ValueError):
            self.monte_carlo_paths = 500_000
        if self.monte_carlo_paths <= 0:
            raise ValueError("monte_carlo_paths must be a positive integer.")
        if self.monte_carlo_precision is not None:
            try:
                self.monte_carlo_precision = float(self.monte_carlo_precision)
            except (TypeError, ValueError):
                self.monte_carlo_precision = None
            if self.monte_carlo_precision is not None and self.monte_carlo_precision <= 0:
                raise ValueError("monte_carlo_precision must be positive when set.")
        if self.monte_carlo_max_paths is not None:
            try:
                self.monte_carlo_max_paths = int(self.monte_carlo_max_paths)
            except (TypeError, ValueError):
                self.monte_carlo_max_paths = None
            if self.monte_carlo_max_paths is not None and self.monte_carlo_max_paths <= 0:
                raise ValueError("monte_carlo_max_paths must be positive when set.")
        self.direction_bootstrap_enabled = bool(self.direction_bootstrap_enabled)
        try:
            self.direction_bootstrap_paths = int(self.direction_bootstrap_paths)
        except (TypeError, ValueError):
            self.direction_bootstrap_paths = 2_000_000
        if self.direction_bootstrap_paths <= 0:
            raise ValueError("direction_bootstrap_paths must be a positive integer.")

        if self.direction_bootstrap_workers is not None:
            try:
                self.direction_bootstrap_workers = int(self.direction_bootstrap_workers)
            except (TypeError, ValueError):
                self.direction_bootstrap_workers = None
            if self.direction_bootstrap_workers is not None and self.direction_bootstrap_workers <= 0:
                self.direction_bootstrap_workers = None

        try:
            self.direction_bootstrap_blend = float(self.direction_bootstrap_blend)
        except (TypeError, ValueError):
            self.direction_bootstrap_blend = 0.65
        self.direction_bootstrap_blend = max(0.0, min(1.0, self.direction_bootstrap_blend))
        try:
            self.price_backfill_page_days = max(1, int(self.price_backfill_page_days))
        except (TypeError, ValueError):
            self.price_backfill_page_days = 365
        if self.price_backfill_page_size is not None:
            try:
                self.price_backfill_page_size = int(self.price_backfill_page_size)
            except (TypeError, ValueError):
                self.price_backfill_page_size = None
            if self.price_backfill_page_size is not None and self.price_backfill_page_size <= 0:
                self.price_backfill_page_size = None
        self.min_samples_per_horizon = _coerce_min_samples_per_horizon(
            self.min_samples_per_horizon
        )
        self.use_cached_training_data = bool(self.use_cached_training_data)
        if isinstance(self.buy_zone, Mapping):
            self.buy_zone = BuyZoneConfirmationSettings.from_mapping(self.buy_zone)
        elif not isinstance(self.buy_zone, BuyZoneConfirmationSettings):
            self.buy_zone = BuyZoneConfirmationSettings()

    def ensure_directories(self) -> None:
        """Ensure that data and model directories exist."""

        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        if self.database_url.startswith("sqlite:///"):
            Path(self.database_url.replace("sqlite:///", "")).parent.mkdir(
                parents=True, exist_ok=True
            )

    @property
    def price_cache_path(self) -> Path:
        return self.data_dir / f"{self.ticker}_{self.interval}_prices.csv"

    @property
    def news_cache_path(self) -> Path:
        return self.data_dir / f"{self.ticker}_news.csv"

    @property
    def model_path(self) -> Path:
        return self.model_path_for("close")

    @property
    def metrics_path(self) -> Path:
        return self.metrics_path_for("close")

    @property
    def training_cache_path(self) -> Path:
        root = self.training_cache_dir or (self.data_dir / "training_cache")
        root.mkdir(parents=True, exist_ok=True)
        return root / f"{self.ticker}_{self.interval}_training.parquet"

    @property
    def training_metadata_path(self) -> Path:
        root = self.training_cache_dir or (self.data_dir / "training_cache")
        root.mkdir(parents=True, exist_ok=True)
        return root / f"{self.ticker}_{self.interval}_training.json"

    @property
    def database_path(self) -> Path:
        if not self.database_url.startswith("sqlite:///"):
            raise ValueError("database_path is only available for SQLite URLs.")
        return Path(self.database_url.replace("sqlite:///", ""))

    def model_path_for(self, target: str, horizon: Optional[int] = None) -> Path:
        """Return the filesystem path for a persisted model of ``target``."""

        resolved_horizon = self.resolve_horizon(horizon)
        return self.models_dir / self._build_filename(
            target, resolved_horizon, suffix=".joblib"
        )

    def metrics_path_for(self, target: str, horizon: Optional[int] = None) -> Path:
        """Return the filesystem path for metrics associated with ``target``."""

        resolved_horizon = self.resolve_horizon(horizon)
        return self.models_dir / self._build_filename(
            target, resolved_horizon, suffix="_metrics.json"
        )

    def preprocessor_path_for(self, target: str, horizon: Optional[int] = None) -> Path:
        """Return the filesystem path for the preprocessing pipeline of ``target``."""

        resolved_horizon = self.resolve_horizon(horizon)
        return self.models_dir / self._build_filename(
            target, resolved_horizon, suffix="_preprocessor.joblib"
        )

    def resolve_horizon(self, horizon: Optional[int]) -> int:
        """Validate and resolve ``horizon`` against configured horizons."""

        if horizon is None:
            return self.prediction_horizons[0]
        value = self._coerce_horizon_value(horizon)
        if value <= 0:
            raise ValueError("horizon must be a positive integer")
        if value not in self.prediction_horizons:
            raise ValueError(
                f"horizon {value} is not configured. Available: {self.prediction_horizons}."
            )
        return value

    def resolve_tolerance_band(self, horizon: Optional[int]) -> float | None:
        """Return the configured tolerance band for a given horizon."""

        if horizon is None:
            return None
        try:
            resolved = int(horizon)
        except (TypeError, ValueError):
            return None

        band = self.forecast_tolerance_bands.get(resolved)
        if band is None or not math.isfinite(band) or band <= 0:
            return None
        return float(band)

    def _coerce_horizon_value(self, horizon: Any) -> int:
        if isinstance(horizon, int):
            return horizon
        if isinstance(horizon, str):
            cleaned = horizon.strip().lower()
            if not cleaned:
                raise ValueError("horizon must be an integer or supported label")
            try:
                return int(cleaned)
            except ValueError:
                pass
            match = re.fullmatch(r"(?P<value>\d+)\s*(?P<unit>[a-z]+)", cleaned)
            if match:
                base = int(match.group("value"))
                unit = match.group("unit")
                mapped = HORIZON_UNIT_TO_DAYS.get(unit)
                if mapped:
                    return base * mapped
        raise ValueError("horizon must be an integer or supported label")

    @property
    def default_horizon(self) -> int:
        return self.prediction_horizons[0]

    # ------------------------------------------------------------------
    # Normalisation helpers
    # ------------------------------------------------------------------
    def _normalise_collection(
        self,
        values: Sequence[str] | Iterable[str] | None,
        fallback: Sequence[str],
    ) -> tuple[str, ...]:
        items: list[str] = []
        if values is not None:
            for raw in values:
                if not raw:
                    continue
                name = str(raw).strip().lower()
                if name and name not in items:
                    items.append(name)
        if not items:
            items = [str(item).strip().lower() for item in fallback]
        return tuple(items)

    def _normalise_model_params(
        self, params: Mapping[str, Mapping[str, Any]] | None
    ) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        if params:
            for key, value in params.items():
                result[str(key)] = dict(value)
        result.setdefault("global", {})
        return result

    def _normalise_horizons(
        self, horizons: Sequence[int] | Iterable[int] | None
    ) -> tuple[int, ...]:
        values: list[int] = []
        if horizons is not None:
            for raw in horizons:
                if raw is None:
                    continue
                try:
                    value = int(raw)
                except (TypeError, ValueError):
                    continue
                if value <= 0:
                    continue
                if value not in values:
                    values.append(value)
        if not values:
            values = list(DEFAULT_PREDICTION_HORIZONS)
        values.sort()
        return tuple(values)

    def _build_filename(self, target: str, horizon: int, suffix: str) -> str:
        clean_target = str(target).strip().lower() or "default"
        clean_target = clean_target.replace(" ", "_")
        return f"{self.ticker}_{self.model_type}_{clean_target}_h{int(horizon)}{suffix}"

    @staticmethod
    def _normalise_strings(
        values: Sequence[str] | Iterable[str] | None, *, lower: bool = False
    ) -> tuple[str, ...]:
        if values is None:
            return tuple()
        cleaned: list[str] = []
        for raw in values:
            token = str(raw or "").strip()
            if not token:
                continue
            token = token.lower() if lower else token
            if token not in cleaned:
                cleaned.append(token)
        return tuple(cleaned)


def load_environment() -> None:
    """Load configuration from an optional ``.env`` file."""

    load_dotenv()


def build_config(
    ticker: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    interval: str = "1d",
    model_type: str = "random_forest",
    data_dir: Optional[str] = None,
    models_dir: Optional[str] = None,
    news_api_key: Optional[str] = None,
    news_limit: int = 50,
    sentiment: bool = True,
    database_url: Optional[str] = None,
    feature_sets: Optional[Mapping[str, Any] | Iterable[str] | str] = None,
    feature_toggles: Optional[FeatureToggles | Mapping[str, Any] | Iterable[str] | str] = None,
    price_feature_toggles: Optional[Mapping[str, Any] | Iterable[str] | str] = None,
    macro_merge_symbols: Optional[Iterable[str] | str] = None,
    prediction_targets: Optional[Iterable[str] | str] = None,
    prediction_horizons: Optional[Iterable[int] | str] = None,
    model_params: Optional[Mapping[str, Mapping[str, Any]]] = None,
    test_size: Optional[float] = None,
    shuffle_training: Optional[bool] = None,
    backtest_strategy: Optional[str] = None,
    backtest_window: Optional[int] = None,
    backtest_step: Optional[int] = None,
    volatility_window: Optional[int] = None,
    research_api_keys: Optional[Iterable[str] | str] = None,
    research_allow_list: Optional[Iterable[str] | str] = None,
    price_provider_priority: Optional[Iterable[str] | str] = None,
    disabled_providers: Optional[Iterable[str] | str] = None,
    memory_cache_seconds: Optional[float] = None,
    k_stop: Optional[float] = None,
    time_series_baselines: Optional[Iterable[str] | str] = None,
    time_series_params: Optional[Mapping[str, Mapping[str, Any]] | str] = None,
    buy_zone: Optional[Mapping[str, Any]] = None,
    evaluation_strategy: Optional[str] = None,
    evaluation_folds: Optional[int] = None,
    evaluation_slippage_bps: Optional[float] = None,
    evaluation_fee_bps: Optional[float] = None,
    evaluation_fixed_cost: Optional[float] = None,
    tuning_enabled: Optional[bool] = None,
    tuning_method: Optional[str] = None,
    tuning_iterations: Optional[int] = None,
    tuning_folds: Optional[int] = None,
    tuning_n_jobs: Optional[int] = None,
    monte_carlo_paths: Optional[int] = None,
    monte_carlo_precision: Optional[float] = None,
    monte_carlo_max_paths: Optional[int] = None,
    direction_bootstrap_enabled: Optional[bool] = None,
    direction_bootstrap_paths: Optional[int] = None,
    direction_bootstrap_workers: Optional[int] = None,
    direction_bootstrap_blend: Optional[float] = None,
    min_samples_per_horizon: Optional[int] = None,
    forecast_tolerance_bands: Optional[Mapping[int, Any] | str] = None,
) -> PredictorConfig:
    """Build a :class:`PredictorConfig` instance from string parameters."""

    start_dt = (
        date.fromisoformat(start_date)
        if start_date
        else date.today() - timedelta(days=365)
    )
    end_dt = date.fromisoformat(end_date) if end_date else None

    load_environment()

    db_url = (
        database_url
        or os.getenv("STOCK_PREDICTOR_DATABASE_URL")
        or _default_database_url()
    )

    research_keys_value = research_api_keys or os.getenv(
        "STOCK_PREDICTOR_RESEARCH_API_KEYS"
    )
    research_allow_value = research_allow_list or os.getenv(
        "STOCK_PREDICTOR_RESEARCH_ALLOW_LIST"
    )
    provider_priority_value = price_provider_priority or os.getenv(
        "STOCK_PREDICTOR_PRICE_PROVIDERS"
    )
    disabled_providers_value = disabled_providers or os.getenv(
        "STOCK_PREDICTOR_DISABLED_PROVIDERS"
    )

    def _coerce_provider_tokens(value: Optional[Iterable[str] | str]) -> tuple[str, ...]:
        if value is None:
            return tuple()
        if isinstance(value, str):
            tokens = [part.strip() for part in value.split(",")]
        else:
            tokens = [str(item).strip() for item in value]
        return PredictorConfig._normalise_strings(tokens, lower=True)

    memory_cache_value = memory_cache_seconds or os.getenv(
        "STOCK_PREDICTOR_MEMORY_CACHE_SECONDS"
    )
    memory_cache_float: float | None = None
    if memory_cache_value is not None:
        try:
            memory_cache_float = float(memory_cache_value)
        except (TypeError, ValueError):
            memory_cache_float = None

    monte_carlo_paths_value = monte_carlo_paths or os.getenv(
        "STOCK_PREDICTOR_MONTE_CARLO_PATHS"
    )
    monte_carlo_paths_int: int | None = None
    if monte_carlo_paths_value is not None:
        try:
            monte_carlo_paths_int = int(monte_carlo_paths_value)
        except (TypeError, ValueError):
            monte_carlo_paths_int = None

    monte_carlo_precision_value = monte_carlo_precision or os.getenv(
        "STOCK_PREDICTOR_MONTE_CARLO_PRECISION"
    )
    monte_carlo_precision_float: float | None = None
    if monte_carlo_precision_value is not None:
        try:
            monte_carlo_precision_float = float(monte_carlo_precision_value)
        except (TypeError, ValueError):
            monte_carlo_precision_float = None

    monte_carlo_max_paths_value = monte_carlo_max_paths or os.getenv(
        "STOCK_PREDICTOR_MONTE_CARLO_MAX_PATHS"
    )
    monte_carlo_max_paths_int: int | None = None
    if monte_carlo_max_paths_value is not None:
        try:
            monte_carlo_max_paths_int = int(monte_carlo_max_paths_value)
        except (TypeError, ValueError):
            monte_carlo_max_paths_int = None

    bootstrap_enabled_value = direction_bootstrap_enabled
    if bootstrap_enabled_value is None:
        bootstrap_enabled_env = os.getenv("STOCK_PREDICTOR_DIRECTION_BOOTSTRAP_ENABLED")
        if isinstance(bootstrap_enabled_env, str):
            bootstrap_enabled_value = bootstrap_enabled_env.strip().lower() in {
                "1",
                "true",
                "yes",
                "y",
                "on",
            }

    bootstrap_paths_value = direction_bootstrap_paths or os.getenv(
        "STOCK_PREDICTOR_DIRECTION_BOOTSTRAP_PATHS"
    )
    bootstrap_paths_int: int | None = None
    if bootstrap_paths_value is not None:
        try:
            bootstrap_paths_int = int(bootstrap_paths_value)
        except (TypeError, ValueError):
            bootstrap_paths_int = None

    bootstrap_workers_value = direction_bootstrap_workers or os.getenv(
        "STOCK_PREDICTOR_DIRECTION_BOOTSTRAP_WORKERS"
    )
    bootstrap_workers_int: int | None = None
    if bootstrap_workers_value is not None:
        try:
            bootstrap_workers_int = int(bootstrap_workers_value)
        except (TypeError, ValueError):
            bootstrap_workers_int = None

    bootstrap_blend_value = direction_bootstrap_blend or os.getenv(
        "STOCK_PREDICTOR_DIRECTION_BOOTSTRAP_BLEND"
    )
    bootstrap_blend_float: float | None = None
    if bootstrap_blend_value is not None:
        try:
            bootstrap_blend_float = float(bootstrap_blend_value)
        except (TypeError, ValueError):
            bootstrap_blend_float = None

    min_samples_value = min_samples_per_horizon or os.getenv(
        "STOCK_PREDICTOR_MIN_SAMPLES_PER_HORIZON"
    )
    min_samples_int: int | None = None
    if min_samples_value is not None:
        min_samples_int = _coerce_min_samples_per_horizon(min_samples_value)

    tolerance_value = forecast_tolerance_bands or os.getenv(
        "STOCK_PREDICTOR_FORECAST_TOLERANCE_BANDS"
    )
    tolerance_bands = _coerce_tolerance_bands(tolerance_value)

    stop_loss_value = k_stop
    if stop_loss_value is None:
        stop_loss_env = os.getenv("STOCK_PREDICTOR_STOP_LOSS_K")
        if stop_loss_env:
            try:
                stop_loss_value = float(stop_loss_env)
            except (TypeError, ValueError):
                stop_loss_value = None
    baselines_value = time_series_baselines or os.getenv(
        "STOCK_PREDICTOR_TIME_SERIES_BASELINES"
    )
    baseline_models = _coerce_iterable(baselines_value, ())
    params_value = time_series_params or os.getenv("STOCK_PREDICTOR_TIME_SERIES_PARAMS")
    baseline_params: dict[str, dict[str, Any]] = {}
    if isinstance(params_value, str):
        try:
            loaded = json.loads(params_value)
            if isinstance(loaded, Mapping):
                baseline_params = dict(loaded)  # type: ignore[arg-type]
        except json.JSONDecodeError:
            baseline_params = {}
    elif isinstance(params_value, Mapping):
        baseline_params = dict(params_value)
    if price_feature_toggles is None and feature_toggles is not None:
        price_feature_toggles = feature_toggles

    config_kwargs: dict[str, Any] = {
        "ticker": ticker.upper(),
        "start_date": start_dt,
        "end_date": end_dt,
        "interval": interval,
        "model_type": model_type,
        "data_dir": Path(data_dir).expanduser().resolve()
        if data_dir
        else DEFAULT_DATA_DIR,
        "models_dir": Path(models_dir).expanduser().resolve()
        if models_dir
        else DEFAULT_MODELS_DIR,
        "news_api_key": news_api_key
        or os.getenv("FINANCIALMODELINGPREP_API_KEY")
        or os.getenv("NEWS_API_KEY")
        or os.getenv("ALPHAVANTAGE_API_KEY"),
        "news_limit": news_limit,
        "sentiment": sentiment,
        "database_url": db_url,
        "feature_toggles": _coerce_feature_toggles(
            feature_toggles if feature_toggles is not None else feature_sets
        ),
        "price_feature_toggles": _coerce_price_feature_toggles(
            price_feature_toggles
        ),
        "macro_merge_symbols": _coerce_iterable(
            macro_merge_symbols, DEFAULT_MACRO_MERGE_SYMBOLS
        ),
        "prediction_targets": _coerce_iterable(
            prediction_targets, DEFAULT_PREDICTION_TARGETS
        ),
        "prediction_horizons": _coerce_int_iterable(
            prediction_horizons, DEFAULT_PREDICTION_HORIZONS
        ),
        "model_params": model_params or {"global": {}},
        "test_size": test_size if test_size is not None else DEFAULT_TEST_SIZE,
        "shuffle_training": _coerce_bool(shuffle_training, default=False),
        "backtest_strategy": backtest_strategy or DEFAULT_BACKTEST_STRATEGY,
        "backtest_window": backtest_window
        if backtest_window is not None
        else DEFAULT_BACKTEST_WINDOW,
        "backtest_step": backtest_step if backtest_step is not None else DEFAULT_BACKTEST_STEP,
        "volatility_window":
        volatility_window if volatility_window is not None else DEFAULT_VOLATILITY_WINDOW,
        "evaluation_strategy": (evaluation_strategy or "time_series"),
        "evaluation_folds": evaluation_folds if evaluation_folds is not None else 5,
        "research_api_keys": _coerce_iterable(
            research_keys_value,
            (),
        ),
        "research_allow_list": _coerce_iterable(
            research_allow_value,
            (),
        ),
        "price_provider_priority": _coerce_provider_tokens(provider_priority_value),
        "disabled_providers": _coerce_provider_tokens(disabled_providers_value),
        "memory_cache_seconds": memory_cache_float,
        "time_series_baselines": baseline_models,
        "time_series_params": baseline_params,
        "buy_zone": buy_zone,
        "evaluation_slippage_bps": evaluation_slippage_bps
        if evaluation_slippage_bps is not None
        else 0.0,
        "evaluation_fee_bps": evaluation_fee_bps if evaluation_fee_bps is not None else 0.0,
        "evaluation_fixed_cost": evaluation_fixed_cost
        if evaluation_fixed_cost is not None
        else 0.0,
        "tuning_enabled": _coerce_bool(tuning_enabled, default=True),
        "tuning_method": tuning_method or "random",
        "tuning_iterations": tuning_iterations if tuning_iterations is not None else 10,
        "tuning_folds": tuning_folds,
        "tuning_n_jobs": tuning_n_jobs,
    }
    if monte_carlo_paths_int is not None:
        config_kwargs["monte_carlo_paths"] = monte_carlo_paths_int
    if monte_carlo_precision_float is not None:
        config_kwargs["monte_carlo_precision"] = monte_carlo_precision_float
    if monte_carlo_max_paths_int is not None:
        config_kwargs["monte_carlo_max_paths"] = monte_carlo_max_paths_int
    if bootstrap_enabled_value is not None:
        config_kwargs["direction_bootstrap_enabled"] = bool(bootstrap_enabled_value)
    if bootstrap_paths_int is not None:
        config_kwargs["direction_bootstrap_paths"] = bootstrap_paths_int
    if bootstrap_workers_int is not None:
        config_kwargs["direction_bootstrap_workers"] = bootstrap_workers_int
    if bootstrap_blend_float is not None:
        config_kwargs["direction_bootstrap_blend"] = bootstrap_blend_float
    if stop_loss_value is not None:
        config_kwargs["k_stop"] = stop_loss_value
    if min_samples_int is not None:
        config_kwargs["min_samples_per_horizon"] = min_samples_int
    if tolerance_bands:
        config_kwargs["forecast_tolerance_bands"] = tolerance_bands

    config = PredictorConfig(**config_kwargs)
    config.ensure_directories()
    return config


def _coerce_feature_toggles(
    value: Optional[FeatureToggles | Mapping[str, Any] | Iterable[str] | str],
    default: Optional[FeatureToggles | Mapping[str, bool]] = None,
) -> FeatureToggles:
    default_map: dict[str, bool]
    if isinstance(default, FeatureToggles):
        default_map = default.asdict()
    else:
        default_map = dict(default or default_feature_toggles().asdict())

    toggles = FeatureToggles.from_any(value, defaults=default_map)

    implemented = {name for name, spec in FEATURE_REGISTRY.items() if spec.implemented}
    for name in list(toggles):
        if name not in implemented:
            toggles[name] = False

    return toggles


def _coerce_price_feature_toggles(
    value: Optional[Mapping[str, Any] | Iterable[str] | str],
    default: Optional[Mapping[str, bool]] = None,
) -> dict[str, bool]:
    defaults = dict(default or default_price_feature_toggles())
    if value is None:
        return defaults

    expanded = derive_price_feature_toggles(value)
    defaults.update(expanded)
    return defaults


def _coerce_iterable(
    value: Optional[Iterable[str] | str], default: Sequence[str]
) -> tuple[str, ...]:
    if value is None:
        return tuple(default)
    if isinstance(value, str):
        candidates = [part.strip() for part in value.split(",")]
    else:
        candidates = [str(item).strip() for item in value]
    return tuple(filter(None, candidates)) or tuple(default)


def _coerce_bool(value: Optional[object], *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y", "on"}:
            return True
        if normalized in {"false", "0", "no", "n", "off"}:
            return False
    return bool(value)


def _coerce_int_iterable(
    value: Optional[Iterable[int] | str], default: Sequence[int]
) -> tuple[int, ...]:
    if value is None:
        return tuple(default)
    if isinstance(value, str):
        tokens = [part.strip() for part in value.split(",")]
    else:
        tokens = list(value)
    result: list[int] = []
    for token in tokens:
        try:
            number = int(token)
        except (TypeError, ValueError):
            continue
        result.append(number)
    return tuple(result) if result else tuple(default)


def _coerce_tolerance_bands(
    value: Optional[Mapping[int | str, Any] | Iterable[Any] | str]
) -> dict[int, float]:
    """Parse tolerance bands from mappings, iterable pairs, or a compact string."""

    if value is None:
        return {}

    if isinstance(value, Mapping):
        items = value.items()
    elif isinstance(value, str):
        tokens = [token.strip() for token in value.split(",") if token.strip()]
        pairs: list[tuple[str, str]] = []
        for token in tokens:
            if ":" not in token:
                continue
            key_str, val_str = token.split(":", 1)
            pairs.append((key_str.strip(), val_str.strip()))
        items = pairs
    else:
        try:
            items = list(value)
        except Exception:
            return {}

    bands: dict[int, float] = {}
    for key, raw_value in items:
        try:
            horizon_key = int(key)
        except (TypeError, ValueError):
            continue
        try:
            band_value = float(raw_value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(band_value) or band_value <= 0:
            continue
        bands[horizon_key] = float(band_value)

    return bands


def load_config_from_mapping(payload: Mapping[str, Any]) -> PredictorConfig:
    """Create a :class:`PredictorConfig` from a mapping (JSON/YAML)."""

    if "ticker" not in payload:
        raise KeyError("Configuration mapping must include a 'ticker' value.")

    load_environment()

    known_fields = {field.name for field in fields(PredictorConfig)}
    data: dict[str, Any] = {
        key: value for key, value in payload.items() if key in known_fields
    }

    toggles_input = (
        payload.get("feature_toggles")
        or payload.get("feature_groups")
        or payload.get("feature_sets")
    )
    if toggles_input is not None:
        data["feature_toggles"] = _coerce_feature_toggles(toggles_input)
    elif "feature_toggles" in data:
        data["feature_toggles"] = _coerce_feature_toggles(data["feature_toggles"])

    if "price_feature_toggles" in payload:
        data["price_feature_toggles"] = _coerce_price_feature_toggles(
            payload.get("price_feature_toggles")
        )
    elif "price_feature_toggles" in data:
        data["price_feature_toggles"] = _coerce_price_feature_toggles(
            data["price_feature_toggles"]
        )

    if "macro_merge_symbols" in payload:
        data["macro_merge_symbols"] = _coerce_iterable(
            payload.get("macro_merge_symbols"), DEFAULT_MACRO_MERGE_SYMBOLS
        )

    if "sentiment" in data:
        data["sentiment"] = _coerce_bool(data["sentiment"], default=True)
    if "shuffle_training" in data:
        data["shuffle_training"] = _coerce_bool(data["shuffle_training"], default=False)

    if "start_date" in data and isinstance(data["start_date"], str) and data["start_date"]:
        data["start_date"] = date.fromisoformat(data["start_date"])
    if "end_date" in data and isinstance(data["end_date"], str) and data["end_date"]:
        data["end_date"] = date.fromisoformat(data["end_date"])

    for key in ("data_dir", "models_dir"):
        if key in data and data[key]:
            data[key] = Path(str(data[key])).expanduser().resolve()

    if "prediction_targets" in data:
        data["prediction_targets"] = _coerce_iterable(
            data["prediction_targets"], DEFAULT_PREDICTION_TARGETS
        )
    if "prediction_horizons" in data:
        data["prediction_horizons"] = _coerce_int_iterable(
            data["prediction_horizons"], DEFAULT_PREDICTION_HORIZONS
        )

    if "model_params" not in data or data["model_params"] is None:
        data["model_params"] = {"global": {}}

    if "time_series_baselines" in data:
        data["time_series_baselines"] = _coerce_iterable(
            data["time_series_baselines"], ()
        )

    if "time_series_params" in payload and "time_series_params" not in data:
        params_payload = payload.get("time_series_params")
        if isinstance(params_payload, str):
            try:
                params_payload = json.loads(params_payload)
            except json.JSONDecodeError:
                params_payload = None
        if params_payload is not None:
            data["time_series_params"] = params_payload

    if "k_stop" in data and data["k_stop"] is not None:
        try:
            data["k_stop"] = float(data["k_stop"])
        except (TypeError, ValueError):
            data.pop("k_stop")
    if "min_samples_per_horizon" in payload:
        data["min_samples_per_horizon"] = _coerce_min_samples_per_horizon(
            payload.get("min_samples_per_horizon")
        )
    if "monte_carlo_paths" in data:
        try:
            data["monte_carlo_paths"] = int(data["monte_carlo_paths"])
        except (TypeError, ValueError):
            data.pop("monte_carlo_paths")
    if "monte_carlo_precision" in data:
        try:
            data["monte_carlo_precision"] = float(data["monte_carlo_precision"])
        except (TypeError, ValueError):
            data.pop("monte_carlo_precision")
    if "monte_carlo_max_paths" in data:
        try:
            data["monte_carlo_max_paths"] = int(data["monte_carlo_max_paths"])
        except (TypeError, ValueError):
            data.pop("monte_carlo_max_paths")
    if "direction_bootstrap_enabled" in data:
        data["direction_bootstrap_enabled"] = _coerce_bool(
            data["direction_bootstrap_enabled"], default=True
        )
    if "direction_bootstrap_paths" in data:
        try:
            data["direction_bootstrap_paths"] = int(data["direction_bootstrap_paths"])
        except (TypeError, ValueError):
            data.pop("direction_bootstrap_paths")
    if "direction_bootstrap_workers" in data:
        try:
            workers_val = int(data["direction_bootstrap_workers"])
            if workers_val > 0:
                data["direction_bootstrap_workers"] = workers_val
            else:
                data.pop("direction_bootstrap_workers")
        except (TypeError, ValueError):
            data.pop("direction_bootstrap_workers")
    if "direction_bootstrap_blend" in data:
        try:
            data["direction_bootstrap_blend"] = float(data["direction_bootstrap_blend"])
        except (TypeError, ValueError):
            data.pop("direction_bootstrap_blend")

    config = PredictorConfig(**data)  # type: ignore[arg-type]
    config.ensure_directories()
    return config


def load_config_from_file(path: str | Path) -> PredictorConfig:
    """Load configuration from a JSON or YAML file."""

    resolved = Path(path).expanduser().resolve()
    with resolved.open("r", encoding="utf-8") as handle:
        if resolved.suffix.lower() in {".yaml", ".yml"}:
            try:
                import yaml  # type: ignore[import-not-found]
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise RuntimeError(
                    "PyYAML is required to load YAML configuration files."
                ) from exc
            payload = yaml.safe_load(handle) or {}
        else:
            payload = json.load(handle)

    if not isinstance(payload, Mapping):
        raise TypeError("Configuration file must define a mapping of values.")

    return load_config_from_mapping(payload)
