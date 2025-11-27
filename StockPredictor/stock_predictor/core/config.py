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

from .features import FEATURE_REGISTRY, default_feature_toggles

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
DEFAULT_TARGET_GAIN_PCT = 0.03


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
    feature_toggles: dict[str, bool] = field(default_factory=default_feature_toggles)
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
    shuffle_training: bool = True
    backtest_strategy: str = DEFAULT_BACKTEST_STRATEGY
    backtest_window: int = DEFAULT_BACKTEST_WINDOW
    backtest_step: int = DEFAULT_BACKTEST_STEP
    backtest_slippage_bps: float = 1.0
    backtest_fee_bps: float = 1.0
    backtest_neutral_threshold: float = 0.001
    evaluation_strategy: str = "holdout"
    evaluation_folds: int = 5
    evaluation_window: int = DEFAULT_BACKTEST_WINDOW
    evaluation_step: int = DEFAULT_BACKTEST_STEP
    direction_confidence_threshold: float = 0.5
    volatility_window: int = DEFAULT_VOLATILITY_WINDOW
    target_gain_pct: float = DEFAULT_TARGET_GAIN_PCT
    risk_free_rate: float = 0.0
    research_api_keys: tuple[str, ...] = field(default_factory=tuple)
    research_allow_list: tuple[str, ...] = field(default_factory=tuple)
    sentiment_confidence_adjustment: bool = False
    sentiment_confidence_window: int = 7
    sentiment_confidence_weight: float = 0.2
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
    time_series_baselines: tuple[str, ...] = field(default_factory=tuple)
    time_series_params: dict[str, dict[str, Any]] = field(default_factory=dict)
    buy_zone: BuyZoneConfirmationSettings = field(
        default_factory=BuyZoneConfirmationSettings
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
        self.evaluation_strategy = str(self.evaluation_strategy).strip().lower() or "holdout"
        if not self.sentiment:
            self.feature_toggles["sentiment"] = False
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
    feature_toggles: Optional[Mapping[str, Any] | Iterable[str] | str] = None,
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
        "prediction_targets": _coerce_iterable(
            prediction_targets, DEFAULT_PREDICTION_TARGETS
        ),
        "prediction_horizons": _coerce_int_iterable(
            prediction_horizons, DEFAULT_PREDICTION_HORIZONS
        ),
        "model_params": model_params or {"global": {}},
        "test_size": test_size if test_size is not None else DEFAULT_TEST_SIZE,
        "shuffle_training": _coerce_bool(shuffle_training, default=True),
        "backtest_strategy": backtest_strategy or DEFAULT_BACKTEST_STRATEGY,
        "backtest_window": backtest_window
        if backtest_window is not None
        else DEFAULT_BACKTEST_WINDOW,
        "backtest_step": backtest_step if backtest_step is not None else DEFAULT_BACKTEST_STEP,
        "volatility_window":
        volatility_window if volatility_window is not None else DEFAULT_VOLATILITY_WINDOW,
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
    }
    if stop_loss_value is not None:
        config_kwargs["k_stop"] = stop_loss_value

    config = PredictorConfig(**config_kwargs)
    config.ensure_directories()
    return config


def _coerce_feature_toggles(
    value: Optional[Mapping[str, Any] | Iterable[str] | str],
    default: Optional[Mapping[str, bool]] = None,
) -> dict[str, bool]:
    defaults = dict(default or default_feature_toggles())
    if value is None:
        return defaults

    toggles: dict[str, bool] = {}
    if isinstance(value, Mapping):
        for key, enabled in value.items():
            name = str(key).strip().lower()
            if name in defaults:
                toggles[name] = bool(enabled)
    else:
        if isinstance(value, str):
            tokens = [part.strip() for part in value.split(",")]
        else:
            tokens = [str(item).strip() for item in value]
        for token in tokens:
            if not token:
                continue
            name = token.lower()
            if name in defaults:
                toggles[name] = True

    defaults.update(toggles)

    implemented = {name for name, spec in FEATURE_REGISTRY.items() if spec.implemented}
    for name in list(defaults):
        if name not in implemented:
            defaults[name] = False

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

    if "sentiment" in data:
        data["sentiment"] = _coerce_bool(data["sentiment"], default=True)
    if "shuffle_training" in data:
        data["shuffle_training"] = _coerce_bool(data["shuffle_training"], default=True)

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
