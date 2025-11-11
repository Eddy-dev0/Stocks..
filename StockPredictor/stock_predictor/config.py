"""Configuration utilities for the stock predictor package."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_DATABASE_PATH = DEFAULT_DATA_DIR / "market_data.sqlite"


def _default_database_url() -> str:
    return f"sqlite:///{DEFAULT_DATABASE_PATH}"


DEFAULT_FEATURE_SETS: tuple[str, ...] = (
    "technical",
    "elliott",
    "fundamental",
    "macro",
    "sentiment",
)

DEFAULT_PREDICTION_TARGETS: tuple[str, ...] = (
    "close",
    "direction",
    "return",
    "volatility",
)

DEFAULT_TEST_SIZE = 0.2
DEFAULT_BACKTEST_STRATEGY = "rolling"
DEFAULT_BACKTEST_WINDOW = 252
DEFAULT_BACKTEST_STEP = 21


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
    feature_sets: tuple[str, ...] = field(default_factory=lambda: DEFAULT_FEATURE_SETS)
    prediction_targets: tuple[str, ...] = field(
        default_factory=lambda: DEFAULT_PREDICTION_TARGETS
    )
    model_params: dict[str, dict[str, Any]] = field(
        default_factory=lambda: {"global": {}}
    )
    test_size: float = DEFAULT_TEST_SIZE
    shuffle_training: bool = True
    backtest_strategy: str = DEFAULT_BACKTEST_STRATEGY
    backtest_window: int = DEFAULT_BACKTEST_WINDOW
    backtest_step: int = DEFAULT_BACKTEST_STEP

    def __post_init__(self) -> None:
        self.ticker = self.ticker.upper()
        self.feature_sets = self._normalise_collection(
            self.feature_sets, DEFAULT_FEATURE_SETS
        )
        self.prediction_targets = self._normalise_collection(
            self.prediction_targets, DEFAULT_PREDICTION_TARGETS
        )
        self.model_params = self._normalise_model_params(self.model_params)
        self.shuffle_training = bool(self.shuffle_training)
        self.backtest_strategy = (
            str(self.backtest_strategy).strip().lower() or DEFAULT_BACKTEST_STRATEGY
        )
        if not 0 < self.test_size < 1:
            raise ValueError("test_size must be between 0 and 1.")
        if self.backtest_window <= 0:
            raise ValueError("backtest_window must be positive.")
        if self.backtest_step <= 0:
            raise ValueError("backtest_step must be positive.")

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
        return self.models_dir / f"{self.ticker}_{self.model_type}.joblib"

    @property
    def metrics_path(self) -> Path:
        return self.models_dir / f"{self.ticker}_{self.model_type}_metrics.json"

    @property
    def database_path(self) -> Path:
        if not self.database_url.startswith("sqlite:///"):
            raise ValueError("database_path is only available for SQLite URLs.")
        return Path(self.database_url.replace("sqlite:///", ""))

    def model_path_for(self, target: str) -> Path:
        """Return the filesystem path for a persisted model of ``target``."""

        return self.models_dir / self._build_filename(target, suffix=".joblib")

    def metrics_path_for(self, target: str) -> Path:
        """Return the filesystem path for metrics associated with ``target``."""

        return self.models_dir / self._build_filename(
            target, suffix="_metrics.json"
        )

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

    def _build_filename(self, target: str, suffix: str) -> str:
        clean_target = str(target).strip().lower() or "default"
        clean_target = clean_target.replace(" ", "_")
        return f"{self.ticker}_{self.model_type}_{clean_target}{suffix}"


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
    feature_sets: Optional[Iterable[str] | str] = None,
    prediction_targets: Optional[Iterable[str] | str] = None,
    model_params: Optional[Mapping[str, Mapping[str, Any]]] = None,
    test_size: Optional[float] = None,
    shuffle_training: Optional[bool] = None,
    backtest_strategy: Optional[str] = None,
    backtest_window: Optional[int] = None,
    backtest_step: Optional[int] = None,
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

    config = PredictorConfig(
        ticker=ticker.upper(),
        start_date=start_dt,
        end_date=end_dt,
        interval=interval,
        model_type=model_type,
        data_dir=Path(data_dir).expanduser().resolve()
        if data_dir
        else DEFAULT_DATA_DIR,
        models_dir=Path(models_dir).expanduser().resolve()
        if models_dir
        else DEFAULT_MODELS_DIR,
        news_api_key=news_api_key
        or os.getenv("FINANCIALMODELINGPREP_API_KEY")
        or os.getenv("NEWS_API_KEY")
        or os.getenv("ALPHAVANTAGE_API_KEY"),
        news_limit=news_limit,
        sentiment=sentiment,
        database_url=db_url,
        feature_sets=_coerce_iterable(feature_sets, DEFAULT_FEATURE_SETS),
        prediction_targets=_coerce_iterable(
            prediction_targets, DEFAULT_PREDICTION_TARGETS
        ),
        model_params=model_params or {"global": {}},
        test_size=test_size if test_size is not None else DEFAULT_TEST_SIZE,
        shuffle_training=_coerce_bool(shuffle_training, default=True),
        backtest_strategy=backtest_strategy or DEFAULT_BACKTEST_STRATEGY,
        backtest_window=backtest_window
        if backtest_window is not None
        else DEFAULT_BACKTEST_WINDOW,
        backtest_step=backtest_step if backtest_step is not None else DEFAULT_BACKTEST_STEP,
    )
    config.ensure_directories()
    return config


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
