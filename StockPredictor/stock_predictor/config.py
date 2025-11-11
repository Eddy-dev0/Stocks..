"""Configuration utilities for the stock predictor package."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"


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
    feature_sets: list[str] = field(
        default_factory=lambda: ["technical", "elliott", "fundamental", "sentiment", "macro"]
    )
    prediction_targets: list[str] = field(
        default_factory=lambda: ["close", "return", "direction"]
    )
    model_params: dict[str, dict[str, object]] = field(default_factory=dict)
    test_size: float = 0.2
    shuffle_training: bool = False
    backtest_strategy: str = "rolling"
    backtest_window: int = 252
    backtest_step: int = 20

    def ensure_directories(self) -> None:
        """Ensure that data and model directories exist."""

        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)

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

    def model_path_for(self, target: str) -> Path:
        return self.models_dir / f"{self.ticker}_{self.model_type}_{target}.joblib"

    def metrics_path_for(self, target: str) -> Path:
        return self.models_dir / f"{self.ticker}_{self.model_type}_{target}_metrics.json"

    @property
    def database_path(self) -> Path:
        return self.models_dir / "experiments.sqlite"


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
    feature_sets: Optional[list[str]] = None,
    prediction_targets: Optional[list[str]] = None,
    model_params: Optional[dict[str, dict[str, object]]] = None,
    test_size: float = 0.2,
    shuffle_training: bool = False,
    backtest_strategy: str = "rolling",
    backtest_window: int = 252,
    backtest_step: int = 20,
) -> PredictorConfig:
    """Build a :class:`PredictorConfig` instance from string parameters."""

    start_dt = (
        date.fromisoformat(start_date)
        if start_date
        else date.today() - timedelta(days=365)
    )
    end_dt = date.fromisoformat(end_date) if end_date else None

    load_environment()

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
        feature_sets=feature_sets or ["technical", "elliott", "fundamental", "sentiment", "macro"],
        prediction_targets=prediction_targets or ["close", "return", "direction"],
        model_params=model_params or {},
        test_size=test_size,
        shuffle_training=shuffle_training,
        backtest_strategy=backtest_strategy,
        backtest_window=backtest_window,
        backtest_step=backtest_step,
    )
    config.ensure_directories()
    return config
