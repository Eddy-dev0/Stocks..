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
DEFAULT_DATABASE_PATH = DEFAULT_DATA_DIR / "market_data.sqlite"


def _default_database_url() -> str:
    return f"sqlite:///{DEFAULT_DATABASE_PATH}"


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
    )
    config.ensure_directories()
    return config
