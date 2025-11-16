from __future__ import annotations

from datetime import date
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stock_predictor.core import PredictorConfig, StockPredictorAI
from stock_predictor.core.features import FeatureAssembler
from stock_predictor.providers.database import Database


def _price_frame(rows: int = 160) -> pd.DataFrame:
    dates = pd.date_range("2024-01-02", periods=rows, freq="B")
    base = np.linspace(100.0, 120.0, rows)
    frame = pd.DataFrame(
        {
            "Date": dates,
            "Open": base + 0.5,
            "High": base + 1.0,
            "Low": base - 1.0,
            "Close": base,
            "Adj Close": base,
            "Volume": np.linspace(1_000_000, 1_200_000, rows),
        }
    )
    return frame


def _fundamental_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "AsOf": pd.to_datetime(["2024-02-01", "2024-05-01"]),
            "Metric": ["Revenue", "Revenue"],
            "Value": [100_000_000.0, 125_000_000.0],
        }
    )


def test_feature_assembler_emits_fundamental_columns_when_data_available() -> None:
    price_df = _price_frame()
    fundamentals_df = _fundamental_frame()

    assembler = FeatureAssembler(feature_toggles={"fundamental": True}, horizons=(1,))
    result = assembler.build(
        price_df,
        news_df=None,
        sentiment_enabled=False,
        fundamentals_df=fundamentals_df,
    )

    columns = set(result.features.columns)
    assert "Fundamental_Revenue_Latest" in columns
    assert "Fundamental_Revenue_PctChange_63" in columns
    assert result.metadata["feature_groups"]["fundamental"]["executed"] is True


class _DatabaseBackedFetcher:
    def __init__(self, price_df: pd.DataFrame, database: Database, ticker: str) -> None:
        self._price_df = price_df
        self._database = database
        self._ticker = ticker

    def fetch_price_data(self, force: bool = False) -> pd.DataFrame:
        return self._price_df.copy()

    def fetch_news_data(self, force: bool = False) -> pd.DataFrame:
        return pd.DataFrame()

    def fetch_fundamentals(self, force: bool = False) -> pd.DataFrame:
        return self._database.get_fundamentals(self._ticker)

    def get_data_sources(self) -> list[str]:
        return ["database"]

    def refresh_all(self, force: bool = False) -> dict[str, str]:
        return {}


def test_stock_predictor_features_surface_database_fundamentals(tmp_path: Path) -> None:
    price_df = _price_frame()
    db_path = tmp_path / "market.sqlite"
    database = Database(f"sqlite:///{db_path}")
    records = [
        {
            "Ticker": "TEST",
            "Statement": "",
            "Period": "",
            "AsOf": date(2024, 2, 1),
            "Metric": "Revenue",
            "Value": 100_000_000.0,
        },
        {
            "Ticker": "TEST",
            "Statement": "",
            "Period": "",
            "AsOf": date(2024, 5, 1),
            "Metric": "Revenue",
            "Value": 125_000_000.0,
        },
    ]
    database.upsert_fundamentals(records)

    config = PredictorConfig(
        ticker="TEST",
        sentiment=False,
        prediction_targets=("close",),
        prediction_horizons=(1,),
        data_dir=tmp_path,
        models_dir=tmp_path / "models",
        database_url=f"sqlite:///{db_path}",
    )
    config.ensure_directories()

    fetcher = _DatabaseBackedFetcher(price_df, database, config.ticker)
    predictor = StockPredictorAI(config)
    predictor.fetcher = fetcher

    features, _, _ = predictor.prepare_features()

    assert "Fundamental_Revenue_Latest" in features.columns
    latest_value = features["Fundamental_Revenue_Latest"].iloc[-1]
    assert latest_value == 125_000_000.0
