from datetime import date
from pathlib import Path

import pandas as pd

from stock_predictor.core.config import PredictorConfig
from stock_predictor.core.training_data import TrainingDatasetBuilder
from stock_predictor.providers.database import Database


def _sample_price_history(start: date, periods: int) -> pd.DataFrame:
    dates = pd.date_range(start=start, periods=periods, freq="B")
    return pd.DataFrame(
        {
            "Date": dates,
            "Open": [100 + i for i in range(periods)],
            "High": [101 + i for i in range(periods)],
            "Low": [99 + i for i in range(periods)],
            "Close": [100.5 + i for i in range(periods)],
            "Adj Close": [100.5 + i for i in range(periods)],
            "Volume": [1_000_000 + i for i in range(periods)],
        }
    )


def test_builder_persists_and_loads_cache(tmp_path: Path) -> None:
    db_file = tmp_path / "training.db"
    database = Database(f"sqlite:///{db_file}")
    config = PredictorConfig(
        ticker="AAPL",
        interval="1d",
        start_date=date(2023, 1, 2),
        end_date=date(2023, 3, 1),
        data_dir=tmp_path,
        min_samples_per_horizon=1,
    )

    prices = _sample_price_history(config.start_date, 40)
    database.upsert_prices("AAPL", "1d", prices)

    builder = TrainingDatasetBuilder(config, database=database)
    dataset = builder.build(force=True)

    assert not dataset.features.empty
    assert dataset.targets
    assert config.training_cache_path.exists()

    # Second invocation should hit the cache
    cached = builder.build(force=False)
    assert cached.features.equals(dataset.features)
    assert cached.metadata["feature_columns"] == dataset.metadata["feature_columns"]
