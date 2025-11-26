from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stock_predictor.core import PredictorConfig, StockPredictorAI


class _StaticFetcher:
    def __init__(self, price_df: pd.DataFrame) -> None:
        self._price_df = price_df

    def fetch_price_data(self, force: bool = False) -> pd.DataFrame:
        return self._price_df.copy()

    def fetch_news_data(self, force: bool = False) -> pd.DataFrame:
        return pd.DataFrame()

    def get_data_sources(self) -> list[str]:  # pragma: no cover - interface stub
        return ["dummy"]

    def refresh_all(self, force: bool = False) -> dict[str, str]:  # pragma: no cover - interface stub
        return {}


def _synthetic_price_frame(rows: int = 120) -> pd.DataFrame:
    dates = pd.date_range("2022-01-01", periods=rows, freq="B")
    base_trend = np.linspace(50.0, 80.0, rows)
    oscillation = np.sin(np.linspace(0, 10, rows)) * 1.5
    close = base_trend + oscillation
    high = close + 1.25
    low = close - 1.25
    volume = np.linspace(500_000, 600_000, rows)
    return pd.DataFrame(
        {
            "Date": dates,
            "Open": close,
            "High": high,
            "Low": low,
            "Close": close,
            "Adj Close": close,
            "Volume": volume,
        }
    )


def test_target_hit_probability_training_and_inference(tmp_path: Path) -> None:
    price_df = _synthetic_price_frame(90)
    fetcher = _StaticFetcher(price_df)

    config = PredictorConfig(
        ticker="TEST",
        sentiment=False,
        prediction_targets=("target_hit",),
        prediction_horizons=(5,),
        data_dir=tmp_path,
        models_dir=tmp_path / "models",
        target_return_threshold=0.05,
        model_params={"global": {"random_state": 7}},
    )
    config.ensure_directories()

    trainer = StockPredictorAI(config)
    trainer.fetcher = fetcher
    report = trainer.train_model(targets=("target_hit",))
    assert "target_hit" in report["targets"]

    predictor = StockPredictorAI(config)
    predictor.fetcher = fetcher
    prediction = predictor.predict(targets=("target_hit",))
    probability = prediction.get("target_hit_probability")
    assert probability is not None
    assert 0.0 <= float(probability) <= 100.0
