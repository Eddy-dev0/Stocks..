"""Smoke tests covering training and prediction flows with volatility support."""

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

    def get_data_sources(self) -> list[str]:
        return ["dummy"]

    def refresh_all(self, force: bool = False) -> dict[str, str]:
        return {}


def _synthetic_price_frame(rows: int = 90) -> pd.DataFrame:
    dates = pd.date_range("2022-01-01", periods=rows, freq="B")
    base = np.linspace(100.0, 110.0, rows)
    oscillation = np.sin(np.linspace(0, 8, rows)) * 2.5
    close = base + oscillation
    frame = pd.DataFrame(
        {
            "Date": dates,
            "Open": close + 0.5,
            "High": close + 1.0,
            "Low": close - 1.0,
            "Close": close,
            "Adj Close": close,
            "Volume": np.linspace(1_000_000, 1_200_000, rows),
        }
    )
    return frame


def test_training_and_predict_handles_missing_volatility_model(tmp_path: Path) -> None:
    price_df = _synthetic_price_frame(80)
    fetcher = _StaticFetcher(price_df)

    config = PredictorConfig(
        ticker="TEST",
        sentiment=False,
        prediction_targets=("direction", "return", "volatility"),
        prediction_horizons=(1,),
        data_dir=tmp_path,
        models_dir=tmp_path / "models",
        model_params={
            "global": {"n_estimators": 10, "random_state": 42},
            "preprocessing": {"clip_outliers": False},
        },
        volatility_window=10,
    )
    config.ensure_directories()

    trainer = StockPredictorAI(config)
    trainer.fetcher = fetcher
    report = trainer.train_model(targets=("direction", "return", "volatility"))
    assert "volatility" in report["targets"]

    for target in ("direction", "return", "volatility"):
        assert config.metrics_path_for(target, 1).exists()

    vol_model_path = config.model_path_for("volatility", 1)
    assert vol_model_path.exists()
    vol_model_path.unlink()

    predictor = StockPredictorAI(config)
    predictor.fetcher = fetcher
    prediction = predictor.predict(targets=("direction", "return", "volatility"))

    assert config.model_path_for("volatility", 1).exists()
    metrics = prediction.get("training_metrics", {}) or {}
    assert "volatility" in metrics
