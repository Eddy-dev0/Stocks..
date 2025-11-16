from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stock_predictor.core import PredictorConfig, StockPredictorAI
from stock_predictor.core.ml_preprocessing import PreprocessingBuilder


def test_preprocessing_pipeline_handles_non_finite_values():
    df = pd.DataFrame(
        {
            "a": [1.0, np.nan, np.inf, -np.inf, 5.0],
            "b": [10.0, 20.0, 30.0, np.nan, 50.0],
        }
    )
    builder = PreprocessingBuilder()
    pipeline = builder.fit(df)
    transformed = pipeline.transform(df)
    assert np.isfinite(transformed.to_numpy()).all()


def test_preprocessing_pipeline_reduces_dimensions_with_pca():
    rng = np.random.default_rng(42)
    df = pd.DataFrame(rng.normal(size=(60, 5)), columns=[f"f{i}" for i in range(5)])
    builder = PreprocessingBuilder(
        clip_outliers=False,
        correlation_threshold=1.0,
        use_pca=True,
        pca_components=2,
    )
    pipeline = builder.fit(df)
    transformed = pipeline.transform(df)
    assert transformed.shape[1] <= 2


class _DummyFetcher:
    def __init__(self, price_df: pd.DataFrame, fundamentals_df: pd.DataFrame | None = None) -> None:
        self._price_df = price_df
        self._fundamentals_df = fundamentals_df

    def fetch_price_data(self) -> pd.DataFrame:
        return self._price_df.copy()

    def fetch_news_data(self) -> pd.DataFrame:
        return pd.DataFrame()

    def fetch_fundamentals(self) -> pd.DataFrame:
        if self._fundamentals_df is None:
            return pd.DataFrame()
        return self._fundamentals_df.copy()

    def get_data_sources(self) -> list[str]:
        return ["dummy"]

    def refresh_all(self, force: bool = False) -> dict[str, str]:
        return {}


def _make_price_frame(rows: int = 90) -> pd.DataFrame:
    dates = pd.date_range("2021-01-01", periods=rows, freq="D")
    base = np.linspace(100.0, 120.0, rows)
    frame = pd.DataFrame(
        {
            "Date": dates,
            "Open": base + 0.5,
            "High": base + 1.0,
            "Low": base - 1.0,
            "Close": base,
            "Adj Close": base,
            "Volume": np.linspace(1_000_000, 1_500_000, rows),
        }
    )
    return frame


def test_prediction_reuses_saved_preprocessor(tmp_path):
    price_df = _make_price_frame()
    config = PredictorConfig(
        ticker="TEST",
        sentiment=False,
        prediction_targets=("close",),
        prediction_horizons=(1,),
        data_dir=tmp_path,
        models_dir=tmp_path,
        model_params={
            "global": {"n_estimators": 10, "random_state": 42},
            "preprocessing": {"clip_outliers": False},
        },
    )
    config.ensure_directories()

    trainer = StockPredictorAI(config)
    trainer.fetcher = _DummyFetcher(price_df)
    trainer.train_model()

    preprocessor_path = config.preprocessor_path_for("close", 1)
    pipeline = joblib.load(preprocessor_path)
    setattr(pipeline, "saved_marker", "persisted")
    joblib.dump(pipeline, preprocessor_path)

    predictor = StockPredictorAI(config)
    predictor.fetcher = _DummyFetcher(price_df)
    result = predictor.predict(targets=("close",))

    loaded_pipeline = predictor.preprocessors.get(("close", 1))
    assert loaded_pipeline is not None
    assert getattr(loaded_pipeline, "saved_marker", None) == "persisted"
    assert "predictions" in result
