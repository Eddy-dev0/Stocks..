"""Regression tests verifying metadata refresh during prediction."""

from __future__ import annotations

from pathlib import Path
import sys
import types

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stock_predictor.core import PredictorConfig, StockPredictorAI


class _MutableFetcher:
    """Fetcher stub that allows price data to be updated between calls."""

    def __init__(self, price_df: pd.DataFrame) -> None:
        self._price_df = price_df.copy()
        self._fundamentals_df = pd.DataFrame()

    def set_price_data(self, price_df: pd.DataFrame) -> None:
        self._price_df = price_df.copy()

    def fetch_price_data(self, force: bool = False) -> pd.DataFrame:
        return self._price_df.copy()

    def fetch_news_data(self, force: bool = False) -> pd.DataFrame:
        return pd.DataFrame()

    def fetch_fundamentals(self, force: bool = False) -> pd.DataFrame:
        return self._fundamentals_df.copy()

    def get_data_sources(self) -> list[str]:
        return ["stub"]

    def refresh_all(self, force: bool = False) -> dict[str, str]:
        return {}


class _ConstantModel:
    """Simple regression model used to avoid training in tests."""

    def __init__(self) -> None:
        self.named_steps: dict[str, object] = {}

    def predict(self, _features):
        return [0.0]


def _install_minimal_prepare(predictor: StockPredictorAI, tracker: dict[str, int]) -> None:
    """Install a deterministic ``prepare_features`` implementation for tests."""

    def _minimal_prepare(self, price_df=None, news_df=None):
        tracker["count"] += 1
        if price_df is None:
            price_df = self.fetcher.fetch_price_data()
        working = price_df.copy()
        working["Date"] = pd.to_datetime(working["Date"], errors="coerce")
        working = working.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
        latest_row = working.iloc[-1]
        latest_date = pd.to_datetime(latest_row["Date"])
        latest_close = float(latest_row.get("Close", 0.0))
        features = pd.DataFrame({"bias": [1.0] * len(working)})
        latest_features = features.iloc[[-1]].copy()
        horizons = tuple(self.config.prediction_horizons)
        target_dates = {
            int(h): latest_date + pd.tseries.offsets.BDay(int(h)) for h in horizons
        }
        metadata = {
            "latest_features": latest_features,
            "raw_feature_columns": list(latest_features.columns),
            "feature_columns": list(latest_features.columns),
            "latest_close": latest_close,
            "latest_date": latest_date,
            "horizons": horizons,
            "target_dates": target_dates,
        }
        self.metadata = metadata
        self.preprocessor_templates = {}
        return features, {}, {}

    predictor.prepare_features = types.MethodType(_minimal_prepare, predictor)


def test_predict_refreshes_metadata_when_prices_advance(tmp_path: Path) -> None:
    """New price rows should trigger feature regeneration and metadata updates."""

    initial_prices = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "Close": [100.0, 101.0],
        }
    )

    fetcher = _MutableFetcher(initial_prices)

    config = PredictorConfig(
        ticker="TEST",
        sentiment=False,
        prediction_targets=("close",),
        prediction_horizons=(1,),
        data_dir=tmp_path,
        models_dir=tmp_path / "models",
        model_params={"global": {}},
    )
    config.ensure_directories()

    predictor = StockPredictorAI(config)
    predictor.fetcher = fetcher
    predictor.models[("close", 1)] = _ConstantModel()

    prepare_calls = {"count": 0}
    _install_minimal_prepare(predictor, prepare_calls)

    first_result = predictor.predict(targets=("close",), horizon=1)

    assert prepare_calls["count"] == 1

    first_as_of = pd.to_datetime(first_result["market_data_as_of"])
    first_target = pd.to_datetime(first_result["target_date"])
    assert first_as_of == pd.Timestamp("2024-01-03")
    assert first_target > first_as_of

    updated_prices = pd.concat(
        [
            initial_prices,
            pd.DataFrame(
                {
                    "Date": [pd.Timestamp("2024-01-04")],
                    "Close": [102.0],
                }
            ),
        ],
        ignore_index=True,
    )
    fetcher.set_price_data(updated_prices)

    second_result = predictor.predict(targets=("close",), horizon=1)

    assert prepare_calls["count"] == 2

    second_as_of = pd.to_datetime(second_result["market_data_as_of"])
    second_target = pd.to_datetime(second_result["target_date"])

    assert second_as_of == pd.Timestamp("2024-01-04")
    assert second_as_of > first_as_of
    assert second_target > first_target

    latest_metadata_date = pd.to_datetime(predictor.metadata.get("latest_date"))
    assert latest_metadata_date == second_as_of

    target_dates = predictor.metadata.get("target_dates", {})
    refreshed_target = pd.to_datetime(target_dates.get(1))
    assert refreshed_target is not None
    assert refreshed_target > latest_metadata_date
    assert refreshed_target == second_as_of + pd.tseries.offsets.BDay(1)


def test_predict_refreshes_with_mismatched_timezones(tmp_path: Path) -> None:
    """Mixed timezone metadata should not prevent refresh detection."""

    initial_prices = pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-02", periods=2, freq="D", tz="UTC"),
            "Close": [100.0, 101.0],
        }
    )

    fetcher = _MutableFetcher(initial_prices)

    config = PredictorConfig(
        ticker="TEST",
        sentiment=False,
        prediction_targets=("close",),
        prediction_horizons=(1,),
        data_dir=tmp_path,
        models_dir=tmp_path / "models",
        model_params={"global": {}},
    )
    config.ensure_directories()

    predictor = StockPredictorAI(config)
    predictor.fetcher = fetcher
    predictor.models[("close", 1)] = _ConstantModel()

    prepare_calls = {"count": 0}
    _install_minimal_prepare(predictor, prepare_calls)

    first_result = predictor.predict(targets=("close",), horizon=1)
    first_as_of = pd.to_datetime(first_result["market_data_as_of"])
    assert first_as_of == pd.Timestamp("2024-01-03", tz="UTC")

    # Simulate historical metadata persisted without timezone information.
    predictor.metadata["latest_date"] = pd.Timestamp("2024-01-03")

    updated_prices = pd.concat(
        [
            initial_prices,
            pd.DataFrame(
                {
                    "Date": [pd.Timestamp("2024-01-04", tz="UTC")],
                    "Close": [102.0],
                }
            ),
        ],
        ignore_index=True,
    )
    fetcher.set_price_data(updated_prices)

    second_result = predictor.predict(targets=("close",), horizon=1)

    assert prepare_calls["count"] == 2
    second_as_of = pd.to_datetime(second_result["market_data_as_of"])
    assert second_as_of == pd.Timestamp("2024-01-04", tz="UTC")
    assert second_as_of > first_as_of

    refreshed_date = pd.to_datetime(predictor.metadata.get("latest_date"))
    assert refreshed_date == second_as_of
