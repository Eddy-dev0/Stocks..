from __future__ import annotations

import logging
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stock_predictor.core import PredictorConfig
from stock_predictor.core.features import FeatureToggles
from stock_predictor.core.modeling.exceptions import InsufficientSamplesError
from stock_predictor.core.modeling.multi_horizon_engine import (
    HorizonArtifacts,
    MultiHorizonModelingEngine,
)


class _StubDatabase:
    def __init__(self, price_df: pd.DataFrame) -> None:
        self._price_df = price_df

    def get_prices(self, **_kwargs: object) -> pd.DataFrame:
        return self._price_df.copy()

    def get_news(self, *_args: object, **_kwargs: object) -> pd.DataFrame:
        return pd.DataFrame()

    def get_indicators(self, *_args: object, **_kwargs: object) -> pd.DataFrame:
        return pd.DataFrame()


class _StubFetcher:
    def __init__(self, price_df: pd.DataFrame) -> None:
        self._price_df = price_df
        self.database = _StubDatabase(price_df)

    def fetch_price_data(self, force: bool = False) -> pd.DataFrame:  # noqa: ARG002
        return self._price_df.copy()

    def fetch_news_data(self, force: bool = False) -> pd.DataFrame:  # noqa: ARG002
        return pd.DataFrame()


class _StubFeatureAssembler:
    def build(self, price_df: pd.DataFrame, *_args: object, **_kwargs: object):
        features = pd.DataFrame(
            {"feature": np.arange(len(price_df), dtype=float)},
            index=pd.to_datetime(price_df["Date"]),
        )
        return type("FeatureResult", (), {"features": features})()


class _NotFittedPreprocessor:
    def transform(self, _frame: pd.DataFrame):
        raise NotFittedError("not fitted")


class _ConstantModel:
    def __init__(self, value: float) -> None:
        self.value = value

    def predict(self, _transformed):
        return np.array([self.value])


def _price_frame(start: str = "2023-01-01", periods: int = 3) -> pd.DataFrame:
    dates = pd.date_range(start, periods=periods, freq="D")
    return pd.DataFrame({"Date": dates, "Close": np.linspace(100, 101, periods)})


def _engine(tmp_path: Path, price_df: pd.DataFrame) -> MultiHorizonModelingEngine:
    config = PredictorConfig(
        ticker="CACHE",
        sentiment=False,
        prediction_horizons=(1,),
        data_dir=tmp_path,
        models_dir=tmp_path / "models",
        feature_toggles=FeatureToggles(),
        model_params={"global": {"random_state": 0}},
    )
    config.ensure_directories()
    return MultiHorizonModelingEngine(
        config,
        fetcher=_StubFetcher(price_df),
        feature_assembler=_StubFeatureAssembler(),
    )


def test_predict_latest_skips_retrain_until_new_data(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    price_df = _price_frame(periods=4)
    engine = _engine(tmp_path, price_df)

    load_calls = 0
    train_calls = 0

    def _fake_load(_horizon: int) -> HorizonArtifacts:
        nonlocal load_calls
        load_calls += 1
        raise FileNotFoundError

    def _fake_train(*, horizon: int | None = None, **_kwargs: object):
        nonlocal train_calls
        train_calls += 1
        raise InsufficientSamplesError(
            "insufficient",
            horizons=(horizon or 1,),
            targets=("close_h",),
            sample_counts={1: {"close_h": 0}},
            missing_targets={1: {"close_h": 0}},
        )

    engine._load_horizon_artefacts = _fake_load  # type: ignore[method-assign]
    engine.train = _fake_train  # type: ignore[method-assign]

    with caplog.at_level(logging.INFO):
        first = engine.predict_latest(horizon=1)
        second = engine.predict_latest(horizon=1)

    assert train_calls == 1
    assert load_calls == 1
    assert first["status"] == "no_data"
    assert second["status"] == "no_data"
    warning_messages = [record.message for record in caplog.records if record.levelno == logging.WARNING]
    assert len(warning_messages) == 1

    extended_prices = _price_frame(start="2023-01-10", periods=4)
    engine.fetcher._price_df = extended_prices  # type: ignore[attr-defined]
    with caplog.at_level(logging.INFO):
        engine.predict_latest(horizon=1)
    assert train_calls == 2


def test_not_fitted_preprocessor_uses_cache(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    price_df = _price_frame(periods=5)
    engine = _engine(tmp_path, price_df)

    def _load(_horizon: int) -> HorizonArtifacts:
        return HorizonArtifacts(
            horizon=1,
            preprocessor=_NotFittedPreprocessor(),
            models={"close_h": _ConstantModel(1.0)},
            metrics={},
            sample_counts={"close_h": 0},
        )

    train_calls = 0

    def _train(*, horizon: int | None = None, **_kwargs: object):
        nonlocal train_calls
        train_calls += 1
        raise InsufficientSamplesError(
            "insufficient",
            horizons=(horizon or 1,),
            targets=("close_h",),
            sample_counts={1: {"close_h": 0}},
            missing_targets={1: {"close_h": 0}},
        )

    engine._load_horizon_artefacts = _load  # type: ignore[method-assign]
    engine.train = _train  # type: ignore[method-assign]

    with caplog.at_level(logging.DEBUG):
        first = engine.predict_latest(horizon=1)
        second = engine.predict_latest(horizon=1)

    assert train_calls == 1
    assert first["status"] == "no_data"
    assert second["status"] == "no_data"
    warning_count = len([record for record in caplog.records if record.levelno == logging.WARNING])
    info_count = len([record for record in caplog.records if record.levelno == logging.INFO])
    debug_count = len([record for record in caplog.records if record.levelno == logging.DEBUG])
    availability_warnings = [
        record
        for record in caplog.records
        if record.levelno == logging.WARNING and "Prediction unavailable" in record.message
    ]
    assert warning_count >= 1
    assert len(availability_warnings) == 1
    assert info_count >= 0
    assert debug_count >= 1


def test_build_dataset_downgrades_repeated_insufficient_logs(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    price_df = _price_frame(periods=2)
    engine = _engine(tmp_path, price_df)
    setattr(engine.config, "min_samples_per_horizon", 5)

    with caplog.at_level(logging.DEBUG):
        engine.build_dataset(horizons=(1,), targets=("close_h",))
    first_levels = [record.levelno for record in caplog.records if "Insufficient samples for horizon" in record.message]
    assert logging.WARNING in first_levels

    caplog.clear()
    with caplog.at_level(logging.DEBUG):
        engine.build_dataset(horizons=(1,), targets=("close_h",))
    second_levels = [record.levelno for record in caplog.records if "Insufficient samples for horizon" in record.message]
    assert logging.WARNING not in second_levels
    assert logging.DEBUG in second_levels
