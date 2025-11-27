from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stock_predictor.core import PredictorConfig, StockPredictorAI


class _RecordingEstimator:
    def __init__(self) -> None:
        self.seen_X: pd.DataFrame | None = None
        self.seen_y: pd.Series | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "_RecordingEstimator":
        self.seen_X = X.copy()
        self.seen_y = y.copy()
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(X))


class _RecordingFactory:
    def __init__(self, estimator: _RecordingEstimator) -> None:
        self._estimator = estimator

    def create(self, task: str, *, calibrate: bool = False, calibration_params=None) -> Pipeline:
        return Pipeline([("estimator", self._estimator)])


def test_holdout_uses_leading_rows_for_training(tmp_path: Path) -> None:
    rows = 10
    features = pd.DataFrame({"feature": np.arange(rows)})
    target = pd.Series(np.arange(rows), name="return")

    config = PredictorConfig(
        ticker="HOLD",  # holdout strategy is default but spelled out for clarity
        sentiment=False,
        prediction_targets=("return",),
        prediction_horizons=(1,),
        data_dir=tmp_path,
        models_dir=tmp_path / "models",
        evaluation_strategy="holdout",
        test_size=0.3,
        shuffle_training=False,
    )
    predictor = StockPredictorAI(config)

    estimator = _RecordingEstimator()
    factory = _RecordingFactory(estimator)
    evaluation = predictor._evaluate_model(
        factory,
        features,
        target,
        "regression",
        "return",
        calibrate_flag=False,
        preprocessor=None,
    )

    expected_split_idx = max(1, int(len(features) * (1 - config.test_size)))
    expected_split_idx = min(expected_split_idx, len(features) - 1)

    assert estimator.seen_X is not None and estimator.seen_y is not None
    pd.testing.assert_frame_equal(estimator.seen_X, features.iloc[:expected_split_idx])
    pd.testing.assert_series_equal(estimator.seen_y, target.iloc[:expected_split_idx])

    assert evaluation["splits"][0]["train_size"] == expected_split_idx
    assert evaluation["splits"][0]["test_size"] == len(features) - expected_split_idx
