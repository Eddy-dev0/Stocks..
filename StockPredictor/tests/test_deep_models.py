import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import torch
except Exception:  # pragma: no cover - optional dependency
    torch = None

from stock_predictor.core.models import ModelFactory

pytestmark = pytest.mark.skipif(torch is None, reason="PyTorch is not installed")


def _synthetic_series(samples: int = 64) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(42)
    base = np.linspace(0, 5, samples)
    noise = rng.normal(scale=0.1, size=samples)
    drift = np.sin(np.linspace(0, 3, samples))
    y = base + drift + noise
    X = pd.DataFrame({"close": y, "volume": base * 2 + rng.normal(scale=0.05, size=samples)})
    return X, pd.Series(y)


def test_lstm_regressor_produces_predictions():
    X, y = _synthetic_series(samples=50)
    factory = ModelFactory(
        "lstm",
        {
            "sequence_length": 5,
            "hidden_size": 16,
            "num_layers": 1,
            "epochs": 3,
            "batch_size": 8,
            "lr": 5e-3,
        },
    )
    model = factory.create("regression")
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == y.shape
    assert np.isfinite(preds).all()


def test_transformer_factory_integration():
    X, y = _synthetic_series(samples=40)
    factory = ModelFactory(
        "transformer",
        {
            "sequence_length": 6,
            "hidden_size": 32,
            "epochs": 2,
            "batch_size": 4,
            "nhead": 2,
            "dim_feedforward": 64,
        },
    )
    model = factory.create("regression")
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == y.shape
    # ensure predictions reflect correlation with target trend
    assert np.corrcoef(preds, y.to_numpy())[0, 1] > 0.5


def test_lstm_classifier_outputs_probabilities():
    X, y = _synthetic_series(samples=32)
    labels = (y > y.median()).astype(int)
    factory = ModelFactory(
        "lstm",
        {
            "sequence_length": 4,
            "hidden_size": 8,
            "num_layers": 1,
            "epochs": 2,
            "batch_size": 4,
            "lr": 1e-3,
        },
    )
    model = factory.create("classification")
    model.fit(X, labels)
    proba = model.predict_proba(X)
    assert proba.shape[0] == len(labels)
    assert proba.shape[1] == 2
    row_sums = np.sum(proba, axis=1)
    assert np.allclose(row_sums, 1, atol=1e-3)
