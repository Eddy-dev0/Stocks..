from __future__ import annotations

"""Tests for drift-based fallbacks using compounded log returns."""

import math
from pathlib import Path
import sys

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stock_predictor.core import StockPredictorAI  # noqa: E402  pylint: disable=wrong-import-position
from stock_predictor.core.modeling.main import _historical_drift_volatility  # noqa: E402  pylint: disable=wrong-import-position


def test_compounded_drift_matches_log_return_growth() -> None:
    price_df = pd.DataFrame({"Close": [100.0, 100.1, 100.2]})
    drift_per_step, _ = _historical_drift_volatility(price_df)

    assert drift_per_step is not None

    horizon = 5.0
    predicted_return, growth_factor = StockPredictorAI._compound_log_drift(drift_per_step, horizon)

    expected_exponent = drift_per_step * horizon
    anchor_price = 101.0

    assert predicted_return == pytest.approx(math.expm1(expected_exponent))
    assert growth_factor == pytest.approx(math.exp(expected_exponent))
    assert anchor_price * growth_factor == pytest.approx(anchor_price * (1 + predicted_return))
