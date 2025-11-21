"""Unit tests for macro explanation logic."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stock_predictor.core import StockPredictorAI


def test_macro_reasons_reads_macro_context_columns() -> None:
    predictor = StockPredictorAI.__new__(StockPredictorAI)
    predictor.metadata = {}

    feature_row = pd.Series(
        {
            "Volatility_21": 0.04,
            "Trend_Slope_21": 0.25,
            "Trend_Curvature_63": -0.15,
        }
    )

    reasons = predictor._macro_reasons(feature_row)

    assert any("volatility" in reason.lower() for reason in reasons)
    assert any("trend slope positive" in reason.lower() for reason in reasons)
    assert any("trend curvature turning lower" in reason.lower() for reason in reasons)


def test_macro_reasons_include_beta_context() -> None:
    predictor = StockPredictorAI.__new__(StockPredictorAI)
    predictor.metadata = {}

    feature_row = pd.Series({"Beta_SP500_63": 1.8})

    reasons = predictor._macro_reasons(feature_row)

    assert any("beta" in reason.lower() for reason in reasons)
