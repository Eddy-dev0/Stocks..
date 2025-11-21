"""Tests for beta-driven risk guidance and rationales."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stock_predictor.core import StockPredictorAI


class _Config:
    backtest_neutral_threshold = 0.0


def test_recommendation_includes_beta_guidance() -> None:
    predictor = StockPredictorAI.__new__(StockPredictorAI)
    predictor.config = _Config()
    predictor.metadata = {
        "latest_features": pd.DataFrame(
            [
                {
                    "Beta_SP500_21": 1.65,
                    "Beta_VIX_63": 0.55,
                }
            ]
        )
    }

    recommendation = predictor._generate_recommendation({"predicted_return": 0.02})

    beta_guidance = recommendation.get("risk_guidance", {}).get("beta")
    assert beta_guidance is not None
    assert beta_guidance["sp500"]["value"] == 1.65
    assert recommendation.get("risk_rationale")


def test_recommendation_handles_missing_beta_gracefully() -> None:
    predictor = StockPredictorAI.__new__(StockPredictorAI)
    predictor.config = _Config()
    predictor.metadata = {}

    recommendation = predictor._generate_recommendation({"predicted_return": 0.02})

    assert "beta" not in recommendation.get("risk_guidance", {})
    assert recommendation.get("risk_rationale") == []
