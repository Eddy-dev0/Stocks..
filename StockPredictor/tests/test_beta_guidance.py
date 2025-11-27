"""Tests for beta-driven risk guidance and rationales."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stock_predictor.core import StockPredictorAI
from stock_predictor.core.features import _build_cross_sectional_betas


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


def test_missing_benchmark_betas_not_marked_defensive() -> None:
    dates = pd.date_range("2023-01-02", periods=40, freq="B")
    price_df = pd.DataFrame(
        {
            "Date": dates,
            "Close": np.linspace(100.0, 110.0, len(dates)),
            "Close_^GSPC": np.nan,
            "Close_^VIX": np.nan,
        }
    )

    beta_block = _build_cross_sectional_betas(price_df)
    assert beta_block is not None
    beta_columns = [col for col in beta_block.frame.columns if col.startswith("Beta_")]
    assert beta_columns
    for column in beta_columns:
        assert beta_block.frame[column].isna().all()

    predictor = StockPredictorAI.__new__(StockPredictorAI)
    predictor.config = _Config()
    predictor.metadata = {"latest_features": beta_block.frame.iloc[[-1]][beta_columns]}

    recommendation = predictor._generate_recommendation({"predicted_return": 0.02})

    assert "beta" not in recommendation.get("risk_guidance", {})
    rationales = recommendation.get("risk_rationale") or []
    assert any("unavailable" in rationale.lower() for rationale in rationales)
    assert not any("defensive" in rationale.lower() for rationale in rationales)
