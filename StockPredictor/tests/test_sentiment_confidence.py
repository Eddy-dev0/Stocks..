"""Tests for sentiment-driven confidence adjustments."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stock_predictor.core import PredictorConfig, StockPredictorAI


def _sentiment_frame(values: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-01", periods=len(values), freq="D"),
            "Sentiment_Avg": values,
        }
    )


def test_sentiment_boosts_confidence_and_direction_probabilities():
    config = PredictorConfig(
        ticker="TEST",
        sentiment_confidence_adjustment=True,
        sentiment_confidence_weight=0.5,
        sentiment_confidence_window=5,
    )
    predictor = StockPredictorAI(config)
    predictor.metadata["sentiment_daily"] = _sentiment_frame([0.2, 0.25, 0.3, 0.35, 0.4])

    combined_confidence, up_prob, down_prob, factor = predictor._apply_sentiment_adjustment(
        0.4, 0.55, 0.45
    )

    assert factor is not None and factor > 0
    assert combined_confidence is not None and combined_confidence > 0.4
    assert up_prob is not None and down_prob is not None
    assert up_prob > 0.55
    assert down_prob < 0.45


def test_negative_sentiment_dampens_confidence():
    config = PredictorConfig(
        ticker="TEST",
        sentiment_confidence_adjustment=True,
        sentiment_confidence_weight=0.5,
        sentiment_confidence_window=3,
    )
    predictor = StockPredictorAI(config)
    predictor.metadata["sentiment_daily"] = _sentiment_frame([-0.5, -0.45, -0.55])

    combined_confidence, up_prob, down_prob, factor = predictor._apply_sentiment_adjustment(
        0.8, 0.6, 0.4
    )

    assert factor is not None and factor < 0
    assert combined_confidence is not None and combined_confidence < 0.8
    assert up_prob is not None and down_prob is not None
    assert up_prob < 0.6
    assert down_prob > 0.4


def test_disabled_adjustment_returns_inputs():
    config = PredictorConfig(ticker="TEST", sentiment_confidence_adjustment=False)
    predictor = StockPredictorAI(config)
    predictor.metadata["sentiment_daily"] = _sentiment_frame([0.6, 0.7, 0.8])

    combined_confidence, up_prob, down_prob, factor = predictor._apply_sentiment_adjustment(
        0.5, 0.6, 0.4
    )

    assert factor is None
    assert combined_confidence == 0.5
    assert up_prob == 0.6
    assert down_prob == 0.4
