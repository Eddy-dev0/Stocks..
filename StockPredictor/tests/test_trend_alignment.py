"""Tests for multi-timeframe trend alignment utilities."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stock_predictor.core import PredictorConfig, StockPredictorAI
from stock_predictor.core.indicator_bundle import compute_multi_timeframe_trends


def _uptrend_frame(rows: int = 120) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=rows, freq="D")
    base = np.linspace(100, 140, rows)
    return pd.DataFrame(
        {
            "Date": dates,
            "Open": base,
            "High": base + 1,
            "Low": base - 1,
            "Close": base + np.sin(np.linspace(0, 1.5, rows)),
            "Volume": np.linspace(1_000_000, 1_500_000, rows),
        }
    )


def test_multi_timeframe_trends_identify_weekly_bullish_bias() -> None:
    frame = _uptrend_frame()
    summary = compute_multi_timeframe_trends(frame)

    weekly = summary.get("timeframes", {}).get("weekly", {})
    assert weekly
    assert weekly.get("bias") == "bullish"
    assert weekly.get("strength", 0) > 0.5
    assert weekly.get("slopes", {}).get("sma50", 0) > 0


def test_trend_alignment_boosts_confidence_on_agreement() -> None:
    config = PredictorConfig(ticker="TEST")
    predictor = StockPredictorAI(config)
    predictor.metadata["trend_summary"] = {
        "base_timeframe": "daily",
        "timeframes": {
            "daily": {"bias": "bullish", "strength": 0.7},
            "weekly": {"bias": "bullish", "strength": 0.6},
        },
    }

    adjusted, note = predictor._apply_trend_alignment_adjustment(0.5)

    assert adjusted is not None and adjusted > 0.5
    assert note is not None and "support" in note.lower()
    assert predictor.metadata.get("trend_alignment_note") == note


def test_trend_alignment_penalizes_conflicting_trends() -> None:
    config = PredictorConfig(ticker="TEST")
    predictor = StockPredictorAI(config)
    predictor.metadata["trend_summary"] = {
        "base_timeframe": "daily",
        "timeframes": {
            "daily": {"bias": "bullish", "strength": 0.7},
            "weekly": {"bias": "bearish", "strength": 0.6},
        },
    }

    adjusted, note = predictor._apply_trend_alignment_adjustment(0.8)

    assert adjusted is not None and adjusted < 0.8
    assert note is not None and "conflict" in note.lower()


def test_explanation_surfaces_trend_alignment_note() -> None:
    config = PredictorConfig(ticker="TEST")
    predictor = StockPredictorAI(config)
    note = "Weekly trend bullish supports short-term call"
    predictor.metadata["trend_alignment_note"] = note
    predictor.metadata["data_sources"] = {"prices": "test-cache"}
    predictor.metadata["latest_features"] = pd.DataFrame([
        {"Date": pd.Timestamp("2024-01-01"), "Close": 100.0}
    ])

    prediction = {
        "expected_change": 0.01,
        "expected_change_pct": 0.01,
        "target_date": None,
        "horizon": 5,
        "trend_alignment_note": note,
    }

    explanation = predictor._build_prediction_explanation(prediction, {})

    assert explanation is not None
    assert note in explanation.get("summary", "") or note in explanation.get("technical_reasons", [])
