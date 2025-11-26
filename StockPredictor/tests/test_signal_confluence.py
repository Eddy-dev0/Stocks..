from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stock_predictor.core.indicator_bundle import evaluate_signal_confluence


def test_confluence_detects_aligned_signals() -> None:
    frame = pd.DataFrame(
        {
            "EMA_12": [100, 102, 104, 105],
            "EMA_26": [98, 99, 101, 102],
            "EMA_50": [95, 96, 97, 98],
            "RSI_14": [52, 55, 58, 62],
            "OBV": [1_000_000, 1_050_000, 1_120_000, 1_200_000],
        }
    )

    assessment = evaluate_signal_confluence(frame)

    assert assessment.passed is True
    assert assessment.score > 0.6
    assert assessment.components["ema_alignment"] > 0
    assert assessment.components["rsi_bullish"] > 0
    assert assessment.components["obv_slope"] > 0


def test_confluence_blocks_when_signals_conflict() -> None:
    frame = pd.DataFrame(
        {
            "EMA_12": [104, 103, 102, 101],
            "EMA_26": [102, 102, 103, 104],
            "EMA_50": [100, 100, 101, 102],
            "RSI_14": [48, 46, 44, 42],
            "OBV": [1_000_000, 980_000, 960_000, 940_000],
        }
    )

    assessment = evaluate_signal_confluence(frame)

    assert assessment.passed is False
    assert assessment.score < 0.6
    assert assessment.components["ema_alignment"] < 1
