from __future__ import annotations

from datetime import datetime, timedelta

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stock_predictor.screener.pattern_engine.detect_pattern import detect_pattern
from stock_predictor.screener.pattern_engine.swing_points import compute_atr, find_swing_points
from stock_predictor.screener.pattern_engine.types import Candle


def _candles_from_close(close_values: list[float]) -> list[Candle]:
    base = datetime(2025, 1, 1)
    out: list[Candle] = []
    for i, c in enumerate(close_values):
        out.append(
            Candle(
                timestamp=base + timedelta(hours=i),
                open=c,
                high=c * 1.01,
                low=c * 0.99,
                close=c,
                volume=1_000_000 + i * 1000,
            )
        )
    return out


def test_swing_points_and_atr() -> None:
    candles = _candles_from_close([100, 102, 98, 103, 97, 104, 99, 105, 100, 106, 101, 107])
    atr = compute_atr(candles)
    swings = find_swing_points(candles, left_bars=1, right_bars=1, min_move_atr=0.1)
    assert atr > 0
    assert len(swings) >= 4


def test_detect_double_bottom() -> None:
    candles = _candles_from_close([110, 106, 101, 104, 108, 103, 101.2, 105, 109, 112, 114, 116])
    det = detect_pattern(candles, "Double Bottom")
    assert det is not None


def test_detect_double_top() -> None:
    candles = _candles_from_close([100, 104, 108, 106, 103, 107.8, 108.1, 105, 102, 99, 97, 96])
    det = detect_pattern(candles, "Double Top")
    assert det is not None


def test_detect_head_and_shoulders() -> None:
    candles = _candles_from_close([100, 104, 108, 103, 111, 106, 107.8, 103, 101, 98, 97, 96])
    det = detect_pattern(candles, "Head and Shoulders")
    assert det is not None


def test_detect_triangle_and_flag() -> None:
    tri = _candles_from_close([100, 101, 102, 101.5, 102, 102.1, 102, 102.1, 102, 102.2, 102.3, 102.5, 102.7])
    tri_det = detect_pattern(tri, "Ascending Triangle")
    assert tri_det is not None

    flag = _candles_from_close([100, 102, 104, 106, 108, 110, 112, 114, 113, 112, 111, 110, 109, 108, 107])
    flag_det = detect_pattern(flag, "Flag")
    assert flag_det is not None
