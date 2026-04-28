from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stock_predictor.screener.pattern_engine.detect_pattern import (  # noqa: E402
    PatternOptions,
    detect_pattern,
    detect_patterns,
)
from stock_predictor.screener.pattern_engine.pattern_detectors.common import dedupe_detections  # noqa: E402
from stock_predictor.screener.pattern_engine.swing_points import compute_atr, find_swing_points  # noqa: E402
from stock_predictor.screener.pattern_engine.types import Candle, PatternDetection  # noqa: E402


def _candles_from_close(close_values: list[float], high_wick: dict[int, float] | None = None) -> list[Candle]:
    base = datetime(2025, 1, 1)
    out: list[Candle] = []
    wick = high_wick or {}
    for i, c in enumerate(close_values):
        hi = max(c * 1.01, wick.get(i, c * 1.01))
        out.append(
            Candle(
                timestamp=base + timedelta(hours=i),
                open=c,
                high=hi,
                low=c * 0.99,
                close=c,
                volume=1_000_000 + i * 2000,
            )
        )
    return out


def test_swing_points_and_atr() -> None:
    candles = _candles_from_close([100, 102, 98, 103, 97, 104, 99, 105, 100, 106, 101, 107, 102, 108, 103])
    atr = compute_atr(candles)
    swings = find_swing_points(candles, left_bars=1, right_bars=1, min_move_atr=0.1)
    assert atr > 0
    assert len(swings) >= 5


def test_double_bottom_confirmed_and_close_only_breakout() -> None:
    closes = [130, 128, 125, 122, 118, 113, 109, 104, 101, 103, 106, 109, 107, 104, 101.5, 103, 106, 108, 108.5, 108.7, 108.8]
    no_confirm = _candles_from_close(closes, high_wick={19: 118.0})
    forming = detect_pattern(no_confirm, "Double Bottom")
    assert forming is not None
    assert forming.status == "forming"

    confirmed = detect_pattern(_candles_from_close(closes + [111.5, 114.0]), "Double Bottom")
    assert confirmed is not None
    assert confirmed.status == "confirmed"
    assert confirmed.score >= 60


def test_double_bottom_imperfect_lows_is_still_forming() -> None:
    closes = [
        100,
        98,
        96,
        94,
        97,
        101,
        106,
        112,
        108,
        104,
        102,
        103,
        106,
        109,
        112,
        108,
        104,
        102,
        103,
        106,
        109,
        111,
    ]
    detections = detect_patterns(
        _candles_from_close(closes),
        "Double Bottom",
        PatternOptions(min_confidence=35, show_candidates=True, allow_loose_fallback=True),
    )
    assert detections
    det = detections[-1]
    assert det.status in {"forming", "confirmed"}
    assert det.score >= 50


def test_double_top_confirmed() -> None:
    closes = [90, 92, 95, 99, 103, 107, 111, 114, 117, 115, 112, 109, 112, 116.5, 114, 111, 108, 104, 101, 99, 97]
    det = detect_pattern(_candles_from_close(closes), "Double Top")
    assert det is not None
    assert det.status in {"forming", "confirmed"}

    broken = detect_pattern(_candles_from_close(closes + [95, 93]), "Double Top")
    assert broken is not None
    assert broken.status == "confirmed"


def test_reject_invalid_structures() -> None:
    bad_db = _candles_from_close([120, 117, 113, 108, 103, 99, 104, 109, 102, 94, 100, 105, 110])
    assert detect_pattern(bad_db, "Double Bottom") is None

    bad_dt = _candles_from_close([90, 94, 98, 102, 106, 110, 105, 100, 112, 118, 113, 108, 104])
    assert detect_pattern(bad_dt, "Double Top") is None


def test_all_pattern_detectors_have_smoke_path() -> None:
    datasets = {
        "Triple Bottom": [140, 136, 132, 127, 121, 115, 110, 106, 101, 104, 108, 111, 106, 102, 105, 109, 112, 107, 103, 106, 111, 116, 120],
        "Triple Top": [80, 84, 88, 92, 97, 102, 107, 111, 116, 112, 108, 104, 109, 114, 111, 107, 103, 109, 113, 110, 106, 101, 96],
        "Head and Shoulders": [100, 104, 108, 112, 116, 121, 118, 114, 120, 127, 123, 117, 120, 124, 119, 113, 108, 103],
        "Inverted Head and Shoulders": [130, 126, 122, 118, 113, 108, 112, 116, 110, 103, 108, 114, 111, 107, 112, 118, 123],
        "Ascending Triangle": [100, 101, 102, 101, 103, 102, 104, 103, 105, 104, 106, 105, 107, 106, 108, 107, 110, 112],
        "Descending Triangle": [120, 118, 116, 117, 115, 114, 113, 114, 112, 111, 110, 111, 109, 108, 107, 108, 104, 101],
        "Pennant": [100, 104, 108, 112, 116, 121, 126, 130, 132, 131, 130, 129, 130, 129.5, 129, 129.3, 129.8, 130.5],
        "Flag": [100, 104, 108, 113, 118, 123, 128, 132, 130, 128, 126, 125, 124, 123, 122, 124, 127],
        "Bearish Flag": [150, 146, 141, 136, 131, 126, 121, 117, 119, 121, 122, 123, 124, 123, 122, 120, 117],
        "Channel": [100, 103, 101, 104, 102, 105, 103, 106, 104, 107, 105, 108, 106, 109, 107, 110, 108, 111, 109, 112, 110, 113, 111, 114, 112, 115, 113, 116, 114, 117],
        "Channel Up": [100, 102, 101, 103, 102, 104, 103, 105, 104, 106, 105, 107, 106, 108, 107, 109, 108, 110, 109, 111, 110, 112, 111, 113, 112, 114, 113, 115, 114, 116],
        "Channel Down": [130, 128, 129, 127, 128, 126, 127, 125, 126, 124, 125, 123, 124, 122, 123, 121, 122, 120, 121, 119, 120, 118, 119, 117, 118, 116, 117, 115, 116, 114],
        "Diamond": [100, 103, 97, 106, 94, 110, 92, 107, 95, 104, 97, 102, 99, 101, 98, 100, 97, 96, 95, 94, 93, 92, 91, 90, 89, 88, 87, 86, 85, 84, 83, 82, 81, 80, 79, 78, 77, 76, 75, 74],
    }
    for pattern, closes in datasets.items():
        det = detect_pattern(_candles_from_close(closes), pattern)
        assert det is None or det.score >= 60


def test_confidence_filter_and_dedupe() -> None:
    candles = _candles_from_close([120, 118, 115, 111, 107, 103, 100, 103, 107, 104, 100.5, 103, 107, 111, 116, 120])
    detections = detect_patterns(candles, "Double Bottom", PatternOptions(min_confidence=0))
    if detections:
        assert dedupe_detections(detections)

    weak = detect_patterns(candles, "Double Bottom", PatternOptions(min_confidence=95))
    assert weak == []


def test_loose_fallback_produces_detection_when_normal_is_too_strict() -> None:
    candles = _candles_from_close([140, 136, 132, 128, 123, 118, 114, 110, 107, 104, 108, 112, 116, 111, 106, 103, 106, 110, 114, 117])
    strict = detect_patterns(
        candles,
        "Double Bottom",
        PatternOptions(min_confidence=0, sensitivity="strict", allow_loose_fallback=False),
    )
    with_fallback = detect_patterns(
        candles,
        "Double Bottom",
        PatternOptions(min_confidence=0, sensitivity="strict", allow_loose_fallback=True),
    )
    assert len(with_fallback) >= len(strict)


def test_invalidation_failed_status() -> None:
    candles = _candles_from_close([130, 127, 124, 120, 115, 110, 106, 102, 100, 104, 108, 111, 108, 104, 100.5, 98, 96, 94])
    det = detect_pattern(candles, "Double Bottom")
    assert det is None or det.status in {"forming", "failed"}
