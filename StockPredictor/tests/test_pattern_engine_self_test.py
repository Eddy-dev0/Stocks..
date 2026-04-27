from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stock_predictor.screener.pattern_engine.detect_pattern import PatternOptions, detect_patterns
from stock_predictor.screener.pattern_engine.pattern_test_fixtures import (
    create_bearish_flag_fixture,
    create_double_bottom_fixture,
    create_flag_fixture,
)


def test_double_bottom_fixture_detects() -> None:
    detections = detect_patterns(create_double_bottom_fixture(), "Double Bottom", PatternOptions(min_confidence=0, active_lookback_bars=40))
    assert detections
    assert detections[-1].status in {"candidate", "forming", "confirmed"}


def test_flag_fixture_detects() -> None:
    detections = detect_patterns(create_flag_fixture(), "Flag", PatternOptions(min_confidence=0, active_lookback_bars=40))
    assert detections


def test_bearish_flag_fixture_detects() -> None:
    detections = detect_patterns(create_bearish_flag_fixture(), "Bearish Flag", PatternOptions(min_confidence=0, active_lookback_bars=40))
    assert detections
