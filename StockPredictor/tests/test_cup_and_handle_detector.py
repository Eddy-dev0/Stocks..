from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stock_predictor.screener.pattern_engine.pattern_detectors.cup_and_handle import (
    CupAndHandleOptions,
    detect_cup_and_handle_candidates,
    detect_cup_and_handle_details,
)
from stock_predictor.screener.pattern_engine.types import Candle


def _build_candles(closes: list[float], *, base_volume: float = 1_000_000) -> list[Candle]:
    base = datetime(2026, 1, 1)
    candles: list[Candle] = []
    for i, close in enumerate(closes):
        candles.append(
            Candle(
                timestamp=base + timedelta(hours=i),
                open=close * 0.998,
                high=close * 1.01,
                low=close * 0.99,
                close=close,
                volume=base_volume,
            )
        )
    return candles


def _base_pattern(*, with_breakout: bool, breakout_volume: float = 2_000_000) -> list[Candle]:
    pre = [90, 92, 94, 96, 97, 98, 99, 100]
    left_rim = [110]
    down = [108, 106, 104, 102, 100, 97, 95, 94, 93, 92]
    bottom = [91, 91, 90.8, 90.5, 91]
    up = [92, 93, 95, 97, 99, 101, 103, 105, 107, 109]
    handle = [108.5, 107.5, 106.8, 106.2, 106.6, 107.2]
    closes = pre + left_rim + down + bottom + up + handle
    if with_breakout:
        closes += [108.5, 112.0]
    candles = _build_candles(closes)
    if with_breakout:
        candles[-1] = Candle(
            timestamp=candles[-1].timestamp,
            open=candles[-1].open,
            high=candles[-1].high,
            low=candles[-1].low,
            close=candles[-1].close,
            volume=breakout_volume,
        )
    return candles


def _opts() -> CupAndHandleOptions:
    return CupAndHandleOptions(min_confidence=50.0, scan_window_min=40, scan_window_max=250)


def test_confirmed_cup_and_handle() -> None:
    det = detect_cup_and_handle_details(_base_pattern(with_breakout=True), _opts())
    assert det is not None
    assert det["status"] == "confirmed"


def test_forming_cup_and_handle_without_breakout() -> None:
    det = detect_cup_and_handle_details(_base_pattern(with_breakout=False), _opts())
    assert det is not None
    assert det["status"] == "forming"


def test_v_shaped_spike_is_rejected() -> None:
    closes = [95, 97, 99, 102, 110, 95, 110, 108, 107, 106, 107, 108, 109, 110, 111, 112]
    det = detect_cup_and_handle_details(_build_candles(closes), _opts())
    assert det is None


def test_handle_below_cup_midpoint_is_rejected() -> None:
    candles = _base_pattern(with_breakout=False)
    closes = [c.close for c in candles]
    closes[-3:] = [98, 97, 96]
    det = detect_cup_and_handle_details(_build_candles(closes), _opts())
    assert det is None


def test_handle_retraces_more_than_half_is_rejected() -> None:
    candles = _base_pattern(with_breakout=False)
    closes = [c.close for c in candles]
    closes[-6:] = [108, 104, 101, 99, 98, 100]
    det = detect_cup_and_handle_details(_build_candles(closes), _opts())
    assert det is None


def test_right_rim_far_from_left_rim_is_rejected() -> None:
    candles = _base_pattern(with_breakout=False)
    closes = [c.close for c in candles]
    closes[-16:] = [98, 97, 96, 95, 94, 93, 93, 94, 95, 96, 95, 94, 93, 92, 92, 93]
    det = detect_cup_and_handle_details(_build_candles(closes), _opts())
    assert det is None


def test_breakout_with_higher_volume_increases_confidence() -> None:
    low_vol = detect_cup_and_handle_details(_base_pattern(with_breakout=True, breakout_volume=850_000), _opts())
    high_vol = detect_cup_and_handle_details(_base_pattern(with_breakout=True, breakout_volume=2_500_000), _opts())
    assert low_vol is not None and high_vol is not None
    assert float(high_vol["confidence"]) > float(low_vol["confidence"])


def test_overlapping_detections_are_deduplicated() -> None:
    candles = _base_pattern(with_breakout=True)
    trailing = _build_candles([111.5, 111.8, 112.0, 112.2], base_volume=1_500_000)
    shifted = [
        Candle(
            timestamp=c.timestamp + timedelta(hours=len(candles)),
            open=c.open,
            high=c.high,
            low=c.low,
            close=c.close,
            volume=c.volume,
        )
        for c in trailing
    ]
    many = detect_cup_and_handle_candidates(candles + shifted, _opts())
    assert len(many) == 1
