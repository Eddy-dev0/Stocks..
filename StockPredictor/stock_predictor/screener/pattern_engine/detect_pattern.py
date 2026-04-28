from __future__ import annotations

from dataclasses import dataclass, replace

from .pattern_detectors.channels import detect_channel, detect_channel_down, detect_channel_up
from .pattern_detectors.common import dedupe_detections
from .pattern_detectors.cup_and_handle import detect_cup_and_handle
from .pattern_detectors.diamond import detect_diamond
from .pattern_detectors.double_bottom import detect_double_bottom
from .pattern_detectors.double_top import detect_double_top
from .pattern_detectors.flags import detect_bearish_flag, detect_flag, detect_pennant
from .pattern_detectors.head_and_shoulders import (
    detect_head_and_shoulders,
    detect_inverted_head_and_shoulders,
)
from .pattern_detectors.triangles import detect_ascending_triangle, detect_descending_triangle
from .pattern_detectors.triple_patterns import detect_triple_bottom, detect_triple_top
from .swing_points import find_swing_points
from .types import Candle, PatternDetection, PatternType


SWING_SENSITIVITY_PRESETS: dict[str, dict[str, float]] = {
    "loose": {"left_bars": 2, "right_bars": 2, "min_move_atr": 0.25, "min_move_percent": 0.002},
    "normal": {"left_bars": 3, "right_bars": 3, "min_move_atr": 0.5, "min_move_percent": 0.003},
    "strict": {"left_bars": 5, "right_bars": 5, "min_move_atr": 0.8, "min_move_percent": 0.005},
}


@dataclass(frozen=True)
class PatternOptions:
    min_confidence: float = 50.0
    min_confidence_candidate: float = 35.0
    min_confidence_display: float = 50.0
    min_confidence_confirmed: float = 45.0
    active_lookback_bars: int = 20
    sensitivity: str = "normal"
    allow_loose_fallback: bool = True
    show_candidates: bool = False
    debug: bool = False
    left_bars: int | None = None
    right_bars: int | None = None
    min_move_atr: float | None = None
    min_move_percent: float | None = None


def _resolve_swing_config(cfg: PatternOptions) -> tuple[int, int, float, float]:
    preset = SWING_SENSITIVITY_PRESETS.get(cfg.sensitivity, SWING_SENSITIVITY_PRESETS["normal"])
    return (
        int(cfg.left_bars if cfg.left_bars is not None else preset["left_bars"]),
        int(cfg.right_bars if cfg.right_bars is not None else preset["right_bars"]),
        float(cfg.min_move_atr if cfg.min_move_atr is not None else preset["min_move_atr"]),
        float(cfg.min_move_percent if cfg.min_move_percent is not None else preset["min_move_percent"]),
    )


def _run_detector(pattern_type: PatternType, candles: list[Candle], swings, sensitivity: str) -> PatternDetection | None:
    mapping = {
        "Double Bottom": lambda: detect_double_bottom(candles, swings, sensitivity=sensitivity),
        "Double Top": lambda: detect_double_top(candles, swings, sensitivity=sensitivity),
        "Triple Bottom": lambda: detect_triple_bottom(candles, swings),
        "Triple Top": lambda: detect_triple_top(candles, swings),
        "Head and Shoulders": lambda: detect_head_and_shoulders(candles, swings),
        "Inverted Head and Shoulders": lambda: detect_inverted_head_and_shoulders(candles, swings),
        "Ascending Triangle": lambda: detect_ascending_triangle(candles, swings),
        "Descending Triangle": lambda: detect_descending_triangle(candles, swings),
        "Pennant": lambda: detect_pennant(candles),
        "Flag": lambda: detect_flag(candles),
        "Bearish Flag": lambda: detect_bearish_flag(candles),
        "Channel": lambda: detect_channel(candles),
        "Channel Up": lambda: detect_channel_up(candles),
        "Channel Down": lambda: detect_channel_down(candles),
        "Cup and Handle": lambda: detect_cup_and_handle(candles),
        "Diamond": lambda: detect_diamond(candles),
    }
    return mapping[pattern_type]()


def is_pattern_active(detection: PatternDetection, candle_count: int, active_lookback_bars: int) -> bool:
    active_from = max(0, candle_count - max(active_lookback_bars, 1))
    if detection.status == "forming":
        return detection.end_index >= active_from
    if detection.status == "confirmed":
        return (detection.breakout_index if detection.breakout_index is not None else detection.end_index) >= active_from
    if detection.status == "candidate":
        return detection.end_index >= active_from
    return False


def detect_patterns(
    candles: list[Candle], pattern_type: PatternType, options: PatternOptions | None = None
) -> list[PatternDetection]:
    cfg = options or PatternOptions()
    passes = [cfg.sensitivity]
    if cfg.allow_loose_fallback and cfg.sensitivity != "loose":
        passes.append("loose")

    scan_start = max(0, len(candles) - 500)
    scan_candles = candles[scan_start:]
    if not scan_candles:
        return []

    detections: list[PatternDetection] = []
    for pass_sensitivity in passes:
        pass_cfg = replace(cfg, sensitivity=pass_sensitivity)
        left_bars, right_bars, min_move_atr, min_move_percent = _resolve_swing_config(pass_cfg)
        min_window = max(20, left_bars + right_bars + 10)
        start_end = max(min_window, len(scan_candles) - cfg.active_lookback_bars)
        pass_hits = 0
        for end in range(start_end, len(scan_candles) + 1):
            window = scan_candles[:end]
            swings = find_swing_points(
                window,
                left_bars=left_bars,
                right_bars=right_bars,
                min_move_atr=min_move_atr,
                min_move_percent=min_move_percent,
            )
            det = _run_detector(pattern_type, window, swings, sensitivity=pass_sensitivity)
            if det is None:
                continue
            shifted = replace(
                det,
                start_index=det.start_index + scan_start,
                end_index=det.end_index + scan_start,
                signal_index=(det.signal_index + scan_start) if det.signal_index is not None else None,
                breakout_index=(det.breakout_index + scan_start) if det.breakout_index is not None else None,
                score=(det.score - 10.0 if pass_sensitivity == "loose" and cfg.sensitivity != "loose" else det.score),
            )
            detections.append(shifted)
            pass_hits += 1
        if pass_hits > 0 and pass_sensitivity != "loose":
            break

    filtered: list[PatternDetection] = []
    for det in dedupe_detections(detections):
        if det.score < cfg.min_confidence_candidate:
            continue
        if det.status == "forming" and det.score < cfg.min_confidence_display:
            det = replace(det, status="candidate")
        if det.status == "confirmed" and det.score < cfg.min_confidence_confirmed:
            det = replace(det, status="forming")
        if det.status == "candidate" and not (cfg.show_candidates or cfg.debug):
            continue
        if is_pattern_active(det, len(candles), cfg.active_lookback_bars) and det.score >= cfg.min_confidence:
            filtered.append(det)
    return filtered


def detect_pattern(candles: list[Candle], pattern_type: PatternType) -> PatternDetection | None:
    detections = detect_patterns(candles, pattern_type, PatternOptions())
    return detections[-1] if detections else None
