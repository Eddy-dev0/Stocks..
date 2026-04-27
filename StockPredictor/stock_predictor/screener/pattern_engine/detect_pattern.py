from __future__ import annotations

from dataclasses import dataclass

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


@dataclass(frozen=True)
class PatternOptions:
    min_confidence: float = 60.0
    left_bars: int = 3
    right_bars: int = 3
    min_move_atr: float = 0.5
    min_move_percent: float = 0.003


def detect_patterns(
    candles: list[Candle], pattern_type: PatternType, options: PatternOptions | None = None
) -> list[PatternDetection]:
    cfg = options or PatternOptions()
    swings = find_swing_points(
        candles,
        left_bars=cfg.left_bars,
        right_bars=cfg.right_bars,
        min_move_atr=cfg.min_move_atr,
        min_move_percent=cfg.min_move_percent,
    )
    mapping = {
        "Double Bottom": lambda: detect_double_bottom(candles, swings),
        "Double Top": lambda: detect_double_top(candles, swings),
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
    det = mapping[pattern_type]()
    out = [det] if det is not None else []
    return [d for d in dedupe_detections(out) if d.score >= cfg.min_confidence]


def detect_pattern(candles: list[Candle], pattern_type: PatternType) -> PatternDetection | None:
    detections = detect_patterns(candles, pattern_type, PatternOptions())
    return detections[-1] if detections else None
