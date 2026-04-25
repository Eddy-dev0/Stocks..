from __future__ import annotations

from .pattern_detectors.channels import detect_channel, detect_channel_down, detect_channel_up
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


def detect_pattern(candles: list[Candle], pattern_type: PatternType) -> PatternDetection | None:
    swings = find_swing_points(candles, left_bars=1, right_bars=1, min_move_atr=0.1)
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
    return mapping[pattern_type]()
