from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

PatternType = Literal[
    "Double Bottom",
    "Double Top",
    "Triple Bottom",
    "Triple Top",
    "Head and Shoulders",
    "Inverted Head and Shoulders",
    "Ascending Triangle",
    "Descending Triangle",
    "Pennant",
    "Flag",
    "Bearish Flag",
    "Channel",
    "Channel Up",
    "Channel Down",
    "Cup and Handle",
    "Diamond",
]

PatternStatus = Literal["forming", "confirmed", "failed", "expired"]
PatternDirection = Literal["bullish", "bearish", "neutral"]


@dataclass(frozen=True)
class Candle:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True)
class SwingPoint:
    index: int
    timestamp: datetime
    price: float
    kind: Literal["high", "low"]


@dataclass(frozen=True)
class Trendline:
    start_index: int
    start_price: float
    end_index: int
    end_price: float
    slope: float
    intercept: float


@dataclass(frozen=True)
class PatternKeyPoint:
    index: int
    price: float
    type: str


@dataclass(frozen=True)
class PatternDetection:
    pattern_type: PatternType
    status: PatternStatus
    direction: PatternDirection
    score: float
    start_index: int
    end_index: int
    breakout_level: float
    invalidation_level: float
    neckline_level: float | None = None
    signal_index: int | None = None
    breakout_index: int | None = None
    support_level: float | None = None
    resistance_level: float | None = None
    trendline_upper: Trendline | None = None
    trendline_lower: Trendline | None = None
    trendline_neckline: Trendline | None = None
    key_points: tuple[PatternKeyPoint, ...] = field(default_factory=tuple)
    explanation: str = ""


@dataclass(frozen=True)
class TradeQuality:
    occurrences: int
    successes: int
    success_rate: float
    average_move_percent: float
    median_move_percent: float
    rating: str
    sample_warning: bool
