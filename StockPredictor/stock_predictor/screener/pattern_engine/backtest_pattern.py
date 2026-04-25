from __future__ import annotations

from statistics import median

from .detect_pattern import detect_pattern
from .types import Candle, PatternType


def find_historical_occurrences(candles: list[Candle], pattern_type: PatternType, min_score: float = 60.0) -> list[int]:
    hits: list[int] = []
    for i in range(60, len(candles) - 1):
        det = detect_pattern(candles[: i + 1], pattern_type)
        if det and det.status == "confirmed" and det.score >= min_score:
            hits.append(i)
    return hits


def calculate_moves(
    candles: list[Candle],
    indexes: list[int],
    *,
    forward_bars: int,
    bullish: bool,
) -> list[float]:
    moves: list[float] = []
    for idx in indexes:
        if idx + 1 >= len(candles):
            continue
        entry = candles[idx].close
        fw = candles[idx + 1 : idx + 1 + forward_bars]
        if not fw:
            continue
        px = max(c.high for c in fw) if bullish else min(c.low for c in fw)
        mv = (px - entry) / max(entry, 1e-9) if bullish else (entry - px) / max(entry, 1e-9)
        moves.append(max(0.0, mv * 100))
    return moves


def split_train_test(items: list[int], ratio: float = 0.7) -> tuple[list[int], list[int]]:
    n = int(len(items) * ratio)
    return items[:n], items[n:]


def summarize_moves(moves: list[float]) -> tuple[float, float]:
    if not moves:
        return 0.0, 0.0
    avg = sum(moves) / len(moves)
    return avg, float(median(moves))
