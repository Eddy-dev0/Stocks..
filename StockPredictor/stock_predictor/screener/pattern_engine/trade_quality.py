from __future__ import annotations

from dataclasses import dataclass

from .backtest_pattern import calculate_moves, find_historical_occurrences, split_train_test, summarize_moves
from .types import Candle, PatternType, TradeQuality


def _rating(success_rate: float, occurrences: int) -> str:
    if occurrences < 20:
        return "Unrated"
    if success_rate >= 0.65 and occurrences >= 50:
        return "A+"
    if success_rate >= 0.58 and occurrences >= 40:
        return "A"
    if success_rate >= 0.52 and occurrences >= 30:
        return "B"
    if success_rate >= 0.45 and occurrences >= 20:
        return "C"
    return "D"


@dataclass(frozen=True)
class TradeQualityOptions:
    forward_bars: int = 20
    min_occurrences: int = 20
    use_walk_forward: bool = True
    confirmation_only: bool = True


def calculate_trade_quality(
    candles: list[Candle],
    pattern_type: PatternType,
    direction: str,
    options: TradeQualityOptions | None = None,
) -> TradeQuality:
    opts = options or TradeQualityOptions()
    bullish = direction == "bullish"
    all_hits = find_historical_occurrences(candles, pattern_type)
    if len(all_hits) < opts.min_occurrences:
        return TradeQuality(0, 0, 0.0, 0.0, 0.0, "Unrated", True)

    if opts.use_walk_forward:
        train, test = split_train_test(all_hits)
        train_moves = calculate_moves(candles, train, forward_bars=opts.forward_bars, bullish=bullish)
        target, _ = summarize_moves(train_moves)
        test_moves = calculate_moves(candles, test, forward_bars=opts.forward_bars, bullish=bullish)
        successes = sum(1 for m in test_moves if m >= target)
        occ = len(test_moves)
        avg, med = summarize_moves(test_moves)
    else:
        moves = calculate_moves(candles, all_hits, forward_bars=opts.forward_bars, bullish=bullish)
        avg, med = summarize_moves(moves)
        target = avg
        successes = sum(1 for m in moves if m >= target)
        occ = len(moves)

    success_rate = (successes / occ) if occ else 0.0
    return TradeQuality(
        occurrences=occ,
        successes=successes,
        success_rate=success_rate,
        average_move_percent=avg,
        median_move_percent=med,
        rating=_rating(success_rate, occ),
        sample_warning=occ < opts.min_occurrences,
    )
