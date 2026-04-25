from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

from ..market_data.provider import MarketDataProvider
from ..pattern_engine.detect_pattern import detect_pattern
from ..pattern_engine.score_pattern import score_pattern
from ..pattern_engine.trade_quality import TradeQualityOptions, calculate_trade_quality
from ..pattern_engine.types import PatternType
from .backtest_cache import BacktestCache
from .pattern_scan_scheduler import parallel_map


@dataclass(frozen=True)
class ScreenerFilters:
    min_score: float = 60.0
    min_occurrences: int = 20
    min_volume: float = 0.0
    status: str = "all"


class ScreenerService:
    def __init__(self, provider: MarketDataProvider, cache: BacktestCache | None = None) -> None:
        self.provider = provider
        self.cache = cache or BacktestCache()

    def scan_market(
        self,
        pattern_type: PatternType,
        market_type: str,
        *,
        start_date: date,
        end_date: date,
        filters: ScreenerFilters,
    ) -> list[dict[str, object]]:
        symbols = self.provider.get_universe(market_type)

        def scan_symbol(symbol: str) -> dict[str, object] | None:
            candles = self.provider.get_historical_bars(symbol, "1h", start_date, end_date)
            if len(candles) < 80:
                return None
            detection = detect_pattern(candles, pattern_type)
            if not detection:
                return None
            final_score = score_pattern(detection, candles)
            if final_score < filters.min_score:
                return None
            if filters.status != "all" and detection.status != filters.status:
                return None
            if candles[-1].volume < filters.min_volume:
                return None
            cache_key = f"{symbol}:{pattern_type}:{filters.min_occurrences}"
            quality = self.cache.get(cache_key)
            if quality is None:
                quality = calculate_trade_quality(
                    candles,
                    pattern_type,
                    detection.direction,
                    TradeQualityOptions(min_occurrences=filters.min_occurrences),
                )
                self.cache.set(cache_key, quality)
            meta = self.provider.get_symbol_metadata(symbol)
            return {
                "id": f"{symbol}-{pattern_type}-{detection.end_index}",
                "symbol": symbol,
                "name": meta.get("name", symbol),
                "marketType": meta.get("market_type", "stock"),
                "patternType": detection.pattern_type,
                "status": detection.status,
                "timeframe": "1h",
                "detectedAt": candles[-1].timestamp.strftime("%Y-%m-%d %H:%M"),
                "score": round(final_score, 1),
                "tradeQuality": {
                    "rating": quality.rating,
                    "successes": quality.successes,
                    "occurrences": quality.occurrences,
                    "successRate": round(quality.success_rate, 4),
                    "averageMovePercent": round(quality.average_move_percent, 2),
                    "medianMovePercent": round(quality.median_move_percent, 2),
                    "sampleWarning": quality.sample_warning,
                },
                "levels": {
                    "breakout": detection.breakout_level,
                    "invalidation": detection.invalidation_level,
                    "neckline": detection.neckline_level,
                },
                "volume": candles[-1].volume,
                "lastPrice": candles[-1].close,
                "lastUpdated": datetime.utcnow().strftime("%H:%M:%S"),
            }

        rows = [item for item in parallel_map(symbols, scan_symbol) if item is not None]
        rows.sort(
            key=lambda x: (
                1 if x["status"] == "confirmed" else 0,
                x["tradeQuality"]["successes"],
                x["score"],
                x["volume"],
            ),
            reverse=True,
        )
        return rows
