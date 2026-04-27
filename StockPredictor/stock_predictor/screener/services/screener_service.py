from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from typing import Callable

from ..market_data.provider import MarketDataProvider
from ..market_data.symbol_universe import SymbolInfo, SymbolUniverseService
from ..pattern_engine.detect_pattern import detect_pattern
from ..pattern_engine.score_pattern import score_pattern
from ..pattern_engine.trade_quality import TradeQualityOptions, calculate_trade_quality
from ..pattern_engine.types import PatternType
from .backtest_cache import BacktestCache


@dataclass(frozen=True)
class ScreenerFilters:
    min_score: float = 60.0
    min_occurrences: int = 20
    min_volume: float = 0.0
    status: str = "all"


@dataclass(frozen=True)
class ScanOptions:
    pattern_type: PatternType
    market_filter: str = "all"
    timeframe: str = "1h"
    min_confidence: float = 60.0
    min_trade_quality: int = 0
    lookback_bars: int = 500
    max_symbols: int | None = None
    include_futures: bool = True


class ScreenerService:
    def __init__(
        self,
        provider: MarketDataProvider,
        cache: BacktestCache | None = None,
        universe_service: SymbolUniverseService | None = None,
    ) -> None:
        self.provider = provider
        self.cache = cache or BacktestCache()
        self.universe_service = universe_service or SymbolUniverseService()

    def _resolve_universe(self, options: ScanOptions) -> list[SymbolInfo]:
        market_filter = options.market_filter
        if market_filter not in {"all", "stock", "future"}:
            market_filter = "all"
        universe = self.universe_service.get_universe(market_filter)
        if not options.include_futures:
            universe = [item for item in universe if item.market_type != "future"]
        if options.max_symbols is not None and options.max_symbols > 0:
            return universe[: options.max_symbols]
        return universe

    def scan_market(
        self,
        pattern_type: PatternType,
        market_type: str,
        *,
        start_date: object | None = None,
        end_date: object | None = None,
        timeframe: str = "1h",
        force_refresh_data: bool = False,
        filters: ScreenerFilters | None = None,
        progress_callback: Callable[[int, int, str], None] | None = None,
        max_symbols: int | None = None,
    ) -> list[dict[str, object]]:
        """Backwards-compatible API that now performs a market-wide scan."""
        _ = (start_date, end_date, force_refresh_data)
        active_filters = filters or ScreenerFilters()
        options = ScanOptions(
            pattern_type=pattern_type,
            market_filter=market_type,
            timeframe=timeframe,
            min_confidence=active_filters.min_score,
            min_trade_quality=active_filters.min_occurrences,
            max_symbols=max_symbols,
        )
        return self.scan_market_wide(options, filters=active_filters, progress_callback=progress_callback)

    def scan_market_wide(
        self,
        options: ScanOptions,
        *,
        filters: ScreenerFilters | None = None,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> list[dict[str, object]]:
        active_filters = filters or ScreenerFilters(
            min_score=options.min_confidence,
            min_occurrences=options.min_trade_quality,
        )
        universe = self._resolve_universe(options)
        total = len(universe)
        if total == 0:
            return []

        def scan_symbol(info: SymbolInfo) -> dict[str, object] | None:
            try:
                candles = self.provider.get_historical_bars(info.symbol, options.timeframe, options.lookback_bars)
                if len(candles) < 80:
                    return None
                detection = detect_pattern(candles, options.pattern_type)
                if not detection:
                    return None
                final_score = score_pattern(detection, candles)
                if final_score < active_filters.min_score:
                    return None
                if active_filters.status != "all" and detection.status != active_filters.status:
                    return None
                if candles[-1].volume < active_filters.min_volume:
                    return None
                cache_key = f"{info.symbol}:{options.pattern_type}:{active_filters.min_occurrences}"
                quality = self.cache.get(cache_key)
                if quality is None:
                    quality = calculate_trade_quality(
                        candles,
                        options.pattern_type,
                        detection.direction,
                        TradeQualityOptions(min_occurrences=active_filters.min_occurrences),
                    )
                    self.cache.set(cache_key, quality)
                if quality.occurrences < options.min_trade_quality:
                    return None
                return {
                    "id": f"{info.symbol}-{options.pattern_type}-{detection.end_index}",
                    "symbol": info.symbol,
                    "name": info.name,
                    "marketType": info.market_type,
                    "patternType": detection.pattern_type,
                    "direction": detection.direction.capitalize(),
                    "status": detection.status,
                    "timeframe": options.timeframe,
                    "signalTime": candles[-1].timestamp.strftime("%Y-%m-%d %H:%M"),
                    "detectedAt": candles[-1].timestamp.strftime("%Y-%m-%d %H:%M"),
                    "confidence": round(final_score, 1),
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
                    "close": candles[-1].close,
                    "lastPrice": candles[-1].close,
                    "volume": candles[-1].volume,
                    "explanation": f"{detection.pattern_type} detected on {options.timeframe} candles.",
                    "lastUpdated": datetime.utcnow().strftime("%H:%M:%S"),
                }
            except Exception:
                return None

        rows: list[dict[str, object]] = []
        serial_scan = bool(getattr(self.provider, "serial_scan", False))
        if serial_scan:
            for done_count, info in enumerate(universe, start=1):
                item = scan_symbol(info)
                if item is not None:
                    rows.append(item)
                if progress_callback:
                    progress_callback(done_count, total, f"Scanning {done_count} / {total} symbols")
        else:
            with ThreadPoolExecutor(max_workers=8) as pool:
                futures = [pool.submit(scan_symbol, info) for info in universe]
                for done_count, future in enumerate(as_completed(futures), start=1):
                    item = future.result()
                    if item is not None:
                        rows.append(item)
                    if progress_callback:
                        progress_callback(done_count, total, f"Scanning {done_count} / {total} symbols")
        rows.sort(
            key=lambda x: (
                x["tradeQuality"]["successRate"],
                x["confidence"],
                x["volume"],
            ),
            reverse=True,
        )
        return rows
