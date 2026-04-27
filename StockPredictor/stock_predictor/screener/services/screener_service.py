from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from statistics import median
from typing import Callable

from ..market_data.provider import MarketDataProvider
from ..market_data.symbol_universe import SymbolInfo, SymbolUniverseService
from ..pattern_engine.detect_pattern import PatternOptions, detect_patterns, is_pattern_active
from ..pattern_engine.score_pattern import score_pattern
from ..pattern_engine.swing_points import find_swing_points
from ..pattern_engine.trade_quality import TradeQualityOptions, calculate_trade_quality
from ..pattern_engine.types import Candle, PatternType
from .backtest_cache import BacktestCache


@dataclass(frozen=True)
class ScreenerFilters:
    min_score: float = 50.0
    min_occurrences: int = 20
    min_volume: float = 0.0
    status: str = "all"


@dataclass(frozen=True)
class ScanOptions:
    pattern_type: PatternType
    market_filter: str = "all"
    timeframe: str = "1h"
    min_confidence: float = 50.0
    min_trade_quality: int = 0
    lookback_bars: int = 500
    min_candles_required: int = 120
    active_lookback_bars: int = 20
    max_symbols: int | None = None
    include_futures: bool = True


@dataclass
class ScreenerDebugStats:
    totalSymbols: int = 0
    scannedSymbols: int = 0
    symbolsWithData: int = 0
    symbolsWithoutData: int = 0
    symbolsWithEnoughCandles: int = 0
    symbolsWithoutEnoughCandles: int = 0
    totalCandlesLoaded: int = 0
    averageCandlesPerSymbol: float = 0.0
    patternCandidates: int = 0
    formingPatterns: int = 0
    confirmedPatterns: int = 0
    rejectedByReason: dict[str, int] = field(default_factory=dict)
    sampleRejects: list[dict[str, str]] = field(default_factory=list)
    pipeline: dict[str, int] = field(
        default_factory=lambda: {
            "rawDetections": 0,
            "activeDetections": 0,
            "afterConfidenceFilter": 0,
            "afterMarketFilter": 0,
            "afterTradeQualityFilter": 0,
            "displayedResults": 0,
        }
    )
    symbolDiagnostics: list[dict[str, float | str | int]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


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
        self.last_debug_stats = ScreenerDebugStats()

    def get_last_debug_stats(self) -> ScreenerDebugStats:
        return self.last_debug_stats

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

    def _record_reject(self, debug: ScreenerDebugStats, symbol: str, reason: str, detail: str) -> None:
        debug.rejectedByReason[reason] = debug.rejectedByReason.get(reason, 0) + 1
        if len(debug.sampleRejects) < 30:
            debug.sampleRejects.append({"symbol": symbol, "reason": reason, "detail": detail[:280]})

    def _min_candles_for_pattern(self, pattern_type: PatternType) -> int:
        mapping = {
            "Double Bottom": 80,
            "Double Top": 80,
            "Triple Bottom": 120,
            "Triple Top": 120,
            "Head and Shoulders": 120,
            "Inverted Head and Shoulders": 120,
            "Ascending Triangle": 100,
            "Descending Triangle": 100,
            "Flag": 60,
            "Bearish Flag": 60,
            "Pennant": 60,
            "Channel": 100,
            "Channel Up": 100,
            "Channel Down": 100,
            "Cup and Handle": 150,
            "Diamond": 120,
        }
        return mapping.get(pattern_type, 120)

    def validate_candles(self, symbol: str, candles: list[Candle], timeframe: str, min_required: int) -> tuple[list[Candle], list[str], str | None]:
        if not isinstance(candles, list):
            return [], [], "NO_DATA"
        if not candles:
            return [], [], "NO_DATA"
        ordered = sorted(candles, key=lambda c: c.timestamp)
        warnings: list[str] = []
        seen: set[datetime] = set()
        clean: list[Candle] = []
        for candle in ordered:
            if candle.timestamp in seen:
                continue
            seen.add(candle.timestamp)
            if not all(isinstance(v, (int, float)) for v in [candle.open, candle.high, candle.low, candle.close]):
                continue
            if candle.high < candle.low or candle.high < max(candle.open, candle.close) or candle.low > min(candle.open, candle.close):
                continue
            clean.append(candle)
        if len(clean) < min_required:
            return clean, warnings, "NOT_ENOUGH_CANDLES"
        if timeframe == "1h" and len(clean) >= 10:
            spacings = [
                (clean[i].timestamp - clean[i - 1].timestamp).total_seconds() / 60
                for i in range(1, len(clean))
                if clean[i].timestamp > clean[i - 1].timestamp
            ]
            if spacings:
                med = median(spacings)
                if med > 90 or med < 30:
                    warnings.append(f"Symbol {symbol} appears to have non-1h candle spacing.")
        return clean, warnings, None

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
        active_filters = filters or ScreenerFilters(min_score=options.min_confidence, min_occurrences=options.min_trade_quality)
        universe = self._resolve_universe(options)
        debug = ScreenerDebugStats(totalSymbols=len(universe))
        if not universe:
            self.last_debug_stats = debug
            return []

        def scan_symbol(info: SymbolInfo) -> tuple[dict[str, object] | None, dict[str, object]]:
            metrics: dict[str, object] = {"symbol": info.symbol, "candles": 0, "rejects": [], "warnings": [], "pipeline": Counter()}
            try:
                candles = self.provider.get_historical_bars(info.symbol, options.timeframe, options.lookback_bars)
            except Exception as error:
                metrics["rejects"].append(("DATA_ERROR", str(error)))
                return None, metrics
            metrics["candles"] = len(candles)
            if not candles:
                metrics["rejects"].append(("NO_DATA", "provider returned no candles"))
                return None, metrics
            min_required = max(1, min(options.min_candles_required, self._min_candles_for_pattern(options.pattern_type)))
            valid_candles, warnings, reject_reason = self.validate_candles(info.symbol, candles, options.timeframe, min_required)
            metrics["warnings"] = warnings
            if reject_reason is not None:
                metrics["rejects"].append((reject_reason, f"candle_count={len(valid_candles)}"))
                return None, metrics

            swing_points = find_swing_points(valid_candles)
            swing_high_count = len([s for s in swing_points if s.kind == "high"])
            swing_low_count = len([s for s in swing_points if s.kind == "low"])
            metrics["diag"] = {
                "symbol": info.symbol,
                "candleCount": len(valid_candles),
                "swingHighCount": swing_high_count,
                "swingLowCount": swing_low_count,
                "lastClose": valid_candles[-1].close,
            }
            if not swing_points:
                metrics["rejects"].append(("NO_SWING_POINTS", "swing detector returned 0 points"))
            if swing_high_count < 5 or swing_low_count < 5:
                metrics["warnings"].append("Swing detector too strict or data too flat.")

            detections = detect_patterns(
                valid_candles,
                options.pattern_type,
                PatternOptions(
                    min_confidence=35,
                    min_confidence_candidate=35,
                    min_confidence_display=50,
                    min_confidence_confirmed=45,
                    active_lookback_bars=options.active_lookback_bars,
                    sensitivity="normal",
                ),
            )
            metrics["pipeline"]["rawDetections"] = len(detections)
            if not detections:
                metrics["rejects"].append(("NO_PATTERN_STRUCTURE", "detector found no detections"))
                return None, metrics

            active = [d for d in detections if d.status != "failed" and is_pattern_active(d, len(valid_candles), options.active_lookback_bars)]
            metrics["pipeline"]["activeDetections"] = len(active)
            if not active:
                metrics["rejects"].append(("NO_BREAKOUT", "no active pattern in lookback window"))
                return None, metrics

            after_conf = [d for d in active if d.score >= active_filters.min_score]
            metrics["pipeline"]["afterConfidenceFilter"] = len(after_conf)
            if not after_conf:
                metrics["rejects"].append(("BELOW_CONFIDENCE", f"all < min {active_filters.min_score}"))
                return None, metrics

            after_market = list(after_conf)
            metrics["pipeline"]["afterMarketFilter"] = len(after_market)
            if active_filters.status != "all":
                after_market = [d for d in after_market if d.status == active_filters.status]
            if not after_market:
                metrics["rejects"].append(("UI_FILTERED_OUT", f"status filter={active_filters.status}"))
                return None, metrics

            detection = max(after_market, key=lambda d: d.score)
            final_score = score_pattern(detection, valid_candles)
            if valid_candles[-1].volume < active_filters.min_volume:
                metrics["rejects"].append(("UI_FILTERED_OUT", "min volume filter"))
                return None, metrics

            cache_key = f"{info.symbol}:{options.pattern_type}:{active_filters.min_occurrences}"
            quality = self.cache.get(cache_key)
            if quality is None:
                quality = calculate_trade_quality(
                    valid_candles,
                    options.pattern_type,
                    detection.direction,
                    TradeQualityOptions(min_occurrences=active_filters.min_occurrences),
                )
                self.cache.set(cache_key, quality)

            if quality.occurrences < options.min_trade_quality:
                metrics["rejects"].append(("UI_FILTERED_OUT", "trade quality occurrence filter"))
                return None, metrics
            metrics["pipeline"]["afterTradeQualityFilter"] = 1
            metrics["pipeline"]["displayedResults"] = 1

            return {
                "id": f"{info.symbol}-{options.pattern_type}-{detection.end_index}",
                "symbol": info.symbol,
                "name": info.name,
                "marketType": info.market_type,
                "patternType": detection.pattern_type,
                "direction": detection.direction.capitalize(),
                "status": detection.status,
                "timeframe": options.timeframe,
                "signalTime": valid_candles[(detection.breakout_index or detection.end_index)].timestamp.strftime("%Y-%m-%d %H:%M"),
                "detectedAt": valid_candles[-1].timestamp.strftime("%Y-%m-%d %H:%M"),
                "confidence": round(final_score, 1),
                "score": round(final_score, 1),
                "scoreBreakdown": detection.score_breakdown.__dict__,
                "tradeQuality": {
                    "rating": quality.rating,
                    "successes": quality.successes,
                    "occurrences": quality.occurrences,
                    "successRate": round(quality.success_rate, 4),
                    "averageMovePercent": round(quality.average_move_percent, 2),
                    "medianMovePercent": round(quality.median_move_percent, 2),
                    "sampleWarning": quality.sample_warning,
                },
                "close": valid_candles[-1].close,
                "lastPrice": valid_candles[-1].close,
                "volume": valid_candles[-1].volume,
                "explanation": detection.explanation or f"{detection.pattern_type} detected on {options.timeframe} candles.",
                "lastUpdated": datetime.utcnow().strftime("%H:%M:%S"),
            }, metrics

        rows: list[dict[str, object]] = []
        serial_scan = bool(getattr(self.provider, "serial_scan", False))
        total = len(universe)

        def consume_result(item: tuple[dict[str, object] | None, dict[str, object]]) -> None:
            row, metrics = item
            debug.scannedSymbols += 1
            candles_count = int(metrics.get("candles", 0))
            debug.totalCandlesLoaded += candles_count
            if candles_count > 0:
                debug.symbolsWithData += 1
            else:
                debug.symbolsWithoutData += 1
            for warning in metrics.get("warnings", []):
                if len(debug.warnings) < 50:
                    debug.warnings.append(str(warning))
            diag = metrics.get("diag")
            if isinstance(diag, dict) and len(debug.symbolDiagnostics) < 50:
                debug.symbolDiagnostics.append(diag)
            for key, value in dict(metrics.get("pipeline", {})).items():
                debug.pipeline[key] = debug.pipeline.get(key, 0) + int(value)
            if diag is not None:
                debug.symbolsWithEnoughCandles += 1
            for reason, detail in metrics.get("rejects", []):
                if reason == "NOT_ENOUGH_CANDLES":
                    debug.symbolsWithoutEnoughCandles += 1
                self._record_reject(debug, str(metrics.get("symbol", "?")), reason, detail)
            if row is not None:
                rows.append(row)
                status = str(row.get("status", ""))
                if status == "candidate":
                    debug.patternCandidates += 1
                elif status == "forming":
                    debug.formingPatterns += 1
                elif status == "confirmed":
                    debug.confirmedPatterns += 1

        if serial_scan:
            for done_count, info in enumerate(universe, start=1):
                consume_result(scan_symbol(info))
                if progress_callback:
                    progress_callback(done_count, total, f"Scanning {done_count} / {total} symbols")
        else:
            with ThreadPoolExecutor(max_workers=8) as pool:
                futures = [pool.submit(scan_symbol, info) for info in universe]
                for done_count, future in enumerate(as_completed(futures), start=1):
                    consume_result(future.result())
                    if progress_callback:
                        progress_callback(done_count, total, f"Scanning {done_count} / {total} symbols")

        scanned_for_avg = max(debug.scannedSymbols, 1)
        debug.averageCandlesPerSymbol = round(debug.totalCandlesLoaded / scanned_for_avg, 2)
        rows.sort(key=lambda x: (x["tradeQuality"]["successRate"], x["confidence"], x["volume"]), reverse=True)
        self.last_debug_stats = debug
        return rows
