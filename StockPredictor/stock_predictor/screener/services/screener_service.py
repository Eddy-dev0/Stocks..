from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from statistics import median
from typing import Any, Callable

from ..market_data.provider import (
    MarketDataError,
    MarketDataProvider,
    normalize_symbol_for_provider,
    timeframe_period_hint,
)
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
    status: str = "forming_confirmed"
    sensitivity: str = "normal"
    show_candidates: bool = False
    debug: bool = False


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
    sensitivity: str = "normal"
    show_candidates: bool = False
    debug: bool = False
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
    providerStatus: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    timeframeUsage: dict[str, int] = field(default_factory=lambda: {"1h": 0, "60m": 0, "30m": 0, "15m": 0, "45m": 0})
    unsupportedSymbols: int = 0
    dataErrors: int = 0


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

    @staticmethod
    def scale_bars_for_timeframe(base_bars_1h: int, timeframe: str) -> int:
        if timeframe in {"1h", "60m"}:
            return base_bars_1h
        if timeframe == "30m":
            return base_bars_1h * 2
        if timeframe == "15m":
            return base_bars_1h * 4
        if timeframe == "45m":
            return int((base_bars_1h * 60 + 44) // 45)
        return base_bars_1h

    @staticmethod
    def get_min_candles(timeframe: str) -> int:
        if timeframe in {"1h", "60m", "45m"}:
            return 120
        if timeframe == "30m":
            return 240
        if timeframe == "15m":
            return 480
        return 120

    @staticmethod
    def aggregate_candles(candles: list[Candle], target_minutes: int = 45) -> list[Candle]:
        if target_minutes != 45:
            return candles
        group_size = 3
        ordered = sorted(candles, key=lambda c: c.timestamp)
        out: list[Candle] = []
        for idx in range(0, len(ordered), group_size):
            group = ordered[idx : idx + group_size]
            if len(group) < group_size:
                continue
            out.append(
                Candle(
                    timestamp=group[0].timestamp,
                    open=group[0].open,
                    high=max(c.high for c in group),
                    low=min(c.low for c in group),
                    close=group[-1].close,
                    volume=sum(c.volume for c in group),
                )
            )
        return out

    def validate_candles(
        self, symbol: str, candles: list[Candle], timeframe: str, min_required: int
    ) -> tuple[list[Candle], list[str], str | None, float | None]:
        if not isinstance(candles, list):
            return [], [], "NO_DATA", None
        if not candles:
            return [], [], "NO_DATA", None
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
        if len(clean) != len(ordered):
            warnings.append(f"Dropped {len(ordered)-len(clean)} invalid/duplicate candles for {symbol}.")
        if len(clean) < min_required:
            return clean, warnings, "NOT_ENOUGH_CANDLES", None
        if len(clean) >= 10:
            spacings = [
                (clean[i].timestamp - clean[i - 1].timestamp).total_seconds() / 60
                for i in range(1, len(clean))
                if clean[i].timestamp > clean[i - 1].timestamp
            ]
            if spacings:
                med = median(spacings)
                expected = {"1h": 60, "60m": 60, "45m": 45, "30m": 30, "15m": 15}.get(timeframe, 60)
                if med > expected * 1.8 or med < expected * 0.5:
                    warnings.append(f"Symbol {symbol} appears to have non-{timeframe} candle spacing.")
                return clean, warnings, None, med
        return clean, warnings, None, None

    def get_bars_with_fallback(self, symbol: str, lookback_bars: int) -> dict[str, Any]:
        attempts = [
            {"timeframe": "1h", "period": "60d"},
            {"timeframe": "60m", "period": "60d"},
            {"timeframe": "30m", "period": "30d"},
            {"timeframe": "15m", "period": "15d"},
        ]
        errors: list[str] = []
        for attempt in attempts:
            timeframe = str(attempt["timeframe"])
            try:
                candles = self.provider.get_historical_bars(symbol, timeframe, lookback_bars, options={"period": attempt["period"]})
                minimum = self.get_min_candles(timeframe)
                validation = self.validate_candles(symbol, candles, timeframe, minimum)
                valid_candles, warnings, reason, _spacing = validation
                if reason is None and len(valid_candles) >= minimum:
                    return {
                        "candles": valid_candles,
                        "timeframeUsed": timeframe,
                        "providerSymbol": symbol,
                        "warning": f"Fallback timeframe used: {timeframe}" if timeframe not in {"1h", "60m"} else None,
                        "warnings": warnings,
                    }
                errors.append(f"{timeframe}: {reason or 'validation_failed'}")
            except Exception as error:
                errors.append(f"{timeframe}: {error}")
        raise MarketDataError(symbol, f"No usable intraday candles. Attempts: {' | '.join(errors)}")

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
        active_lookback_bars: int = 20,
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
            active_lookback_bars=active_lookback_bars,
            sensitivity=active_filters.sensitivity,
            show_candidates=active_filters.show_candidates,
            debug=active_filters.debug,
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
        provider_status = getattr(self.provider, "provider_status", None)
        if callable(provider_status):
            debug.providerStatus = provider_status()
        if not universe:
            self.last_debug_stats = debug
            return []

        def scan_symbol(info: SymbolInfo) -> tuple[dict[str, object] | None, dict[str, object]]:
            provider_status_name = str(debug.providerStatus.get("provider", "Yahoo"))
            provider_symbol, normalize_error = normalize_symbol_for_provider(provider_status_name, info.symbol)
            if normalize_error is not None:
                metrics = {"symbol": info.symbol, "providerSymbol": "-", "candles": 0, "rejects": [("UNSUPPORTED_SYMBOL", normalize_error)], "warnings": [], "pipeline": Counter()}
                metrics["diag"] = {"symbol": info.symbol, "providerSymbol": "-", "status": "UNSUPPORTED_SYMBOL", "candles": 0, "firstTime": "-", "lastTime": "-", "lastClose": "-", "medianSpacingMinutes": "-", "error": normalize_error}
                return None, metrics
            provider_symbol = provider_symbol or self.provider.normalize_symbol(info.symbol)
            metrics: dict[str, object] = {"symbol": info.symbol, "providerSymbol": provider_symbol, "candles": 0, "rejects": [], "warnings": [], "pipeline": Counter(), "timeframeUsed": options.timeframe}
            try:
                if options.timeframe == "auto":
                    fetched = self.get_bars_with_fallback(provider_symbol, options.lookback_bars)
                    candles = fetched["candles"]
                    metrics["timeframeUsed"] = fetched["timeframeUsed"]
                    if fetched.get("warning"):
                        metrics["warnings"].append(fetched["warning"])
                elif options.timeframe == "45m":
                    base = self.provider.get_historical_bars(provider_symbol, "15m", options.lookback_bars, options={"period": timeframe_period_hint("15m")})
                    candles = self.aggregate_candles(base, 45)
                    metrics["timeframeUsed"] = "45m"
                else:
                    candles = self.provider.get_historical_bars(provider_symbol, options.timeframe, options.lookback_bars, options={"period": timeframe_period_hint(options.timeframe)})
            except Exception as error:
                metrics["diag"] = {"symbol": info.symbol, "providerSymbol": provider_symbol, "status": "ERROR", "candles": 0, "firstTime": "-", "lastTime": "-", "lastClose": "-", "medianSpacingMinutes": "-", "error": str(error)}
                reason = "UNSUPPORTED_SYMBOL" if isinstance(error, MarketDataError) and "unsupported" in str(error).lower() else "DATA_ERROR"
                metrics["rejects"].append((reason, str(error)))
                return None, metrics
            metrics["candles"] = len(candles)
            if not candles:
                metrics["diag"] = {"symbol": info.symbol, "providerSymbol": provider_symbol, "status": "ERROR", "candles": 0, "firstTime": "-", "lastTime": "-", "lastClose": "-", "medianSpacingMinutes": "-", "error": "No data returned"}
                metrics["rejects"].append(("NO_DATA", "provider returned no candles"))
                return None, metrics
            used_timeframe = str(metrics.get("timeframeUsed", options.timeframe))
            min_required = max(1, min(options.min_candles_required, self.get_min_candles(used_timeframe)))
            valid_candles, warnings, reject_reason, median_spacing = self.validate_candles(info.symbol, candles, used_timeframe, min_required)
            metrics["warnings"] = warnings
            if reject_reason is not None:
                metrics["diag"] = {"symbol": info.symbol, "providerSymbol": provider_symbol, "status": "INVALID", "candles": len(valid_candles), "firstTime": valid_candles[0].timestamp.strftime("%Y-%m-%d %H:%M") if valid_candles else "-", "lastTime": valid_candles[-1].timestamp.strftime("%Y-%m-%d %H:%M") if valid_candles else "-", "lastClose": round(valid_candles[-1].close, 4) if valid_candles else "-", "medianSpacingMinutes": round(median_spacing, 2) if median_spacing is not None else "-", "error": reject_reason}
                metrics["rejects"].append((reject_reason, f"candle_count={len(valid_candles)}"))
                return None, metrics

            swing_points = find_swing_points(valid_candles)
            swing_high_count = len([s for s in swing_points if s.kind == "high"])
            swing_low_count = len([s for s in swing_points if s.kind == "low"])
            metrics["diag"] = {
                "symbol": info.symbol,
                "providerSymbol": provider_symbol,
                "status": "OK",
                "candles": len(valid_candles),
                "firstTime": valid_candles[0].timestamp.strftime("%Y-%m-%d %H:%M"),
                "lastTime": valid_candles[-1].timestamp.strftime("%Y-%m-%d %H:%M"),
                "lastClose": round(valid_candles[-1].close, 4),
                "medianSpacingMinutes": round(median_spacing, 2) if median_spacing is not None else "-",
                "error": "",
                "swingHighCount": swing_high_count,
                "swingLowCount": swing_low_count,
                "timeframeUsed": used_timeframe,
            }
            if not swing_points:
                metrics["rejects"].append(("NO_SWING_POINTS", "swing detector returned 0 points"))
            if swing_high_count < 5 or swing_low_count < 5:
                metrics["warnings"].append("Swing detector too strict or data too flat.")

            detections = detect_patterns(
                valid_candles,
                options.pattern_type,
                PatternOptions(
                    min_confidence=35 if options.debug else active_filters.min_score,
                    min_confidence_candidate=35,
                    min_confidence_display=50,
                    min_confidence_confirmed=45,
                    active_lookback_bars=self.scale_bars_for_timeframe(options.active_lookback_bars, used_timeframe),
                    sensitivity=options.sensitivity,
                    allow_loose_fallback=True,
                    show_candidates=options.show_candidates,
                    debug=options.debug,
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
            status_map = {
                "forming_confirmed": {"forming", "confirmed"},
                "confirmed_only": {"confirmed"},
                "all": {"candidate", "forming", "confirmed"},
            }
            allowed_statuses = status_map.get(active_filters.status, {"forming", "confirmed"})
            after_market = [d for d in after_market if d.status in allowed_statuses]
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
                metrics["warnings"].append("trade quality sample below preferred threshold")
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
                "timeframe": used_timeframe,
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
            tf_used = str(metrics.get("timeframeUsed", options.timeframe))
            if tf_used in debug.timeframeUsage:
                debug.timeframeUsage[tf_used] += 1
            for warning in metrics.get("warnings", []):
                if len(debug.warnings) < 50:
                    debug.warnings.append(str(warning))
            diag = metrics.get("diag")
            if isinstance(diag, dict) and len(debug.symbolDiagnostics) < 50:
                debug.symbolDiagnostics.append(diag)
            for key, value in dict(metrics.get("pipeline", {})).items():
                debug.pipeline[key] = debug.pipeline.get(key, 0) + int(value)
            if isinstance(diag, dict) and str(diag.get("status")) == "OK":
                debug.symbolsWithEnoughCandles += 1
            for reason, detail in metrics.get("rejects", []):
                if reason == "UNSUPPORTED_SYMBOL":
                    debug.unsupportedSymbols += 1
                if reason in {"DATA_ERROR", "NO_DATA"}:
                    debug.dataErrors += 1
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
        status_rank = {"confirmed": 2, "forming": 1, "candidate": 0}
        rows.sort(
            key=lambda x: (
                status_rank.get(str(x.get("status")), -1),
                float(x.get("confidence", 0.0)),
                float(x["tradeQuality"]["successRate"]),
                str(x.get("signalTime", "")),
            ),
            reverse=True,
        )
        self.last_debug_stats = debug
        return rows
