from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from types import SimpleNamespace
import threading
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stock_predictor.screener.market_data.symbol_universe import SymbolInfo
from stock_predictor.screener.pattern_engine.types import Candle
from stock_predictor.screener.services.screener_service import (
    ScanOptions,
    ScreenerFilters,
    ScreenerService,
)


@dataclass
class _Quality:
    occurrences: int = 25
    successes: int = 18
    success_rate: float = 0.72
    average_move_percent: float = 2.4
    median_move_percent: float = 1.9
    rating: str = "A"
    sample_warning: bool = False


class _Provider:
    def __init__(self, candles: dict[str, list[Candle]]) -> None:
        self._candles = candles

    def get_universe(self, market_type: str) -> list[str]:
        return list(self._candles)

    def get_historical_bars(self, symbol: str, timeframe: str, lookback: int) -> list[Candle]:
        return self._candles.get(symbol, [])

    def get_historical_bars_batch(self, symbols: list[str], timeframe: str, lookback: int) -> dict[str, list[Candle]]:
        return {symbol: self.get_historical_bars(symbol, timeframe, lookback) for symbol in symbols}

    def subscribe_live_bars(self, symbols: list[str], timeframe: str, callback):
        return None

    def normalize_symbol(self, symbol: str) -> str:
        return symbol

    def get_symbol_metadata(self, symbol: str):
        return {"symbol": symbol, "name": symbol, "market_type": "stock"}


class _SerialProvider(_Provider):
    serial_scan = True


class _Universe:
    def get_universe(self, market_filter: str) -> list[SymbolInfo]:
        return [
            SymbolInfo("AAPL", "Apple", "stock"),
            SymbolInfo("MSFT", "Microsoft", "stock"),
            SymbolInfo("TSLA", "Tesla", "stock"),
        ]


def _candles_from_close(close_values: list[float]) -> list[Candle]:
    base = datetime(2025, 1, 1)
    out: list[Candle] = []
    for i, c in enumerate(close_values):
        out.append(
            Candle(
                timestamp=base + timedelta(hours=i),
                open=c,
                high=c * 1.01,
                low=c * 0.99,
                close=c,
                volume=1_000_000 + i * 100,
            )
        )
    return out


def test_scan_market_returns_only_matching_symbols(monkeypatch) -> None:
    candles = {
        "AAPL": _candles_from_close([110, 106, 101, 104, 108, 103, 101.2, 105, 109, 112, 114, 116] * 8),
        "MSFT": _candles_from_close([115, 110, 104, 108, 111, 106, 103.8, 107, 110, 113, 116, 118] * 8),
        "TSLA": _candles_from_close([100, 102, 104, 106, 108, 110, 109, 108, 107, 106, 105, 104] * 8),
    }
    provider = _Provider(candles)
    service = ScreenerService(provider, universe_service=_Universe())

    monkeypatch.setattr(
        "stock_predictor.screener.services.screener_service.calculate_trade_quality",
        lambda *_args, **_kwargs: _Quality(),
    )
    monkeypatch.setattr(
        "stock_predictor.screener.services.screener_service.detect_patterns",
        lambda _candles, _pattern, _opts: []
        if _candles[-1].close == 104
        else [
            SimpleNamespace(
                pattern_type="Double Bottom",
                status="confirmed",
                direction="bullish",
                end_index=90,
                breakout_index=90,
                score=88.0,
                explanation="",
                score_breakdown=SimpleNamespace(structure=0, total=88),
            )
        ],
    )
    monkeypatch.setattr(
        "stock_predictor.screener.services.screener_service.score_pattern",
        lambda _detection, _candles: 88.0,
    )

    rows = service.scan_market(
        "Double Bottom",
        "stock",
        filters=ScreenerFilters(min_score=0, min_occurrences=0),
    )

    symbols = {row["symbol"] for row in rows}
    assert symbols == {"AAPL", "MSFT"}


def test_progress_callback_runs_on_caller_thread(monkeypatch) -> None:
    candles = {
        "AAPL": _candles_from_close([110, 106, 101, 104, 108, 103, 101.2, 105, 109, 112, 114, 116] * 8),
        "MSFT": _candles_from_close([115, 110, 104, 108, 111, 106, 103.8, 107, 110, 113, 116, 118] * 8),
    }
    provider = _Provider(candles)
    service = ScreenerService(provider, universe_service=_Universe())

    monkeypatch.setattr(
        "stock_predictor.screener.services.screener_service.calculate_trade_quality",
        lambda *_args, **_kwargs: _Quality(),
    )
    monkeypatch.setattr(
        "stock_predictor.screener.services.screener_service.detect_patterns",
        lambda _candles, _pattern, _opts: [SimpleNamespace(
            pattern_type="Double Bottom",
            status="confirmed",
            direction="bullish",
            end_index=90,
            breakout_index=90,
            score=88.0,
            explanation="",
            score_breakdown=SimpleNamespace(structure=0, total=88),
        )],
    )
    monkeypatch.setattr(
        "stock_predictor.screener.services.screener_service.score_pattern",
        lambda _detection, _candles: 88.0,
    )

    caller_thread = threading.get_ident()
    callback_threads: list[int] = []

    def _on_progress(_done: int, _total: int, _message: str) -> None:
        callback_threads.append(threading.get_ident())

    service.scan_market(
        "Double Bottom",
        "stock",
        filters=ScreenerFilters(min_score=0, min_occurrences=0),
        progress_callback=_on_progress,
    )

    assert callback_threads
    assert set(callback_threads) == {caller_thread}


def test_serial_scan_provider_uses_single_thread(monkeypatch) -> None:
    candles = {
        "AAPL": _candles_from_close([110, 106, 101, 104, 108, 103, 101.2, 105, 109, 112, 114, 116] * 8),
        "MSFT": _candles_from_close([115, 110, 104, 108, 111, 106, 103.8, 107, 110, 113, 116, 118] * 8),
    }
    provider = _SerialProvider(candles)
    service = ScreenerService(provider, universe_service=_Universe())

    monkeypatch.setattr(
        "stock_predictor.screener.services.screener_service.calculate_trade_quality",
        lambda *_args, **_kwargs: _Quality(),
    )

    caller_thread = threading.get_ident()
    detector_threads: list[int] = []

    def _detect(_candles, _pattern, _opts):
        detector_threads.append(threading.get_ident())
        return [SimpleNamespace(
            pattern_type="Double Bottom",
            status="confirmed",
            direction="bullish",
            end_index=90,
            breakout_index=90,
            score=88.0,
            explanation="",
            score_breakdown=SimpleNamespace(structure=0, total=88),
        )]

    monkeypatch.setattr(
        "stock_predictor.screener.services.screener_service.detect_patterns",
        _detect,
    )
    monkeypatch.setattr(
        "stock_predictor.screener.services.screener_service.score_pattern",
        lambda _detection, _candles: 88.0,
    )

    service.scan_market(
        "Double Bottom",
        "stock",
        filters=ScreenerFilters(min_score=0, min_occurrences=0),
    )

    assert detector_threads
    assert set(detector_threads) == {caller_thread}


def test_empty_provider_data_is_reported_in_debug_stats() -> None:
    provider = _Provider({"AAPL": [], "MSFT": []})
    service = ScreenerService(provider, universe_service=_Universe())
    rows = service.scan_market("Double Bottom", "stock")
    debug = service.get_last_debug_stats()
    assert rows == []
    assert debug.rejectedByReason.get("NO_DATA", 0) >= 2


def test_pattern_without_breakout_can_be_forming(monkeypatch) -> None:
    candles = {"AAPL": _candles_from_close([100 + ((-1) ** i) for i in range(140)])}
    provider = _Provider(candles)
    service = ScreenerService(provider, universe_service=_Universe())
    monkeypatch.setattr(
        "stock_predictor.screener.services.screener_service.detect_patterns",
        lambda *_args, **_kwargs: [
            SimpleNamespace(
                pattern_type="Double Bottom",
                status="forming",
                direction="bullish",
                end_index=130,
                breakout_index=None,
                score=55.0,
                explanation="",
                score_breakdown=SimpleNamespace(structure=55, total=55),
            )
        ],
    )
    monkeypatch.setattr(
        "stock_predictor.screener.services.screener_service.score_pattern",
        lambda _d, _c: 55.0,
    )
    monkeypatch.setattr(
        "stock_predictor.screener.services.screener_service.calculate_trade_quality",
        lambda *_args, **_kwargs: _Quality(),
    )
    rows = service.scan_market("Double Bottom", "stock", filters=ScreenerFilters(min_score=50, min_occurrences=0))
    assert rows
    assert rows[0]["status"] == "forming"

def test_scan_aborts_when_known_liquid_symbols_have_no_data() -> None:
    class _EmptyProvider(_Provider):
        def provider_status(self):
            return {"provider": "Alpaca", "feed": "iex"}

    provider = _EmptyProvider({})
    service = ScreenerService(provider, universe_service=_Universe())

    rows = service.scan_market("Double Bottom", "stock")
    debug = service.get_last_debug_stats()

    assert rows == []
    assert any("Provider failed on 10 known liquid symbols" in warning for warning in debug.warnings)
