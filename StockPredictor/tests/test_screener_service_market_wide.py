from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from types import SimpleNamespace
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stock_predictor.screener.market_data.symbol_universe import SymbolInfo
from stock_predictor.screener.pattern_engine.types import Candle
from stock_predictor.screener.services.screener_service import (
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
        "stock_predictor.screener.services.screener_service.detect_pattern",
        lambda _candles, _pattern: SimpleNamespace(
            pattern_type="Double Bottom",
            status="confirmed",
            direction="bullish",
            end_index=10,
        )
        if _candles is not candles["TSLA"]
        else None,
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
