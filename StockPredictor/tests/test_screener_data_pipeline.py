from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stock_predictor.screener.market_data.provider import FrontendAPIMarketDataProvider, MockPatternMarketDataProvider
from stock_predictor.screener.market_data.symbol_universe import SymbolInfo
from stock_predictor.screener.pattern_engine.types import Candle
from stock_predictor.screener.services.screener_service import ScreenerFilters, ScreenerService


def _candles(count: int = 150) -> list[Candle]:
    base = datetime(2026, 1, 1)
    return [
        Candle(
            timestamp=base + timedelta(hours=i),
            open=100 + i * 0.1,
            high=100 + i * 0.1 + 1,
            low=100 + i * 0.1 - 1,
            close=100 + i * 0.1,
            volume=1_000_000 + i,
        )
        for i in range(count)
    ]


def test_provider_normalizes_yahoo_symbols() -> None:
    provider = FrontendAPIMarketDataProvider(request_fn=lambda *_a, **_k: None)
    assert provider.normalize_symbol("BRK.B") == "BRK-B"
    assert provider.normalize_symbol("ES") == "ES=F"


def test_provider_fetches_1h_from_yfinance_fallback(monkeypatch) -> None:
    class _FakeYF:
        @staticmethod
        def download(**_kwargs):
            idx = pd.date_range("2026-02-01", periods=140, freq="h", tz="UTC")
            return pd.DataFrame(
                {
                    "Open": [100.0] * 140,
                    "High": [101.0] * 140,
                    "Low": [99.0] * 140,
                    "Close": [100.5] * 140,
                    "Volume": [1_000_000] * 140,
                },
                index=idx.rename("Datetime"),
            )

    monkeypatch.setitem(sys.modules, "yfinance", _FakeYF())
    provider = FrontendAPIMarketDataProvider(request_fn=lambda *_a, **_k: {"data": {"indicator_rows": 1}})
    bars = provider.get_historical_bars("AAPL", "1h", 500)
    assert len(bars) >= 120


def test_validate_candles_marks_1h_data_valid() -> None:
    service = ScreenerService(MockPatternMarketDataProvider())
    clean, _warnings, reason, spacing = service.validate_candles("AAPL", _candles(140), "1h", 120)
    assert reason is None
    assert len(clean) == 140
    assert spacing is not None
    assert 59 <= spacing <= 61


def test_mock_provider_screener_returns_fixture_results(monkeypatch) -> None:
    class _Universe:
        def get_universe(self, _market_filter: str) -> list[SymbolInfo]:
            return [
                SymbolInfo("AAPL", "Apple", "stock"),
                SymbolInfo("MSFT", "Microsoft", "stock"),
                SymbolInfo("TSLA", "Tesla", "stock"),
                SymbolInfo("NVDA", "NVIDIA", "stock"),
                SymbolInfo("AMZN", "Amazon", "stock"),
            ]

    provider = MockPatternMarketDataProvider()
    service = ScreenerService(provider, universe_service=_Universe())

    monkeypatch.setattr(
        "stock_predictor.screener.services.screener_service.detect_patterns",
        lambda _candles, _pattern, _opts: [
            SimpleNamespace(
                pattern_type="Double Bottom",
                status="confirmed",
                direction="bullish",
                end_index=len(_candles) - 1,
                breakout_index=len(_candles) - 1,
                score=80.0,
                explanation="fixture",
                score_breakdown=SimpleNamespace(structure=80, total=80),
            )
        ],
    )
    monkeypatch.setattr("stock_predictor.screener.services.screener_service.score_pattern", lambda *_a, **_k: 80.0)
    monkeypatch.setattr(
        "stock_predictor.screener.services.screener_service.calculate_trade_quality",
        lambda *_a, **_k: SimpleNamespace(
            occurrences=30,
            successes=20,
            success_rate=0.67,
            average_move_percent=2.0,
            median_move_percent=1.0,
            rating="A",
            sample_warning=False,
        ),
    )

    rows = service.scan_market("Double Bottom", "stock", filters=ScreenerFilters(min_score=0, min_occurrences=0))
    assert len(rows) >= 4


def test_scan_continues_when_single_symbol_fails() -> None:
    class _Provider(MockPatternMarketDataProvider):
        def get_historical_bars(self, symbol: str, timeframe: str, lookback: int, options=None):
            if symbol == "MSFT":
                raise RuntimeError("boom")
            return super().get_historical_bars(symbol, timeframe, lookback, options=options)

    class _Universe:
        def get_universe(self, _market_filter: str) -> list[SymbolInfo]:
            return [SymbolInfo("AAPL", "Apple", "stock"), SymbolInfo("MSFT", "Microsoft", "stock")]

    service = ScreenerService(_Provider(), universe_service=_Universe())
    _ = service.scan_market("Double Bottom", "stock")
    debug = service.get_last_debug_stats()
    assert debug.scannedSymbols == 2
    assert any(str(row.get("symbol")) == "MSFT" and str(row.get("status")) == "ERROR" for row in debug.symbolDiagnostics)


def test_get_bars_with_fallback_uses_30m_when_1h_empty() -> None:
    class _Provider(MockPatternMarketDataProvider):
        def get_historical_bars(self, symbol: str, timeframe: str, lookback: int, options=None):
            if timeframe in {"1h", "60m"}:
                return []
            if timeframe == "30m":
                return _candles(260)
            return []

    service = ScreenerService(_Provider())
    result = service.get_bars_with_fallback("AAPL", 500)
    assert result["timeframeUsed"] == "30m"
    assert "Fallback timeframe used: 30m" in str(result["warning"])
    assert len(result["candles"]) >= 240


def test_get_bars_with_fallback_uses_15m_when_30m_empty() -> None:
    class _Provider(MockPatternMarketDataProvider):
        def get_historical_bars(self, symbol: str, timeframe: str, lookback: int, options=None):
            if timeframe in {"1h", "60m", "30m"}:
                return []
            if timeframe == "15m":
                return _candles(520)
            return []

    service = ScreenerService(_Provider())
    result = service.get_bars_with_fallback("AAPL", 500)
    assert result["timeframeUsed"] == "15m"
    assert len(result["candles"]) >= 480


def test_aggregate_candles_15m_to_45m() -> None:
    source = _candles(9)
    out = ScreenerService.aggregate_candles(source, 45)
    assert len(out) == 3
    assert out[0].open == source[0].open
    assert out[0].close == source[2].close
    assert out[0].high == max(source[i].high for i in range(3))
    assert out[0].low == min(source[i].low for i in range(3))
    assert out[0].volume == sum(source[i].volume for i in range(3))


def test_scan_market_auto_timeframe_reports_fallback(monkeypatch) -> None:
    class _Provider(MockPatternMarketDataProvider):
        def get_historical_bars(self, symbol: str, timeframe: str, lookback: int, options=None):
            if timeframe in {"1h", "60m"}:
                return []
            if timeframe == "30m":
                return _candles(260)
            return []

        def provider_status(self):
            return {"provider": "Yahoo", "mode": "Live", "configured": "yes"}

    class _Universe:
        def get_universe(self, _market_filter: str) -> list[SymbolInfo]:
            return [SymbolInfo("AAPL", "Apple", "stock")]

    monkeypatch.setattr(
        "stock_predictor.screener.services.screener_service.detect_patterns",
        lambda _candles, _pattern, _opts: [
            SimpleNamespace(
                pattern_type="Double Bottom",
                status="forming",
                direction="bullish",
                end_index=len(_candles) - 1,
                breakout_index=len(_candles) - 1,
                score=75.0,
                explanation="ok",
                score_breakdown=SimpleNamespace(structure=75, total=75),
            )
        ],
    )
    monkeypatch.setattr("stock_predictor.screener.services.screener_service.score_pattern", lambda *_a, **_k: 75.0)
    monkeypatch.setattr(
        "stock_predictor.screener.services.screener_service.calculate_trade_quality",
        lambda *_a, **_k: SimpleNamespace(
            occurrences=30,
            successes=20,
            success_rate=0.67,
            average_move_percent=2.0,
            median_move_percent=1.0,
            rating="A",
            sample_warning=False,
        ),
    )
    service = ScreenerService(_Provider(), universe_service=_Universe())
    rows = service.scan_market("Double Bottom", "stock", timeframe="auto", filters=ScreenerFilters(min_score=0, min_occurrences=0))
    assert rows
    assert rows[0]["timeframe"] == "30m"
