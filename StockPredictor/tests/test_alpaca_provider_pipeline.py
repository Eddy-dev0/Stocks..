from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stock_predictor.screener.market_data.provider import to_alpaca_timeframe
from stock_predictor.ui_app import AlpacaMarketDataProvider, StockPredictorDesktopApp


def test_to_alpaca_timeframe_mapping() -> None:
    assert to_alpaca_timeframe("1h") == "1Hour"
    assert to_alpaca_timeframe("30m") == "30Min"
    assert to_alpaca_timeframe("15m") == "15Min"


def test_provider_self_test_fails_when_all_symbols_empty() -> None:
    class _Provider:
        def get_historical_bars(self, *_args, **_kwargs):
            return []

    app = StockPredictorDesktopApp.__new__(StockPredictorDesktopApp)
    app.screener_status_var = type("Var", (), {"set": lambda self, value: setattr(self, "value", value)})()
    app._resolve_provider = lambda: _Provider()

    StockPredictorDesktopApp._on_test_provider(app)

    assert app.provider_self_test_passed is False
    assert "Live provider failed" in app.screener_status_var.value


def test_alpaca_403_error_message(monkeypatch) -> None:
    provider = AlpacaMarketDataProvider(feed="sip")
    provider.api_key = "k"
    provider.api_secret = "s"

    def _fake_request(*_args, **_kwargs):
        return [], {"httpStatus": 403, "feed": "sip", "rawResponseSnippet": "forbidden"}

    monkeypatch.setattr(provider, "_request", _fake_request)
    try:
        provider.get_historical_bars("AAPL", "1h", 500)
    except Exception as exc:
        assert "feed not authorized" in str(exc)


def test_alpaca_401_error_message(monkeypatch) -> None:
    provider = AlpacaMarketDataProvider(feed="iex")
    provider.api_key = "k"
    provider.api_secret = "s"

    def _fake_request(*_args, **_kwargs):
        return [], {"httpStatus": 401, "feed": "iex", "rawResponseSnippet": "unauthorized"}

    monkeypatch.setattr(provider, "_request", _fake_request)
    try:
        provider.get_historical_bars("AAPL", "1h", 500)
    except Exception as exc:
        assert "invalid or missing" in str(exc)
