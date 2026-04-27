from __future__ import annotations

from types import SimpleNamespace

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stock_predictor.ui_app import StockPredictorDesktopApp
from stock_predictor.ui_app import YahooMarketDataProvider


class _DummyVar:
    def __init__(self, value: object | None = None) -> None:
        self.value = value

    def set(self, value: object) -> None:
        self.value = value

    def get(self) -> object:
        return self.value


class _DummyRoot:
    def update_idletasks(self) -> None:
        return None


class _DummyListbox:
    def curselection(self):
        return (0,)

    def get(self, index: int) -> str:
        return "Double Bottom"


class _DummyPlaceholder:
    def __init__(self) -> None:
        self.text = ""
        self.visible = False

    def configure(self, **kwargs) -> None:
        self.text = str(kwargs.get("text", self.text))

    def grid(self, **kwargs) -> None:
        self.visible = True

    def grid_remove(self) -> None:
        self.visible = False


class _DummyTree:
    def __init__(self) -> None:
        self.rows: dict[str, tuple[object, ...]] = {}
        self._counter = 0
        self.visible = False
        self._selection: tuple[str, ...] = ()

    def get_children(self):
        return tuple(self.rows.keys())

    def delete(self, item: str) -> None:
        self.rows.pop(item, None)

    def insert(self, _parent: str, _index: str, values: tuple[object, ...]):
        self._counter += 1
        iid = f"row-{self._counter}"
        self.rows[iid] = values
        return iid

    def grid_remove(self) -> None:
        self.visible = False

    def grid(self, **kwargs) -> None:
        self.visible = True

    def selection(self):
        return self._selection

    def item(self, item: str, _field: str):
        return self.rows[item]


class _DummyTickerVar:
    def __init__(self) -> None:
        self.value = ""

    def set(self, value: str) -> None:
        self.value = value


class _ServiceStub:
    def __init__(self, rows):
        self.rows = rows

    def scan_market(self, *_args, **kwargs):
        progress_callback = kwargs.get("progress_callback")
        if progress_callback is not None:
            progress_callback(3, 10, "Scanning 3 / 10 symbols")
        return list(self.rows)

    def get_last_debug_stats(self):
        return SimpleNamespace(
            scannedSymbols=10,
            symbolsWithData=9,
            symbolsWithEnoughCandles=8,
            pipeline={"rawDetections": 7, "activeDetections": 4, "displayedResults": len(self.rows), "afterConfidenceFilter": 3},
        )


def _build_app_with_rows(rows) -> StockPredictorDesktopApp:
    app = StockPredictorDesktopApp.__new__(StockPredictorDesktopApp)
    app.root = _DummyRoot()
    app.screener_tree = _DummyTree()
    app.screener_placeholder = _DummyPlaceholder()
    app.screener_pattern_listbox = _DummyListbox()
    app.screener_status_var = _DummyVar()
    app.screener_progress_var = _DummyVar()
    app.screener_last_scan_var = _DummyVar()
    app.screener_row_symbol_map = {}
    app.screener_service = _ServiceStub(rows)
    app.config = SimpleNamespace(ticker="AAPL")
    app.ticker_var = _DummyTickerVar()
    app._set_status = lambda _msg: None
    app._on_refresh_called = False
    app._on_refresh = lambda: setattr(app, "_on_refresh_called", True)
    return app


def test_screener_status_is_market_wide_not_current_ticker() -> None:
    app = _build_app_with_rows(
        [
            {"name": "Apple", "symbol": "AAPL", "marketType": "stock", "patternType": "Double Bottom", "direction": "Bullish", "confidence": 82.1, "tradeQuality": {"rating": "A+", "successes": 90, "occurrences": 140}, "signalTime": "2026-04-25 13:00"},
            {"name": "Microsoft", "symbol": "MSFT", "marketType": "stock", "patternType": "Double Bottom", "direction": "Bullish", "confidence": 88.4, "tradeQuality": {"rating": "A", "successes": 72, "occurrences": 110}, "signalTime": "2026-04-25 13:00"},
        ]
    )

    app._update_screener_view(force_refresh_data=True)

    assert app.screener_status_var.value.startswith("Detected 2 symbols matching Double Bottom on the 1h timeframe.")
    assert app.screener_progress_var.value == "2 Treffer, 10 Aktien gescannt"
    assert "AAPL" not in app.screener_status_var.value


def test_screener_row_click_sets_main_ticker() -> None:
    app = _build_app_with_rows(
        [{"name": "Microsoft", "symbol": "MSFT", "marketType": "stock", "patternType": "Double Bottom", "direction": "Bullish", "confidence": 88.4, "tradeQuality": {"rating": "A", "successes": 72, "occurrences": 110}, "signalTime": "2026-04-25 13:00"}]
    )
    app._update_screener_view(force_refresh_data=True)

    row_id = next(iter(app.screener_tree.rows))
    app.screener_tree._selection = (row_id,)

    app._on_screener_row_selected(None)

    assert app.ticker_var.value == "MSFT"
    assert app._on_refresh_called is True


def test_yahoo_provider_normalizes_dot_ticker(monkeypatch) -> None:
    calls: list[str] = []

    class _FakeTicker:
        def __init__(self, symbol: str) -> None:
            calls.append(symbol)

        def history(self, **_kwargs):
            import pandas as pd

            return pd.DataFrame(
                {
                    "Open": [1.0],
                    "High": [1.1],
                    "Low": [0.9],
                    "Close": [1.05],
                    "Volume": [1000],
                },
                index=pd.Index([pd.Timestamp("2026-04-25 10:00:00+00:00")], name="Datetime"),
            )

    class _FakeYFinance:
        Ticker = _FakeTicker

    monkeypatch.setitem(sys.modules, "yfinance", _FakeYFinance())

    provider = YahooMarketDataProvider()
    bars = provider.get_historical_bars("MOG.A", "1h", 500)

    assert bars
    assert calls == ["MOG-A"]
