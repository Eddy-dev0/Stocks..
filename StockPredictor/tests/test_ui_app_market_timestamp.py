from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stock_predictor.ui_app import StockPredictorDesktopApp


class _DummyVar:
    def __init__(self) -> None:
        self.value = None

    def set(self, value: object) -> None:
        self.value = value


def _build_stubbed_app() -> StockPredictorDesktopApp:
    app = StockPredictorDesktopApp.__new__(StockPredictorDesktopApp)
    app.market_timezone = ZoneInfo("America/New_York")
    app._now = lambda: pd.Timestamp("2024-05-01 15:00", tz=ZoneInfo("UTC"))
    app.metric_vars = {
        key: _DummyVar()
        for key in (
            "ticker",
            "as_of",
            "last_close",
            "predicted_close",
            "expected_low",
            "stop_loss",
            "expected_change",
            "direction",
        )
    }
    app.stop_loss_var = app.metric_vars["stop_loss"]
    app.sentiment_label_var = _DummyVar()
    app.sentiment_score_var = _DummyVar()
    app.current_market_timestamp = pd.Timestamp("2024-05-01 15:00", tz="UTC")
    app.currency_symbol = "$"
    app.price_decimal_places = 2
    app.expected_low_multiplier = 1.0
    app.current_forecast_date = None
    app.current_prediction = {
        "ticker": "TSLA",
        "market_data_as_of": pd.Timestamp("2024-05-01 15:00"),
    }
    app.status_var = _DummyVar()
    app.config = SimpleNamespace(ticker="TSLA")
    app._reference_last_price = lambda prediction: (None, None)
    app._convert_currency = lambda value: value
    app._compute_forecast_date = lambda target=None: None
    app._compute_expected_low = lambda prediction, multiplier=None: None
    app._resolve_sentiment_snapshot = lambda prediction: (None, None)
    app._recompute_pnl = lambda: None
    return app


def test_naive_market_timestamp_localizes_from_utc() -> None:
    app = _build_stubbed_app()

    app._update_metrics()

    assert app.metric_vars["as_of"].value == "2024-05-01 11:00 EDT"
