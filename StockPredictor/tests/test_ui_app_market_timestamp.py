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
    app.application = SimpleNamespace(
        pipeline=SimpleNamespace(
            fetcher=SimpleNamespace(fetch_live_price=lambda force=False: (None, None))
        )
    )
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
    app.market_timestamp_live_estimate = False
    app.current_market_price = None
    app.price_history = None
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


def test_market_timestamp_from_price_history() -> None:
    app = _build_stubbed_app()
    app.current_market_timestamp = None
    app.price_history = pd.DataFrame(
        {
            "Date": pd.date_range(
                "2024-04-30 15:00", periods=2, freq="D", tz=ZoneInfo("UTC")
            ),
            "Close": [100.0, 101.0],
        }
    )
    app.price_history.attrs["cache_timestamp"] = pd.Timestamp.now(tz="UTC")

    app._update_metrics()

    assert app.metric_vars["as_of"].value == "2024-05-01 11:00 EDT"


def test_market_timestamp_live_estimate_for_stale_cache() -> None:
    app = _build_stubbed_app()
    app.current_market_timestamp = None
    app.price_history = pd.DataFrame(
        {
            "Date": pd.date_range(
                "2024-04-30 15:00", periods=2, freq="D", tz=ZoneInfo("UTC")
            ),
            "Close": [100.0, 101.0],
        }
    )
    app.price_history.attrs["cache_timestamp"] = pd.Timestamp(
        "2024-04-01 12:00", tz=ZoneInfo("UTC")
    )

    app._update_metrics()

    assert app.metric_vars["as_of"].value == "2024-05-01 11:00 EDT (live estimate)"
    assert "live estimate" in app.status_var.value


def test_last_price_prefers_price_history() -> None:
    app = _build_stubbed_app()
    app.current_market_price = None
    app.price_history = pd.DataFrame(
        {
            "Date": [pd.Timestamp("2024-05-01 15:00", tz=ZoneInfo("UTC"))],
            "Close": [321.5],
        }
    )

    app._update_metrics()

    assert app.metric_vars["last_close"].value == "$321.50 (cached)"


def test_last_price_prefers_live_fetcher_over_history() -> None:
    app = _build_stubbed_app()
    app.current_market_price = None
    app.price_history = pd.DataFrame(
        {
            "Date": [pd.Timestamp("2024-05-01 15:00", tz=ZoneInfo("UTC"))],
            "Close": [321.5],
        }
    )
    app.application.pipeline.fetcher.fetch_live_price = lambda force=False: (
        333.25,
        pd.Timestamp("2024-05-01 15:00", tz=ZoneInfo("UTC")),
    )

    app._update_metrics()

    assert app.metric_vars["last_close"].value == "$333.25"
