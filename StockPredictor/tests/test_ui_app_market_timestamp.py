from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stock_predictor.core.clock import app_clock
from stock_predictor.ui_app import StockPredictorDesktopApp


class _DummyVar:
    def __init__(self) -> None:
        self.value = None

    def set(self, value: object) -> None:
        self.value = value


class _DummyWidget:
    def __init__(self) -> None:
        self.grid_calls: list[dict[str, object]] = []
        self.removed = False

    def grid(self, **kwargs: object) -> None:
        self.grid_calls.append(kwargs)
        self.removed = False

    def grid_remove(self) -> None:
        self.removed = True


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
            "daily_change",
            "direction",
        )
    }
    app.stop_loss_var = app.metric_vars["stop_loss"]
    app.stop_loss_label = _DummyWidget()
    app.stop_loss_value = _DummyWidget()
    app._stop_loss_label_grid_options = {"row": 2, "column": 0, "sticky": "e"}
    app._stop_loss_value_grid_options = {"row": 2, "column": 1, "sticky": "e"}
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
    app.analysis_override_date = None
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


def test_market_timestamp_override_uses_historical_close() -> None:
    app = _build_stubbed_app()
    app.current_market_timestamp = None
    app.price_history = pd.DataFrame(
        {
            "Date": pd.date_range(
                "2024-04-29 15:00", periods=2, freq="D", tz=ZoneInfo("UTC")
            ),
            "Close": [100.0, 101.0],
        }
    )
    app.price_history.attrs["cache_timestamp"] = pd.Timestamp(
        "2024-04-01 12:00", tz=ZoneInfo("UTC")
    )

    app_clock.set_test_date(pd.Timestamp("2024-05-02").date())
    try:
        app._update_metrics()
    finally:
        app_clock.clear_test_date()

    assert "historical close" in app.metric_vars["as_of"].value
    assert "live estimate" not in (app.status_var.value or "")


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


def test_stop_loss_hidden_without_volatility() -> None:
    app = _build_stubbed_app()
    app.current_prediction = {
        "ticker": "TSLA",
        "market_data_as_of": pd.Timestamp("2024-05-01 15:00"),
    }

    app._update_metrics()

    assert app.stop_loss_label.removed is True
    assert app.stop_loss_value.removed is True
    assert app.stop_loss_var.value is None


def test_stop_loss_visible_with_volatility() -> None:
    app = _build_stubbed_app()
    app.current_prediction = {
        "ticker": "TSLA",
        "market_data_as_of": pd.Timestamp("2024-05-01 15:00"),
        "predicted_volatility": 0.02,
        "stop_loss": 287.5,
    }

    app._update_metrics()

    assert app.stop_loss_label.removed is False
    assert app.stop_loss_value.removed is False
    assert app.stop_loss_var.value == "$287.50"
