from __future__ import annotations

from datetime import date

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stock_predictor.ui_app import StockPredictorDesktopApp


def _build_app() -> StockPredictorDesktopApp:
    app = StockPredictorDesktopApp.__new__(StockPredictorDesktopApp)
    app.selected_horizon_code = "1w"
    app.selected_horizon_label = "1 Week"
    app.selected_horizon_offset = 5
    app.current_prediction = {}
    app.price_history = pd.DataFrame()
    app.market_holidays = []
    app.current_forecast_date = None
    app._resolve_market_holidays = lambda: []
    base = pd.Timestamp(date(2025, 11, 14))
    app._forecast_base_date = lambda: base
    return app


def test_week_horizon_uses_trading_week() -> None:
    app = _build_app()
    forecast = app._compute_forecast_date()
    assert forecast is not None
    assert forecast.date() == date(2025, 11, 21)


def test_tomorrow_rolls_over_weekend() -> None:
    app = _build_app()
    app.selected_horizon_code = "1d"
    app.selected_horizon_label = "Tomorrow"
    app.selected_horizon_offset = 1

    forecast = app._compute_forecast_date()
    assert forecast is not None
    assert forecast.date() == date(2025, 11, 17)


def test_rejects_weekend_target_date(caplog: pytest.LogCaptureFixture) -> None:
    app = _build_app()

    with caplog.at_level("WARNING"):
        forecast = app._compute_forecast_date(date(2025, 11, 15))

    assert forecast is None
    assert "weekend" in caplog.text.lower()


def test_rejects_past_target_date(caplog: pytest.LogCaptureFixture) -> None:
    app = _build_app()

    with caplog.at_level("WARNING"):
        forecast = app._compute_forecast_date(date(2025, 11, 13))

    assert forecast is None
    assert "precedes" in caplog.text.lower()
