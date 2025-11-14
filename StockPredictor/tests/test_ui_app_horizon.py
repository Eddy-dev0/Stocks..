from __future__ import annotations

from datetime import date

import sys
from pathlib import Path

import pandas as pd

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


def test_week_horizon_uses_calendar_week() -> None:
    app = _build_app()
    forecast = app._compute_forecast_date()
    assert forecast is not None
    assert forecast.date() == date(2025, 11, 21)
