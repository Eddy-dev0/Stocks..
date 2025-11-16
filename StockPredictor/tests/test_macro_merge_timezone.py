"""Tests for timezone normalization when merging macro indicators."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stock_predictor.core.config import PredictorConfig
from stock_predictor.core.modeling import StockPredictorAI


def test_merge_macro_columns_handles_mixed_timezones() -> None:
    config = PredictorConfig(ticker="SAP", market_timezone="Europe/Berlin")
    predictor = StockPredictorAI(config)

    price_dates = pd.date_range("2024-01-01", periods=3, freq="D", tz="Europe/Berlin")
    price_df = pd.DataFrame({"Date": price_dates, "Close": [100.0, 101.0, 102.0]})

    macro_dates = pd.date_range("2024-01-01", periods=3, freq="D")
    macro_df = pd.DataFrame({"Date": macro_dates, "Close_GER": [1.5, 1.6, 1.7]})

    merged = predictor._merge_macro_columns(price_df, macro_df)

    assert merged["Close_GER"].tolist() == [1.5, 1.6, 1.7]
    assert getattr(merged["Date"].dt.tz, "key", None) is None
