"""Regression tests for price refresh scheduling."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stock_predictor.core.config import PredictorConfig
from stock_predictor.core.pipeline import MarketDataETL, US_MARKET_TIMEZONE
from stock_predictor.providers.base import (
    DatasetType,
    PriceBar,
    ProviderFetchSummary,
    ProviderResult,
)
from stock_predictor.providers.database import Database


def test_refresh_prices_downloads_when_cache_is_stale(monkeypatch, tmp_path) -> None:
    """Stale cached data should trigger a download and refresh metadata."""

    db_file = tmp_path / "prices.db"
    database = Database(f"sqlite:///{db_file}")
    config = PredictorConfig(ticker="AAPL", start_date=date(2024, 3, 18))
    etl = MarketDataETL(config, database)

    cached_dates = pd.date_range(start="2024-03-18", end="2024-03-25", freq="B")
    cached_prices = pd.DataFrame(
        {
            "Date": cached_dates,
            "Open": [100 + idx for idx in range(len(cached_dates))],
            "High": [101 + idx for idx in range(len(cached_dates))],
            "Low": [99 + idx for idx in range(len(cached_dates))],
            "Close": [100.5 + idx for idx in range(len(cached_dates))],
            "Adj Close": [100.5 + idx for idx in range(len(cached_dates))],
            "Volume": [1_000_000 + idx for idx in range(len(cached_dates))],
        }
    )
    database.upsert_prices("AAPL", "1d", cached_prices)

    old_refresh = datetime(2024, 3, 25, 12, 0, 0)
    database.set_refresh_timestamp("AAPL", "1d", "prices", timestamp=old_refresh)

    expected_session = date(2024, 3, 27)
    monkeypatch.setattr(
        MarketDataETL,
        "_latest_trading_session",
        staticmethod(lambda reference=None: expected_session),
    )

    new_timestamp = pd.Timestamp("2024-03-27 16:00:00", tz=US_MARKET_TIMEZONE)
    price_bar = PriceBar(
        symbol="AAPL",
        timestamp=new_timestamp.to_pydatetime(),
        open=110.0,
        high=111.0,
        low=109.0,
        close=110.5,
        adj_close=110.5,
        volume=1_500_000,
    )
    summary = ProviderFetchSummary(
        results=[
            ProviderResult(
                dataset_type=DatasetType.PRICES,
                source="stub",
                records=[price_bar],
            )
        ],
        failures=[],
    )

    def fake_fetch(self, dataset, params=None):
        assert dataset == DatasetType.PRICES
        return summary

    monkeypatch.setattr(MarketDataETL, "_fetch_dataset", fake_fetch)

    result = etl.refresh_prices()

    assert result.downloaded is True
    assert len(result.data) == len(cached_dates) + 1
    assert result.data["Date"].dt.tz is US_MARKET_TIMEZONE
    assert result.data.iloc[-1]["Date"] == new_timestamp

    refreshed_ts = database.get_refresh_timestamp("AAPL", "1d", "prices")
    assert refreshed_ts is not None
    assert refreshed_ts > old_refresh


def test_latest_trading_session_with_midday_utc_reference_returns_previous_day() -> None:
    """Midday UTC references should map to the prior completed U.S. session."""

    reference = pd.Timestamp("2024-03-27 12:00:00", tz="UTC")

    session = MarketDataETL._latest_trading_session(reference=reference)

    assert session == date(2024, 3, 26)
