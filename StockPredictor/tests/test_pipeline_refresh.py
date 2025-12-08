"""Regression tests for price refresh scheduling."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import sys

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stock_predictor.core.config import PredictorConfig
from stock_predictor.core.data_pipeline import AsyncDataPipeline, PRICE_LOOKBACK_DAYS
from stock_predictor.core.pipeline import MarketDataETL, NoPriceDataError
from stock_predictor.providers.base import (
    DatasetType,
    PriceBar,
    ProviderFetchSummary,
    ProviderResult,
    SentimentSignal,
)
from stock_predictor.providers.database import Database


def test_refresh_prices_downloads_when_cache_is_stale(monkeypatch, tmp_path) -> None:
    """Stale cached data should trigger a download and refresh metadata."""

    db_file = tmp_path / "prices.db"
    database = Database(f"sqlite:///{db_file}")
    config = PredictorConfig(ticker="AAPL", start_date=date(2024, 3, 18), data_dir=tmp_path)
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
        lambda self, reference=None: expected_session,
    )

    market_tz = etl.market_timezone
    new_timestamp = pd.Timestamp("2024-03-27 16:00:00", tz=market_tz)
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

    def fake_fetch(self, dataset, params=None, *, providers=None):
        assert dataset == DatasetType.PRICES
        return summary

    monkeypatch.setattr(MarketDataETL, "_fetch_dataset", fake_fetch)

    result = etl.refresh_prices()

    assert result.downloaded is True
    assert len(result.data) == len(cached_dates) + 1
    assert getattr(result.data["Date"].dt.tz, "key", None) == getattr(market_tz, "key", None)
    assert result.data.iloc[-1]["Date"] == new_timestamp

    refreshed_ts = database.get_refresh_timestamp("AAPL", "1d", "prices")
    assert refreshed_ts is not None
    assert refreshed_ts > old_refresh


def test_refresh_prices_raises_when_database_cache_expired(monkeypatch, tmp_path) -> None:
    db_file = tmp_path / "prices.db"
    database = Database(f"sqlite:///{db_file}")
    config = PredictorConfig(ticker="MSFT", start_date=date(2024, 3, 18), data_dir=tmp_path)
    etl = MarketDataETL(config, database)

    cached_dates = pd.date_range(start="2024-03-18", end="2024-03-20", freq="B")
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
    database.upsert_prices("MSFT", "1d", cached_prices)
    expired_timestamp = datetime.now(timezone.utc) - timedelta(days=2)
    database.set_refresh_timestamp(
        "MSFT", "1d", "prices", timestamp=expired_timestamp.replace(tzinfo=None)
    )

    summary = ProviderFetchSummary(results=[], failures=[])

    def fake_fetch(self, dataset, params=None, *, providers=None):  # pragma: no cover - deterministic stub
        assert dataset == DatasetType.PRICES
        return summary

    monkeypatch.setattr(MarketDataETL, "_fetch_dataset", fake_fetch)

    with pytest.raises(NoPriceDataError) as excinfo:
        etl.refresh_prices()

    message = str(excinfo.value)
    assert "Database price cache" in message
    assert "Cached data unavailable" in message


def test_refresh_prices_raises_when_local_cache_expired(monkeypatch, tmp_path) -> None:
    db_file = tmp_path / "prices.db"
    database = Database(f"sqlite:///{db_file}")
    config = PredictorConfig(ticker="TSLA", start_date=date(2024, 3, 18), data_dir=tmp_path)
    etl = MarketDataETL(config, database)

    cache_path = config.price_cache_path
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    expired_timestamp = datetime.now(timezone.utc) - timedelta(days=2)
    cached_dates = pd.date_range(start="2024-03-18", end="2024-03-19", freq="B")
    cache_frame = pd.DataFrame(
        {
            "Date": cached_dates,
            "Open": [210 + idx for idx in range(len(cached_dates))],
            "High": [211 + idx for idx in range(len(cached_dates))],
            "Low": [209 + idx for idx in range(len(cached_dates))],
            "Close": [210.5 + idx for idx in range(len(cached_dates))],
            "Adj Close": [210.5 + idx for idx in range(len(cached_dates))],
            "Volume": [500_000 + idx for idx in range(len(cached_dates))],
            "CacheTimestamp": expired_timestamp.isoformat(),
        }
    )
    cache_frame.to_csv(cache_path, index=False)

    summary = ProviderFetchSummary(results=[], failures=[])

    def fake_fetch(self, dataset, params=None, *, providers=None):  # pragma: no cover - deterministic stub
        assert dataset == DatasetType.PRICES
        return summary

    monkeypatch.setattr(MarketDataETL, "_fetch_dataset", fake_fetch)

    with pytest.raises(NoPriceDataError) as excinfo:
        etl.refresh_prices()

    message = str(excinfo.value)
    assert "Local price cache" in message
    assert "Cached data unavailable" in message


def test_latest_trading_session_with_midday_utc_reference_returns_previous_day() -> None:
    """Midday UTC references should map to the prior completed U.S. session."""

    reference = pd.Timestamp("2024-03-27 12:00:00", tz="UTC")

    config = PredictorConfig(ticker="AAPL")
    etl = MarketDataETL(config)
    session = etl._latest_trading_session(reference=reference)

    assert session == date(2024, 3, 26)


class _NullRegistry:
    async def aclose(self) -> None:  # pragma: no cover - helper
        return None

    async def fetch_all(self, request):  # pragma: no cover - helper
        raise RuntimeError("fetch_all should not be invoked in these tests")


def test_default_params_use_cached_price_start(tmp_path) -> None:
    db_file = tmp_path / "prices.db"
    database = Database(f"sqlite:///{db_file}")
    config = PredictorConfig(ticker="AAPL", start_date=date(2023, 1, 1))

    cached_dates = pd.date_range("2024-01-01", periods=5, freq="B")
    cached_prices = pd.DataFrame({"Date": cached_dates, "Close": [100 + i for i in range(5)]})
    database.upsert_prices("AAPL", "1d", cached_prices)

    pipeline = AsyncDataPipeline(config, registry=_NullRegistry(), database=database)

    params = pipeline._default_params(DatasetType.PRICES)

    expected_start = (cached_dates[-1].date() + timedelta(days=1)).isoformat()
    assert params["start"] == expected_start
    assert params["interval"] == "1d"


def test_default_params_use_conservative_lookback_when_cache_empty(monkeypatch, tmp_path) -> None:
    db_file = tmp_path / "prices.db"
    database = Database(f"sqlite:///{db_file}")

    config = PredictorConfig(ticker="MSFT", start_date=date(2020, 1, 1))

    class _FixedDate(date):
        @classmethod
        def today(cls) -> date:  # pragma: no cover - deterministic override
            return date(2024, 4, 15)

    monkeypatch.setattr("stock_predictor.core.data_pipeline.date", _FixedDate)

    pipeline = AsyncDataPipeline(config, registry=_NullRegistry(), database=database)

    params = pipeline._default_params(DatasetType.PRICES)

    assert params["start"] == config.start_date.isoformat()


def test_price_params_paginate_backfill(tmp_path) -> None:
    db_file = tmp_path / "prices.db"
    database = Database(f"sqlite:///{db_file}")

    config = PredictorConfig(
        ticker="MSFT",
        start_date=date(2020, 1, 1),
        end_date=date(2020, 2, 15),
        price_backfill_page_days=10,
    )

    pipeline = AsyncDataPipeline(config, registry=_NullRegistry(), database=database)
    params = pipeline._default_params(DatasetType.PRICES)
    batches = pipeline._price_param_batches(params)

    assert len(batches) == 5
    assert batches[0]["start"] == date(2020, 1, 1).isoformat()
    assert batches[-1]["end"] == config.end_date.isoformat()


def test_refresh_sentiment_overrides_placeholder_cache(monkeypatch, tmp_path) -> None:
    db_file = tmp_path / "sentiment.db"
    database = Database(f"sqlite:///{db_file}")
    config = PredictorConfig(ticker="AAPL", data_dir=tmp_path)
    etl = MarketDataETL(config, database)

    placeholder_record = {
        "Ticker": "AAPL",
        "AsOf": datetime(2024, 1, 1, tzinfo=timezone.utc),
        "Provider": "placeholder",
        "SignalType": "news_sentiment",
        "Score": 0.0,
        "Magnitude": None,
        "Payload": {"note": "placeholder"},
    }
    database.upsert_sentiment_signals([placeholder_record])

    calls = {"count": 0}

    def fake_fetch(self, dataset, params=None, *, providers=None):
        if dataset != DatasetType.SENTIMENT:
            return ProviderFetchSummary(results=[], failures=[])
        calls["count"] += 1
        record = SentimentSignal(
            symbol="AAPL",
            provider="mock_provider",
            signal_type="news_sentiment",
            as_of=datetime(2024, 1, 2, tzinfo=timezone.utc),
            score=0.6,
            magnitude=None,
            payload={"articles": 3},
        )
        return ProviderFetchSummary(
            results=[
                ProviderResult(
                    dataset_type=DatasetType.SENTIMENT, source="stub", records=[record]
                )
            ],
            failures=[],
        )

    monkeypatch.setattr(MarketDataETL, "_fetch_dataset", fake_fetch)

    result = etl.refresh_sentiment_signals(force=False)

    assert calls["count"] == 1
    assert result.downloaded is True
    assert "mock_provider" in set(result.data["Provider"])


def test_refresh_sentiment_rebuilds_from_zero_payload(monkeypatch, tmp_path) -> None:
    db_file = tmp_path / "sentiment.db"
    database = Database(f"sqlite:///{db_file}")
    config = PredictorConfig(ticker="MSFT", data_dir=tmp_path)
    etl = MarketDataETL(config, database)

    zero_payload_record = {
        "Ticker": "MSFT",
        "AsOf": datetime(2024, 2, 1, tzinfo=timezone.utc),
        "Provider": "vader",
        "SignalType": "news_sentiment",
        "Score": 0.0,
        "Magnitude": None,
        "Payload": {"articles": 0},
    }
    database.upsert_sentiment_signals([zero_payload_record])

    calls = {"count": 0}

    def fake_fetch(self, dataset, params=None, *, providers=None):
        if dataset != DatasetType.SENTIMENT:
            return ProviderFetchSummary(results=[], failures=[])
        calls["count"] += 1
        record = SentimentSignal(
            symbol="MSFT",
            provider="fresh_provider",
            signal_type="news_sentiment",
            as_of=datetime(2024, 2, 2, tzinfo=timezone.utc),
            score=-0.4,
            magnitude=None,
            payload={"articles": 1},
        )
        return ProviderFetchSummary(
            results=[
                ProviderResult(
                    dataset_type=DatasetType.SENTIMENT, source="stub", records=[record]
                )
            ],
            failures=[],
        )

    monkeypatch.setattr(MarketDataETL, "_fetch_dataset", fake_fetch)

    result = etl.refresh_sentiment_signals(force=False)

    assert calls["count"] == 1
    assert result.downloaded is True
    assert "fresh_provider" in set(result.data["Provider"])
