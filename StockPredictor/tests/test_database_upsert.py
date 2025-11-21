"""Regression tests for database upsert helpers."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from sqlalchemy import select

from stock_predictor.providers.database import Database, Price


def test_upsert_indicators_handles_large_payload(tmp_path) -> None:
    """Large indicator batches should be committed without parameter errors."""

    db_file = tmp_path / "indicators.db"
    database = Database(f"sqlite:///{db_file}")

    base_date = datetime(2024, 1, 1)
    records = []
    for idx in range(500):
        records.append(
            {
                "Ticker": "NVDA",
                "Interval": "1d",
                "Date": base_date + timedelta(days=idx),
                "Indicator": f"feature_{idx}",
                "Value": float(idx),
                "Category": "technical",
                "Extra": {"sequence": idx},
            }
        )

    inserted = database.upsert_indicators("NVDA", "1d", records)

    assert inserted == len(records)


def test_upsert_prices_batches_large_payload(tmp_path) -> None:
    """Price upserts should honour dialect parameter limits via batching."""

    db_file = tmp_path / "prices.db"
    database = Database(f"sqlite:///{db_file}")

    dates = pd.date_range("2024-01-01", periods=1200, freq="D")
    frame = pd.DataFrame(
        {
            "Date": dates,
            "Open": [float(i) for i in range(len(dates))],
            "High": [float(i + 1) for i in range(len(dates))],
            "Low": [float(i - 1) for i in range(len(dates))],
            "Close": [float(i + 0.5) for i in range(len(dates))],
            "Adj Close": [float(i + 0.25) for i in range(len(dates))],
            "Volume": [float(i * 10) for i in range(len(dates))],
        }
    )

    inserted = database.upsert_prices("AAPL", "1d", frame)

    assert inserted == len(frame)

    with database.session() as session:
        rows = (
            session.execute(
                select(Price).where(Price.ticker == "AAPL", Price.interval == "1d")
            )
            .scalars()
            .all()
        )

    assert len(rows) == len(frame)


def test_upsert_prices_appends_without_duplicates(tmp_path) -> None:
    """Repeated price upserts should not duplicate historical dates."""

    db_file = tmp_path / "prices.db"
    database = Database(f"sqlite:///{db_file}")

    initial_dates = pd.date_range("2024-02-01", periods=3, freq="B")
    initial = pd.DataFrame(
        {
            "Date": initial_dates,
            "Open": [100.0, 101.0, 102.0],
            "High": [101.0, 102.0, 103.0],
            "Low": [99.0, 100.0, 101.0],
            "Close": [100.5, 101.5, 102.5],
            "Adj Close": [100.5, 101.5, 102.5],
            "Volume": [1_000_000, 1_100_000, 1_200_000],
        }
    )

    follow_on = pd.DataFrame(
        {
            "Date": pd.date_range("2024-02-05", periods=3, freq="B"),
            "Open": [103.0, 104.0, 105.0],
            "High": [104.0, 105.0, 106.0],
            "Low": [102.0, 103.0, 104.0],
            "Close": [103.5, 104.5, 105.5],
            "Adj Close": [103.5, 104.5, 105.5],
            "Volume": [1_300_000, 1_400_000, 1_500_000],
        }
    )

    database.upsert_prices("AAPL", "1d", initial)
    database.upsert_prices("AAPL", "1d", follow_on)

    prices = database.get_prices("AAPL", "1d")

    assert len(prices) == 5
    assert prices.iloc[-1]["Date"].date() == follow_on.iloc[-1]["Date"].date()
