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
