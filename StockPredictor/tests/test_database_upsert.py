"""Regression tests for database upsert helpers."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stock_predictor.providers.database import Database


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
