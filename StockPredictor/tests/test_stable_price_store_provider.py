"""Tests for the stable price store provider's timezone handling."""

from __future__ import annotations

import asyncio
import logging
from datetime import timezone
from pathlib import Path
import sys
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stock_predictor.providers.adapters import StablePriceStoreProvider
from stock_predictor.providers.base import DatasetType, PriceBar, ProviderRequest


def test_stable_price_store_normalizes_timezones(tmp_path: Path) -> None:
    """Mixed timezone inputs should filter without raising and return UTC bars."""

    async def _runner() -> None:
        frame = pd.DataFrame(
            {
                "Ticker": ["AAPL", "AAPL"],
                "Date": [
                    "2024-01-01T09:30:00-05:00",
                    "2024-01-02T09:30:00-05:00",
                ],
                "Open": [100.0, 101.0],
                "High": [101.0, 102.0],
                "Low": [99.0, 100.0],
                "Close": [100.5, 101.5],
                "Adj Close": [100.5, 101.5],
                "Volume": [1000, 2000],
            }
        )
        path = tmp_path / "store.csv"
        frame.to_csv(path, index=False)

        provider = StablePriceStoreProvider(path=str(path))
        request = ProviderRequest(
            dataset_type=DatasetType.PRICES,
            symbol="AAPL",
            params={"start": "2024-01-02", "end": "2024-01-03"},
        )

        result = await provider._fetch(request)

        assert len(result.records) == 1
        bar = result.records[0]
        assert isinstance(bar, PriceBar)
        assert bar.timestamp.tzinfo == timezone.utc
        assert bar.timestamp.isoformat() == "2024-01-02T14:30:00+00:00"

    asyncio.run(_runner())


def test_stable_price_store_warns_on_empty_filtered_result(
    tmp_path: Path, caplog: Any
) -> None:
    """If filters remove all rows, a warning should be logged without retries."""

    async def _runner() -> None:
        frame = pd.DataFrame(
            {
                "Ticker": ["AAPL"],
                "Date": ["2024-01-01"],
                "Open": [100.0],
                "High": [101.0],
                "Low": [99.0],
                "Close": [100.5],
                "Adj Close": [100.5],
                "Volume": [1000],
            }
        )
        path = tmp_path / "store.csv"
        frame.to_csv(path, index=False)

        provider = StablePriceStoreProvider(path=str(path))
        request = ProviderRequest(
            dataset_type=DatasetType.PRICES,
            symbol="AAPL",
            params={"start": "2025-01-01", "end": "2025-01-02"},
        )

        caplog.set_level(logging.WARNING)
        result = await provider._fetch(request)

        assert result.records == []
        assert any(
            "StablePriceStoreProvider returned no rows" in record.message
            for record in caplog.records
        )

    asyncio.run(_runner())

