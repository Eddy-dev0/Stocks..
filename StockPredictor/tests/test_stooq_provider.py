"""Unit tests for the Stooq provider adapter."""

from __future__ import annotations

import asyncio
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stock_predictor.providers.adapters import StooqProvider
from stock_predictor.providers.base import DatasetType, PriceBar, ProviderRequest


class _DummyResponse:
    """Simple stand-in for httpx.Response in tests."""

    def __init__(self, text: str) -> None:
        self.text = text

    def raise_for_status(self) -> None:  # pragma: no cover - kept for interface parity
        return None


class _DummyClient:
    """Async client returning predetermined payloads and recording URLs."""

    def __init__(self, response_text: str, url_log: list[str]) -> None:
        self._response_text = response_text
        self._url_log = url_log

    async def get(self, url: str, *_: Any, **__: Any) -> _DummyResponse:
        self._url_log.append(url)
        return _DummyResponse(self._response_text)

    async def aclose(self) -> None:  # pragma: no cover - interface parity
        return None


def test_stooq_appends_us_suffix_for_plain_symbols() -> None:
    """Plain tickers should be converted to their US-qualified counterparts."""

    async def _runner() -> None:
        provider = StooqProvider()
        url_log: list[str] = []
        provider._client = _DummyClient(
            "Date,Open,High,Low,Close,Volume\n2024-01-02,1,1,1,1,100\n",
            url_log,
        )

        request = ProviderRequest(dataset_type=DatasetType.PRICES, symbol="AAPL", params={})
        result = await provider._fetch(request)

        assert url_log == ["https://stooq.com/q/d/l/?s=aapl.us&i=d"]
        assert result.records
        assert isinstance(result.records[0], PriceBar)

        await provider.aclose()

    asyncio.run(_runner())


def test_stooq_filters_out_nan_rows() -> None:
    """Rows containing NaN values should be removed before producing price bars."""

    async def _runner() -> None:
        provider = StooqProvider()
        csv_text = """Date,Open,High,Low,Close,Volume
2024-01-02,1,1.5,0.9,1.2,100
2024-01-03,,,,,
2024-01-04,2,2.5,1.8,2.1,150
"""
        url_log: list[str] = []
        provider._client = _DummyClient(csv_text, url_log)

        request = ProviderRequest(dataset_type=DatasetType.PRICES, symbol="MSFT.us", params={})
        result = await provider._fetch(request)

        assert url_log == ["https://stooq.com/q/d/l/?s=msft.us&i=d"]
        assert len(result.records) == 2
        assert all(isinstance(bar, PriceBar) for bar in result.records)
        assert {bar.timestamp.date().isoformat() for bar in result.records} == {
            "2024-01-02",
            "2024-01-04",
        }

        await provider.aclose()

    asyncio.run(_runner())
