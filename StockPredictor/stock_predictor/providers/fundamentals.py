"""Asynchronous helper for fetching headline fundamental metrics."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping, MutableMapping

import yfinance as yf

LOGGER = logging.getLogger(__name__)

CACHE_TTL = timedelta(hours=6)


SECTOR_BASELINES: dict[str, dict[str, float]] = {
    "Communication Services": {"pe": 18.0, "debt_to_equity": 1.1, "growth": 0.06},
    "Consumer Cyclical": {"pe": 20.0, "debt_to_equity": 1.4, "growth": 0.07},
    "Consumer Defensive": {"pe": 18.0, "debt_to_equity": 1.2, "growth": 0.05},
    "Energy": {"pe": 12.0, "debt_to_equity": 0.9, "growth": 0.04},
    "Financial Services": {"pe": 14.0, "debt_to_equity": 2.0, "growth": 0.05},
    "Healthcare": {"pe": 18.0, "debt_to_equity": 0.8, "growth": 0.07},
    "Industrials": {"pe": 17.0, "debt_to_equity": 1.1, "growth": 0.06},
    "Real Estate": {"pe": 16.0, "debt_to_equity": 2.4, "growth": 0.04},
    "Technology": {"pe": 24.0, "debt_to_equity": 0.7, "growth": 0.10},
    "Utilities": {"pe": 16.0, "debt_to_equity": 1.4, "growth": 0.04},
    "default": {"pe": 18.0, "debt_to_equity": 1.0, "growth": 0.06},
}


def _coerce_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if numeric != numeric:  # NaN check
        return None
    return numeric


@dataclass(frozen=True)
class FundamentalSnapshot:
    """Normalized headline fundamental metrics for a ticker."""

    ticker: str
    as_of: datetime
    pe_ratio: float | None
    earnings_growth: float | None
    debt_to_equity: float | None
    eps: float | None
    sector: str | None
    sector_pe: float | None
    sector_growth: float | None
    sector_debt_to_equity: float | None
    source: str = "yfinance"
    raw: Mapping[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker,
            "as_of": self.as_of,
            "pe_ratio": self.pe_ratio,
            "earnings_growth": self.earnings_growth,
            "debt_to_equity": self.debt_to_equity,
            "eps": self.eps,
            "sector": self.sector,
            "sector_pe": self.sector_pe,
            "sector_growth": self.sector_growth,
            "sector_debt_to_equity": self.sector_debt_to_equity,
            "source": self.source,
            "raw": self.raw,
        }


def _baseline_for_sector(sector: str | None) -> dict[str, float]:
    if not sector:
        return SECTOR_BASELINES["default"].copy()
    return SECTOR_BASELINES.get(sector, SECTOR_BASELINES["default"]).copy()


def normalize_fundamentals_payload(
    ticker: str, info: Mapping[str, Any]
) -> FundamentalSnapshot:
    """Normalize a yfinance info payload into a stable snapshot."""

    pe_ratio = _coerce_float(
        info.get("trailingPE")
        or info.get("forwardPE")
        or info.get("priceToBook")
    )
    earnings_growth = _coerce_float(
        info.get("earningsQuarterlyGrowth")
        or info.get("earningsGrowth")
        or info.get("revenueGrowth")
    )
    debt_to_equity = _coerce_float(info.get("debtToEquity"))
    eps = _coerce_float(info.get("trailingEps") or info.get("forwardEps"))
    sector = info.get("sector") or None

    sector_baseline = _baseline_for_sector(sector)
    sector_pe = sector_baseline.get("pe")
    sector_growth = sector_baseline.get("growth")
    sector_debt = sector_baseline.get("debt_to_equity")

    return FundamentalSnapshot(
        ticker=ticker,
        as_of=datetime.now(timezone.utc),
        pe_ratio=pe_ratio,
        earnings_growth=earnings_growth,
        debt_to_equity=debt_to_equity,
        eps=eps,
        sector=sector,
        sector_pe=sector_pe,
        sector_growth=sector_growth,
        sector_debt_to_equity=sector_debt,
        raw=dict(info),
    )


class FundamentalsClient:
    """Async fundamentals fetcher with basic in-memory caching."""

    def __init__(self, ticker: str, *, ttl: timedelta = CACHE_TTL) -> None:
        self.ticker = ticker
        self.ttl = ttl
        self._cache: MutableMapping[str, FundamentalSnapshot] = {}
        self._updated_at: MutableMapping[str, datetime] = {}

    def _cached(self, *, force: bool = False) -> FundamentalSnapshot | None:
        if force:
            return None
        snapshot = self._cache.get(self.ticker)
        timestamp = self._updated_at.get(self.ticker)
        if not snapshot or not timestamp:
            return None
        if datetime.now(timezone.utc) - timestamp > self.ttl:
            return None
        return snapshot

    async def fetch(self, *, force: bool = False) -> FundamentalSnapshot:
        cached = self._cached(force=force)
        if cached:
            return cached

        snapshot = await asyncio.to_thread(self._download_snapshot)
        self._cache[self.ticker] = snapshot
        self._updated_at[self.ticker] = datetime.now(timezone.utc)
        return snapshot

    def _download_snapshot(self) -> FundamentalSnapshot:
        ticker = yf.Ticker(self.ticker)
        yf_logger = logging.getLogger("yfinance")
        previous_level = yf_logger.level
        # ``yfinance`` logs HTTP errors even when the library recovers by
        # returning an empty payload. Suppress those noise-level errors while we
        # handle failures gracefully ourselves.
        yf_logger.setLevel(logging.CRITICAL)

        try:
            info_fetcher = getattr(ticker, "get_info", None)
            if callable(info_fetcher):
                info: Mapping[str, Any] = info_fetcher() or {}
            else:
                info = getattr(ticker, "info", {}) or {}
        except Exception as exc:
            LOGGER.info("Fundamentals request failed for %s: %s", self.ticker, exc)
            info = {}
        finally:
            yf_logger.setLevel(previous_level)

        if not info:
            LOGGER.debug("No fundamentals returned for %s", self.ticker)
        return normalize_fundamentals_payload(self.ticker, info)

    def fetch_sync(self, *, force: bool = False) -> dict[str, Any]:
        """Synchronous wrapper for async fetch used by the core pipeline."""

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.fetch(force=force)).as_dict()

        if loop.is_running():
            return asyncio.run_coroutine_threadsafe(self.fetch(force=force), loop).result().as_dict()
        return loop.run_until_complete(self.fetch(force=force)).as_dict()


__all__ = [
    "FundamentalSnapshot",
    "FundamentalsClient",
    "normalize_fundamentals_payload",
]
