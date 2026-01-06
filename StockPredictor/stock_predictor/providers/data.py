"""Facade for interacting with the persistent market data store."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd

from .config import PredictorConfig
from .database import Database
from .etl import MarketDataETL
from ..core.pipeline import DEFAULT_CACHE_MAX_AGE

LOGGER = logging.getLogger(__name__)


class DataFetcher:
    """High level API for retrieving cached market data."""

    def __init__(self, config: PredictorConfig) -> None:
        self.config = config
        self.database = Database(config.database_url)
        self.etl = MarketDataETL(config, database=self.database)
        self._sources: dict[str, str] = {
            "prices": "database",
            "indicators": "database",
            "news": "database",
            "corporate_events": "database",
            "options_surface": "database",
            "sentiment_signals": "database",
            "esg_metrics": "database",
            "ownership_flows": "database",
        }

    # ------------------------------------------------------------------
    # Public API used by the pipeline
    # ------------------------------------------------------------------
    def fetch_price_data(self, force: bool = False) -> pd.DataFrame:
        """Return price data for the configured ticker, refreshing if required."""

        result = self.etl.refresh_prices(force=force)
        self._set_source("prices", result.downloaded or force)
        return result.data

    def fetch_news_data(self, force: bool = False) -> pd.DataFrame:
        """Return cached news articles, refreshing via the API when necessary."""

        result = self.etl.refresh_news(force=force)
        self._set_source("news", result.downloaded or force)
        return result.data

    def fetch_indicator_data(self, category: Optional[str] = None) -> pd.DataFrame:
        """Fetch indicator data for the current ticker and interval."""

        return self.database.get_indicators(
            ticker=self.config.ticker,
            interval=self.config.interval,
            category=category,
        )

    def fetch_fundamentals(self) -> pd.DataFrame:
        """Fetch cached fundamentals for the configured ticker."""

        return self.database.get_fundamentals(self.config.ticker)

    def fetch_corporate_events(self, force: bool = False) -> pd.DataFrame:
        """Fetch corporate action events, refreshing the cache if necessary."""

        result = self.etl.refresh_corporate_events(force=force)
        self._set_source("corporate_events", result.downloaded or force)
        return result.data

    def fetch_options_surface(self, force: bool = False) -> pd.DataFrame:
        """Fetch option surface metrics for the configured ticker."""

        result = self.etl.refresh_options_surface(force=force)
        self._set_source("options_surface", result.downloaded or force)
        return result.data

    def fetch_sentiment_signals(self, force: bool = False) -> pd.DataFrame:
        """Fetch aggregated sentiment and alternative signals."""

        result = self.etl.refresh_sentiment_signals(force=force)
        self._set_source("sentiment_signals", result.downloaded or force)
        return result.data

    def fetch_esg_metrics(self, force: bool = False) -> pd.DataFrame:
        """Fetch ESG metrics for the configured ticker."""

        result = self.etl.refresh_esg_metrics(force=force)
        self._set_source("esg_metrics", result.downloaded or force)
        return result.data

    def fetch_ownership_flows(self, force: bool = False) -> pd.DataFrame:
        """Fetch ownership and flow datasets for the configured ticker."""

        result = self.etl.refresh_ownership_flows(force=force)
        self._set_source("ownership_flows", result.downloaded or force)
        return result.data

    def refresh_all(self, force: bool = False) -> dict[str, int]:
        """Refresh all supported datasets and return a summary of inserted rows."""

        summary = self.etl.refresh_all(force=force)
        if summary.get("downloaded"):
            for key in self._sources:
                self._sources[key] = "remote"
        return summary

    def refresh_data(self, force: bool = False) -> dict[str, int]:
        """Refresh price data and indicators when cache staleness requires it."""

        needs_price_refresh = force or self._is_price_cache_stale()
        if needs_price_refresh:
            price_result = self.etl.refresh_prices(force=force)
            price_frame = price_result.data
            self._set_source("prices", price_result.downloaded or force)
        else:
            price_frame = self.database.get_prices(
                ticker=self.config.ticker,
                interval=self.config.interval,
                start=self.config.start_date,
                end=self.config.end_date,
            )
            self._set_source("prices", False)

        price_rows = len(price_frame.index)
        indicator_rows = 0
        if needs_price_refresh:
            indicator_rows = self.etl.refresh_indicators(
                price_frame=price_frame, force=force
            )
        self._sources["indicators"] = "local" if indicator_rows else "database"

        return {"price_rows": price_rows, "indicator_rows": indicator_rows}

    def fetch_live_price(self, force: bool = False) -> tuple[float | None, pd.Timestamp | None]:
        """Fetch the most recent intraday price and timestamp."""

        price, timestamp = self.etl.fetch_live_price(force=force)
        self._set_source("prices", True)
        return price, timestamp

    # ------------------------------------------------------------------
    # Convenience helpers for the UI layer
    # ------------------------------------------------------------------
    @property
    def last_price_source(self) -> str:
        return self._sources.get("prices", "database")

    @property
    def last_news_source(self) -> str:
        return self._sources.get("news", "database")

    def get_last_source(self, category: str) -> Optional[str]:
        """Return the last known data source for a dataset category."""

        return self._sources.get(category)

    def get_indicator_columns(self) -> list[str]:
        """Return the list of indicator columns persisted for the active context."""

        try:
            columns = self.database.get_indicator_columns(
                self.config.ticker, self.config.interval
            )
        except Exception as exc:  # pragma: no cover - defensive guard around persistence
            LOGGER.debug("Failed to fetch indicator column metadata: %s", exc)
            return []
        return columns

    def get_last_updated(self, category: str) -> Optional[datetime]:
        """Return the last refresh timestamp recorded for the given category."""

        interval = self._interval_for_category(category)
        return self.database.get_refresh_timestamp(
            self.config.ticker, interval, category
        )

    def get_data_sources(self) -> list[dict[str, object]]:
        """Return metadata describing the data sources touched during the ETL run."""

        return self.etl.list_sources()

    def _interval_for_category(self, category: str) -> str:
        if category in {"prices", "indicators", "macro"}:
            return self.config.interval
        return "global"

    def _set_source(self, category: str, downloaded: bool) -> None:
        self._sources[category] = "remote" if downloaded else "database"

    def _cache_max_age(self) -> timedelta:
        return getattr(self.config, "cache_expiry", DEFAULT_CACHE_MAX_AGE)

    def _is_timestamp_fresh(self, timestamp: datetime | None) -> bool:
        if timestamp is None:
            return False
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        else:
            timestamp = timestamp.astimezone(timezone.utc)
        return datetime.now(timezone.utc) - timestamp <= self._cache_max_age()

    def _is_price_cache_stale(self) -> bool:
        refresh_ts = self.database.get_refresh_timestamp(
            self.config.ticker, self.config.interval, "prices"
        )
        if self._is_timestamp_fresh(refresh_ts):
            return False

        cache_path = self.config.price_cache_path
        if cache_path.exists():
            try:
                cache_timestamp = datetime.fromtimestamp(
                    cache_path.stat().st_mtime, timezone.utc
                )
            except OSError:
                return True
            return not self._is_timestamp_fresh(cache_timestamp)
        return True


__all__ = ["DataFetcher"]
