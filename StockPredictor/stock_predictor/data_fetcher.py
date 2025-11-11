"""Facade for interacting with the persistent market data store."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

import pandas as pd

from .config import PredictorConfig
from .database import Database
from .etl import MarketDataETL

LOGGER = logging.getLogger(__name__)


class DataFetcher:
    """High level API for retrieving cached market data."""

    def __init__(self, config: PredictorConfig) -> None:
        self.config = config
        self.database = Database(config.database_url)
        self.etl = MarketDataETL(config, database=self.database)
        self._price_source = "database"
        self._news_source = "database"

    # ------------------------------------------------------------------
    # Public API used by the pipeline
    # ------------------------------------------------------------------
    def fetch_price_data(self, force: bool = False) -> pd.DataFrame:
        """Return price data for the configured ticker, refreshing if required."""

        result = self.etl.refresh_prices(force=force)
        self._price_source = "remote" if result.downloaded or force else "database"
        return result.data

    def fetch_news_data(self, force: bool = False) -> pd.DataFrame:
        """Return cached news articles, refreshing via the API when necessary."""

        result = self.etl.refresh_news(force=force)
        self._news_source = "remote" if result.downloaded or force else "database"
        return result.data

    def fetch_indicator_data(self, category: Optional[str] = None) -> pd.DataFrame:
        """Fetch indicator data for the current ticker and interval."""

        return self.database.get_indicators(
            ticker=self.config.ticker,
            interval=self.config.interval,
            progress=False,
            auto_adjust=False,
        )

    def fetch_fundamentals(self) -> pd.DataFrame:
        """Fetch cached fundamentals for the configured ticker."""

        return self.database.get_fundamentals(self.config.ticker)

    def refresh_all(self, force: bool = False) -> dict[str, int]:
        """Refresh all supported datasets and return a summary of inserted rows."""

        summary = self.etl.refresh_all(force=force)
        if summary.get("downloaded"):
            self._price_source = "remote"
            self._news_source = "remote"
        return summary

    # ------------------------------------------------------------------
    # Convenience helpers for the UI layer
    # ------------------------------------------------------------------
    @property
    def last_price_source(self) -> str:
        return self._price_source

    @property
    def last_news_source(self) -> str:
        return self._news_source

    def get_last_updated(self, category: str) -> Optional[datetime]:
        """Return the last refresh timestamp recorded for the given category."""

        interval = self._interval_for_category(category)
        return self.database.get_refresh_timestamp(
            self.config.ticker, interval, category
        )

    def get_data_sources(self) -> list[str]:
        """Return a human-readable list of data sources touched during the ETL run."""

        return self.etl.list_sources()

    def _interval_for_category(self, category: str) -> str:
        if category in {"prices", "indicators", "macro"}:
            return self.config.interval
        return "global"


__all__ = ["DataFetcher"]

