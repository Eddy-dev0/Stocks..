"""Utilities for downloading market and news data from the internet."""

from __future__ import annotations

import json
import logging

import pandas as pd
import requests
import yfinance as yf

from .config import PredictorConfig

LOGGER = logging.getLogger(__name__)


class DataFetcher:
    """Download financial time series data and optionally related news."""

    NEWS_ENDPOINT = "https://financialmodelingprep.com/api/v3/stock_news"

    def __init__(self, config: PredictorConfig) -> None:
        self.config = config

    def fetch_price_data(self, force: bool = False) -> pd.DataFrame:
        """Fetch historical price data for the configured ticker."""

        cache_path = self.config.price_cache_path
        if cache_path.exists() and not force:
            LOGGER.info("Loading cached price data from %s", cache_path)
            return pd.read_csv(cache_path, parse_dates=["Date"])

        LOGGER.info(
            "Downloading price data for %s (%s - %s)",
            self.config.ticker,
            self.config.start_date,
            self.config.end_date or "today",
        )
        df = yf.download(
            tickers=self.config.ticker,
            start=self.config.start_date.isoformat(),
            end=self.config.end_date.isoformat() if self.config.end_date else None,
            interval=self.config.interval,
            progress=False,
        )
        if df.empty:
            raise RuntimeError(
                f"No price data returned for ticker {self.config.ticker}."
            )

        df = df.reset_index().rename(columns=str.title)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_path, index=False)
        LOGGER.info("Saved price data to %s", cache_path)
        return df

    def fetch_news_data(self, force: bool = False) -> pd.DataFrame:
        """Fetch recent news articles for the configured ticker."""

        cache_path = self.config.news_cache_path
        if cache_path.exists() and not force:
            LOGGER.info("Loading cached news data from %s", cache_path)
            return pd.read_csv(cache_path, parse_dates=["publishedDate"])

        if not self.config.news_api_key:
            LOGGER.warning(
                "No API key configured for news download. Skipping news collection."
            )
            return pd.DataFrame()

        params = {
            "tickers": self.config.ticker,
            "limit": self.config.news_limit,
            "apikey": self.config.news_api_key,
        }
        LOGGER.info(
            "Downloading up to %s news articles for %s",
            self.config.news_limit,
            self.config.ticker,
        )
        try:
            response = requests.get(self.NEWS_ENDPOINT, params=params, timeout=30)
            response.raise_for_status()
            articles = response.json()
        except requests.RequestException as exc:
            LOGGER.error("Failed to download news data: %s", exc)
            return pd.DataFrame()
        except ValueError as exc:  # json decoding error
            LOGGER.error("Failed to parse news response: %s", exc)
            return pd.DataFrame()
        if not isinstance(articles, list):
            LOGGER.error("Unexpected response for news data: %s", json.dumps(articles)[:200])
            return pd.DataFrame()

        if not articles:
            LOGGER.warning("No news articles returned for %s", self.config.ticker)
            return pd.DataFrame()

        df = pd.DataFrame(articles)
        if "publishedDate" in df.columns:
            df["publishedDate"] = pd.to_datetime(df["publishedDate"], errors="coerce")
            df = df.dropna(subset=["publishedDate"])
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_path, index=False)
        LOGGER.info("Saved news data to %s", cache_path)
        return df

    def download_all(self, force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Download price and news data in one call."""

        prices = self.fetch_price_data(force=force)
        news = self.fetch_news_data(force=force)
        return prices, news
