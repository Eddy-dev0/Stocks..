"""Extract-transform-load helpers for market data ingestion."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, Tuple

import pandas as pd
import requests
import yfinance as yf

from .config import PredictorConfig
from .database import Database
from .preprocessing import compute_price_features

LOGGER = logging.getLogger(__name__)

TECHNICAL_INDICATORS = [
    "Return_1d",
    "LogReturn_1d",
    "SMA_5",
    "SMA_10",
    "EMA_9",
    "Volatility_5",
    "Volume_Change",
]

MACRO_SYMBOLS: dict[str, str] = {
    "^GSPC": "S&P 500",
    "^VIX": "CBOE Volatility Index",
    "DX-Y.NYB": "US Dollar Index",
}


@dataclass(slots=True)
class RefreshResult:
    """Container capturing ETL refresh outcomes."""

    data: pd.DataFrame
    downloaded: bool


class MarketDataETL:
    """Services that orchestrate downloading, normalising and storing data."""

    NEWS_ENDPOINT = "https://financialmodelingprep.com/api/v3/stock_news"

    def __init__(self, config: PredictorConfig, database: Database | None = None) -> None:
        self.config = config
        self.database = database or Database(config.database_url)

    # ------------------------------------------------------------------
    # Public refresh API
    # ------------------------------------------------------------------
    def refresh_prices(self, force: bool = False) -> RefreshResult:
        existing = self.database.get_prices(
            ticker=self.config.ticker,
            interval=self.config.interval,
            start=self.config.start_date,
            end=self.config.end_date,
        )
        if not force and self._covers_requested_range(existing):
            LOGGER.debug("Price data already present for %s", self.config.ticker)
            return RefreshResult(existing, downloaded=False)

        LOGGER.info(
            "Downloading price data for %s (%s - %s)",
            self.config.ticker,
            self.config.start_date,
            self.config.end_date or "today",
        )
        downloaded = self._download_price_data()
        if downloaded.empty:
            raise RuntimeError(
                f"No price data returned for ticker {self.config.ticker}."
            )
        self.database.upsert_prices(self.config.ticker, self.config.interval, downloaded)
        self.database.set_refresh_timestamp(
            self.config.ticker, self.config.interval, "prices"
        )
        refreshed = self.database.get_prices(
            ticker=self.config.ticker,
            interval=self.config.interval,
            start=self.config.start_date,
            end=self.config.end_date,
        )
        return RefreshResult(refreshed, downloaded=True)

    def refresh_indicators(
        self, price_frame: pd.DataFrame | None = None, force: bool = False
    ) -> int:
        if price_frame is None or price_frame.empty:
            price_frame = self.database.get_prices(
                ticker=self.config.ticker,
                interval=self.config.interval,
                start=self.config.start_date,
                end=self.config.end_date,
            )
        if price_frame.empty:
            LOGGER.warning("Cannot compute indicators without price data.")
            return 0

        LOGGER.info("Computing technical indicators for %s", self.config.ticker)
        enriched = compute_price_features(price_frame)
        records: list[dict[str, object]] = []
        for _, row in enriched.iterrows():
            for indicator in TECHNICAL_INDICATORS:
                if indicator not in row:
                    continue
                records.append(
                    {
                        "Date": row["Date"],
                        "Indicator": indicator,
                        "Value": row[indicator],
                        "Category": "technical",
                    }
                )
        inserted = self.database.upsert_indicators(
            ticker=self.config.ticker,
            interval=self.config.interval,
            records=records,
        )
        if inserted:
            self.database.set_refresh_timestamp(
                self.config.ticker, self.config.interval, "indicators"
            )
        return inserted

    def refresh_fundamentals(self, force: bool = False) -> int:
        existing = self.database.get_fundamentals(self.config.ticker)
        if not force and not existing.empty:
            return 0

        LOGGER.info("Downloading fundamentals for %s", self.config.ticker)
        ticker = yf.Ticker(self.config.ticker)

        records: list[dict[str, object]] = []
        statement_sources: Dict[str, Tuple[str, pd.DataFrame | None]] = {
            "income_statement_annual": ("annual", getattr(ticker, "financials", None)),
            "income_statement_quarterly": (
                "quarterly",
                getattr(ticker, "quarterly_financials", None),
            ),
            "balance_sheet_annual": ("annual", getattr(ticker, "balance_sheet", None)),
            "balance_sheet_quarterly": (
                "quarterly",
                getattr(ticker, "quarterly_balance_sheet", None),
            ),
            "cashflow_annual": ("annual", getattr(ticker, "cashflow", None)),
            "cashflow_quarterly": (
                "quarterly",
                getattr(ticker, "quarterly_cashflow", None),
            ),
        }

        for statement_name, (period, frame) in statement_sources.items():
            normalized = self._normalize_statement(frame, statement_name, period)
            records.extend(normalized)

        info = getattr(ticker, "info", {}) or {}
        if info:
            as_of = datetime.utcnow().date()
            for key, value in info.items():
                record = {
                    "Ticker": self.config.ticker,
                    "Statement": "company_profile",
                    "Period": "latest",
                    "AsOf": as_of,
                    "Metric": key,
                    "Value": value if isinstance(value, (int, float)) else None,
                    "Raw": value if not isinstance(value, (int, float)) else None,
                }
                records.append(record)

        inserted = self.database.upsert_fundamentals(records)
        if inserted:
            self.database.set_refresh_timestamp(
                self.config.ticker, "global", "fundamentals"
            )
        return inserted

    def refresh_macro(self, force: bool = False) -> int:
        existing = self.database.get_indicators(
            ticker=self.config.ticker,
            interval=self.config.interval,
            category="macro",
        )
        if not force and not existing.empty:
            return 0

        start = self.config.start_date
        end = self.config.end_date
        LOGGER.info("Downloading macro indicators: %s", ", ".join(MACRO_SYMBOLS))
        data = yf.download(
            tickers=list(MACRO_SYMBOLS.keys()),
            start=start.isoformat() if start else None,
            end=end.isoformat() if end else None,
            interval="1d",
            progress=False,
            group_by="ticker",
        )
        if data.empty:
            LOGGER.warning("No macro data returned from yfinance.")
            return 0

        records: list[dict[str, object]] = []
        for symbol, label in MACRO_SYMBOLS.items():
            series = None
            if (symbol, "Close") in data.columns:
                series = data[(symbol, "Close")]
            elif "Close" in data.columns:
                # Single symbol returns a flat frame
                series = data["Close"]
            elif symbol in data:
                series = data[symbol].get("Close") if isinstance(data[symbol], pd.DataFrame) else data[symbol]
            if series is None:
                continue
            series = series.dropna()
            for idx, value in series.items():
                records.append(
                    {
                        "Date": idx,
                        "Indicator": f"macro:{symbol}",
                        "Value": value,
                        "Category": "macro",
                        "Extra": {"name": label},
                    }
                )

        inserted = self.database.upsert_indicators(
            ticker=self.config.ticker,
            interval=self.config.interval,
            records=records,
        )
        if inserted:
            self.database.set_refresh_timestamp(
                self.config.ticker, self.config.interval, "macro"
            )
        return inserted

    def refresh_news(self, force: bool = False) -> RefreshResult:
        existing = self.database.get_news(self.config.ticker)
        if not force and not existing.empty:
            return RefreshResult(existing, downloaded=False)

        if not self.config.news_api_key:
            LOGGER.info(
                "No news API key configured; returning cached database entries only."
            )
            return RefreshResult(existing, downloaded=False)

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
            return RefreshResult(existing, downloaded=False)
        except ValueError as exc:  # json decoding error
            LOGGER.error("Failed to parse news response: %s", exc)
            return RefreshResult(existing, downloaded=False)

        if not isinstance(articles, list):
            LOGGER.error("Unexpected response for news data: %s", json.dumps(articles)[:200])
            return RefreshResult(existing, downloaded=False)
        if not articles:
            LOGGER.warning("No news articles returned for %s", self.config.ticker)
            return RefreshResult(existing, downloaded=True)

        frame = pd.DataFrame(articles)
        if "publishedDate" in frame.columns:
            frame["publishedDate"] = pd.to_datetime(
                frame["publishedDate"], errors="coerce"
            )
            frame = frame.dropna(subset=["publishedDate"])
        records = []
        for row in frame.itertuples(index=False):
            records.append(
                {
                    "Ticker": self.config.ticker,
                    "PublishedAt": getattr(row, "publishedDate", None),
                    "Title": getattr(row, "title", None),
                    "Summary": getattr(row, "text", getattr(row, "summary", None)),
                    "Url": getattr(row, "url", None),
                    "Source": getattr(row, "site", getattr(row, "source", None)),
                }
            )

        inserted = self.database.upsert_news(records)
        if inserted:
            self.database.set_refresh_timestamp(
                self.config.ticker, "global", "news"
            )
        refreshed = self.database.get_news(self.config.ticker)
        return RefreshResult(refreshed, downloaded=True)

    def refresh_all(self, force: bool = False) -> dict[str, int]:
        prices_result = self.refresh_prices(force=force)
        indicators_count = self.refresh_indicators(prices_result.data, force=force)
        fundamentals_count = self.refresh_fundamentals(force=force)
        macro_count = self.refresh_macro(force=force)
        news_result = self.refresh_news(force=force if self.config.sentiment else False)

        return {
            "prices": int(len(prices_result.data)),
            "indicators": int(indicators_count),
            "fundamentals": int(fundamentals_count),
            "macro_indicators": int(macro_count),
            "news": int(len(news_result.data)),
            "downloaded": int(prices_result.downloaded or news_result.downloaded or force),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _download_price_data(self) -> pd.DataFrame:
        df = yf.download(
            tickers=self.config.ticker,
            start=self.config.start_date.isoformat() if self.config.start_date else None,
            end=self.config.end_date.isoformat() if self.config.end_date else None,
            interval=self.config.interval,
            progress=False,
        )
        if df.empty:
            return pd.DataFrame()
        df = df.reset_index()
        if isinstance(df.columns, pd.MultiIndex):
            # Flatten multi-level columns returned by yfinance when requesting a
            # single ticker so downstream code can access fields using the
            # expected names (e.g. "Open", "Close"). We only keep the price
            # field component as the ticker level is redundant once flattened.
            df.columns = df.columns.get_level_values(0)
        # Ensure the first column is always normalised to "Date" even when
        # yfinance labels it differently (e.g. "Datetime").
        df = df.rename(columns={df.columns[0]: "Date"})
        df.columns = [col.title() if isinstance(col, str) else col for col in df.columns]
        return df

    def _covers_requested_range(self, frame: pd.DataFrame) -> bool:
        if frame.empty:
            return False
        start = self.config.start_date
        end = self.config.end_date
        frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
        if frame["Date"].isna().all():
            return False
        min_date = frame["Date"].min().date()
        max_date = frame["Date"].max().date()
        if start and min_date > start:
            return False
        if end and max_date < end:
            return False
        if not end:
            today = datetime.utcnow().date()
            if (today - max_date).days > 2:
                return False
        return True

    def _normalize_statement(
        self, frame: pd.DataFrame | None, statement_name: str, period: str
    ) -> Iterable[dict[str, object]]:
        if frame is None or frame.empty:
            return []
        tidy = frame.copy()
        tidy.columns = pd.to_datetime(tidy.columns, errors="coerce")
        tidy = tidy.dropna(axis=1, how="all")
        if tidy.empty:
            return []
        stacked = tidy.stack().reset_index()
        stacked.columns = ["Metric", "AsOf", "Value"]
        stacked["AsOf"] = pd.to_datetime(stacked["AsOf"], errors="coerce")
        stacked = stacked.dropna(subset=["AsOf"])
        records = []
        for row in stacked.itertuples(index=False):
            records.append(
                {
                    "Ticker": self.config.ticker,
                    "Statement": statement_name,
                    "Period": period,
                    "AsOf": row.AsOf.date(),
                    "Metric": row.Metric,
                    "Value": row.Value,
                    "Raw": None,
                }
            )
        return records


__all__ = ["MarketDataETL", "RefreshResult"]

