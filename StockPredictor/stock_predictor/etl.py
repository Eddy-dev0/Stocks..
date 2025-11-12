"""Extract-transform-load helpers for market data ingestion."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Callable, DefaultDict, Dict, Iterable, Tuple

import pandas as pd
import requests
import yfinance as yf
from urllib.error import HTTPError

from .config import PredictorConfig
from .database import Database
from .preprocessing import compute_price_features
from .sentiment import aggregate_daily_sentiment, attach_sentiment

LOGGER = logging.getLogger(__name__)

YF_PRICES_MISSING_ERROR = getattr(yf, "YFPricesMissingError", None)
YF_TZ_MISSING_ERROR = getattr(yf, "YFTzMissingError", None)

YFINANCE_DOWNLOAD_ERRORS: tuple[type[Exception], ...] = tuple(
    exc
    for exc in (YF_PRICES_MISSING_ERROR, YF_TZ_MISSING_ERROR)
    if isinstance(exc, type) and issubclass(exc, Exception)
)


class NoPriceDataError(RuntimeError):
    """Raised when a ticker fails to yield any price observations."""

    def __init__(self, ticker: str, message: str = "") -> None:
        self.ticker = ticker
        base = f"No price data returned for ticker {ticker}."
        if message:
            base = f"{base} Details: {message}"
        super().__init__(base)


TICKER_HINTS: dict[str, str] = {
    "RHEINMETALL": "Try RHM.DE (XETRA).",
    "RHM": "Try RHM.DE (XETRA).",
    "S&P": "Try ^GSPC for the S&P 500 index.",
    "SP500": "Try ^GSPC for the S&P 500 index.",
    "NYSE": "Try ^NYA for NYSE composite.",
}

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
        self._source_log: DefaultDict[str, set[str]] = defaultdict(set)

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
        if existing.empty:
            LOGGER.info(
                "No cached price data found for %s; attempting to download fresh data.",
                self.config.ticker,
            )
        if not force and self._covers_requested_range(existing):
            LOGGER.debug("Price data already present for %s", self.config.ticker)
            self._record_source(
                "database",
                f"Cached price history for {self.config.ticker} ({self.config.interval} interval)",
            )
            return RefreshResult(existing, downloaded=False)

        LOGGER.info(
            "Downloading price data for %s (%s - %s)",
            self.config.ticker,
            self.config.start_date,
            self.config.end_date or "today",
        )
        try:
            downloaded = self._download_price_data()
        except HTTPError as exc:
            LOGGER.error("HTTP error while downloading prices for %s: %s", self.config.ticker, exc)
            raise NoPriceDataError(
                self.config.ticker,
                self._compose_error_message(str(exc)),
            ) from exc
        except Exception as exc:  # pylint: disable=broad-except
            if YFINANCE_DOWNLOAD_ERRORS and isinstance(exc, YFINANCE_DOWNLOAD_ERRORS):  # type: ignore[arg-type]
                LOGGER.error("Failed to download prices for %s: %s", self.config.ticker, exc)
                raise NoPriceDataError(
                    self.config.ticker,
                    self._compose_error_message(str(exc)),
                ) from exc
            raise

        if downloaded.empty:
            LOGGER.error(
                "No price data returned for ticker %s in requested range",
                self.config.ticker,
            )
            raise NoPriceDataError(
                self.config.ticker,
                self._compose_error_message("Empty dataframe from data provider."),
            )
        self.database.upsert_prices(self.config.ticker, self.config.interval, downloaded)
        self.database.set_refresh_timestamp(
            self.config.ticker, self.config.interval, "prices"
        )
        self._record_source(
            "yfinance",
            f"Price history for {self.config.ticker} ({self.config.interval} interval)",
        )
        refreshed = self.database.get_prices(
            ticker=self.config.ticker,
            interval=self.config.interval,
            start=self.config.start_date,
            end=self.config.end_date,
        )
        if refreshed.empty:
            raise NoPriceDataError(
                self.config.ticker,
                self._compose_error_message("Cached dataset is empty after refresh."),
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
            self._record_source("local", "Derived technical indicators from price history")
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
            self._record_source("yfinance", f"Fundamental statements for {self.config.ticker}")
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
            auto_adjust=False,
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
            self._record_source(
                "yfinance",
                "Macro indicators: "
                + ", ".join(f"{symbol} ({name})" for symbol, name in MACRO_SYMBOLS.items()),
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
            self._record_source("news_api", f"News and sentiment articles for {self.config.ticker}")
        refreshed = self.database.get_news(self.config.ticker)
        return RefreshResult(refreshed, downloaded=True)

    def refresh_corporate_events(self, force: bool = False) -> RefreshResult:
        existing = self.database.get_corporate_events(self.config.ticker)
        if not force and not existing.empty:
            self._record_source(
                "database", f"Cached corporate events for {self.config.ticker}"
            )
            return RefreshResult(existing, downloaded=False)

        LOGGER.info("Refreshing corporate action history for %s", self.config.ticker)
        ticker = yf.Ticker(self.config.ticker)
        records: list[dict[str, Any]] = []
        downloaded = False

        actions = self._safe_download(lambda: getattr(ticker, "actions", pd.DataFrame()))
        if isinstance(actions, pd.DataFrame) and not actions.empty:
            actions = actions.reset_index().rename(columns={actions.index.name or "index": "Date"})
            actions["Date"] = pd.to_datetime(actions["Date"], errors="coerce")
            actions = actions.dropna(subset=["Date"])
            for row in actions.itertuples(index=False):
                dividend = getattr(row, "Dividends", None)
                if dividend not in (None, 0, 0.0):
                    records.append(
                        {
                            "Ticker": self.config.ticker,
                            "EventType": "dividend",
                            "EventDate": row.Date,
                            "Reference": "dividend",
                            "Value": dividend,
                            "Currency": None,
                            "Details": {"amount": dividend},
                            "Source": "yfinance",
                        }
                    )
                split = getattr(row, "Stock_Splits", getattr(row, "Stock Splits", None))
                if split not in (None, 0, 0.0):
                    records.append(
                        {
                            "Ticker": self.config.ticker,
                            "EventType": "split",
                            "EventDate": row.Date,
                            "Reference": "stock_split",
                            "Value": split,
                            "Currency": None,
                            "Details": {"ratio": split},
                            "Source": "yfinance",
                        }
                    )
            downloaded = downloaded or bool(records)

        earnings = self._safe_download(lambda: ticker.get_earnings_dates(limit=8))
        if isinstance(earnings, pd.DataFrame) and not earnings.empty:
            earnings = earnings.reset_index()
            if "Earnings Date" in earnings.columns:
                earnings["EventDate"] = pd.to_datetime(
                    earnings["Earnings Date"], errors="coerce"
                )
            elif earnings.columns[0] != "EventDate":
                earnings["EventDate"] = pd.to_datetime(
                    earnings[earnings.columns[0]], errors="coerce"
                )
            earnings = earnings.dropna(subset=["EventDate"])
            for row in earnings.itertuples(index=False):
                details = row._asdict()
                details.pop("EventDate", None)
                reference = str(details.get("Symbol") or details.get("Company") or "earnings")
                records.append(
                    {
                        "Ticker": self.config.ticker,
                        "EventType": "earnings",
                        "EventDate": row.EventDate,
                        "Reference": reference,
                        "Value": self._safe_float(details.get("EPS Estimate")),
                        "Currency": None,
                        "Details": details,
                        "Source": "yfinance",
                    }
                )
            downloaded = True

        if not records:
            records.append(
                {
                    "Ticker": self.config.ticker,
                    "EventType": "placeholder",
                    "EventDate": datetime.utcnow().date(),
                    "Reference": "placeholder",
                    "Value": None,
                    "Currency": None,
                    "Details": {
                        "note": "Placeholder corporate event generated because no provider data was returned."
                    },
                    "Source": "placeholder",
                }
            )

        inserted = self.database.upsert_corporate_events(records)
        if inserted:
            self.database.set_refresh_timestamp(
                self.config.ticker, "global", "corporate_events"
            )
            provider = "yfinance" if downloaded else "placeholder"
            self._record_source(
                provider, f"Corporate events for {self.config.ticker}"
            )
        refreshed = self.database.get_corporate_events(self.config.ticker)
        return RefreshResult(refreshed, downloaded=downloaded)

    def refresh_options_surface(self, force: bool = False) -> RefreshResult:
        existing = self.database.get_option_surface(self.config.ticker)
        if not force and not existing.empty:
            self._record_source(
                "database", f"Cached option surface for {self.config.ticker}"
            )
            return RefreshResult(existing, downloaded=False)

        LOGGER.info("Refreshing option surface snapshot for %s", self.config.ticker)
        ticker = yf.Ticker(self.config.ticker)
        records: list[dict[str, Any]] = []
        downloaded = False
        as_of = datetime.utcnow()

        expirations = self._safe_download(lambda: list(getattr(ticker, "options", []) or []))
        if expirations:
            expiration = expirations[0]
            chain = self._safe_download(lambda: ticker.option_chain(expiration))
            if chain:
                metrics = {
                    "lastPrice": "last_price",
                    "bid": "bid",
                    "ask": "ask",
                    "impliedVolatility": "implied_volatility",
                    "openInterest": "open_interest",
                    "volume": "volume",
                }
                for option_type, frame in (("call", chain.calls), ("put", chain.puts)):
                    if frame is None or frame.empty:
                        continue
                    subset = frame.head(50).copy()
                    subset["expiration"] = pd.to_datetime(expiration, errors="coerce")
                    subset = subset.dropna(subset=["expiration", "strike"])
                    for row in subset.itertuples(index=False):
                        strike = self._safe_float(getattr(row, "strike", None))
                        if strike is None:
                            continue
                        extra = {
                            "contract": getattr(row, "contractSymbol", None),
                            "lastTradeDate": str(getattr(row, "lastTradeDate", "")),
                        }
                        bid = self._safe_float(getattr(row, "bid", None))
                        ask = self._safe_float(getattr(row, "ask", None))
                        if bid is not None and ask is not None:
                            records.append(
                                {
                                    "Ticker": self.config.ticker,
                                    "AsOf": as_of,
                                    "Expiration": getattr(row, "expiration"),
                                    "Strike": strike,
                                    "OptionType": option_type,
                                    "Metric": "mid_price",
                                    "Value": (bid + ask) / 2,
                                    "Source": "yfinance",
                                    "Extra": extra,
                                }
                            )
                        for raw_metric, alias in metrics.items():
                            value = getattr(row, raw_metric, None)
                            if value is None:
                                continue
                            if isinstance(value, str) and not value.strip():
                                continue
                            if isinstance(value, (float, int)) and pd.isna(value):
                                continue
                            records.append(
                                {
                                    "Ticker": self.config.ticker,
                                    "AsOf": as_of,
                                    "Expiration": getattr(row, "expiration"),
                                    "Strike": strike,
                                    "OptionType": option_type,
                                    "Metric": alias,
                                    "Value": self._safe_float(value),
                                    "Source": "yfinance",
                                    "Extra": extra,
                                }
                            )
                downloaded = bool(records)

        if not records:
            records.append(
                {
                    "Ticker": self.config.ticker,
                    "AsOf": as_of,
                    "Expiration": as_of.date(),
                    "Strike": 0.0,
                    "OptionType": "call",
                    "Metric": "placeholder",
                    "Value": None,
                    "Source": "placeholder",
                    "Extra": {
                        "note": "Placeholder option surface point generated because provider data was unavailable."
                    },
                }
            )

        inserted = self.database.upsert_option_surface(records)
        if inserted:
            self.database.set_refresh_timestamp(
                self.config.ticker, "global", "options"
            )
            provider = "yfinance" if downloaded else "placeholder"
            self._record_source(provider, f"Option surface snapshot for {self.config.ticker}")
        refreshed = self.database.get_option_surface(self.config.ticker)
        return RefreshResult(refreshed, downloaded=downloaded)

    def refresh_sentiment_signals(self, force: bool = False) -> RefreshResult:
        existing = self.database.get_sentiment_signals(self.config.ticker)
        if not force and not existing.empty:
            self._record_source(
                "database", f"Cached sentiment signals for {self.config.ticker}"
            )
            return RefreshResult(existing, downloaded=False)

        news = self.database.get_news(self.config.ticker)
        if news.empty and self.config.sentiment:
            news_result = self.refresh_news(force=False)
            news = news_result.data

        records: list[dict[str, Any]] = []
        downloaded = False
        if not news.empty:
            working = news.copy()
            if "publishedDate" in working.columns:
                working["publishedDate"] = pd.to_datetime(
                    working["publishedDate"], errors="coerce"
                )
            elif "PublishedAt" in working.columns:
                working["publishedDate"] = pd.to_datetime(
                    working["PublishedAt"], errors="coerce"
                )
            else:
                working["publishedDate"] = pd.NaT
            working = working.dropna(subset=["publishedDate"])
            if "Summary" in working.columns:
                working["text"] = working["Summary"].fillna("")
            elif "summary" in working.columns:
                working["text"] = working["summary"].fillna("")
            elif "Title" in working.columns:
                working["text"] = working["Title"].fillna("")
            elif "title" in working.columns:
                working["text"] = working["title"].fillna("")
            else:
                working["text"] = ""
            scored = attach_sentiment(working)
            aggregated = aggregate_daily_sentiment(scored)
            counts_series = (
                scored.groupby(scored["publishedDate"].dt.date)["publishedDate"].count()
                if "publishedDate" in scored.columns
                else pd.Series(dtype=int)
            )
            counts = counts_series.to_dict()
            for row in aggregated.itertuples(index=False):
                row_date = getattr(row, "Date", None)
                if pd.isna(row_date):
                    continue
                count = int(counts.get(row_date.date(), 0))
                records.append(
                    {
                        "Ticker": self.config.ticker,
                        "AsOf": row_date,
                        "Provider": "vader",
                        "SignalType": "news_sentiment",
                        "Score": getattr(row, "sentiment", 0.0),
                        "Magnitude": None,
                        "Payload": {"articles": count},
                    }
                )
            downloaded = bool(records)

        if not records:
            records.append(
                {
                    "Ticker": self.config.ticker,
                    "AsOf": datetime.utcnow(),
                    "Provider": "placeholder",
                    "SignalType": "news_sentiment",
                    "Score": 0.0,
                    "Magnitude": None,
                    "Payload": {
                        "note": "Placeholder sentiment score generated because no news data was available."
                    },
                }
            )

        inserted = self.database.upsert_sentiment_signals(records)
        if inserted:
            self.database.set_refresh_timestamp(
                self.config.ticker, "global", "sentiment"
            )
            provider = "analytics" if downloaded else "placeholder"
            self._record_source(
                provider, f"Sentiment signals for {self.config.ticker}"
            )
        refreshed = self.database.get_sentiment_signals(self.config.ticker)
        return RefreshResult(refreshed, downloaded=downloaded)

    def refresh_esg_metrics(self, force: bool = False) -> RefreshResult:
        existing = self.database.get_esg_metrics(self.config.ticker)
        if not force and not existing.empty:
            self._record_source(
                "database", f"Cached ESG metrics for {self.config.ticker}"
            )
            return RefreshResult(existing, downloaded=False)

        LOGGER.info("Refreshing ESG metrics for %s", self.config.ticker)
        ticker = yf.Ticker(self.config.ticker)
        records: list[dict[str, Any]] = []
        downloaded = False
        today = datetime.utcnow().date()

        sustainability = self._safe_download(lambda: getattr(ticker, "sustainability", None))
        if isinstance(sustainability, pd.DataFrame) and not sustainability.empty:
            frame = sustainability.reset_index()
            frame.columns = ["Metric", "Value"] if frame.shape[1] >= 2 else ["Metric"]
            for row in frame.itertuples(index=False):
                metric = getattr(row, "Metric", None)
                if metric is None:
                    continue
                value = getattr(row, "Value", None)
                records.append(
                    {
                        "Ticker": self.config.ticker,
                        "AsOf": today,
                        "Provider": "yfinance",
                        "Metric": str(metric),
                        "Value": self._safe_float(value),
                        "Raw": row._asdict(),
                    }
                )
            downloaded = bool(records)

        if not records:
            records.append(
                {
                    "Ticker": self.config.ticker,
                    "AsOf": today,
                    "Provider": "placeholder",
                    "Metric": "esg_placeholder",
                    "Value": None,
                    "Raw": {
                        "note": "Placeholder ESG metrics generated because provider data was unavailable."
                    },
                }
            )

        inserted = self.database.upsert_esg_metrics(records)
        if inserted:
            self.database.set_refresh_timestamp(
                self.config.ticker, "global", "esg"
            )
            provider = "yfinance" if downloaded else "placeholder"
            self._record_source(provider, f"ESG metrics for {self.config.ticker}")
        refreshed = self.database.get_esg_metrics(self.config.ticker)
        return RefreshResult(refreshed, downloaded=downloaded)

    def refresh_ownership_flows(self, force: bool = False) -> RefreshResult:
        existing = self.database.get_ownership_flows(self.config.ticker)
        if not force and not existing.empty:
            self._record_source(
                "database", f"Cached ownership data for {self.config.ticker}"
            )
            return RefreshResult(existing, downloaded=False)

        LOGGER.info("Refreshing ownership and flow data for %s", self.config.ticker)
        ticker = yf.Ticker(self.config.ticker)
        records: list[dict[str, Any]] = []
        downloaded = False

        for holder_type, accessor in (
            ("institutional", "institutional_holders"),
            ("mutual_fund", "mutualfund_holders"),
        ):
            frame = self._safe_download(lambda attr=accessor: getattr(ticker, attr, None))
            if isinstance(frame, pd.DataFrame) and not frame.empty:
                frame = frame.copy()
                frame.columns = [col.replace(" ", "_").lower() for col in frame.columns]
                as_of_col = None
                for candidate in ("date_reported", "report_date", "asofdate"):
                    if candidate in frame.columns:
                        as_of_col = candidate
                        break
                for row in frame.itertuples(index=False):
                    holder = getattr(row, "holder", None) or getattr(row, "organization", None)
                    if holder is None:
                        continue
                    as_of_val = getattr(row, as_of_col) if as_of_col else None
                    as_of_date = self._normalize_date(as_of_val)
                    if as_of_date is None:
                        as_of_date = datetime.utcnow().date()
                    row_dict = row._asdict()
                    for metric_key, metric_name in (
                        ("shares", "shares"),
                        ("pct_held", "percent_held"),
                        ("value", "market_value"),
                    ):
                        if metric_key in row_dict and row_dict[metric_key] not in (None, ""):
                            records.append(
                                {
                                    "Ticker": self.config.ticker,
                                    "AsOf": as_of_date,
                                    "Holder": holder,
                                    "HolderType": holder_type,
                                    "Metric": metric_name,
                                    "Value": self._safe_float(row_dict[metric_key]),
                                    "Raw": row_dict,
                                    "Source": "yfinance",
                                }
                            )
                downloaded = downloaded or bool(records)

        major_holders = self._safe_download(lambda: getattr(ticker, "major_holders", None))
        if isinstance(major_holders, pd.DataFrame) and not major_holders.empty:
            for row in major_holders.itertuples(index=False):
                if len(row) < 2:
                    continue
                percent_raw = row[0]
                description = str(row[1])
                percent = None
                if isinstance(percent_raw, str) and percent_raw.endswith("%"):
                    percent_value = self._safe_float(percent_raw.replace("%", ""))
                    if percent_value is not None:
                        percent = percent_value / 100
                else:
                    percent = self._safe_float(percent_raw)
                if percent is None:
                    continue
                records.append(
                    {
                        "Ticker": self.config.ticker,
                        "AsOf": datetime.utcnow().date(),
                        "Holder": description,
                        "HolderType": "major",
                        "Metric": "percent_shares_outstanding",
                        "Value": percent,
                        "Raw": {"percent": percent, "description": description},
                        "Source": "yfinance",
                    }
                )
            downloaded = True

        if not records:
            records.append(
                {
                    "Ticker": self.config.ticker,
                    "AsOf": datetime.utcnow().date(),
                    "Holder": "placeholder",
                    "HolderType": "unknown",
                    "Metric": "shares",
                    "Value": None,
                    "Raw": {
                        "note": "Placeholder ownership row generated because provider data was unavailable."
                    },
                    "Source": "placeholder",
                }
            )

        inserted = self.database.upsert_ownership_flows(records)
        if inserted:
            self.database.set_refresh_timestamp(
                self.config.ticker, "global", "ownership"
            )
            provider = "yfinance" if downloaded else "placeholder"
            self._record_source(provider, f"Ownership and flow data for {self.config.ticker}")
        refreshed = self.database.get_ownership_flows(self.config.ticker)
        return RefreshResult(refreshed, downloaded=downloaded)

    def refresh_all(self, force: bool = False) -> dict[str, int]:
        prices_result = self.refresh_prices(force=force)
        indicators_count = self.refresh_indicators(prices_result.data, force=force)
        fundamentals_count = self.refresh_fundamentals(force=force)
        macro_count = self.refresh_macro(force=force)
        news_result = self.refresh_news(force=force if self.config.sentiment else False)
        corporate_events = self.refresh_corporate_events(force=force)
        options_surface = self.refresh_options_surface(force=force)
        sentiment_signals = self.refresh_sentiment_signals(force=force)
        esg_metrics = self.refresh_esg_metrics(force=force)
        ownership_data = self.refresh_ownership_flows(force=force)

        downloaded_flag = (
            prices_result.downloaded
            or news_result.downloaded
            or corporate_events.downloaded
            or options_surface.downloaded
            or sentiment_signals.downloaded
            or esg_metrics.downloaded
            or ownership_data.downloaded
            or force
        )

        return {
            "prices": int(len(prices_result.data)),
            "indicators": int(indicators_count),
            "fundamentals": int(fundamentals_count),
            "macro_indicators": int(macro_count),
            "news": int(len(news_result.data)),
            "corporate_events": int(len(corporate_events.data)),
            "options_surface": int(len(options_surface.data)),
            "sentiment_signals": int(len(sentiment_signals.data)),
            "esg_metrics": int(len(esg_metrics.data)),
            "ownership_flows": int(len(ownership_data.data)),
            "downloaded": int(downloaded_flag),
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
            auto_adjust=False,
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

    def list_sources(self) -> list[str]:
        entries: list[str] = []
        for provider, descriptions in self._source_log.items():
            for description in sorted(descriptions):
                entries.append(f"{provider}: {description}")
        return entries

    def _record_source(self, provider: str, description: str) -> None:
        self._source_log[provider].add(description)

    @staticmethod
    def _normalize_date(value: Any) -> date | None:
        if value is None:
            return None
        parsed = pd.to_datetime(value, errors="coerce")
        if pd.isna(parsed):
            return None
        return parsed.date()

    def _compose_error_message(self, message: str | None) -> str:
        parts: list[str] = []
        if message:
            parts.append(message)
        hint = self._ticker_hint()
        if hint:
            parts.append(f"Hint: {hint}")
        return " ".join(parts)

    def _ticker_hint(self) -> str:
        ticker_upper = (self.config.ticker or "").upper()
        return TICKER_HINTS.get(ticker_upper, "")

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

    @staticmethod
    def _safe_float(value: Any) -> float | None:
        try:
            if value is None:
                return None
            if isinstance(value, str) and not value.strip():
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _safe_download(fetcher: Callable[[], Any]) -> Any:
        try:
            return fetcher()
        except Exception as exc:  # pylint: disable=broad-except
            LOGGER.debug("Optional data fetch failed: %s", exc)
            return None


__all__ = ["MarketDataETL", "RefreshResult", "NoPriceDataError"]

