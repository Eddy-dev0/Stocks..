"""Extract-transform-load helpers for market data ingestion."""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, time as dt_time, timedelta, timezone, tzinfo
import re
import unicodedata
from typing import (
    Any,
    Awaitable,
    Callable,
    DefaultDict,
    Dict,
    Iterable,
    Mapping,
    Sequence,
    Tuple,
    TypeVar,
    ClassVar,
)

import pandas as pd

from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .config import PredictorConfig
from .database import Database
from .preprocessing import compute_price_features
from .sentiment import aggregate_daily_sentiment, attach_sentiment
from stock_predictor.providers.base import (
    DatasetType,
    EconomicIndicator,
    NewsArticle,
    PriceBar,
    ProviderConfigurationError,
    ProviderFailure,
    ProviderFetchSummary,
    ProviderRequest,
    ProviderResult,
    SentimentSignal,
    build_default_registry,
)

LOGGER = logging.getLogger(__name__)

T = TypeVar("T")

DEFAULT_MARKET_TIMEZONE = ZoneInfo("America/New_York")
US_MARKET_CLOSE = dt_time(16, 0)
US_MARKET_BUSINESS_DAY = CustomBusinessDay(calendar=USFederalHolidayCalendar())
CACHE_MAX_AGE = timedelta(hours=24)

_TZ_ALIAS_MAP: dict[str, str] = {
    # Central European variants (German)
    "cet": "Europe/Paris",
    "cest": "Europe/Paris",
    "central european time": "Europe/Paris",
    "central european standard time": "Europe/Paris",
    "central european summer time": "Europe/Paris",
    "mez": "Europe/Berlin",
    "mesz": "Europe/Berlin",
    "mitteleuropaeische zeit": "Europe/Berlin",
    "mitteleuropaeische sommerzeit": "Europe/Berlin",
    "mitteleuropaische zeit": "Europe/Berlin",
    "mitteleuropaische sommerzeit": "Europe/Berlin",
    "mitteleuropaeische standardzeit": "Europe/Berlin",
    "mitteleuropaeische winterzeit": "Europe/Berlin",
    "mitteleuropaeischer zeit": "Europe/Berlin",
    "mitteleuropaeischer sommerzeit": "Europe/Berlin",
    "mitteleuropaeische normalzeit": "Europe/Berlin",
    "mitteleuropaeische standard time": "Europe/Berlin",
    "mitteleuropaeische daylight time": "Europe/Berlin",
    "mitteleuropaeische sommernzeit": "Europe/Berlin",
    "mitteleuropaeische winterzeit": "Europe/Berlin",
    "mitteleuropäische zeit": "Europe/Berlin",
    "mitteleuropäische sommerzeit": "Europe/Berlin",
    "mitteleuropäische standardzeit": "Europe/Berlin",
    "mitteleuropäische winterzeit": "Europe/Berlin",
    "mitteleuropäischer zeit": "Europe/Berlin",
    "mitteleuropäischer sommerzeit": "Europe/Berlin",
    "mitteleuropäische normalzeit": "Europe/Berlin",
    "mitteleuropäische standard time": "Europe/Berlin",
    "mitteleuropäische daylight time": "Europe/Berlin",
    "hec": "Europe/Paris",
    "haec": "Europe/Paris",
    "heure d'europe centrale": "Europe/Paris",
    "heure d’europe centrale": "Europe/Paris",
    "heure deurope centrale": "Europe/Paris",
    "heure d'ete d'europe centrale": "Europe/Paris",
    "heure d’été d’europe centrale": "Europe/Paris",
    "heure dete deurope centrale": "Europe/Paris",
    "heure normale d'europe centrale": "Europe/Paris",
    "heure normale d’europe centrale": "Europe/Paris",
    "heure normale deurope centrale": "Europe/Paris",
    "heure d'hiver d'europe centrale": "Europe/Paris",
    "heure d’hiver d’europe centrale": "Europe/Paris",
    "heure dhiver deurope centrale": "Europe/Paris",
}


def resolve_market_timezone(config: PredictorConfig | None = None) -> ZoneInfo:
    """Return the market timezone configured for the application context."""

    tz_key = getattr(config, "market_timezone", None)
    timezone = _coerce_zoneinfo(tz_key)
    if timezone is not None:
        return timezone

    local_timezone = datetime.now().astimezone().tzinfo
    timezone = _coerce_zoneinfo(_tz_name(local_timezone))
    if timezone is not None:
        return timezone

    if isinstance(local_timezone, ZoneInfo):
        return local_timezone

    LOGGER.debug(
        "Falling back to default market timezone %s", DEFAULT_MARKET_TIMEZONE.key
    )
    return DEFAULT_MARKET_TIMEZONE


def _normalise_timezone_key(value: str) -> str:
    normalised = unicodedata.normalize("NFKD", value)
    normalised = normalised.replace("’", "'").replace("`", "'")
    normalised = "".join(ch for ch in normalised if not unicodedata.combining(ch))
    normalised = normalised.casefold()
    normalised = re.sub(r"[\s\-_]+", " ", normalised)
    normalised = re.sub(r"[^a-z0-9/+' ]+", " ", normalised)
    normalised = re.sub(r"\s+", " ", normalised)
    return normalised.strip()


def _coerce_zoneinfo(value: str | None) -> ZoneInfo | None:
    if not value:
        return None
    candidate = value.strip()
    if not candidate:
        return None
    try:
        return ZoneInfo(candidate)
    except ZoneInfoNotFoundError:
        pass

    alias = _TZ_ALIAS_MAP.get(_normalise_timezone_key(candidate))
    if alias:
        try:
            timezone = ZoneInfo(alias)
            LOGGER.debug("Translated timezone alias '%s' to '%s'.", candidate, alias)
            return timezone
        except ZoneInfoNotFoundError:  # pragma: no cover - defensive guard
            LOGGER.warning("Unable to resolve timezone alias '%s' -> '%s'.", candidate, alias)

    LOGGER.warning("Unknown timezone '%s'; ignoring configured value.", candidate)
    return None


def _tz_name(tz: tzinfo | None) -> str | None:
    if tz is None:
        return None
    for attr in ("key", "zone"):
        name = getattr(tz, attr, None)
        if isinstance(name, str) and name:
            return name
    try:
        now = datetime.now(tz)
    except Exception:  # pragma: no cover - defensive guard
        now = datetime.now()
    name = tz.tzname(now)
    if isinstance(name, str) and name:
        return name
    return None

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


SOURCE_DESCRIPTION_OVERRIDES: dict[str, str] = {
    "alpha_vantage": "Alpha Vantage",
    "csv_loader": "CSV Price Loader",
    "database": "Local database cache",
    "finnhub": "Finnhub",
    "fred": "Federal Reserve Economic Data (FRED)",
    "gdelt": "GDELT Project",
    "local": "Local computation",
    "newsapi": "NewsAPI.org",
    "placeholder": "Placeholder data",
    "polygon": "Polygon.io",
    "parquet_loader": "Parquet Price Loader",
    "quandl": "Quandl",
    "reddit": "Reddit",
    "stooq": "Stooq",
    "tiingo": "Tiingo",
    "twitter": "Twitter",
    "unknown": "Unknown provider",
    "yahoo_finance": "Yahoo Finance",
    "yfinance": "Yahoo Finance (yfinance)",
}


@dataclass(slots=True)
class RefreshResult:
    """Container capturing ETL refresh outcomes."""

    data: pd.DataFrame
    downloaded: bool


class MarketDataETL:
    """Services that orchestrate downloading, normalising and storing data.

    To enable local file-based price loaders configure
    :attr:`PredictorConfig.csv_price_loader_path` or
    :attr:`PredictorConfig.parquet_price_loader_path`.
    """

    _memory_cache: ClassVar[
        dict[tuple[str, str, DatasetType], tuple[float, ProviderFetchSummary]]
    ] = {}
    _memory_cache_default_ttl: ClassVar[float] = 900.0

    def __init__(self, config: PredictorConfig, database: Database | None = None) -> None:
        self.config = config
        self.database = database or Database(config.database_url)
        self._source_log: DefaultDict[str, set[str]] = defaultdict(set)
        self._registry = build_default_registry(
            csv_loader_path=self.config.csv_price_loader_path,
            parquet_loader_path=self.config.parquet_price_loader_path,
            config=self.config,
        )
        self.market_timezone = resolve_market_timezone(self.config)
        self._last_price_cache_warning: str | None = None
        self._live_price_cache: tuple[float | None, pd.Timestamp | None, float] | None = None
        ttl = getattr(self.config, "memory_cache_seconds", None)
        if ttl is None:
            ttl = self._memory_cache_default_ttl
        self._memory_cache_ttl = max(60.0, float(ttl))

    # ------------------------------------------------------------------
    # Public refresh API
    # ------------------------------------------------------------------
    def refresh_prices(
        self,
        force: bool = False,
        *,
        providers: Sequence[str] | None = None,
    ) -> RefreshResult:
        stale_notes: list[str] = []
        price_source: str | None = None
        existing = self.database.get_prices(
            ticker=self.config.ticker,
            interval=self.config.interval,
            start=self.config.start_date,
            end=self.config.end_date,
        )
        existing_ts = self.database.get_refresh_timestamp(
            self.config.ticker, self.config.interval, "prices"
        )
        existing_is_fresh = bool(
            existing_ts and self._is_cache_timestamp_fresh(existing_ts)
        )
        if not existing.empty and not existing_is_fresh:
            if existing_ts is None:
                stale_notes.append("Database price cache is missing a refresh timestamp.")
            else:
                if existing_ts.tzinfo is None:
                    display_ts = existing_ts.replace(tzinfo=timezone.utc)
                else:
                    display_ts = existing_ts
                stale_notes.append(
                    f"Database price cache expired (last refreshed {display_ts.isoformat()})."
                )
            existing = existing.iloc[0:0]
            existing_is_fresh = False
        elif not existing.empty and existing_is_fresh:
            price_source = "database"

        cache_path = self.config.price_cache_path
        cache_payload = None
        if existing.empty:
            cache_exists = cache_path.exists()
            cache_payload = self._load_local_price_cache()
            if cache_payload is not None:
                cached_frame, cache_timestamp = cache_payload
                if not cached_frame.empty:
                    LOGGER.info(
                        "Loaded %s cached price rows for %s from local cache (as of %s).",
                        cached_frame.shape[0],
                        self.config.ticker,
                        cache_timestamp.isoformat(),
                    )
                    existing = cached_frame
                    existing_ts = cache_timestamp
                    existing_is_fresh = True
                    price_source = "local"
                else:
                    stale_notes.append("Local price cache contains no data.")
                    existing_is_fresh = False
            elif cache_exists and self._last_price_cache_warning:
                stale_notes.append(self._last_price_cache_warning)
        if not existing.empty and price_source is None:
            price_source = "database"

        if existing.empty:
            if stale_notes:
                LOGGER.info(
                    "Cached price data for %s is unavailable: %s",
                    self.config.ticker,
                    "; ".join(stale_notes),
                )
            else:
                LOGGER.info(
                    "No cached price data found for %s; attempting to download fresh data.",
                    self.config.ticker,
                )

        if (
            not force
            and existing_is_fresh
            and self._covers_requested_range(existing)
        ):
            LOGGER.debug("Price data already present for %s", self.config.ticker)
            prepared = self._apply_provider_timestamps(existing, None)
            if price_source:
                self._record_source("prices", price_source)
            return RefreshResult(prepared, downloaded=False)

        LOGGER.info(
            "Downloading price data for %s (%s - %s)",
            self.config.ticker,
            self.config.start_date,
            self.config.end_date or "today",
        )
        params: Dict[str, Any] = {"interval": self.config.interval}
        if self.config.start_date:
            params["start"] = self.config.start_date.isoformat()
        if self.config.end_date:
            params["end"] = self.config.end_date.isoformat()

        summary = self._fetch_dataset(DatasetType.PRICES, params, providers=providers)
        if summary.failures:
            self._log_provider_failures(
                "prices", summary.failures, bool(summary.results)
            )

        providers = self._collect_providers(summary.results)
        downloaded = self._coalesce_price_results(summary.results)

        if downloaded is None or downloaded.empty:
            if not existing.empty and existing_is_fresh:
                LOGGER.warning(
                    "Providers returned no new price rows for %s; reusing cached data.",
                    self.config.ticker,
                )
                prepared = self._apply_provider_timestamps(existing, None)
                if price_source:
                    self._record_source("prices", price_source)
                return RefreshResult(prepared, downloaded=False)

            detail = (
                self._compose_failure_summary(summary.failures)
                or "Empty dataframe from data providers."
            )
            if stale_notes:
                detail = f"{detail} Cached data unavailable: {'; '.join(stale_notes)}"
            message = self._compose_error_message(detail)
            raise NoPriceDataError(self.config.ticker, message)
        self.database.upsert_prices(self.config.ticker, self.config.interval, downloaded)
        self.database.set_refresh_timestamp(
            self.config.ticker, self.config.interval, "prices"
        )
        self._write_price_cache(downloaded)
        if providers:
            for provider in providers:
                self._record_source("prices", provider)
        else:
            self._record_source("prices", "unknown")
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
        prepared = self._apply_provider_timestamps(refreshed, downloaded)
        return RefreshResult(prepared, downloaded=True)

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
        enriched = compute_price_features(
            price_frame,
            feature_toggles={
                **self.config.feature_toggles.asdict(),
                **self.config.price_feature_toggles,
            },
            macro_symbols=self.config.macro_merge_symbols,
        )
        indicator_columns_attr = enriched.attrs.get("indicator_columns")
        indicator_columns: list[str]
        if isinstance(indicator_columns_attr, (list, tuple, set)):
            indicator_columns = [str(column) for column in indicator_columns_attr]
        else:
            indicator_columns = []
        fallback = [
            column
            for column in TECHNICAL_INDICATORS
            if column not in indicator_columns and column in enriched.columns
        ]
        indicator_columns.extend(fallback)
        indicator_columns = [
            column for column in dict.fromkeys(indicator_columns) if column in enriched.columns
        ]
        if not indicator_columns:
            LOGGER.warning("No indicator columns available to persist for %s", self.config.ticker)
            self.database.set_indicator_columns(
                self.config.ticker, self.config.interval, indicator_columns
            )
            return 0

        indicator_frame = enriched[["Date", *indicator_columns]].copy()
        indicator_frame = indicator_frame.dropna(subset=["Date"])
        melted = indicator_frame.melt(
            id_vars=["Date"], value_vars=indicator_columns, var_name="Indicator", value_name="Value"
        )
        melted["Value"] = pd.to_numeric(melted["Value"], errors="coerce")
        melted = melted.dropna(subset=["Value"])
        if melted.empty:
            self.database.set_indicator_columns(
                self.config.ticker, self.config.interval, indicator_columns
            )
            return 0

        melted["Category"] = "technical"
        records = melted.to_dict("records")
        inserted = self.database.upsert_indicators(
            ticker=self.config.ticker,
            interval=self.config.interval,
            records=records,
        )
        self.database.set_indicator_columns(
            self.config.ticker, self.config.interval, indicator_columns
        )
        if inserted:
            self.database.set_refresh_timestamp(
                self.config.ticker, self.config.interval, "indicators"
            )
            self._record_source("indicators", "local")
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
            self._record_source("macro", "yfinance")
        return inserted

    def refresh_news(self, force: bool = False) -> RefreshResult:
        existing = self.database.get_news(self.config.ticker)
        if not force and not existing.empty:
            self._record_source("news", "database")
            return RefreshResult(existing, downloaded=False)

        LOGGER.info(
            "Downloading up to %s news articles for %s",
            self.config.news_limit,
            self.config.ticker,
        )

        params = {"query": self.config.ticker, "limit": self.config.news_limit or 50}
        summary = self._fetch_dataset(DatasetType.NEWS, params)
        if summary.failures:
            self._log_provider_failures("news", summary.failures, bool(summary.results))
        providers = self._collect_providers(summary.results)
        frame = self._coalesce_news_results(summary.results)

        if frame is None or frame.empty:
            LOGGER.warning("No news articles returned for %s", self.config.ticker)
            for provider in providers:
                self._record_source("news", provider)
            return RefreshResult(existing, downloaded=bool(providers))

        records = frame.to_dict("records")
        inserted = self.database.upsert_news(records)
        if inserted:
            self.database.set_refresh_timestamp(
                self.config.ticker, "global", "news"
            )
            for provider in providers:
                self._record_source("news", provider)
        refreshed = self.database.get_news(self.config.ticker)
        return RefreshResult(refreshed, downloaded=True)

    def refresh_corporate_events(self, force: bool = False) -> RefreshResult:
        existing = self.database.get_corporate_events(self.config.ticker)
        if not force and not existing.empty:
            self._record_source("corporate_events", "database")
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
            self._record_source("corporate_events", provider)
        refreshed = self.database.get_corporate_events(self.config.ticker)
        return RefreshResult(refreshed, downloaded=downloaded)

    def refresh_options_surface(self, force: bool = False) -> RefreshResult:
        existing = self.database.get_option_surface(self.config.ticker)
        if not force and not existing.empty:
            self._record_source("options_surface", "database")
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
            self._record_source("options_surface", provider)
        refreshed = self.database.get_option_surface(self.config.ticker)
        return RefreshResult(refreshed, downloaded=downloaded)

    def refresh_sentiment_signals(self, force: bool = False) -> RefreshResult:
        def _is_placeholder_only(frame: pd.DataFrame) -> bool:
            def _is_zero_payload(payload: Any) -> bool:
                if payload is None:
                    return True
                if isinstance(payload, dict):
                    if not payload:
                        return True
                    if payload.get("articles") == 0:
                        return True
                    if set(payload.keys()) == {"note"}:
                        return True
                return False

            if frame.empty:
                return False

            for row in frame.to_dict("records"):
                provider = str(row.get("Provider", "")).casefold()
                score = row.get("Score")
                payload = row.get("Payload")
                is_placeholder_provider = provider == "placeholder"
                zero_payload = (score is None or score == 0) and _is_zero_payload(payload)
                if not (is_placeholder_provider or zero_payload):
                    return False
            return True

        existing = self.database.get_sentiment_signals(self.config.ticker)
        if not force and not existing.empty and not _is_placeholder_only(existing):
            self._record_source("sentiment", "database")
            return RefreshResult(existing, downloaded=False)

        params = {"query": self.config.ticker, "limit": 200}
        summary = self._fetch_dataset(DatasetType.SENTIMENT, params)
        if summary.failures:
            self._log_provider_failures(
                "sentiment", summary.failures, bool(summary.results)
            )
        provider_results = summary.results
        providers = self._collect_providers(provider_results)

        sentiment_records: list[dict[str, Any]] = []
        for result in provider_results:
            for record in result.records:
                if isinstance(record, SentimentSignal):
                    payload = record.as_record()
                    if payload.get("Ticker"):
                        sentiment_records.append(payload)

        news = self.database.get_news(self.config.ticker)
        if news.empty and self.config.sentiment:
            news_result = self.refresh_news(force=False)
            news = news_result.data

        if not news.empty:
            working = news.copy()
            published_col = None
            for candidate in ("publishedDate", "PublishedAt"):
                if candidate in working.columns:
                    published_col = candidate
                    break
            if published_col:
                working[published_col] = pd.to_datetime(
                    working[published_col], errors="coerce"
                )
            else:
                working["publishedDate"] = pd.NaT
                published_col = "publishedDate"
            working = working.dropna(subset=[published_col])
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
                scored.groupby(scored[published_col].dt.date)[published_col].count()
                if published_col in scored.columns
                else pd.Series(dtype=int)
            )
            counts = counts_series.to_dict()
            for row in aggregated.itertuples(index=False):
                row_date = getattr(row, "Date", None)
                if pd.isna(row_date):
                    continue
                count = int(counts.get(row_date.date(), 0))
                sentiment_records.append(
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
            if aggregated.shape[0]:
                providers.add("vader")

        if not sentiment_records:
            sentiment_records.append(
                {
                    "Ticker": self.config.ticker,
                    "AsOf": datetime.utcnow(),
                    "Provider": "placeholder",
                    "SignalType": "news_sentiment",
                    "Score": 0.0,
                    "Magnitude": None,
                    "Payload": {
                        "note": "Placeholder sentiment score generated because no sentiment providers returned data."
                    },
                }
            )
            providers.add("placeholder")

        frame = pd.DataFrame(sentiment_records)
        if not frame.empty and "AsOf" in frame.columns:
            frame["AsOf"] = pd.to_datetime(frame["AsOf"], errors="coerce")
            frame = frame.dropna(subset=["AsOf"])
            frame = frame.sort_values("AsOf").drop_duplicates(
                subset=["Provider", "SignalType", "AsOf"], keep="last"
            )

        records = frame.to_dict("records") if not frame.empty else []
        inserted = self.database.upsert_sentiment_signals(records)
        if inserted:
            self.database.set_refresh_timestamp(
                self.config.ticker, "global", "sentiment"
            )
            for provider in providers:
                self._record_source("sentiment", provider)
        refreshed = self.database.get_sentiment_signals(self.config.ticker)
        downloaded = bool(records) and providers != {"placeholder"}
        return RefreshResult(refreshed, downloaded=downloaded)

    def refresh_esg_metrics(self, force: bool = False) -> RefreshResult:
        existing = self.database.get_esg_metrics(self.config.ticker)
        if not force and not existing.empty:
            self._record_source("esg_metrics", "database")
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
            self._record_source("esg_metrics", provider)
        refreshed = self.database.get_esg_metrics(self.config.ticker)
        return RefreshResult(refreshed, downloaded=downloaded)

    def refresh_ownership_flows(self, force: bool = False) -> RefreshResult:
        existing = self.database.get_ownership_flows(self.config.ticker)
        if not force and not existing.empty:
            self._record_source("ownership_flows", "database")
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
            self._record_source("ownership_flows", provider)
        refreshed = self.database.get_ownership_flows(self.config.ticker)
        return RefreshResult(refreshed, downloaded=downloaded)

    def fetch_live_price(
        self, *, force: bool = False
    ) -> tuple[float | None, pd.Timestamp | None]:
        """Return the most recent intraday price and timestamp if available."""

        now = time.time()
        if not force and self._live_price_cache is not None:
            cached_price, cached_timestamp, cached_at = self._live_price_cache
            if cached_price is not None and now - cached_at < self._memory_cache_ttl:
                return cached_price, cached_timestamp

        ticker = yf.Ticker(self.config.ticker)
        market_tz = self.market_timezone or DEFAULT_MARKET_TIMEZONE
        price: float | None = None
        timestamp: pd.Timestamp | None = None

        fast_info = getattr(ticker, "fast_info", None)
        if fast_info:
            for key in ("last_price", "lastPrice"):
                candidate = getattr(fast_info, key, None)
                if candidate is None and isinstance(fast_info, Mapping):
                    candidate = fast_info.get(key)
                if candidate is not None:
                    price = self._safe_float(candidate)
                    break
            if isinstance(fast_info, Mapping):
                raw_time = fast_info.get("last_price_time") or fast_info.get("lastPriceTime")
                if raw_time:
                    timestamp = pd.to_datetime(raw_time, errors="coerce")

        history_price: float | None = None
        history_timestamp: pd.Timestamp | None = None
        history = self._safe_download(
            lambda: ticker.history(period="1d", interval="1m", prepost=True)
        )
        if isinstance(history, pd.DataFrame) and not history.empty:
            history = history.dropna(how="all")
            if not history.empty:
                last_bar = history.iloc[-1]
                history_price = self._safe_float(last_bar.get("Close"))
                ts = history.index[-1]
                history_timestamp = pd.to_datetime(ts, errors="coerce")

        # Prefer the most recent data point if both sources are available.
        def _is_newer(candidate: pd.Timestamp | None, reference: pd.Timestamp | None) -> bool:
            if candidate is None:
                return False
            if reference is None:
                return True
            if pd.isna(candidate) or pd.isna(reference):
                return False
            return candidate > reference

        if _is_newer(history_timestamp, timestamp) or price is None:
            price = history_price if history_price is not None else price
            timestamp = history_timestamp if history_timestamp is not None else timestamp

        if timestamp is not None and not pd.isna(timestamp):
            if timestamp.tzinfo is None:
                timestamp = timestamp.tz_localize(market_tz)
            else:
                timestamp = timestamp.tz_convert(market_tz)
        elif price is not None:
            timestamp = pd.Timestamp.now(tz=market_tz)

        if price is not None or timestamp is not None:
            self._live_price_cache = (price, timestamp, now)
        else:
            # Avoid caching failed lookups so that the next request can retry immediately.
            self._live_price_cache = None

        return price, timestamp

    def refresh_all(self, force: bool = False) -> dict[str, int]:
        prices_result = self.refresh_prices(force=force)
        indicators_count = self.refresh_indicators(prices_result.data, force=force)
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
    def _run_async(self, coroutine_factory: Callable[[], Awaitable[T]]) -> T:
        try:
            return asyncio.run(coroutine_factory())
        except RuntimeError as exc:  # pragma: no cover - defensive fallback
            message = str(exc).lower()
            if "event loop" not in message:
                raise
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(coroutine_factory())
            finally:
                loop.close()

    def _fetch_dataset(
        self,
        dataset: DatasetType,
        params: Mapping[str, Any] | None = None,
        *,
        providers: Sequence[str] | None = None,
    ) -> ProviderFetchSummary:
        request_params = dict(params or {})
        if dataset == DatasetType.PRICES:
            loader_path = self._price_loader_path()
            if loader_path and "path" not in request_params:
                request_params["path"] = loader_path
            cache_path = self.config.price_cache_path
            if cache_path and "local_store_path" not in request_params:
                request_params["local_store_path"] = str(cache_path)
        request = ProviderRequest(
            dataset_type=dataset,
            symbol=self.config.ticker,
            params=request_params,
        )
        provider_sequence = self._resolve_provider_sequence(dataset, providers)

        if dataset == DatasetType.PRICES:
            summary = self._fetch_with_rotation(request, provider_sequence)
        else:

            async def _runner() -> ProviderFetchSummary:
                provider_arg: Sequence[str] | None = provider_sequence or None
                return await self._registry.fetch_all(request, providers=provider_arg)

            summary = self._run_async(_runner)

        return self._apply_memory_cache(dataset, summary)

    def preferred_price_providers(self) -> list[str]:
        """Return the resolved provider order for price downloads."""

        return self._resolve_provider_sequence(DatasetType.PRICES, None)

    def _resolve_provider_sequence(
        self, dataset: DatasetType, providers: Sequence[str] | None
    ) -> list[str]:
        """Determine the provider order honouring config priorities."""

        available = [
            provider.name for provider in self._registry.providers_for(dataset)
        ]
        disabled = set(getattr(self.config, "disabled_providers", ()) or ())

        def _normalise(values: Sequence[str] | None) -> list[str]:
            normalised: list[str] = []
            if not values:
                return normalised
            for raw in values:
                token = str(raw or "").strip().lower()
                if not token or token in normalised:
                    continue
                normalised.append(token)
            return normalised

        if providers:
            sequence = [p for p in _normalise(list(providers)) if p in available]
        elif dataset == DatasetType.PRICES:
            priority = _normalise(getattr(self.config, "price_provider_priority", ()))
            sequence = [name for name in priority if name in available]
            remaining = [name for name in available if name not in sequence]
            if "yahoo_finance" in remaining:
                remaining = [name for name in remaining if name != "yahoo_finance"] + [
                    name for name in remaining if name == "yahoo_finance"
                ]
            sequence.extend(remaining)
        else:
            sequence = list(available)

        filtered = [name for name in sequence if name not in disabled]
        return filtered

    def _fetch_with_rotation(
        self, request: ProviderRequest, providers: Sequence[str]
    ) -> ProviderFetchSummary:
        if not providers:
            return ProviderFetchSummary.empty()

        aggregate_results: list[ProviderResult] = []
        aggregate_failures: list[ProviderFailure] = []
        provider_cycle = list(providers)
        max_cycles = 3

        for cycle in range(max_cycles):
            cycle_failures: list[ProviderFailure] = []
            for provider_name in provider_cycle:

                async def _runner(name: str = provider_name) -> ProviderFetchSummary:
                    return await self._registry.fetch_all(request, providers=[name])

                summary = self._run_async(_runner)
                aggregate_results.extend(summary.results)
                aggregate_failures.extend(summary.failures)
                cycle_failures.extend(summary.failures)
                if summary.results:
                    return ProviderFetchSummary(aggregate_results, aggregate_failures)

            if not cycle_failures:
                break

            if not all(self._is_rate_limited_failure(failure) for failure in cycle_failures):
                break

            retry_delay = self._minimum_retry_after(cycle_failures)
            if retry_delay is None or retry_delay <= 0:
                break

            LOGGER.info(
                "All %s providers for %s are rate limited; retrying in %.0fs (attempt %s/%s).",
                request.dataset_type.value,
                request.symbol,
                retry_delay,
                cycle + 1,
                max_cycles,
            )
            time.sleep(retry_delay)

        return ProviderFetchSummary(aggregate_results, aggregate_failures)

    def _apply_memory_cache(
        self, dataset: DatasetType, summary: ProviderFetchSummary
    ) -> ProviderFetchSummary:
        key = self._memory_cache_key(dataset)
        if summary.results or not summary.failures:
            self._store_memory_cache(key, summary)
            return summary

        cached = self._get_memory_cache(key)
        if not cached:
            return summary

        if not summary.failures:
            return cached

        if not all(self._is_rate_limited_failure(failure) for failure in summary.failures):
            return summary

        LOGGER.info(
            "Using cached %s data for %s because providers are rate limited.",
            dataset.value,
            self.config.ticker,
        )
        return cached

    def _memory_cache_key(self, dataset: DatasetType) -> tuple[str, str, DatasetType]:
        return (self.config.ticker, self.config.interval, dataset)

    def _get_memory_cache(
        self, key: tuple[str, str, DatasetType]
    ) -> ProviderFetchSummary | None:
        entry = self._memory_cache.get(key)
        if not entry:
            return None
        expires_at, cached_summary = entry
        if time.monotonic() > expires_at:
            self._memory_cache.pop(key, None)
            return None
        return self._clone_summary(cached_summary, mark_cached=True)

    def _store_memory_cache(
        self, key: tuple[str, str, DatasetType], summary: ProviderFetchSummary
    ) -> None:
        self._memory_cache[key] = (
            time.monotonic() + self._memory_cache_ttl,
            self._clone_summary(summary, mark_cached=False),
        )

    @staticmethod
    def _minimum_retry_after(
        failures: Sequence[ProviderFailure],
    ) -> float | None:
        delays = [
            float(failure.retry_after)
            for failure in failures
            if failure.retry_after and failure.retry_after > 0
        ]
        if not delays:
            return None
        return max(0.0, min(delays))

    @staticmethod
    def _is_rate_limited_failure(failure: ProviderFailure) -> bool:
        if failure.is_rate_limited:
            return True
        return failure.status_code == 429

    @staticmethod
    def _clone_summary(
        summary: ProviderFetchSummary, *, mark_cached: bool
    ) -> ProviderFetchSummary:
        cloned_results: list[ProviderResult] = []
        for result in summary.results:
            copy = result.model_copy(deep=True)
            if mark_cached:
                copy.from_cache = True
            cloned_results.append(copy)
        return ProviderFetchSummary(cloned_results, list(summary.failures))

    @staticmethod
    def _collect_providers(results: Sequence[ProviderResult]) -> set[str]:
        return {result.source for result in results if result.source}

    def _price_loader_path(self) -> str | None:
        """Return the configured path for local price loaders, if any."""

        for path in (
            self.config.csv_price_loader_path,
            self.config.parquet_price_loader_path,
        ):
            if path:
                return str(path)
        return None

    def _log_provider_failures(
        self,
        dataset: str,
        failures: Sequence[ProviderFailure],
        had_success: bool = False,
    ) -> None:
        if not failures:
            return
        log = LOGGER.info if had_success else LOGGER.warning
        log(
            "Partial %s refresh for %s: %s",
            dataset,
            self.config.ticker,
            self._compose_failure_summary(failures),
        )

    @staticmethod
    def _compose_failure_summary(failures: Sequence[ProviderFailure]) -> str:
        parts: list[str] = []
        for failure in failures:
            detail = failure.error
            if failure.status_code:
                detail = f"{detail} (status={failure.status_code})"
            if failure.retry_after:
                detail = f"{detail}; retry_in={int(failure.retry_after)}s"
            parts.append(f"{failure.provider}: {detail}")
        return "; ".join(parts)

    def _is_cache_timestamp_fresh(
        self, timestamp: datetime | pd.Timestamp | None
    ) -> bool:
        if timestamp is None:
            return False
        if isinstance(timestamp, pd.Timestamp):
            timestamp = timestamp.to_pydatetime()
        reference = datetime.now(timezone.utc)
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        else:
            timestamp = timestamp.astimezone(timezone.utc)
        return reference - timestamp <= CACHE_MAX_AGE

    def _load_local_price_cache(self) -> tuple[pd.DataFrame, datetime] | None:
        path = self.config.price_cache_path
        self._last_price_cache_warning = None
        if not path.exists():
            return None
        try:
            frame = pd.read_csv(path)
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.debug("Failed to load local price cache %s: %s", path, exc)
            self._last_price_cache_warning = "Local price cache could not be read."
            return None
        timestamp: datetime | None = None
        if "CacheTimestamp" in frame.columns:
            ts_series = pd.to_datetime(frame["CacheTimestamp"], errors="coerce").dropna()
            if not ts_series.empty:
                timestamp = ts_series.iloc[0].to_pydatetime()
            frame = frame.drop(columns=["CacheTimestamp"])
        if "Date" in frame.columns:
            frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
        if timestamp is None:
            try:
                stat = path.stat()
            except OSError as exc:  # pragma: no cover - defensive
                LOGGER.debug("Unable to stat price cache %s: %s", path, exc)
            else:
                timestamp = datetime.fromtimestamp(stat.st_mtime, timezone.utc)
        if timestamp is None:
            self._last_price_cache_warning = "Local price cache is missing timestamp metadata."
            return None
        if not self._is_cache_timestamp_fresh(timestamp):
            self._last_price_cache_warning = (
                f"Local price cache expired (last updated {timestamp.isoformat()})."
            )
            LOGGER.info(
                "Ignoring local price cache %s with timestamp %s older than %s.",
                path,
                timestamp.isoformat(),
                CACHE_MAX_AGE,
            )
            return None
        frame.attrs["cache_timestamp"] = timestamp
        return frame, timestamp

    def _write_price_cache(self, frame: pd.DataFrame) -> None:
        path = self.config.price_cache_path
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now(timezone.utc)
            payload = frame.copy()
            payload["CacheTimestamp"] = timestamp.isoformat()
            payload.to_csv(path, index=False)
            frame.attrs["cache_timestamp"] = timestamp
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.debug("Unable to persist price cache to %s: %s", path, exc)

    @staticmethod
    def _coalesce_price_results(
        results: Sequence[ProviderResult],
    ) -> pd.DataFrame | None:
        bars: list[dict[str, Any]] = []
        for result in results:
            for record in result.records:
                if isinstance(record, PriceBar):
                    bars.append(record.as_frame_row())
        if not bars:
            return None
        frame = pd.DataFrame(bars)
        if frame.empty:
            return None
        frame = frame.dropna(how="all")
        if frame.empty:
            return None
        frame = frame.sort_values("Date").drop_duplicates(subset=["Date"], keep="last")
        return frame.reset_index(drop=True)

    @staticmethod
    def _coalesce_news_results(
        results: Sequence[ProviderResult],
    ) -> pd.DataFrame | None:
        articles: list[dict[str, Any]] = []
        for result in results:
            for record in result.records:
                if isinstance(record, NewsArticle):
                    articles.append(record.as_record())
        if not articles:
            return None
        frame = pd.DataFrame(articles)
        if frame.empty:
            return None
        if "PublishedAt" in frame.columns:
            frame["PublishedAt"] = pd.to_datetime(frame["PublishedAt"], errors="coerce")
            frame = frame.dropna(subset=["PublishedAt"])
        frame = frame.sort_values("PublishedAt").drop_duplicates(
            subset=["Url", "Title"], keep="last"
        )
        return frame.reset_index(drop=True)


    def list_sources(self) -> list[dict[str, object]]:
        """Return structured metadata describing the data sources consulted."""

        provider_datasets: dict[str, set[str]] = {}
        for dataset, dataset_providers in self._source_log.items():
            for provider in dataset_providers:
                provider_id = provider.strip()
                if not provider_id:
                    continue
                provider_datasets.setdefault(provider_id, set()).add(dataset)

        entries: list[dict[str, object]] = []
        for provider_id in sorted(provider_datasets):
            datasets = provider_datasets[provider_id]
            entry: dict[str, object] = {
                "id": provider_id,
                "description": self._describe_source(provider_id),
            }
            if datasets:
                entry["datasets"] = sorted(datasets)
            entries.append(entry)
        return entries

    def _record_source(self, dataset: str, provider: str) -> None:
        if not provider:
            return
        self._source_log[dataset].add(provider)

    def _describe_source(self, provider_id: str) -> str:
        override = SOURCE_DESCRIPTION_OVERRIDES.get(provider_id)
        if override:
            return override
        try:
            provider = self._registry.get(provider_id)
        except ProviderConfigurationError:
            provider = None
        if provider is not None:
            class_name = provider.__class__.__name__
            if class_name.endswith("Provider"):
                class_name = class_name[:-8]
            return self._humanize_label(class_name)
        return self._humanize_label(provider_id)

    @staticmethod
    def _humanize_label(raw: str) -> str:
        cleaned = (raw or "").replace("_", " ").strip()
        if not cleaned:
            return "Unknown source"
        parts = [part for part in cleaned.split(" ") if part]
        if not parts:
            return "Unknown source"
        return " ".join(part.capitalize() for part in parts)

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

    def _normalize_to_market_tz(self, timestamp: datetime | pd.Timestamp) -> pd.Timestamp:
        ts = pd.Timestamp(timestamp)
        if ts.tzinfo is None:
            try:
                ts = ts.tz_localize("UTC")
            except Exception:
                ts = ts.tz_localize(self.market_timezone)
        else:
            ts = ts.tz_convert("UTC")
        return ts.tz_convert(self.market_timezone)

    def _is_trading_day(self, timestamp: datetime | pd.Timestamp) -> bool:
        ts = self._normalize_to_market_tz(pd.Timestamp(timestamp)).normalize()
        naive = ts.tz_localize(None)
        return US_MARKET_BUSINESS_DAY.is_on_offset(naive)

    def _previous_trading_day(self, timestamp: datetime | pd.Timestamp) -> pd.Timestamp:
        candidate = self._normalize_to_market_tz(pd.Timestamp(timestamp)).normalize()
        for _ in range(10):
            candidate = (candidate - US_MARKET_BUSINESS_DAY).normalize()
            if self._is_trading_day(candidate):
                return candidate
        return candidate

    def _latest_trading_session(
        self, reference: datetime | pd.Timestamp | None = None
    ) -> date:
        basis = pd.Timestamp.utcnow() if reference is None else reference
        current = self._normalize_to_market_tz(basis)
        session = current.normalize()
        if not self._is_trading_day(session):
            session = self._previous_trading_day(session)
        else:
            if (current.hour, current.minute, current.second) < (
                US_MARKET_CLOSE.hour,
                US_MARKET_CLOSE.minute,
                US_MARKET_CLOSE.second,
            ):
                session = self._previous_trading_day(session)
        return session.tz_localize(None).date()

    def _apply_provider_timestamps(
        self, persisted: pd.DataFrame, downloaded: pd.DataFrame | None
    ) -> pd.DataFrame:
        if persisted.empty or "Date" not in persisted.columns:
            return persisted
        frame = persisted.copy()
        try:
            persisted_dates = pd.to_datetime(frame["Date"], errors="coerce")
        except Exception:  # pragma: no cover - defensive
            return frame
        if persisted_dates.isna().all():
            return frame
        if getattr(persisted_dates.dt, "tz", None) is None:
            localized = persisted_dates.dt.tz_localize(self.market_timezone)
        else:
            localized = persisted_dates.dt.tz_convert(self.market_timezone)
        frame["Date"] = localized
        if downloaded is None or downloaded.empty or "Date" not in downloaded.columns:
            return frame
        try:
            provider_dates = pd.to_datetime(downloaded["Date"], errors="coerce").dropna()
        except Exception:  # pragma: no cover - defensive
            return frame
        if provider_dates.empty:
            return frame
        if getattr(provider_dates.dt, "tz", None) is None:
            provider_dates = provider_dates.dt.tz_localize(self.market_timezone)
        else:
            provider_dates = provider_dates.dt.tz_convert(self.market_timezone)
        provider_map: dict[date, pd.Timestamp] = {}
        for ts in provider_dates:
            provider_map[ts.date()] = ts
        if not provider_map:
            return frame
        for trade_date, ts in provider_map.items():
            mask = frame["Date"].dt.date == trade_date
            if mask.any():
                frame.loc[mask, "Date"] = ts
        return frame

    def _covers_requested_range(self, frame: pd.DataFrame) -> bool:
        if frame.empty:
            return False
        start = self.config.start_date
        end = self.config.end_date
        try:
            date_values = pd.to_datetime(frame["Date"], errors="coerce").dropna()
        except Exception:  # pragma: no cover - defensive
            return False
        if date_values.empty:
            return False
        if getattr(date_values.dt, "tz", None) is None:
            market_dates = date_values.dt.tz_localize(self.market_timezone)
        else:
            market_dates = date_values.dt.tz_convert(self.market_timezone)
        min_date = market_dates.min().date()
        max_date = market_dates.max().date()
        if start and min_date > start:
            return False
        if end and max_date < end:
            return False
        if not end:
            latest_session = self._latest_trading_session()
            if max_date < latest_session:
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


__all__ = [
    "MarketDataETL",
    "RefreshResult",
    "NoPriceDataError",
    "resolve_market_timezone",
]

