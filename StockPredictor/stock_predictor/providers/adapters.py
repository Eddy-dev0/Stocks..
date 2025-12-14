"""Provider adapters for third-party data sources."""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timedelta, timezone
from io import StringIO
from pathlib import Path
from typing import Any, List

import pandas as pd

from .base import (
    BaseProvider,
    DatasetType,
    EconomicIndicator,
    NewsArticle,
    PriceBar,
    ProviderRequest,
    ProviderResult,
    SentimentSignal,
)

LOGGER = logging.getLogger(__name__)


def _parse_float(value: Any) -> float | None:
    try:
        if value is None or value == "None":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


class YahooFinanceProvider(BaseProvider):
    """Adapter for Yahoo Finance price data."""

    name = "yahoo_finance"
    supported_datasets = (DatasetType.PRICES,)

    def __init__(
        self,
        *,
        rate_limit_per_sec: float | None = None,
        cooldown_seconds: float | None = None,
        **base_kwargs: Any,
    ) -> None:
        settings: dict[str, Any] = dict(base_kwargs)
        if rate_limit_per_sec is not None:
            settings["rate_limit_per_sec"] = rate_limit_per_sec
        if cooldown_seconds is not None:
            settings["cooldown_seconds"] = cooldown_seconds
        super().__init__(**settings)

    async def _fetch(self, request: ProviderRequest) -> ProviderResult:
        params = {
            "range": request.params.get("range", "1y"),
            "interval": request.params.get("interval", "1d"),
        }
        start = request.params.get("start")
        end = request.params.get("end")
        if start and not params.get("period1"):
            params["period1"] = int(pd.Timestamp(start).timestamp())
        if end and not params.get("period2"):
            params["period2"] = int(pd.Timestamp(end).timestamp())
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{request.symbol}"
        response = await self.client.get(url, params=params)
        response.raise_for_status()
        payload = response.json()
        result_payload = payload.get("chart", {}).get("result", [])
        if not result_payload:
            return ProviderResult(
                dataset_type=request.dataset_type,
                source=self.name,
                records=[],
                metadata={"symbol": request.symbol, "reason": "empty"},
            )
        chart = result_payload[0]
        timestamps: List[int] = chart.get("timestamp", [])
        indicators = chart.get("indicators", {})
        quote = (indicators.get("quote") or [{}])[0]
        adj_close = (indicators.get("adjclose") or [{}])[0]
        opens = quote.get("open") or []
        highs = quote.get("high") or []
        lows = quote.get("low") or []
        closes = quote.get("close") or []
        volumes = quote.get("volume") or []
        adj_closes = adj_close.get("adjclose") or []
        bars: list[PriceBar] = []
        for idx, ts in enumerate(timestamps):
            try:
                timestamp = datetime.fromtimestamp(ts, tz=timezone.utc)
            except (TypeError, ValueError):
                continue
            bars.append(
                PriceBar(
                    symbol=request.symbol,
                    timestamp=timestamp,
                    open=_parse_float(opens[idx] if idx < len(opens) else None),
                    high=_parse_float(highs[idx] if idx < len(highs) else None),
                    low=_parse_float(lows[idx] if idx < len(lows) else None),
                    close=_parse_float(closes[idx] if idx < len(closes) else None),
                    volume=_parse_float(volumes[idx] if idx < len(volumes) else None),
                    adj_close=_parse_float(adj_closes[idx] if idx < len(adj_closes) else None),
                )
            )
        return ProviderResult(
            dataset_type=request.dataset_type,
            source=self.name,
            records=bars,
            metadata={"symbol": request.symbol, "interval": params.get("interval")},
        )


class AlphaVantageProvider(BaseProvider):
    """Adapter for Alpha Vantage time-series API."""

    name = "alpha_vantage"
    supported_datasets = (DatasetType.PRICES, DatasetType.MACRO)

    def __init__(self) -> None:
        super().__init__(rate_limit_per_sec=5 / 60, cache_ttl=600)
        self._api_key = self.require_env("ALPHAVANTAGE_API_KEY")

    async def _fetch(self, request: ProviderRequest) -> ProviderResult:
        function = request.params.get("function")
        if request.dataset_type == DatasetType.PRICES and not function:
            function = "TIME_SERIES_DAILY_ADJUSTED"
        params = {"symbol": request.symbol, "apikey": self._api_key}
        if function:
            params["function"] = function
        params.update({k: v for k, v in request.params.items() if k not in {"function"}})
        response = await self.client.get("https://www.alphavantage.co/query", params=params)
        response.raise_for_status()
        payload = response.json()
        if request.dataset_type == DatasetType.PRICES:
            key = next((k for k in payload.keys() if "Time Series" in k), None)
            records: list[PriceBar] = []
            if key:
                series = payload.get(key, {})
                for date_str, row in series.items():
                    timestamp = pd.Timestamp(date_str).to_pydatetime().replace(tzinfo=timezone.utc)
                    records.append(
                        PriceBar(
                            symbol=request.symbol,
                            timestamp=timestamp,
                            open=_parse_float(row.get("1. open")),
                            high=_parse_float(row.get("2. high")),
                            low=_parse_float(row.get("3. low")),
                            close=_parse_float(row.get("4. close")),
                            adj_close=_parse_float(row.get("5. adjusted close")),
                            volume=_parse_float(row.get("6. volume")),
                        )
                    )
            return ProviderResult(
                dataset_type=request.dataset_type,
                source=self.name,
                records=records,
                metadata={"function": function or "TIME_SERIES_DAILY_ADJUSTED"},
            )
        # Macro/economic indicators
        data_key = payload.get("data", payload.get("observations"))
        records_macro: list[EconomicIndicator] = []
        if isinstance(data_key, list):
            for row in data_key:
                date_str = row.get("date") or row.get("timestamp")
                value = _parse_float(row.get("value"))
                if not date_str:
                    continue
                timestamp = pd.Timestamp(date_str).to_pydatetime().replace(tzinfo=timezone.utc)
                records_macro.append(
                    EconomicIndicator(
                        symbol=request.symbol,
                        name=function or request.params.get("function", "macro"),
                        value=value,
                        as_of=timestamp,
                        extra={"raw": row},
                    )
                )
        return ProviderResult(
            dataset_type=request.dataset_type,
            source=self.name,
            records=records_macro,
            metadata={"function": function},
        )


class FinnhubProvider(BaseProvider):
    """Adapter for Finnhub stock candle API."""

    name = "finnhub"
    supported_datasets = (DatasetType.PRICES, DatasetType.SENTIMENT)

    def __init__(self) -> None:
        super().__init__(rate_limit_per_sec=60 / 60, cache_ttl=300)
        self._api_key = self.require_env("FINNHUB_API_KEY")

    async def _fetch(self, request: ProviderRequest) -> ProviderResult:
        if request.dataset_type == DatasetType.PRICES:
            params = {
                "symbol": request.symbol,
                "resolution": request.params.get("interval", "D"),
                "from": int(pd.Timestamp(request.params.get("start", datetime.now() - timedelta(days=365))).timestamp()),
                "to": int(pd.Timestamp(request.params.get("end", datetime.now())).timestamp()),
                "token": self._api_key,
            }
            response = await self.client.get("https://finnhub.io/api/v1/stock/candle", params=params)
            response.raise_for_status()
            payload = response.json()
            timestamps: list[int] = payload.get("t", [])
            opens = payload.get("o", [])
            highs = payload.get("h", [])
            lows = payload.get("l", [])
            closes = payload.get("c", [])
            volumes = payload.get("v", [])
            bars: list[PriceBar] = []
            for idx, ts in enumerate(timestamps):
                timestamp = datetime.fromtimestamp(ts, tz=timezone.utc)
                bars.append(
                    PriceBar(
                        symbol=request.symbol,
                        timestamp=timestamp,
                        open=_parse_float(opens[idx] if idx < len(opens) else None),
                        high=_parse_float(highs[idx] if idx < len(highs) else None),
                        low=_parse_float(lows[idx] if idx < len(lows) else None),
                        close=_parse_float(closes[idx] if idx < len(closes) else None),
                        volume=_parse_float(volumes[idx] if idx < len(volumes) else None),
                    )
                )
            return ProviderResult(
                dataset_type=request.dataset_type,
                source=self.name,
                records=bars,
                metadata={"resolution": params["resolution"]},
            )
        params = {
            "symbol": request.symbol,
            "token": self._api_key,
        }
        response = await self.client.get("https://finnhub.io/api/v1/news-sentiment", params=params)
        response.raise_for_status()
        payload = response.json()
        signals: list[SentimentSignal] = []
        for key, value in payload.items():
            if not isinstance(value, (int, float)):
                continue
            signals.append(
                SentimentSignal(
                    symbol=request.symbol,
                    provider=self.name,
                    signal_type=key,
                    as_of=datetime.now(timezone.utc),
                    score=float(value),
                )
            )
        return ProviderResult(
            dataset_type=request.dataset_type,
            source=self.name,
            records=signals,
        )


class PolygonProvider(BaseProvider):
    """Adapter for Polygon.io aggregated bars."""

    name = "polygon"
    supported_datasets = (DatasetType.PRICES, DatasetType.NEWS)

    def __init__(self) -> None:
        super().__init__(rate_limit_per_sec=5, cache_ttl=300)
        self._api_key = self.require_env("POLYGON_API_KEY")

    async def _fetch(self, request: ProviderRequest) -> ProviderResult:
        if request.dataset_type == DatasetType.PRICES:
            multiplier = request.params.get("multiplier", 1)
            timespan = request.params.get("timespan", "day")
            start = request.params.get("start")
            end = request.params.get("end") or datetime.now().date().isoformat()
            start = start or (datetime.now().date() - timedelta(days=365)).isoformat()
            url = f"https://api.polygon.io/v2/aggs/ticker/{request.symbol}/range/{multiplier}/{timespan}/{start}/{end}"
            params = {"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": self._api_key}
            response = await self.client.get(url, params=params)
            response.raise_for_status()
            payload = response.json()
            results = payload.get("results", [])
            bars: list[PriceBar] = []
            for row in results:
                timestamp = datetime.fromtimestamp(row.get("t", 0) / 1000, tz=timezone.utc)
                bars.append(
                    PriceBar(
                        symbol=request.symbol,
                        timestamp=timestamp,
                        open=_parse_float(row.get("o")),
                        high=_parse_float(row.get("h")),
                        low=_parse_float(row.get("l")),
                        close=_parse_float(row.get("c")),
                        volume=_parse_float(row.get("v")),
                    )
                )
            return ProviderResult(
                dataset_type=request.dataset_type,
                source=self.name,
                records=bars,
                metadata={"multiplier": multiplier, "timespan": timespan},
            )
        url = "https://api.polygon.io/v2/reference/news"
        params = {
            "ticker": request.symbol,
            "limit": request.params.get("limit", 50),
            "apiKey": self._api_key,
        }
        response = await self.client.get(url, params=params)
        response.raise_for_status()
        payload = response.json()
        articles = payload.get("results", [])
        records = []
        for article in articles:
            published = article.get("published_utc")
            if not published:
                continue
            records.append(
                NewsArticle(
                    symbol=request.symbol,
                    headline=article.get("title", ""),
                    summary=article.get("description"),
                    url=article.get("article_url"),
                    published_at=pd.Timestamp(published).to_pydatetime().replace(tzinfo=timezone.utc),
                    source=article.get("source"),
                )
            )
        return ProviderResult(
            dataset_type=request.dataset_type,
            source=self.name,
            records=records,
        )


class TiingoProvider(BaseProvider):
    """Adapter for Tiingo daily price API."""

    name = "tiingo"
    supported_datasets = (DatasetType.PRICES, DatasetType.NEWS)

    def __init__(self) -> None:
        super().__init__(rate_limit_per_sec=50 / 60, cache_ttl=600)
        self._api_key = self.require_env("TIINGO_API_KEY")

    async def _fetch(self, request: ProviderRequest) -> ProviderResult:
        headers = {"Content-Type": "application/json", "Authorization": f"Token {self._api_key}"}
        if request.dataset_type == DatasetType.PRICES:
            params = {
                "startDate": request.params.get("start"),
                "endDate": request.params.get("end"),
                "resampleFreq": request.params.get("interval", "daily"),
            }
            url = f"https://api.tiingo.com/tiingo/daily/{request.symbol}/prices"
            response = await self.client.get(url, params=params, headers=headers)
            response.raise_for_status()
            payload = response.json()
            bars = []
            for row in payload:
                timestamp = pd.Timestamp(row.get("date")).to_pydatetime().replace(tzinfo=timezone.utc)
                bars.append(
                    PriceBar(
                        symbol=request.symbol,
                        timestamp=timestamp,
                        open=_parse_float(row.get("open")),
                        high=_parse_float(row.get("high")),
                        low=_parse_float(row.get("low")),
                        close=_parse_float(row.get("close")),
                        adj_close=_parse_float(row.get("adjClose")),
                        volume=_parse_float(row.get("volume")),
                    )
                )
            return ProviderResult(dataset_type=request.dataset_type, source=self.name, records=bars)
        url = "https://api.tiingo.com/tiingo/news"
        params = {"tickers": request.symbol, "limit": request.params.get("limit", 50)}
        response = await self.client.get(url, params=params, headers=headers)
        response.raise_for_status()
        payload = response.json()
        records = []
        for row in payload:
            published = row.get("publishedDate")
            if not published:
                continue
            records.append(
                NewsArticle(
                    symbol=request.symbol,
                    headline=row.get("title", ""),
                    summary=row.get("description"),
                    url=row.get("url"),
                    published_at=pd.Timestamp(published).to_pydatetime().replace(tzinfo=timezone.utc),
                    source=row.get("source"),
                )
            )
        return ProviderResult(dataset_type=request.dataset_type, source=self.name, records=records)


class StooqProvider(BaseProvider):
    """Adapter for Stooq CSV downloads."""

    name = "stooq"
    supported_datasets = (DatasetType.PRICES,)

    async def _fetch(self, request: ProviderRequest) -> ProviderResult:
        raw_symbol = request.symbol.strip()
        normalized_symbol = raw_symbol.lower()
        if "." not in raw_symbol:
            normalized_symbol = f"{normalized_symbol}.us"

        url = f"https://stooq.com/q/d/l/?s={normalized_symbol}&i=d"
        response = await self.client.get(url)
        response.raise_for_status()
        frame = pd.read_csv(StringIO(response.text)) if response.text else pd.DataFrame()
        if not frame.empty:
            frame = frame.dropna(how="any")
        bars: list[PriceBar] = []
        for _, row in frame.iterrows():
            timestamp = pd.Timestamp(row["Date"]).to_pydatetime().replace(tzinfo=timezone.utc)
            bars.append(
                PriceBar(
                    symbol=request.symbol,
                    timestamp=timestamp,
                    open=_parse_float(row.get("Open")),
                    high=_parse_float(row.get("High")),
                    low=_parse_float(row.get("Low")),
                    close=_parse_float(row.get("Close")),
                    volume=_parse_float(row.get("Volume")),
                )
            )
        return ProviderResult(dataset_type=request.dataset_type, source=self.name, records=bars)


class FREDProvider(BaseProvider):
    """Adapter for Federal Reserve Economic Data (FRED)."""

    name = "fred"
    supported_datasets = (DatasetType.MACRO,)

    def __init__(self) -> None:
        super().__init__(rate_limit_per_sec=120 / 60, cache_ttl=1800)
        self._api_key = self.require_env("FRED_API_KEY")

    async def _fetch(self, request: ProviderRequest) -> ProviderResult:
        series_id = request.params.get("series_id") or request.symbol
        params = {
            "series_id": series_id,
            "api_key": self._api_key,
            "file_type": "json",
        }
        if request.params.get("start"):
            params["observation_start"] = request.params["start"]
        if request.params.get("end"):
            params["observation_end"] = request.params["end"]
        response = await self.client.get(
            "https://api.stlouisfed.org/fred/series/observations", params=params
        )
        response.raise_for_status()
        payload = response.json()
        observations = payload.get("observations", [])
        records: list[EconomicIndicator] = []
        series_meta = None
        if isinstance(payload.get("seriess"), list) and payload["seriess"]:
            series_meta = payload["seriess"][0]
        series_name = (
            (series_meta or {}).get("title")
            or request.params.get("series_name")
            or series_id
        )
        for row in observations:
            date_str = row.get("date")
            if not date_str:
                continue
            timestamp = pd.Timestamp(date_str).to_pydatetime().replace(tzinfo=timezone.utc)
            records.append(
                EconomicIndicator(
                    symbol=series_id,
                    name=series_name,
                    value=_parse_float(row.get("value")),
                    as_of=timestamp,
                    extra={"realtime_start": row.get("realtime_start"), "realtime_end": row.get("realtime_end")},
                )
            )
        return ProviderResult(dataset_type=request.dataset_type, source=self.name, records=records)


class NewsAPIProvider(BaseProvider):
    """Adapter for NewsAPI headlines."""

    name = "newsapi"
    supported_datasets = (DatasetType.NEWS,)

    def __init__(self) -> None:
        super().__init__(rate_limit_per_sec=30 / 60, cache_ttl=900)
        self._api_key = self.require_env("NEWSAPI_API_KEY")

    async def _fetch(self, request: ProviderRequest) -> ProviderResult:
        params = {
            "q": request.params.get("query", request.symbol),
            "language": request.params.get("language", "en"),
            "pageSize": request.params.get("limit", 50),
            "sortBy": request.params.get("sortBy", "publishedAt"),
        }
        headers = {"Authorization": self._api_key}
        response = await self.client.get("https://newsapi.org/v2/everything", params=params, headers=headers)
        response.raise_for_status()
        payload = response.json()
        articles = payload.get("articles", [])
        records: list[NewsArticle] = []
        for article in articles:
            published_at = article.get("publishedAt")
            if not published_at:
                continue
            records.append(
                NewsArticle(
                    symbol=request.symbol,
                    headline=article.get("title", ""),
                    summary=article.get("description"),
                    url=article.get("url"),
                    published_at=pd.Timestamp(published_at).to_pydatetime().replace(tzinfo=timezone.utc),
                    source=article.get("source", {}).get("name"),
                )
            )
        return ProviderResult(dataset_type=request.dataset_type, source=self.name, records=records)


class GDELTProvider(BaseProvider):
    """Adapter for the GDELT events API."""

    name = "gdelt"
    supported_datasets = (DatasetType.NEWS, DatasetType.SENTIMENT)

    async def _fetch(self, request: ProviderRequest) -> ProviderResult:
        params = {
            "format": "json",
            "query": request.params.get("query", request.symbol),
            "maxrecords": request.params.get("limit", 250),
        }
        response = await self.client.get("https://api.gdeltproject.org/api/v2/doc/doc", params=params)
        response.raise_for_status()
        payload = response.json()
        docs = payload.get("articles", []) or payload.get("documents", [])
        records_news: list[NewsArticle] = []
        for row in docs:
            published = row.get("seendate") or row.get("publishedDate")
            if not published:
                continue
            timestamp = pd.Timestamp(published).to_pydatetime().replace(tzinfo=timezone.utc)
            records_news.append(
                NewsArticle(
                    symbol=request.symbol,
                    headline=row.get("title", ""),
                    summary=row.get("summary") or row.get("excerpt"),
                    url=row.get("url") or row.get("sourceUrl"),
                    published_at=timestamp,
                    source=row.get("source"),
                )
            )
        records_sentiment: list[SentimentSignal] = []
        for row in docs:
            if "tone" in row:
                try:
                    score = float(row["tone"])
                except (TypeError, ValueError):
                    continue
                records_sentiment.append(
                    SentimentSignal(
                        symbol=request.symbol,
                        provider=self.name,
                        signal_type="tone",
                        as_of=datetime.now(timezone.utc),
                        score=score,
                        payload={"source": row.get("source")},
                    )
                )
        if request.dataset_type == DatasetType.SENTIMENT:
            return ProviderResult(dataset_type=request.dataset_type, source=self.name, records=records_sentiment)
        return ProviderResult(dataset_type=request.dataset_type, source=self.name, records=records_news)


class RedditProvider(BaseProvider):
    """Adapter for Reddit search API."""

    name = "reddit"
    supported_datasets = (DatasetType.SENTIMENT, DatasetType.NEWS)

    async def _fetch(self, request: ProviderRequest) -> ProviderResult:
        params = {"q": request.params.get("query", request.symbol), "limit": request.params.get("limit", 100)}
        headers = {"User-Agent": "stock-predictor-bot/1.0"}
        response = await self.client.get("https://www.reddit.com/search.json", params=params, headers=headers)
        response.raise_for_status()
        payload = response.json()
        posts = ((payload.get("data") or {}).get("children") or [])
        records_news: list[NewsArticle] = []
        records_sentiment: list[SentimentSignal] = []
        for post in posts:
            data = post.get("data", {})
            created = data.get("created_utc")
            if created is None:
                continue
            timestamp = datetime.fromtimestamp(created, tz=timezone.utc)
            title = data.get("title", "")
            url = data.get("url")
            records_news.append(
                NewsArticle(
                    symbol=request.symbol,
                    headline=title,
                    summary=data.get("selftext"),
                    url=url,
                    published_at=timestamp,
                    source="reddit",
                )
            )
            if request.dataset_type == DatasetType.SENTIMENT:
                upvotes = data.get("ups")
                ratio = data.get("upvote_ratio")
                records_sentiment.append(
                    SentimentSignal(
                        symbol=request.symbol,
                        provider=self.name,
                        signal_type="engagement",
                        as_of=timestamp,
                        score=_parse_float(upvotes),
                        magnitude=_parse_float(ratio),
                        payload={"subreddit": data.get("subreddit")},
                    )
                )
        if request.dataset_type == DatasetType.SENTIMENT:
            return ProviderResult(dataset_type=request.dataset_type, source=self.name, records=records_sentiment)
        return ProviderResult(dataset_type=request.dataset_type, source=self.name, records=records_news)


class TwitterProvider(BaseProvider):
    """Adapter for Twitter recent search API."""

    name = "twitter"
    supported_datasets = (DatasetType.SENTIMENT,)

    def __init__(self) -> None:
        super().__init__(rate_limit_per_sec=450 / (15 * 60), cache_ttl=300)
        self._bearer_token = self.require_env("TWITTER_BEARER_TOKEN")

    async def _fetch(self, request: ProviderRequest) -> ProviderResult:
        query = request.params.get("query", request.symbol)
        params = {
            "query": query,
            "max_results": min(int(request.params.get("limit", 50)), 100),
            "tweet.fields": "created_at,public_metrics,lang",  # noqa: E231
        }
        headers = {"Authorization": f"Bearer {self._bearer_token}"}
        response = await self.client.get("https://api.twitter.com/2/tweets/search/recent", params=params, headers=headers)
        response.raise_for_status()
        payload = response.json()
        tweets = payload.get("data", [])
        records: list[SentimentSignal] = []
        for tweet in tweets:
            created = tweet.get("created_at")
            if not created:
                continue
            metrics = tweet.get("public_metrics", {})
            records.append(
                SentimentSignal(
                    symbol=request.symbol,
                    provider=self.name,
                    signal_type="tweet",
                    as_of=pd.Timestamp(created).to_pydatetime().replace(tzinfo=timezone.utc),
                    score=_parse_float(metrics.get("like_count")),
                    magnitude=_parse_float(metrics.get("retweet_count")),
                    payload={"id": tweet.get("id"), "text": tweet.get("text"), "lang": tweet.get("lang")},
                )
            )
        return ProviderResult(dataset_type=request.dataset_type, source=self.name, records=records)


class QuandlProvider(BaseProvider):
    """Adapter for Quandl datasets."""

    name = "quandl"
    supported_datasets = (DatasetType.PRICES, DatasetType.MACRO, DatasetType.ALTERNATIVE)

    def __init__(self) -> None:
        super().__init__(rate_limit_per_sec=20 / 60, cache_ttl=900)
        self._api_key = self.require_env("QUANDL_API_KEY")

    async def _fetch(self, request: ProviderRequest) -> ProviderResult:
        dataset = request.params.get("dataset") or request.symbol
        params = {
            "api_key": self._api_key,
        }
        if request.params.get("start"):
            params["start_date"] = request.params["start"]
        if request.params.get("end"):
            params["end_date"] = request.params["end"]
        response = await self.client.get(
            f"https://www.quandl.com/api/v3/datasets/{dataset}.json", params=params
        )
        response.raise_for_status()
        payload = response.json().get("dataset", {})
        column_names = payload.get("column_names", [])
        data = payload.get("data", [])
        records: list[PriceBar | EconomicIndicator] = []
        for row in data:
            if not row:
                continue
            timestamp = pd.Timestamp(row[0]).to_pydatetime().replace(tzinfo=timezone.utc)
            values = dict(zip(column_names[1:], row[1:]))
            if {"Open", "High", "Low", "Close"}.issubset(values.keys()):
                records.append(
                    PriceBar(
                        symbol=request.symbol,
                        timestamp=timestamp,
                        open=_parse_float(values.get("Open")),
                        high=_parse_float(values.get("High")),
                        low=_parse_float(values.get("Low")),
                        close=_parse_float(values.get("Close")),
                        adj_close=_parse_float(values.get("Adj. Close")),
                        volume=_parse_float(values.get("Volume")),
                    )
                )
            else:
                for name, value in values.items():
                    records.append(
                        EconomicIndicator(
                            symbol=request.symbol,
                            name=name,
                            value=_parse_float(value),
                            as_of=timestamp,
                            category="quandl",
                        )
                    )
        return ProviderResult(dataset_type=request.dataset_type, source=self.name, records=records)


class StablePriceStoreProvider(BaseProvider):
    """Load OHLCV bars from a vetted local store for stable backfills."""

    name = "stable_price_store"
    supported_datasets = (DatasetType.PRICES,)

    def __init__(self, path: str | None = None) -> None:
        super().__init__(cache_ttl=0, rate_limit_per_sec=0)
        default_path = path or os.environ.get(
            "STABLE_PRICE_STORE_PATH", "data/stable_price_store.parquet"
        )
        self._default_path = Path(default_path)

    async def _fetch(self, request: ProviderRequest) -> ProviderResult:
        path_value = request.params.get("local_store_path")
        path = Path(path_value) if path_value else self._default_path
        if not path.exists():
            return ProviderResult(
                dataset_type=request.dataset_type,
                source=self.name,
                records=[],
                metadata={"reason": "missing", "path": str(path)},
            )

        loader = pd.read_parquet if path.suffix.lower() in {".parquet", ".pq"} else pd.read_csv
        frame = await asyncio.to_thread(loader, path)
        working = frame.copy()
        if "Ticker" in working.columns:
            working = working[working["Ticker"].str.upper() == request.symbol.upper()]
        date_column = request.params.get("date_column", "Date")
        if date_column in working.columns:
            parsed_dates = pd.to_datetime(working[date_column], errors="coerce")
            if parsed_dates.dt.tz is None:
                parsed_dates = parsed_dates.dt.tz_localize(
                    "UTC", nonexistent="NaT", ambiguous="NaT"
                )
            else:
                parsed_dates = parsed_dates.dt.tz_convert("UTC")
            working[date_column] = parsed_dates
            working = working.dropna(subset=[date_column])
        start = request.params.get("start")
        end = request.params.get("end")

        def _to_utc_timestamp(value: Any | None) -> pd.Timestamp | None:
            if value is None:
                return None
            if isinstance(value, pd.Timestamp):
                return (
                    value.tz_localize("UTC") if value.tzinfo is None else value.tz_convert("UTC")
                )
            return pd.to_datetime(value, utc=True)

        start_ts = _to_utc_timestamp(start)
        end_ts = _to_utc_timestamp(end)
        if date_column in working.columns:
            if start_ts is not None:
                working = working[working[date_column] >= start_ts]
            if end_ts is not None:
                working = working[working[date_column] <= end_ts]
        if date_column in working.columns and working.empty:
            LOGGER.warning(
                "StablePriceStoreProvider returned no rows for %s between %s and %s",
                request.symbol,
                start_ts.isoformat() if start_ts else None,
                end_ts.isoformat() if end_ts else None,
            )
            return ProviderResult(
                dataset_type=request.dataset_type,
                source=self.name,
                records=[],
                metadata={"path": str(path), "reason": "empty"},
            )
        bars: list[PriceBar] = []
        for _, row in working.iterrows():
            timestamp = pd.Timestamp(row.get(date_column)).tz_convert("UTC").to_pydatetime()
            bars.append(
                PriceBar(
                    symbol=request.symbol,
                    timestamp=timestamp,
                    open=_parse_float(row.get("Open")),
                    high=_parse_float(row.get("High")),
                    low=_parse_float(row.get("Low")),
                    close=_parse_float(row.get("Close")),
                    adj_close=_parse_float(row.get("Adj Close")),
                    volume=_parse_float(row.get("Volume")),
                )
            )
        return ProviderResult(
            dataset_type=request.dataset_type,
            source=self.name,
            records=bars,
            metadata={"path": str(path)},
        )


class CSVPriceLoader(BaseProvider):
    """Adapter that loads price data from a CSV file."""

    name = "csv_loader"
    supported_datasets = (DatasetType.PRICES,)

    def __init__(self) -> None:
        super().__init__(cache_ttl=0, rate_limit_per_sec=0)

    async def _fetch(self, request: ProviderRequest) -> ProviderResult:
        path_value = request.params.get("path")
        if not path_value:
            raise ValueError("CSV loader requires a 'path' parameter")
        path = Path(path_value)
        if not path.exists():
            raise FileNotFoundError(path)
        frame = await asyncio.to_thread(pd.read_csv, path)
        bars: list[PriceBar] = []
        date_column = request.params.get("date_column", "Date")
        for _, row in frame.iterrows():
            timestamp = pd.Timestamp(row[date_column]).to_pydatetime().replace(tzinfo=timezone.utc)
            bars.append(
                PriceBar(
                    symbol=request.symbol,
                    timestamp=timestamp,
                    open=_parse_float(row.get("Open")),
                    high=_parse_float(row.get("High")),
                    low=_parse_float(row.get("Low")),
                    close=_parse_float(row.get("Close")),
                    adj_close=_parse_float(row.get("Adj Close")),
                    volume=_parse_float(row.get("Volume")),
                )
            )
        return ProviderResult(dataset_type=request.dataset_type, source=self.name, records=bars)


class ParquetPriceLoader(BaseProvider):
    """Adapter that loads price data from a Parquet file."""

    name = "parquet_loader"
    supported_datasets = (DatasetType.PRICES,)

    def __init__(self) -> None:
        super().__init__(cache_ttl=0, rate_limit_per_sec=0)

    async def _fetch(self, request: ProviderRequest) -> ProviderResult:
        path_value = request.params.get("path")
        if not path_value:
            raise ValueError("Parquet loader requires a 'path' parameter")
        path = Path(path_value)
        if not path.exists():
            raise FileNotFoundError(path)
        frame = await asyncio.to_thread(pd.read_parquet, path)
        bars: list[PriceBar] = []
        date_column = request.params.get("date_column", "Date")
        for _, row in frame.iterrows():
            timestamp = pd.Timestamp(row[date_column]).to_pydatetime().replace(tzinfo=timezone.utc)
            bars.append(
                PriceBar(
                    symbol=request.symbol,
                    timestamp=timestamp,
                    open=_parse_float(row.get("Open")),
                    high=_parse_float(row.get("High")),
                    low=_parse_float(row.get("Low")),
                    close=_parse_float(row.get("Close")),
                    adj_close=_parse_float(row.get("Adj Close")),
                    volume=_parse_float(row.get("Volume")),
                )
            )
        return ProviderResult(dataset_type=request.dataset_type, source=self.name, records=bars)


__all__ = [
    "AlphaVantageProvider",
    "CSVPriceLoader",
    "FREDProvider",
    "FinnhubProvider",
    "GDELTProvider",
    "NewsAPIProvider",
    "ParquetPriceLoader",
    "PolygonProvider",
    "QuandlProvider",
    "StablePriceStoreProvider",
    "RedditProvider",
    "StooqProvider",
    "TiingoProvider",
    "TwitterProvider",
    "YahooFinanceProvider",
]
