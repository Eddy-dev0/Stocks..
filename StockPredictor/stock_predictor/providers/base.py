"""Asynchronous provider interfaces and shared utilities."""

from __future__ import annotations

import abc
import asyncio
import logging
import os
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

try:  # Python 3.11+
    from enum import StrEnum as _BaseStrEnum
except ImportError:  # pragma: no cover - fallback for older interpreters
    from enum import Enum as _Enum

    class _BaseStrEnum(str, _Enum):
        """Fallback StrEnum implementation for Python < 3.11."""

        pass

import httpx
from pydantic import BaseModel, ConfigDict, Field

LOGGER = logging.getLogger(__name__)


class ProviderError(RuntimeError):
    """Base class for provider related failures."""


class ProviderAuthenticationError(ProviderError):
    """Raised when a provider cannot authenticate because credentials are missing."""


class ProviderConfigurationError(ProviderError):
    """Raised when the caller supplies unsupported parameters."""


class DatasetType(_BaseStrEnum):
    """Enumeration of supported dataset categories."""

    PRICES = "prices"
    NEWS = "news"
    MACRO = "macro"
    SENTIMENT = "sentiment"
    FUNDAMENTALS = "fundamentals"
    CORPORATE_EVENTS = "corporate_events"
    OPTIONS = "options"
    ALTERNATIVE = "alternative"
    ESG = "esg"
    OWNERSHIP = "ownership"


class RecordModel(BaseModel):
    """Base class for structured provider payloads."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_by_name=True,
        populate_by_name=True,
    )


class PriceBar(RecordModel):
    """OHLCV bar representation."""

    symbol: str
    timestamp: datetime
    open: float | None = None
    high: float | None = None
    low: float | None = None
    close: float | None = None
    adj_close: float | None = Field(default=None, alias="adjClose")
    volume: float | None = None

    def as_frame_row(self) -> dict[str, Any]:
        return {
            "Date": self.timestamp,
            "Open": self.open,
            "High": self.high,
            "Low": self.low,
            "Close": self.close,
            "Adj Close": self.adj_close,
            "Volume": self.volume,
        }


class NewsArticle(RecordModel):
    """Structured representation of a news article."""

    symbol: str
    headline: str
    summary: str | None = None
    url: str | None = None
    published_at: datetime
    source: str | None = None

    def as_record(self) -> dict[str, Any]:
        return {
            "Ticker": self.symbol,
            "Title": self.headline,
            "Summary": self.summary,
            "Url": self.url,
            "PublishedAt": self.published_at,
            "Source": self.source,
        }


class EconomicIndicator(RecordModel):
    """Macro or economic indicator payload."""

    symbol: str
    name: str
    value: float | None
    as_of: datetime
    category: str = "macro"
    extra: dict[str, Any] | None = None

    def as_record(self) -> dict[str, Any]:
        return {
            "Ticker": self.symbol,
            "Indicator": self.name,
            "Value": self.value,
            "Date": self.as_of,
            "Category": self.category,
            "Extra": self.extra,
        }


class SentimentSignal(RecordModel):
    """Structured sentiment payload."""

    symbol: str
    provider: str
    signal_type: str
    as_of: datetime
    score: float | None = None
    magnitude: float | None = None
    payload: dict[str, Any] | None = None

    def as_record(self) -> dict[str, Any]:
        return {
            "Ticker": self.symbol,
            "Provider": self.provider,
            "SignalType": self.signal_type,
            "AsOf": self.as_of,
            "Score": self.score,
            "Magnitude": self.magnitude,
            "Payload": self.payload,
        }


class ProviderRequest(BaseModel):
    """Common request object passed to providers."""

    dataset_type: DatasetType
    symbol: str
    params: Mapping[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_by_name=True,
    )


class ProviderResult(BaseModel):
    """Structured response for provider fetches."""

    dataset_type: DatasetType
    source: str
    records: List[RecordModel]
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = Field(default_factory=dict)
    from_cache: bool = False

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_by_name=True,
    )


class _TTLCache:
    """Minimal TTL cache for provider responses."""

    def __init__(self) -> None:
        self._data: MutableMapping[str, tuple[float, ProviderResult]] = {}
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> ProviderResult | None:
        async with self._lock:
            entry = self._data.get(key)
            if not entry:
                return None
            expires, value = entry
            if time.monotonic() > expires:
                del self._data[key]
                return None
            return value

    async def set(self, key: str, value: ProviderResult, ttl: float) -> None:
        async with self._lock:
            if ttl <= 0:
                return
            self._data[key] = (time.monotonic() + ttl, value)


class _AsyncRateLimiter:
    """Simple coroutine based rate limiter."""

    def __init__(self, rate_per_sec: float) -> None:
        self._interval = 1.0 / rate_per_sec if rate_per_sec > 0 else 0.0
        self._lock = asyncio.Lock()
        self._last_call = 0.0

    async def acquire(self) -> None:
        if self._interval <= 0:
            return
        async with self._lock:
            now = time.monotonic()
            wait_for = self._interval - (now - self._last_call)
            if wait_for > 0:
                await asyncio.sleep(wait_for)
            self._last_call = time.monotonic()


class BaseProvider(abc.ABC):
    """Base class for asynchronous data providers."""

    name: str = "provider"
    supported_datasets: Sequence[DatasetType] = ()

    def __init__(
        self,
        *,
        client: httpx.AsyncClient | None = None,
        rate_limit_per_sec: float = 5.0,
        cache_ttl: float = 300.0,
        retries: int = 3,
        backoff_factor: float = 0.5,
    ) -> None:
        self._client = client
        self._client_owner = client is None
        self._rate_limiter = _AsyncRateLimiter(rate_limit_per_sec)
        self._cache = _TTLCache()
        self._cache_ttl = cache_ttl
        self._retries = max(1, retries)
        self._backoff_factor = backoff_factor

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    async def fetch(self, request: ProviderRequest) -> ProviderResult:
        if request.dataset_type not in self.supported_datasets:
            raise ProviderConfigurationError(
                f"Provider {self.name} does not support dataset {request.dataset_type}."
            )
        cache_key = self._cache_key(request)
        cached = await self._cache.get(cache_key)
        if cached:
            return cached.copy(update={"from_cache": True})

        await self._rate_limiter.acquire()
        last_error: Exception | None = None
        for attempt in range(self._retries):
            try:
                result = await self._fetch(request)
                await self._cache.set(cache_key, result, self._cache_ttl)
                return result
            except Exception as exc:  # pragma: no cover - defensive
                last_error = exc
                wait = self._backoff_factor * (2 ** attempt)
                LOGGER.warning(
                    "Provider %s attempt %s failed: %s", self.name, attempt + 1, exc
                )
                if attempt + 1 >= self._retries:
                    break
                if wait > 0:
                    await asyncio.sleep(wait)
        assert last_error is not None  # for mypy
        raise ProviderError(str(last_error)) from last_error

    async def _fetch(self, request: ProviderRequest) -> ProviderResult:
        raise NotImplementedError

    async def aclose(self) -> None:
        if self._client_owner and self._client is not None:
            await self._client.aclose()

    async def __aenter__(self) -> "BaseProvider":
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        await self.aclose()

    def _cache_key(self, request: ProviderRequest) -> str:
        params = dict(sorted(request.params.items()))
        return f"{self.name}:{request.dataset_type}:{request.symbol}:{tuple(params.items())}"

    @staticmethod
    def require_env(name: str) -> str:
        value = os.environ.get(name)
        if not value:
            raise ProviderAuthenticationError(
                f"Environment variable {name} is required for this provider."
            )
        return value


class ProviderRegistry:
    """Registry that manages provider instances per dataset."""

    def __init__(self) -> None:
        self._providers: Dict[str, BaseProvider] = {}
        self._dataset_index: MutableMapping[DatasetType, list[str]] = defaultdict(list)

    def register(self, provider: BaseProvider) -> None:
        if provider.name in self._providers:
            raise ProviderConfigurationError(
                f"Provider {provider.name!r} already registered."
            )
        self._providers[provider.name] = provider
        for dataset in provider.supported_datasets:
            self._dataset_index[dataset].append(provider.name)

    def get(self, name: str) -> BaseProvider:
        try:
            return self._providers[name]
        except KeyError as exc:
            raise ProviderConfigurationError(f"Unknown provider {name}") from exc

    def providers_for(self, dataset: DatasetType) -> list[BaseProvider]:
        return [self._providers[name] for name in self._dataset_index.get(dataset, [])]

    async def fetch_all(
        self,
        request: ProviderRequest,
        *,
        providers: Sequence[str] | None = None,
    ) -> list[ProviderResult]:
        selected: Iterable[BaseProvider]
        if providers:
            selected = (self.get(name) for name in providers)
        else:
            selected = self.providers_for(request.dataset_type)

        tasks = [provider.fetch(request) for provider in selected]
        if not tasks:
            return []
        results = await asyncio.gather(*tasks, return_exceptions=True)
        final: list[ProviderResult] = []
        for result in results:
            if isinstance(result, Exception):
                LOGGER.warning("Provider fetch failed: %s", result)
                continue
            final.append(result)
        return final

    async def aclose(self) -> None:
        await asyncio.gather(*(provider.aclose() for provider in self._providers.values()))

    async def __aenter__(self) -> "ProviderRegistry":
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        await self.aclose()


def build_default_registry() -> ProviderRegistry:
    """Construct a registry with the default provider set."""

    from .adapters import (  # local import to avoid circular dependencies
        AlphaVantageProvider,
        CSVPriceLoader,
        FREDProvider,
        FinnhubProvider,
        GDELTProvider,
        NewsAPIProvider,
        ParquetPriceLoader,
        PolygonProvider,
        QuandlProvider,
        RedditProvider,
        StooqProvider,
        TiingoProvider,
        TwitterProvider,
        YahooFinanceProvider,
    )

    registry = ProviderRegistry()
    def _try_register(factory: type[BaseProvider]) -> None:
        try:
            registry.register(factory())
        except ProviderAuthenticationError as exc:
            LOGGER.debug("Skipping provider %s: %s", factory.__name__, exc)

    registry.register(YahooFinanceProvider())
    _try_register(AlphaVantageProvider)
    _try_register(FinnhubProvider)
    _try_register(PolygonProvider)
    _try_register(TiingoProvider)
    registry.register(StooqProvider())
    _try_register(FREDProvider)
    _try_register(NewsAPIProvider)
    registry.register(GDELTProvider())
    registry.register(RedditProvider())
    _try_register(TwitterProvider)
    _try_register(QuandlProvider)
    registry.register(CSVPriceLoader())
    registry.register(ParquetPriceLoader())
    return registry


__all__ = [
    "BaseProvider",
    "DatasetType",
    "EconomicIndicator",
    "NewsArticle",
    "PriceBar",
    "ProviderAuthenticationError",
    "ProviderConfigurationError",
    "ProviderError",
    "ProviderRegistry",
    "ProviderRequest",
    "ProviderResult",
    "SentimentSignal",
    "build_default_registry",
]
