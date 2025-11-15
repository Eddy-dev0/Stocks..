"""Asynchronous provider interfaces and shared utilities."""

from __future__ import annotations

import abc
import asyncio
import logging
import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import TYPE_CHECKING, Any, Awaitable, Dict, List, Mapping, MutableMapping, Optional, Sequence

try:  # Python 3.11+
    from enum import StrEnum as _BaseStrEnum
except ImportError:  # pragma: no cover - fallback for older interpreters
    from enum import Enum as _Enum

    class _BaseStrEnum(str, _Enum):
        """Fallback StrEnum implementation for Python < 3.11."""

        pass

import httpx
from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..core.config import PredictorConfig

LOGGER = logging.getLogger(__name__)

# Default Yahoo Finance request rate used when the caller does not explicitly
# configure a throttle. Roughly equates to one request every 40 seconds.
DEFAULT_YAHOO_RATE_LIMIT_PER_SEC = 1.0 / 40.0

# Tracks whether we've already informed the caller that file-based loaders are
# disabled. This avoids repeating the same informational log on every
# registry creation.
_FILE_LOADER_LOGGED = False


class ProviderError(RuntimeError):
    """Base class for provider related failures."""


class ProviderAuthenticationError(ProviderError):
    """Raised when a provider cannot authenticate because credentials are missing."""


class ProviderConfigurationError(ProviderError):
    """Raised when the caller supplies unsupported parameters."""


class ProviderCooldownError(ProviderError):
    """Raised when a provider is temporarily disabled because of rate limiting."""

    def __init__(
        self,
        provider: str,
        symbol: str,
        retry_after: float,
        attempts: int,
        message: str | None = None,
    ) -> None:
        detail = message or f"{provider} temporarily disabled for {symbol}."
        super().__init__(detail)
        self.provider = provider
        self.symbol = symbol
        self.retry_after = retry_after
        self.attempts = attempts


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


@dataclass(slots=True)
class ProviderFailure:
    """Description of a provider call that failed."""

    provider: str
    error: str
    status_code: int | None = None
    retry_after: float | None = None
    attempt_count: int = 0
    is_rate_limited: bool = False


@dataclass(slots=True)
class ProviderFetchSummary:
    """Collects successful results and failures for a provider request."""

    results: List[ProviderResult]
    failures: List[ProviderFailure]

    @classmethod
    def empty(cls) -> "ProviderFetchSummary":
        return cls([], [])

    def __iter__(self) -> Any:
        """Allow legacy callers to iterate over successful results directly."""

        return iter(self.results)


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
    """Base class for asynchronous data providers with defensive retries."""

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
        cooldown_seconds: float = 900.0,
        max_retry_wait: float = 60.0,
        jitter: float = 0.3,
    ) -> None:
        self._client = client
        self._client_owner = client is None
        self._rate_limiter = _AsyncRateLimiter(rate_limit_per_sec)
        self._cache = _TTLCache()
        self._cache_ttl = cache_ttl
        self._retries = max(1, retries)
        self._backoff_factor = backoff_factor
        self._cooldown_seconds = float(max(0.0, cooldown_seconds))
        self._max_retry_wait = float(max(0.0, max_retry_wait))
        self._jitter = float(max(0.0, jitter))
        self._cooldowns: MutableMapping[str, float] = {}
        self._cooldown_lock = asyncio.Lock()
        self._global_cooldown_until: float = 0.0

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

        in_global_cooldown, global_remaining = await self._global_cooldown_remaining()
        if in_global_cooldown:
            raise ProviderCooldownError(
                self.name,
                request.symbol,
                global_remaining,
                attempts=0,
                message=(
                    f"Provider {self.name} is cooling down globally. "
                    f"Retry after {int(global_remaining)}s."
                ),
            )

        in_cooldown, remaining = await self._cooldown_remaining(request)
        if in_cooldown:
            raise ProviderCooldownError(
                self.name,
                request.symbol,
                remaining,
                attempts=0,
                message=(
                    f"Provider {self.name} is cooling down for {request.symbol}. "
                    f"Retry after {int(remaining)}s."
                ),
            )

        cache_key = self._cache_key(request)
        cached = await self._cache.get(cache_key)
        if cached:
            return cached.model_copy(update={"from_cache": True})

        attempt_errors: list[str] = []
        last_error: Exception | None = None

        for attempt in range(1, self._retries + 1):
            await self._rate_limiter.acquire()
            wait = 0.0
            try:
                result = await self._fetch(request)
                await self._cache.set(cache_key, result, self._cache_ttl)
                await self._clear_cooldown(request)
                return result
            except httpx.HTTPStatusError as exc:
                last_error = exc
                status = exc.response.status_code
                attempt_errors.append(f"HTTP {status}")
                retry_after_hint = self._retry_after_header(exc.response)
                wait = self._retry_delay(attempt, exc, retry_after=retry_after_hint)
                if status == 429 and attempt >= self._retries:
                    cooldown_delay = self._cooldown_delay(wait, retry_after_hint)
                    await self._schedule_cooldown(request, cooldown_delay)
                    await self._schedule_global_cooldown(cooldown_delay)
                    LOGGER.warning(
                        "Provider %s hit rate limits for %s after %s attempts; "
                        "cooling down for %.0fs.",
                        self.name,
                        request.symbol,
                        attempt,
                        cooldown_delay,
                    )
                    raise ProviderCooldownError(
                        self.name,
                        request.symbol,
                        cooldown_delay,
                        attempt,
                        str(exc),
                    ) from exc
            except Exception as exc:  # pragma: no cover - defensive
                last_error = exc
                attempt_errors.append(str(exc))
                wait = self._retry_delay(attempt, exc)

            if attempt >= self._retries:
                break

            if wait > 0:
                LOGGER.debug(
                    "Provider %s retrying %s (%s/%s) in %.2fs because %s",
                    self.name,
                    request.symbol,
                    attempt,
                    self._retries,
                    wait,
                    attempt_errors[-1],
                )
                await asyncio.sleep(wait)

        if last_error is not None:
            LOGGER.warning(
                "Provider %s exhausted retries for %s: %s",
                self.name,
                request.symbol,
                "; ".join(attempt_errors) or str(last_error),
            )
            raise ProviderError(str(last_error)) from last_error

        raise ProviderError(
            f"{self.name} failed for {request.symbol} without raising an exception."
        )

    def _cooldown_delay(self, wait: float, retry_after_hint: float | None) -> float:
        """Return a cooldown delay derived from retry hints and jitter."""

        if retry_after_hint is not None:
            jitter_base = max(retry_after_hint, 1.0) * self._jitter
            jitter = random.uniform(0.0, jitter_base) if jitter_base > 0 else 0.0
            return max(0.0, retry_after_hint + jitter)

        if wait > 0:
            return wait

        fallback = max(self._backoff_factor, 0.0)
        if fallback <= 0:
            return 0.0
        jitter = random.uniform(0.0, fallback * self._jitter) if self._jitter > 0 else 0.0
        return fallback + jitter

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

    def _cooldown_key(self, request: ProviderRequest) -> str:
        return f"{request.dataset_type}:{request.symbol}"

    async def _cooldown_remaining(self, request: ProviderRequest) -> tuple[bool, float]:
        async with self._cooldown_lock:
            expires = self._cooldowns.get(self._cooldown_key(request))
        if not expires:
            return False, 0.0
        remaining = expires - time.monotonic()
        return remaining > 0.0, max(0.0, remaining)

    async def _schedule_cooldown(self, request: ProviderRequest, seconds: float) -> None:
        if seconds <= 0:
            return
        async with self._cooldown_lock:
            self._cooldowns[self._cooldown_key(request)] = time.monotonic() + seconds

    async def _clear_cooldown(self, request: ProviderRequest) -> None:
        async with self._cooldown_lock:
            self._cooldowns.pop(self._cooldown_key(request), None)

    async def _global_cooldown_remaining(self) -> tuple[bool, float]:
        async with self._cooldown_lock:
            expires = self._global_cooldown_until
            if expires <= 0:
                return False, 0.0
            remaining = expires - time.monotonic()
            if remaining <= 0:
                self._global_cooldown_until = 0.0
                return False, 0.0
            return True, remaining

    async def _schedule_global_cooldown(self, seconds: float) -> None:
        if seconds <= 0:
            return
        async with self._cooldown_lock:
            self._global_cooldown_until = time.monotonic() + seconds

    async def global_cooldown_remaining(self) -> tuple[bool, float]:
        """Return whether the provider is in a global cooldown window."""

        return await self._global_cooldown_remaining()

    def _retry_delay(
        self,
        attempt: int,
        exc: Exception,
        *,
        retry_after: float | None = None,
    ) -> float:
        base_delay = self._backoff_factor * (2 ** (attempt - 1))
        header_delay = retry_after
        if header_delay is None and isinstance(exc, httpx.HTTPStatusError):
            header_delay = self._retry_after_header(exc.response)
        if header_delay is not None:
            base_delay = max(base_delay, header_delay)
        jitter = random.uniform(0.0, self._jitter * max(base_delay, 1.0))
        total_delay = base_delay + jitter
        if header_delay is not None:
            return total_delay
        return min(self._max_retry_wait, total_delay)

    @staticmethod
    def _retry_after_header(response: httpx.Response) -> float | None:
        header = response.headers.get("Retry-After")
        if not header:
            return None
        try:
            return float(header)
        except ValueError:
            try:
                retry_time = parsedate_to_datetime(header)
            except (TypeError, ValueError):
                return None
            if retry_time is None:
                return None
            return max(
                0.0,
                (retry_time - datetime.now(timezone.utc)).total_seconds(),
            )


class ProviderRegistry:
    """Registry that manages provider instances per dataset."""

    def __init__(self) -> None:
        self._providers: Dict[str, BaseProvider] = {}
        self._dataset_index: MutableMapping[DatasetType, list[str]] = defaultdict(list)
        self._config_warnings: set[str] = set()

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
    ) -> ProviderFetchSummary:
        if providers:
            selected = [self.get(name) for name in providers]
        else:
            selected = self.providers_for(request.dataset_type)

        tasks: list[tuple[BaseProvider, Awaitable[ProviderResult]]] = []
        skipped_failures: list[ProviderFailure] = []
        for provider in selected:
            if (
                provider.name in {"csv_loader", "parquet_loader"}
                and not request.params.get("path")
            ):
                if provider.name not in self._config_warnings:
                    LOGGER.warning(
                        "Provider %s disabled: configure a 'path' value to enable the loader.",
                        provider.name,
                    )
                    self._config_warnings.add(provider.name)
                continue
            in_global_cooldown, remaining = await provider.global_cooldown_remaining()
            if in_global_cooldown:
                skipped_failures.append(
                    ProviderFailure(
                        provider=provider.name,
                        error=(
                            f"Provider {provider.name} is cooling down globally."
                        ),
                        status_code=429,
                        retry_after=remaining,
                        is_rate_limited=True,
                    )
                )
                continue
            tasks.append((provider, provider.fetch(request)))

        if not tasks:
            if skipped_failures:
                self._log_failures(
                    request.symbol,
                    request.dataset_type,
                    skipped_failures,
                    had_success=False,
                )
                return ProviderFetchSummary([], skipped_failures)
            return ProviderFetchSummary.empty()

        raw_results = await asyncio.gather(
            *(task for _, task in tasks), return_exceptions=True
        )
        successes: list[ProviderResult] = []
        failures: list[ProviderFailure] = list(skipped_failures)

        for (provider, _), outcome in zip(tasks, raw_results):
            if isinstance(outcome, ProviderResult):
                successes.append(outcome)
                continue
            failures.append(self._convert_exception(provider, outcome))

        if failures:
            self._log_failures(
                request.symbol,
                request.dataset_type,
                failures,
                had_success=bool(successes),
            )

        return ProviderFetchSummary(successes, failures)

    async def aclose(self) -> None:
        await asyncio.gather(*(provider.aclose() for provider in self._providers.values()))

    async def __aenter__(self) -> "ProviderRegistry":
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        await self.aclose()

    def _convert_exception(
        self,
        provider: BaseProvider,
        outcome: Exception,
    ) -> ProviderFailure:
        if isinstance(outcome, ProviderCooldownError):
            return ProviderFailure(
                provider=provider.name,
                error=str(outcome),
                status_code=429,
                retry_after=outcome.retry_after,
                attempt_count=outcome.attempts,
                is_rate_limited=True,
            )
        if isinstance(outcome, ProviderError):
            return ProviderFailure(
                provider=provider.name,
                error=str(outcome),
            )
        status = getattr(getattr(outcome, "response", None), "status_code", None)
        return ProviderFailure(
            provider=provider.name,
            error=str(outcome),
            status_code=status,
        )

    def _log_failures(
        self,
        symbol: str,
        dataset: DatasetType,
        failures: Sequence[ProviderFailure],
        *,
        had_success: bool,
    ) -> None:
        parts: list[str] = []
        for failure in failures:
            detail = failure.error
            if failure.status_code:
                detail = f"{detail} (status={failure.status_code})"
            if failure.retry_after:
                detail = f"{detail}; retry_in={int(failure.retry_after)}s"
            parts.append(f"{failure.provider}: {detail}")
        log = LOGGER.info if had_success else LOGGER.warning
        log(
            "Provider failures for %s/%s: %s",
            symbol,
            dataset,
            "; ".join(parts),
        )


def build_default_registry(
    *,
    csv_loader_path: str | os.PathLike[str] | None = None,
    parquet_loader_path: str | os.PathLike[str] | None = None,
    config: "PredictorConfig" | None = None,
) -> ProviderRegistry:
    """Construct a registry with the default provider set.

    Parameters
    ----------
    csv_loader_path:
        Optional filesystem path enabling the :class:`CSVPriceLoader`.
        When ``None`` the loader is not registered.
    parquet_loader_path:
        Optional filesystem path enabling the :class:`ParquetPriceLoader`.
        When ``None`` the loader is not registered.
    config:
        Optional :class:`~stock_predictor.core.config.PredictorConfig` supplying
        provider tuning parameters.

    Notes
    -----
    When no Yahoo-specific rate limit is supplied via ``config`` or environment
    variables the registry applies a conservative default of roughly one
    request every 40 seconds. Override this behaviour by setting
    ``PredictorConfig.yahoo_rate_limit_per_second`` (or the corresponding
    ``YAHOO_RATE_LIMIT_PER_SECOND``/``YAHOO_RATE_LIMIT_PER_MINUTE`` environment
    variables) before building the registry.
    """

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

    def _coerce_positive(value: object | None) -> float | None:
        try:
            numeric = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None
        return max(0.0, numeric)

    def _resolve_yahoo_rate(limit_config: "PredictorConfig" | None) -> float | None:
        rate: float | None = None
        if limit_config is not None:
            per_second = getattr(limit_config, "yahoo_rate_limit_per_second", None)
            if per_second is not None:
                coerced = _coerce_positive(per_second)
                if coerced is not None:
                    rate = coerced
            if rate is None:
                per_minute = getattr(limit_config, "yahoo_rate_limit_per_minute", None)
                if per_minute is not None:
                    coerced = _coerce_positive(per_minute)
                    if coerced is not None:
                        rate = coerced / 60.0
        if rate is None:
            for env_key in ("YAHOO_RATE_LIMIT_PER_SECOND", "YAHOO_RATE_LIMIT_PER_SEC"):
                env_rate = os.environ.get(env_key)
                if env_rate:
                    coerced = _coerce_positive(env_rate)
                    if coerced is not None:
                        rate = coerced
                        break
        if rate is None:
            env_rate = os.environ.get("YAHOO_RATE_LIMIT_PER_MINUTE")
            if env_rate:
                coerced = _coerce_positive(env_rate)
                if coerced is not None:
                    rate = coerced / 60.0
        return rate

    def _resolve_yahoo_cooldown(limit_config: "PredictorConfig" | None) -> float | None:
        cooldown: float | None = None
        if limit_config is not None:
            config_value = getattr(limit_config, "yahoo_cooldown_seconds", None)
            if config_value is not None:
                cooldown = _coerce_positive(config_value)
        if cooldown is None:
            env_value = os.environ.get("YAHOO_COOLDOWN_SECONDS")
            if env_value:
                cooldown = _coerce_positive(env_value)
        return cooldown

    yahoo_rate_limit = _resolve_yahoo_rate(config)
    if yahoo_rate_limit is None:
        yahoo_rate_limit = DEFAULT_YAHOO_RATE_LIMIT_PER_SEC
    yahoo_cooldown = _resolve_yahoo_cooldown(config)

    has_csv_loader = csv_loader_path is not None
    has_parquet_loader = parquet_loader_path is not None

    global _FILE_LOADER_LOGGED
    if not has_csv_loader and not has_parquet_loader and not _FILE_LOADER_LOGGED:
        LOGGER.info(
            "Local price file loaders are disabled. Set "
            "'csv_price_loader_path' or 'parquet_price_loader_path' in the "
            "PredictorConfig to enable them."
        )
        _FILE_LOADER_LOGGED = True

    def _try_register(factory: type[BaseProvider]) -> None:
        try:
            registry.register(factory())
        except ProviderAuthenticationError as exc:
            LOGGER.debug("Skipping provider %s: %s", factory.__name__, exc)

    yahoo_kwargs: dict[str, float] = {}
    if yahoo_rate_limit is not None:
        yahoo_kwargs["rate_limit_per_sec"] = yahoo_rate_limit
    if yahoo_cooldown is not None:
        yahoo_kwargs["cooldown_seconds"] = yahoo_cooldown
    registry.register(YahooFinanceProvider(**yahoo_kwargs))
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
    if has_csv_loader:
        registry.register(CSVPriceLoader())
    if has_parquet_loader:
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
    "ProviderCooldownError",
    "ProviderError",
    "ProviderFailure",
    "ProviderFetchSummary",
    "ProviderRegistry",
    "ProviderRequest",
    "ProviderResult",
    "SentimentSignal",
    "DEFAULT_YAHOO_RATE_LIMIT_PER_SEC",
    "build_default_registry",
]
