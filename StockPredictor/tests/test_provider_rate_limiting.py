"""Tests covering provider cooldown behaviours and fallback logic."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path
import sys

import httpx
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stock_predictor.providers.adapters import StooqProvider, YahooFinanceProvider
from stock_predictor.providers.base import (
    DEFAULT_YAHOO_RATE_LIMIT_PER_SEC,
    DatasetType,
    PriceBar,
    ProviderCooldownError,
    ProviderRegistry,
    ProviderRequest,
    ProviderResult,
    build_default_registry,
)


def test_default_registry_uses_conservative_yahoo_rate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Yahoo provider defaults to a slow, safe rate when unspecified."""

    for key in (
        "YAHOO_RATE_LIMIT_PER_SECOND",
        "YAHOO_RATE_LIMIT_PER_SEC",
        "YAHOO_RATE_LIMIT_PER_MINUTE",
        "YAHOO_COOLDOWN_SECONDS",
    ):
        monkeypatch.delenv(key, raising=False)

    registry = build_default_registry()
    yahoo = registry.get("yahoo_finance")

    interval = yahoo._rate_limiter._interval  # type: ignore[attr-defined]
    if interval > 0:
        assert 1.0 / interval == pytest.approx(
            DEFAULT_YAHOO_RATE_LIMIT_PER_SEC, rel=1e-3
        )
    else:
        assert DEFAULT_YAHOO_RATE_LIMIT_PER_SEC == 0.0

    asyncio.run(registry.aclose())


def test_yahoo_global_cooldown_short_circuits(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 429 response should trigger a provider-wide cooldown."""

    async def _runner() -> None:
        provider = YahooFinanceProvider(cooldown_seconds=3, retries=1, jitter=0.0)
        request = ProviderRequest(dataset_type=DatasetType.PRICES, symbol="AAPL", params={})

        call_count = 0

        async def _failing_fetch(self: YahooFinanceProvider, _: ProviderRequest) -> ProviderResult:
            nonlocal call_count
            call_count += 1
            request_obj = httpx.Request("GET", "https://example.com")
            response = httpx.Response(429, request=request_obj, headers={"Retry-After": "120"})
            raise httpx.HTTPStatusError("Too Many Requests", request=request_obj, response=response)

        monkeypatch.setattr(YahooFinanceProvider, "_fetch", _failing_fetch, raising=False)

        with pytest.raises(ProviderCooldownError) as excinfo:
            await provider.fetch(request)

        assert call_count == 1
        assert excinfo.value.retry_after == pytest.approx(120.0)
        assert excinfo.value.attempts == 1

        with pytest.raises(ProviderCooldownError) as cooldown_exc:
            await provider.fetch(request)

        # Second invocation should short-circuit without invoking _fetch again.
        assert call_count == 1
        assert cooldown_exc.value.attempts == 0

        in_cooldown, remaining = await provider.global_cooldown_remaining()
        assert in_cooldown
        assert remaining > 0

        await provider.aclose()

    asyncio.run(_runner())


def test_429_retry_hint_overrides_default_cooldown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retry-After hints should set the cooldown instead of the default window."""

    async def _runner() -> None:
        provider = YahooFinanceProvider(
            cooldown_seconds=900,
            retries=1,
            backoff_factor=1.0,
            jitter=0.0,
        )
        request = ProviderRequest(dataset_type=DatasetType.PRICES, symbol="IBM", params={})

        async def _always_rate_limit(
            self: YahooFinanceProvider, _: ProviderRequest
        ) -> ProviderResult:
            request_obj = httpx.Request("GET", "https://example.com")
            response = httpx.Response(
                429, request=request_obj, headers={"Retry-After": "30"}
            )
            raise httpx.HTTPStatusError(
                "Too Many Requests", request=request_obj, response=response
            )

        monkeypatch.setattr(
            YahooFinanceProvider, "_fetch", _always_rate_limit, raising=False
        )

        with pytest.raises(ProviderCooldownError) as excinfo:
            await provider.fetch(request)

        assert excinfo.value.retry_after == pytest.approx(30.0)

        await provider.aclose()

    asyncio.run(_runner())


def test_retry_uses_full_retry_after_delay(monkeypatch: pytest.MonkeyPatch) -> None:
    """Providers should respect long Retry-After hints when retrying."""

    async def _runner() -> None:
        provider = YahooFinanceProvider(
            cooldown_seconds=3,
            retries=2,
            backoff_factor=1.0,
            max_retry_wait=10.0,
            jitter=0.0,
            rate_limit_per_sec=0.0,
        )
        request = ProviderRequest(dataset_type=DatasetType.PRICES, symbol="MSFT", params={})

        call_count = 0

        async def _fetch(self: YahooFinanceProvider, _: ProviderRequest) -> ProviderResult:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                request_obj = httpx.Request("GET", "https://example.com")
                response = httpx.Response(
                    429,
                    request=request_obj,
                    headers={"Retry-After": "120"},
                )
                raise httpx.HTTPStatusError(
                    "Too Many Requests", request=request_obj, response=response
                )
            bar = PriceBar(
                symbol="MSFT",
                timestamp=datetime.now(timezone.utc),
                open=1.0,
                high=1.1,
                low=0.9,
                close=1.05,
                volume=5,
            )
            return ProviderResult(
                dataset_type=DatasetType.PRICES,
                source=self.name,
                records=[bar],
                metadata={"symbol": "MSFT"},
            )

        sleep_calls: list[float] = []

        async def _fake_sleep(delay: float, result: object | None = None) -> object | None:
            sleep_calls.append(delay)
            return result

        monkeypatch.setattr(YahooFinanceProvider, "_fetch", _fetch, raising=False)
        monkeypatch.setattr(asyncio, "sleep", _fake_sleep)

        await provider.fetch(request)

        assert sleep_calls
        assert sleep_calls[0] == pytest.approx(120.0)

        await provider.aclose()

    asyncio.run(_runner())


def test_retry_after_header_supports_vendor_specific_hints() -> None:
    """Custom rate limit headers should provide retry hints when available."""

    request = httpx.Request("GET", "https://example.com")

    absolute_reset = datetime.now(timezone.utc).timestamp() + 42
    absolute_response = httpx.Response(
        429,
        request=request,
        headers={"X-RateLimit-Reset": str(absolute_reset)},
    )

    relative_response = httpx.Response(
        429,
        request=request,
        headers={"X-RateLimit-Reset-Second": "17"},
    )

    absolute_delay = YahooFinanceProvider._retry_after_header(absolute_response)
    relative_delay = YahooFinanceProvider._retry_after_header(relative_response)

    assert absolute_delay is not None and absolute_delay == pytest.approx(42, rel=0.05)
    assert relative_delay == pytest.approx(17.0)


def test_registry_respects_global_cooldown(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Registry should skip globally cooled-down providers and rely on fallbacks."""

    async def _runner() -> None:
        call_state = {"yahoo": 0}

        async def _yahoo_fail(self: YahooFinanceProvider, _: ProviderRequest) -> ProviderResult:
            call_state["yahoo"] += 1
            request_obj = httpx.Request("GET", "https://example.com")
            response = httpx.Response(429, request=request_obj, headers={"Retry-After": "5"})
            raise httpx.HTTPStatusError("Too Many Requests", request=request_obj, response=response)

        async def _stooq_success(self: StooqProvider, request: ProviderRequest) -> ProviderResult:
            bar = PriceBar(
                symbol=request.symbol,
                timestamp=datetime.now(timezone.utc),
                open=1.0,
                high=1.5,
                low=0.9,
                close=1.2,
                volume=10,
            )
            return ProviderResult(
                dataset_type=request.dataset_type,
                source=self.name,
                records=[bar],
                metadata={"symbol": request.symbol},
            )

        monkeypatch.setattr(YahooFinanceProvider, "_fetch", _yahoo_fail, raising=False)
        monkeypatch.setattr(StooqProvider, "_fetch", _stooq_success, raising=False)

        yahoo = YahooFinanceProvider(cooldown_seconds=10, retries=1)
        stooq = StooqProvider()
        registry = ProviderRegistry()
        registry.register(yahoo)
        registry.register(stooq)

        request = ProviderRequest(dataset_type=DatasetType.PRICES, symbol="AAPL", params={})

        caplog.set_level(logging.INFO)
        summary_first = await registry.fetch_all(request)
        assert len(summary_first.results) == 1
        assert summary_first.results[0].source == "stooq"
        assert summary_first.failures
        assert summary_first.failures[0].provider == "yahoo_finance"
        assert summary_first.failures[0].is_rate_limited
        assert call_state["yahoo"] == 1

        caplog.clear()
        summary_second = await registry.fetch_all(request)
        assert len(summary_second.results) == 1
        assert summary_second.results[0].source == "stooq"
        assert summary_second.failures
        assert summary_second.failures[0].provider == "yahoo_finance"
        assert summary_second.failures[0].is_rate_limited
        # Yahoo should not be invoked again while the global cooldown is active.
        assert call_state["yahoo"] == 1

        # Ensure no warnings are emitted when fallback providers succeed.
        assert not [record for record in caplog.records if record.levelno >= logging.WARNING]

        await yahoo.aclose()
        await stooq.aclose()

    asyncio.run(_runner())
