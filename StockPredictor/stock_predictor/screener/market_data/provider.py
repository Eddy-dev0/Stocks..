from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from statistics import median
from typing import Any, Callable, Protocol

import pandas as pd

from ..pattern_engine.pattern_test_fixtures import (
    create_cup_and_handle_fixture,
    create_double_bottom_fixture,
    create_flag_fixture,
    create_pennant_fixture,
)
from ..pattern_engine.types import Candle
from .symbol_universe import FUTURES_UNIVERSE, STOCK_UNIVERSE, SymbolUniverseService


class MarketDataError(RuntimeError):
    def __init__(self, symbol: str, message: str, cause: Exception | None = None) -> None:
        self.symbol = symbol
        self.cause = cause
        detail = f"{message} [{symbol}]"
        if cause is not None:
            detail = f"{detail}: {cause}"
        super().__init__(detail)


class MarketDataProvider(Protocol):
    def get_universe(self, market_type: str) -> list[str]: ...

    def get_historical_bars(
        self, symbol: str, timeframe: str, lookback: int, options: dict[str, Any] | None = None
    ) -> list[Candle]: ...

    def get_historical_bars_batch(
        self, symbols: list[str], timeframe: str, lookback: int
    ) -> dict[str, list[Candle]]: ...

    def subscribe_live_bars(
        self, symbols: list[str], timeframe: str, callback: Callable[[str, Candle], None]
    ) -> None: ...

    def normalize_symbol(self, symbol: str) -> str: ...

    def get_symbol_metadata(self, symbol: str) -> dict[str, Any]: ...


TIMEFRAME_TO_MINUTES = {"1h": 60, "60m": 60, "45m": 45, "30m": 30, "15m": 15}

YAHOO_FUTURES_MAP = {
    "ES": "ES=F",
    "NQ": "NQ=F",
    "YM": "YM=F",
    "RTY": "RTY=F",
    "CL": "CL=F",
    "GC": "GC=F",
    "SI": "SI=F",
    "NG": "NG=F",
    "ZB": "ZB=F",
    "ZN": "ZN=F",
}

def normalize_symbol_for_provider(provider_name: str, symbol: str) -> tuple[str | None, str | None]:
    provider = provider_name.strip().lower()
    upper = symbol.strip().upper()
    if provider in {"yahoo", "auto", "mock", "not configured"}:
        if upper in YAHOO_FUTURES_MAP:
            return YAHOO_FUTURES_MAP[upper], None
        return upper.replace(".", "-"), None
    if provider == "alpaca":
        if upper in YAHOO_FUTURES_MAP:
            return None, "UNSUPPORTED_SYMBOL"
        return upper.replace(".", "-"), None
    if provider == "polygon":
        return upper.replace(".", "-"), None
    return upper.replace(".", "-"), None


def normalize_timeframe_for_provider(provider_name: str, timeframe: str) -> str:
    provider = provider_name.strip().lower()
    tf = timeframe.strip().lower()
    if provider in {"yahoo", "mock", "auto", "not configured"}:
        if tf in {"1h", "60m"}:
            return "60m"
        if tf in {"30m", "15m", "45m"}:
            return tf
    return tf


def timeframe_period_hint(timeframe: str) -> str:
    tf = timeframe.strip().lower()
    if tf in {"1h", "60m"}:
        return "60d"
    if tf == "30m":
        return "30d"
    if tf == "15m":
        return "15d"
    return "60d"

@dataclass
class FrontendAPIMarketDataProvider:
    request_fn: Callable[..., dict[str, Any] | None]
    universe_service: SymbolUniverseService | None = None

    def __post_init__(self) -> None:
        if self.universe_service is None:
            self.universe_service = SymbolUniverseService()

    def get_universe(self, market_type: str) -> list[str]:
        if self.universe_service is None:
            if market_type == "stock":
                return STOCK_UNIVERSE
            if market_type == "future":
                return FUTURES_UNIVERSE
            return [*STOCK_UNIVERSE, *FUTURES_UNIVERSE]
        return [item.symbol for item in self.universe_service.get_universe(market_type)]

    def _coerce_frame(self, payload: dict[str, Any] | None) -> pd.DataFrame:
        if not payload:
            return pd.DataFrame()
        raw = payload.get("data") if isinstance(payload, dict) else payload
        frame = pd.DataFrame(raw) if isinstance(raw, list) else pd.DataFrame(raw or {})
        cols = {c.lower(): c for c in frame.columns}
        rename = {}
        for key, target in [("date", "timestamp"), ("datetime", "timestamp"), ("time", "timestamp")]:
            if key in cols:
                rename[cols[key]] = target
        for key in ("open", "high", "low", "close", "volume"):
            if key in cols:
                rename[cols[key]] = key
        return frame.rename(columns=rename)

    def _to_candles(self, frame: pd.DataFrame, symbol: str) -> list[Candle]:
        work = frame.copy()
        cols = {c.lower(): c for c in work.columns}
        rename = {cols[k]: k for k in ("open", "high", "low", "close", "volume") if k in cols}
        if "timestamp" not in work.columns:
            for key in ("datetime", "date", "time"):
                if key in cols:
                    rename[cols[key]] = "timestamp"
                    break
        work = work.rename(columns=rename)
        if work.empty or not {"timestamp", "open", "high", "low", "close"}.issubset(work.columns):
            raise MarketDataError(symbol, "OHLC columns missing or empty payload")
        work["timestamp"] = pd.to_datetime(work["timestamp"], errors="coerce", utc=True)
        if "volume" not in work.columns:
            work["volume"] = 0.0
        out: list[Candle] = []
        for row in work.dropna(subset=["timestamp", "open", "high", "low", "close"]).itertuples(index=False):
            out.append(
                Candle(
                    timestamp=row.timestamp.to_pydatetime().replace(tzinfo=None),
                    open=float(row.open),
                    high=float(row.high),
                    low=float(row.low),
                    close=float(row.close),
                    volume=float(getattr(row, "volume", 0.0) or 0.0),
                )
            )
        if not out:
            raise MarketDataError(symbol, "No candle rows after parsing")
        out.sort(key=lambda c: c.timestamp)
        return out

    def _fetch_from_ui_api(self, symbol: str, timeframe: str) -> list[Candle]:
        payload = self.request_fn(
            f"/data/{symbol}",
            params={
                "refresh": True,
                "interval": timeframe,
            },
        )
        frame = self._coerce_frame(payload)
        return self._to_candles(frame, symbol)

    def _extract_symbol_frame(self, frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
        if not isinstance(frame.columns, pd.MultiIndex):
            return frame
        levels = [list(frame.columns.get_level_values(i)) for i in range(frame.columns.nlevels)]
        for level in range(frame.columns.nlevels):
            if symbol in levels[level]:
                return frame.xs(symbol, axis=1, level=level)
        return frame

    def _fetch_from_yfinance(self, symbol: str, timeframe: str, lookback: int, period_hint: str | None = None) -> list[Candle]:
        import yfinance as yf

        provider_symbol = self.normalize_symbol(symbol)
        interval = normalize_timeframe_for_provider("yahoo", timeframe)
        period = period_hint or timeframe_period_hint(timeframe)
        try:
            frame = yf.download(
                tickers=provider_symbol,
                period=period,
                interval=interval,
                auto_adjust=False,
                progress=False,
                threads=False,
            )
        except Exception as exc:
            raise MarketDataError(provider_symbol, "yfinance download failed", exc) from exc
        if frame is None or frame.empty:
            raise MarketDataError(provider_symbol, "No bars returned from yfinance")

        frame = self._extract_symbol_frame(frame, provider_symbol)
        normalized = frame.reset_index()
        if "Datetime" in normalized.columns:
            normalized = normalized.rename(columns={"Datetime": "timestamp"})
        elif "Date" in normalized.columns:
            normalized = normalized.rename(columns={"Date": "timestamp"})

        candles = self._to_candles(normalized, provider_symbol)
        required = min(int(lookback), 120 if timeframe in {"1h", "60m"} else 60)
        if len(candles) < required:
            raise MarketDataError(provider_symbol, f"Not enough candles for {timeframe}: {len(candles)}")
        return candles[-int(lookback) :]

    def get_historical_bars(
        self, symbol: str, timeframe: str, lookback: int, options: dict[str, Any] | None = None
    ) -> list[Candle]:
        provider_symbol = self.normalize_symbol(symbol)
        period_hint = (options or {}).get("period") if isinstance(options, dict) else None
        try:
            return self._fetch_from_ui_api(provider_symbol, normalize_timeframe_for_provider("yahoo", timeframe))
        except Exception:
            return self._fetch_from_yfinance(provider_symbol, timeframe, lookback, period_hint=period_hint)

    def get_historical_bars_batch(
        self, symbols: list[str], timeframe: str, lookback: int
    ) -> dict[str, list[Candle]]:
        bars: dict[str, list[Candle]] = {}
        for symbol in symbols:
            bars[symbol] = self.get_historical_bars(symbol, timeframe, lookback)
        return bars

    def subscribe_live_bars(self, symbols: list[str], timeframe: str, callback: Callable[[str, Candle], None]) -> None:
        _ = (symbols, timeframe, callback)
        return None

    def normalize_symbol(self, symbol: str) -> str:
        normalized, _err = normalize_symbol_for_provider("yahoo", symbol)
        return normalized or symbol.strip().upper()

    def get_symbol_metadata(self, symbol: str) -> dict[str, Any]:
        upper = self.normalize_symbol(symbol)
        if self.universe_service is not None:
            for item in self.universe_service.get_combined_universe():
                if item.symbol == symbol.strip().upper():
                    return {
                        "symbol": item.symbol,
                        "providerSymbol": upper,
                        "name": item.name,
                        "market_type": item.market_type,
                        "exchange": item.exchange,
                        "active": item.active,
                    }
        return {
            "symbol": upper,
            "providerSymbol": upper,
            "name": upper,
            "market_type": "future" if "=F" in upper else "stock",
        }

    def provider_status(self) -> dict[str, str]:
        return {
            "provider": "Yahoo",
            "mode": "Live",
            "configured": "yes",
        }

    def test_data_provider(self, symbols: list[str], timeframe: str = "1h", lookback: int = 500) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for symbol in symbols:
            provider_symbol = self.normalize_symbol(symbol)
            try:
                candles = self.get_historical_bars(provider_symbol, timeframe, lookback, options={"period": timeframe_period_hint(timeframe)})
                spacings = [
                    (candles[i].timestamp - candles[i - 1].timestamp).total_seconds() / 60
                    for i in range(1, len(candles))
                ]
                rows.append(
                    {
                        "symbol": symbol,
                        "providerSymbol": provider_symbol,
                        "status": "OK",
                        "candles": len(candles),
                        "first": candles[0].timestamp.strftime("%Y-%m-%d %H:%M") if candles else "-",
                        "last": candles[-1].timestamp.strftime("%Y-%m-%d %H:%M") if candles else "-",
                        "lastClose": round(candles[-1].close, 4) if candles else "-",
                        "medianSpacingMinutes": round(float(median(spacings)), 2) if spacings else "-",
                        "error": "",
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "symbol": symbol,
                        "providerSymbol": provider_symbol,
                        "status": "ERROR",
                        "candles": 0,
                        "first": "-",
                        "last": "-",
                        "lastClose": "-",
                        "medianSpacingMinutes": "-",
                        "error": str(exc),
                    }
                )
        return rows


class MockPatternMarketDataProvider:
    serial_scan = True

    def __init__(self) -> None:
        fixtures = {
            "AAPL": create_double_bottom_fixture(),
            "MSFT": create_flag_fixture(),
            "TSLA": create_pennant_fixture(),
            "NVDA": create_cup_and_handle_fixture(),
            "AMZN": create_flag_fixture()[:],
        }
        fixtures["AMZN"] = sorted(fixtures["AMZN"], key=lambda c: c.timestamp, reverse=True)
        expanded: dict[str, list[Candle]] = {}
        for key, series in fixtures.items():
            ordered = sorted(series, key=lambda c: c.timestamp)
            stitched: list[Candle] = []
            for repeat in range(10):
                offset = repeat * len(ordered)
                for idx, candle in enumerate(ordered):
                    ts = candle.timestamp + pd.Timedelta(hours=offset).to_pytimedelta()
                    stitched.append(Candle(timestamp=ts, open=candle.open, high=candle.high, low=candle.low, close=candle.close, volume=candle.volume))
            expanded[key] = stitched[:180]
        self._fixtures = expanded

    def get_universe(self, market_type: str) -> list[str]:
        _ = market_type
        return list(self._fixtures)

    def get_historical_bars(self, symbol: str, timeframe: str, lookback: int, options: dict[str, Any] | None = None) -> list[Candle]:
        _ = options
        if timeframe not in {"1h", "60m", "30m", "15m"}:
            raise MarketDataError(symbol, f"Unsupported timeframe {timeframe}")
        normalized = self.normalize_symbol(symbol)
        candles = self._fixtures.get(normalized)
        if candles is None:
            raise MarketDataError(symbol, "No mock candles configured")
        ordered = sorted(candles, key=lambda c: c.timestamp)
        return ordered[-int(lookback) :]

    def get_historical_bars_batch(self, symbols: list[str], timeframe: str, lookback: int) -> dict[str, list[Candle]]:
        return {s: self.get_historical_bars(s, timeframe, lookback) for s in symbols}

    def subscribe_live_bars(self, symbols: list[str], timeframe: str, callback: Callable[[str, Candle], None]) -> None:
        _ = (symbols, timeframe, callback)
        return None

    def normalize_symbol(self, symbol: str) -> str:
        normalized, _ = normalize_symbol_for_provider("mock", symbol)
        return normalized or symbol.strip().upper()

    def provider_status(self) -> dict[str, str]:
        return {"provider": "Mock", "mode": "Mock", "configured": "yes"}

    def get_symbol_metadata(self, symbol: str) -> dict[str, Any]:
        return {"symbol": symbol, "providerSymbol": symbol, "name": symbol, "market_type": "stock"}
