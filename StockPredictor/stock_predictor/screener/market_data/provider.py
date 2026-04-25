from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Callable, Protocol

import pandas as pd

from ..pattern_engine.types import Candle
from .symbol_universe import (
    FUTURES_UNIVERSE,
    STOCK_UNIVERSE,
    SymbolUniverseService,
)


class MarketDataProvider(Protocol):
    def get_universe(self, market_type: str) -> list[str]: ...

    def get_historical_bars(self, symbol: str, timeframe: str, lookback: int) -> list[Candle]: ...

    def get_historical_bars_batch(
        self, symbols: list[str], timeframe: str, lookback: int
    ) -> dict[str, list[Candle]]: ...

    def subscribe_live_bars(
        self, symbols: list[str], timeframe: str, callback: Callable[[str, Candle], None]
    ) -> None: ...

    def normalize_symbol(self, symbol: str) -> str: ...

    def get_symbol_metadata(self, symbol: str) -> dict[str, Any]: ...


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
        frame = frame.rename(columns=rename)
        return frame

    def _resolve_range(self, lookback: int) -> tuple[date, date]:
        end_date = datetime.utcnow().date()
        start_date = end_date - timedelta(days=max(int(lookback), 1))
        return start_date, end_date

    def get_historical_bars(self, symbol: str, timeframe: str, lookback: int) -> list[Candle]:
        start_date, end_date = self._resolve_range(lookback)
        payload = self.request_fn(
            f"/data/{symbol}",
            params={
                "refresh": False,
                "interval": timeframe,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
            },
        )
        frame = self._coerce_frame(payload)
        if frame.empty or not {"timestamp", "open", "high", "low", "close"}.issubset(frame.columns):
            return []
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
        if "volume" not in frame.columns:
            frame["volume"] = 0.0
        out: list[Candle] = []
        for row in frame.dropna(subset=["timestamp", "open", "high", "low", "close"]).itertuples(index=False):
            out.append(
                Candle(
                    timestamp=row.timestamp.to_pydatetime(),
                    open=float(row.open),
                    high=float(row.high),
                    low=float(row.low),
                    close=float(row.close),
                    volume=float(getattr(row, "volume", 0.0)),
                )
            )
        return out

    def get_historical_bars_batch(
        self, symbols: list[str], timeframe: str, lookback: int
    ) -> dict[str, list[Candle]]:
        return {symbol: self.get_historical_bars(symbol, timeframe, lookback) for symbol in symbols}

    def subscribe_live_bars(self, symbols: list[str], timeframe: str, callback: Callable[[str, Candle], None]) -> None:
        return None

    def normalize_symbol(self, symbol: str) -> str:
        return symbol.strip().upper()

    def get_symbol_metadata(self, symbol: str) -> dict[str, Any]:
        upper = self.normalize_symbol(symbol)
        if self.universe_service is not None:
            for item in self.universe_service.get_combined_universe():
                if item.symbol == upper:
                    return {
                        "symbol": item.symbol,
                        "name": item.name,
                        "market_type": item.market_type,
                        "exchange": item.exchange,
                        "active": item.active,
                    }
        return {
            "symbol": upper,
            "name": upper,
            "market_type": "future" if "=F" in upper else "stock",
        }
