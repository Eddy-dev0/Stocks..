from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Callable, Protocol

import pandas as pd

from ..pattern_engine.types import Candle
from .symbol_universe import FUTURES_UNIVERSE, STOCK_UNIVERSE


class MarketDataProvider(Protocol):
    def get_universe(self, market_type: str) -> list[str]: ...
    def get_historical_bars(self, symbol: str, timeframe: str, start_date: date, end_date: date) -> list[Candle]: ...
    def subscribe_live_bars(self, symbols: list[str], timeframe: str, callback: Callable[[str, Candle], None]) -> None: ...
    def normalize_symbol(self, symbol: str) -> str: ...
    def get_symbol_metadata(self, symbol: str) -> dict[str, Any]: ...


@dataclass
class FrontendAPIMarketDataProvider:
    request_fn: Callable[..., dict[str, Any] | None]

    def get_universe(self, market_type: str) -> list[str]:
        if market_type == "stock":
            return STOCK_UNIVERSE
        if market_type == "future":
            return FUTURES_UNIVERSE
        return [*STOCK_UNIVERSE, *FUTURES_UNIVERSE]

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

    def get_historical_bars(self, symbol: str, timeframe: str, start_date: date, end_date: date) -> list[Candle]:
        payload = self.request_fn(
            f"/data/{symbol}",
            params={"refresh": False, "interval": timeframe, "start_date": start_date.isoformat(), "end_date": end_date.isoformat()},
        )
        frame = self._coerce_frame(payload)
        if frame.empty or not {"timestamp", "open", "high", "low", "close"}.issubset(frame.columns):
            return []
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
        if "volume" not in frame.columns:
            frame["volume"] = 0.0
        out: list[Candle] = []
        for row in frame.dropna(subset=["timestamp", "open", "high", "low", "close"]).itertuples(index=False):
            out.append(Candle(timestamp=row.timestamp.to_pydatetime(), open=float(row.open), high=float(row.high), low=float(row.low), close=float(row.close), volume=float(getattr(row, "volume", 0.0))))
        return out

    def subscribe_live_bars(self, symbols: list[str], timeframe: str, callback: Callable[[str, Candle], None]) -> None:
        return None

    def normalize_symbol(self, symbol: str) -> str:
        return symbol.strip().upper()

    def get_symbol_metadata(self, symbol: str) -> dict[str, Any]:
        return {"symbol": symbol, "name": symbol, "market_type": "future" if "=F" in symbol else "stock"}
