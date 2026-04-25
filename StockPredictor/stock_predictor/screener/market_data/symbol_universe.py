from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SymbolInfo:
    symbol: str
    name: str
    market_type: str
    exchange: str | None = None
    active: bool = True


_STOCK_UNIVERSE: tuple[SymbolInfo, ...] = (
    SymbolInfo("AAPL", "Apple", "stock", "NASDAQ"),
    SymbolInfo("MSFT", "Microsoft", "stock", "NASDAQ"),
    SymbolInfo("NVDA", "NVIDIA", "stock", "NASDAQ"),
    SymbolInfo("TSLA", "Tesla", "stock", "NASDAQ"),
    SymbolInfo("AMZN", "Amazon", "stock", "NASDAQ"),
    SymbolInfo("META", "Meta Platforms", "stock", "NASDAQ"),
    SymbolInfo("GOOGL", "Alphabet", "stock", "NASDAQ"),
    SymbolInfo("AMD", "Advanced Micro Devices", "stock", "NASDAQ"),
    SymbolInfo("NFLX", "Netflix", "stock", "NASDAQ"),
    SymbolInfo("JPM", "JPMorgan Chase", "stock", "NYSE"),
    SymbolInfo("XOM", "Exxon Mobil", "stock", "NYSE"),
    SymbolInfo("AVGO", "Broadcom", "stock", "NASDAQ"),
    SymbolInfo("COST", "Costco", "stock", "NASDAQ"),
    SymbolInfo("CRM", "Salesforce", "stock", "NYSE"),
    SymbolInfo("ORCL", "Oracle", "stock", "NYSE"),
    SymbolInfo("INTC", "Intel", "stock", "NASDAQ"),
    SymbolInfo("BA", "Boeing", "stock", "NYSE"),
    SymbolInfo("DIS", "Walt Disney", "stock", "NYSE"),
    SymbolInfo("NKE", "Nike", "stock", "NYSE"),
    SymbolInfo("PYPL", "PayPal", "stock", "NASDAQ"),
    SymbolInfo("WMT", "Walmart", "stock", "NYSE"),
    SymbolInfo("KO", "Coca-Cola", "stock", "NYSE"),
)

_FUTURES_UNIVERSE: tuple[SymbolInfo, ...] = (
    SymbolInfo("ES=F", "S&P 500 E-mini", "future", "CME"),
    SymbolInfo("NQ=F", "Nasdaq 100 E-mini", "future", "CME"),
    SymbolInfo("YM=F", "Dow Jones E-mini", "future", "CBOT"),
    SymbolInfo("RTY=F", "Russell 2000 E-mini", "future", "CME"),
    SymbolInfo("CL=F", "Crude Oil", "future", "NYMEX"),
    SymbolInfo("GC=F", "Gold", "future", "COMEX"),
    SymbolInfo("SI=F", "Silver", "future", "COMEX"),
    SymbolInfo("NG=F", "Natural Gas", "future", "NYMEX"),
    SymbolInfo("ZB=F", "US Treasury Bond", "future", "CBOT"),
    SymbolInfo("ZN=F", "10Y Treasury Note", "future", "CBOT"),
)


class SymbolUniverseService:
    """Provides a configurable universe for market-wide scans."""

    def get_stock_universe(self) -> list[SymbolInfo]:
        return [item for item in _STOCK_UNIVERSE if item.active]

    def get_futures_universe(self) -> list[SymbolInfo]:
        return [item for item in _FUTURES_UNIVERSE if item.active]

    def get_combined_universe(self) -> list[SymbolInfo]:
        return [*self.get_stock_universe(), *self.get_futures_universe()]

    def get_universe(self, market_filter: str = "all") -> list[SymbolInfo]:
        if market_filter == "stock":
            return self.get_stock_universe()
        if market_filter == "future":
            return self.get_futures_universe()
        return self.get_combined_universe()


_DEFAULT_SERVICE = SymbolUniverseService()


def get_stock_universe() -> list[SymbolInfo]:
    return _DEFAULT_SERVICE.get_stock_universe()


def get_futures_universe() -> list[SymbolInfo]:
    return _DEFAULT_SERVICE.get_futures_universe()


def get_combined_universe() -> list[SymbolInfo]:
    return _DEFAULT_SERVICE.get_combined_universe()


# Backwards compatibility for callers that still import plain symbol lists.
STOCK_UNIVERSE = [item.symbol for item in _DEFAULT_SERVICE.get_stock_universe()]
FUTURES_UNIVERSE = [item.symbol for item in _DEFAULT_SERVICE.get_futures_universe()]
