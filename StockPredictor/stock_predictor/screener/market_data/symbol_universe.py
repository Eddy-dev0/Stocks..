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

_ADDITIONAL_STOCK_SYMBOLS: tuple[str, ...] = (
    "CVX", "COP", "EOG", "SLB", "MPC", "PSX", "VLO", "OXY", "HES", "FANG", "DVN", "HAL", "BKR",
    "WMB", "KMI", "OKE", "TRGP", "LNG", "EQT", "CTRA", "APA", "MRO", "CHK", "PR", "MTDR", "CHRD",
    "SM", "RRC", "AR", "CIVI", "CNX", "NOV", "HP", "PTEN", "NBR", "RES", "LBRT", "OII", "WHD",
    "PUMP", "TDW", "WFRD", "FTI", "DK", "PBF", "DINO", "SUN", "ET", "EPD", "MPLX", "PAA", "ENLC",
    "AM", "PAGP", "HESM", "KNTK", "NFE", "VIST", "NRG", "CEG", "GE", "GEV", "CAT", "RTX", "ETN",
    "HON", "LMT", "UNP", "UPS", "DE", "ADP", "TDG", "PH", "MMM", "WM", "ITW", "GD", "EMR", "CTAS",
    "CSX", "NSC", "CARR", "JCI", "PWR", "CPRT", "RSG", "URI", "PCAR", "FAST", "GWW", "DAL", "LUV",
    "UAL", "AAL", "FDX", "EXPD", "JBHT", "CHRW", "ODFL", "XPO", "HUBG", "SAIA", "KNX", "WERN",
    "ARCB", "LSTR", "R", "TXT", "NOC", "HWM", "AXON", "TDY", "HEI", "LHX", "CW", "HII", "SPR",
    "MOG.A", "MOG.B", "BWXT", "KTOS", "AVAV", "OSK", "WAB", "AGCO", "TTC", "ALSN", "LECO", "IR",
    "DOV", "XYL", "IEX", "ROK", "AME", "GNRC", "HUBB", "VRT", "AYI", "AOS", "FELE", "FLS", "GGG",
    "NDSN", "WSO", "SITE", "WCC", "MSM", "AIT", "KBR", "J", "ACM", "FLR", "MTZ", "FIX", "DY",
    "ROAD", "TTEK", "NVEE", "EME", "LLY", "UNH", "JNJ", "ABBV", "MRK", "TMO", "ABT", "ISRG", "AMGN",
    "PFE", "GILD", "DHR", "BMY", "MDT", "SYK", "BSX", "ELV", "CI", "CVS", "HUM", "HCA", "MCK",
    "COR", "CAH", "ZBH", "EW", "IDXX", "DXCM", "VRTX", "REGN", "BIIB", "MRNA", "WST", "COO", "RMD",
    "PODD", "HOLX", "ALGN", "BAX", "BDX", "STE", "TECH", "CRL", "IQV", "LH", "DGX", "WAT", "ILMN",
    "A", "MOH", "CNC", "UHS", "THC", "DVA", "EXAS", "INCY", "NBIX", "BMRN", "ALNY", "SRPT", "RARE",
    "HALO", "BPMC", "ITCI", "VTRS", "ZTS", "ELAN", "RGEN", "MASI", "TNDM", "PEN", "GKOS", "IART",
    "GMED", "NVRO", "AXNX", "SWAV", "GH", "NTRA", "TXG", "PACB", "TWST", "CERT", "SAGE", "CRSP",
    "BEAM", "EDIT", "NTLA", "BLUE", "ARWR", "FOLD", "MDGL", "VKTX", "RXRX", "SDGR", "EXEL", "IONS",
    "IMVT", "MRTX", "ROIV", "DNLI", "CYTK", "CPRX", "EBS", "OPCH", "AMED", "ACHC", "LFST",
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
        merged: dict[str, SymbolInfo] = {
            item.symbol.upper(): item for item in _STOCK_UNIVERSE if item.active
        }
        for symbol in _ADDITIONAL_STOCK_SYMBOLS:
            key = symbol.upper()
            merged.setdefault(
                key,
                SymbolInfo(symbol=symbol, name=symbol, market_type="stock", exchange=None, active=True),
            )
        return list(merged.values())

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
