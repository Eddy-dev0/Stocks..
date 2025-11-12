"""Expose the core ETL implementation under the provider namespace."""

from __future__ import annotations

from ..core.pipeline import MarketDataETL as _MarketDataETL
from ..core.pipeline import RefreshResult

MarketDataETL = _MarketDataETL

__all__ = ["MarketDataETL", "RefreshResult"]
