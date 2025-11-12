"""External data providers and integration adapters."""

from __future__ import annotations

from typing import Any

__all__ = [
    "BaseProvider",
    "DataFetcher",
    "Database",
    "DatasetType",
    "ExperimentTracker",
    "ProviderRegistry",
    "ProviderRequest",
    "ProviderResult",
    "aggregate_daily_sentiment",
    "attach_sentiment",
    "build_default_registry",
]


def __getattr__(name: str) -> Any:  # pragma: no cover - trivial passthrough
    if name == "DataFetcher":
        from stock_predictor.providers.data import DataFetcher as _DataFetcher

        return _DataFetcher
    if name == "BaseProvider":
        from stock_predictor.providers.base import BaseProvider as _BaseProvider

        return _BaseProvider
    if name == "Database":
        from stock_predictor.providers.database import Database as _Database

        return _Database
    if name == "DatasetType":
        from stock_predictor.providers.base import DatasetType as _DatasetType

        return _DatasetType
    if name == "ExperimentTracker":
        from stock_predictor.providers.database import (
            ExperimentTracker as _ExperimentTracker,
        )

        return _ExperimentTracker
    if name == "ProviderRegistry":
        from stock_predictor.providers.base import ProviderRegistry as _ProviderRegistry

        return _ProviderRegistry
    if name == "ProviderRequest":
        from stock_predictor.providers.base import ProviderRequest as _ProviderRequest

        return _ProviderRequest
    if name == "ProviderResult":
        from stock_predictor.providers.base import ProviderResult as _ProviderResult

        return _ProviderResult
    if name == "build_default_registry":
        from stock_predictor.providers.base import (
            build_default_registry as _build_default_registry,
        )

        return _build_default_registry
    if name == "aggregate_daily_sentiment":
        from stock_predictor.providers.sentiment import (
            aggregate_daily_sentiment as _aggregate_daily_sentiment,
        )

        return _aggregate_daily_sentiment
    if name == "attach_sentiment":
        from stock_predictor.providers.sentiment import (
            attach_sentiment as _attach_sentiment,
        )

        return _attach_sentiment
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
