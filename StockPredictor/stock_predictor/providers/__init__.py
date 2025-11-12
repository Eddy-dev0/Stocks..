"""External data providers and integration adapters."""

from __future__ import annotations

from typing import Any

__all__ = [
    "DataFetcher",
    "Database",
    "ExperimentTracker",
    "aggregate_daily_sentiment",
    "attach_sentiment",
]


def __getattr__(name: str) -> Any:  # pragma: no cover - trivial passthrough
    if name == "DataFetcher":
        from stock_predictor.providers.data import DataFetcher as _DataFetcher

        return _DataFetcher
    if name == "Database":
        from stock_predictor.providers.database import Database as _Database

        return _Database
    if name == "ExperimentTracker":
        from stock_predictor.providers.database import (
            ExperimentTracker as _ExperimentTracker,
        )

        return _ExperimentTracker
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
