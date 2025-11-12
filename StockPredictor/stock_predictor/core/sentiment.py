"""Re-export sentiment helpers for backwards compatibility."""

from __future__ import annotations

from ..providers.sentiment import aggregate_daily_sentiment, attach_sentiment

__all__ = ["aggregate_daily_sentiment", "attach_sentiment"]
