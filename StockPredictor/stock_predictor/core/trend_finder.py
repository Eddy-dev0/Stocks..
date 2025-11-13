"""Utilities for discovering trending opportunities across a stock universe."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .config import PredictorConfig
from .modeling import StockPredictorAI

LOGGER = logging.getLogger(__name__)


DEFAULT_TREND_UNIVERSE: tuple[str, ...] = (
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "META",
    "TSLA",
    "NVDA",
    "NFLX",
    "AMD",
    "INTC",
    "ORCL",
    "CRM",
    "ADBE",
    "CSCO",
    "SHOP",
    "JNJ",
    "PFE",
    "KO",
    "PEP",
    "XOM",
)


_TECHNICAL_TOKENS = {
    "technical",
    "indicator",
    "trend",
    "momentum",
    "oscillator",
    "volatility",
    "macro_trend",
    "macro_benchmark",
}
_FUNDAMENTAL_TOKENS = {
    "fundamental",
    "valuation",
    "macro",
    "quality",
    "growth",
    "macro_beta",
}
_SENTIMENT_TOKENS = {
    "sentiment",
    "news",
    "buzz",
    "social",
}


@dataclass(slots=True)
class TrendInsight:
    """Represents an opportunity identified by :class:`TrendFinder`."""

    ticker: str
    horizon: int
    composite_score: float
    technical_score: float | None = None
    fundamental_score: float | None = None
    sentiment_score: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_row(self) -> Mapping[str, Any]:
        """Return a dictionary representation of the insight."""

        return {
            "ticker": self.ticker,
            "horizon": self.horizon,
            "composite_score": self.composite_score,
            "technical_score": self.technical_score,
            "fundamental_score": self.fundamental_score,
            "sentiment_score": self.sentiment_score,
        }


class TrendFinder:
    """Aggregate multi-factor signals to surface trending opportunities."""

    def __init__(
        self,
        base_config: PredictorConfig,
        *,
        universe: Sequence[str] | None = None,
        ai_factory: type[StockPredictorAI] = StockPredictorAI,
    ) -> None:
        self.base_config = base_config
        self.ai_factory = ai_factory
        self._universe = self._normalise_universe(universe or DEFAULT_TREND_UNIVERSE)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def universe(self) -> tuple[str, ...]:
        return self._universe

    def set_universe(self, universe: Sequence[str]) -> None:
        self._universe = self._normalise_universe(universe)

    def update_base_config(self, config: PredictorConfig) -> None:
        self.base_config = config

    def scan(
        self,
        *,
        horizon: int | str | None = None,
        universe: Sequence[str] | None = None,
        limit: int = 5,
    ) -> list[TrendInsight]:
        """Return the strongest opportunities for ``horizon``."""

        if self.base_config is None:
            raise RuntimeError("TrendFinder requires an initial PredictorConfig")

        tickers = self._normalise_universe(universe or self._universe)
        if not tickers:
            return []

        try:
            resolved_horizon = self.base_config.resolve_horizon(
                None if horizon is None else horizon
            )
        except Exception as exc:
            LOGGER.debug("Failed to resolve horizon %s: %s", horizon, exc)
            resolved_horizon = self.base_config.prediction_horizons[0]

        insights: list[TrendInsight] = []
        for ticker in tickers:
            config = replace(self.base_config, ticker=ticker)
            try:
                ai = self.ai_factory(config, horizon=resolved_horizon)
            except Exception as exc:  # pragma: no cover - optional deps
                LOGGER.debug("Unable to create StockPredictorAI for %s: %s", ticker, exc)
                continue

            try:
                features, _, _ = ai.prepare_features()
            except Exception as exc:  # pragma: no cover - optional deps
                LOGGER.debug("Unable to prepare features for %s: %s", ticker, exc)
                continue

            metadata = getattr(ai, "metadata", {}) or {}
            latest_features = metadata.get("latest_features")
            if isinstance(latest_features, pd.DataFrame) and not latest_features.empty:
                latest_row = latest_features.iloc[0]
            elif isinstance(features, pd.DataFrame) and not features.empty:
                latest_row = features.iloc[-1]
            else:
                continue

            category_map: Mapping[str, Any] = metadata.get("feature_categories", {})
            technical_score = self._aggregate_score(latest_row, category_map, _TECHNICAL_TOKENS)
            fundamental_score = self._aggregate_score(
                latest_row, category_map, _FUNDAMENTAL_TOKENS
            )
            sentiment_score = self._aggregate_score(
                latest_row, category_map, _SENTIMENT_TOKENS
            )

            components = [
                value
                for value in (technical_score, fundamental_score, sentiment_score)
                if value is not None
            ]
            if not components:
                continue

            composite = float(np.mean(components))
            insights.append(
                TrendInsight(
                    ticker=ticker,
                    horizon=resolved_horizon,
                    composite_score=composite,
                    technical_score=technical_score,
                    fundamental_score=fundamental_score,
                    sentiment_score=sentiment_score,
                    metadata=metadata,
                )
            )

        insights.sort(key=lambda item: item.composite_score, reverse=True)
        return insights[: max(0, int(limit))]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _normalise_universe(universe: Iterable[str]) -> tuple[str, ...]:
        seen: dict[str, None] = {}
        for token in universe:
            label = str(token).strip().upper()
            if not label:
                continue
            seen[label] = None
        return tuple(seen.keys())

    @staticmethod
    def _aggregate_score(
        row: pd.Series,
        category_map: Mapping[str, Any],
        tokens: set[str],
    ) -> float | None:
        values: list[float] = []
        lower_map = {str(column).strip(): str(category).lower() for column, category in category_map.items()}
        for column, label in lower_map.items():
            if not any(token in label for token in tokens):
                continue
            if column not in row:
                continue
            value = TrendFinder._safe_float(row[column])
            if value is None:
                continue
            values.append(value)

        if not values:
            return None

        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return None
        scaled = np.tanh(arr / 10.0)
        return float(np.mean(scaled))

    @staticmethod
    def _safe_float(value: Any) -> float | None:
        if isinstance(value, pd.Series):
            value = value.dropna()
            if value.empty:
                return None
            value = value.iloc[-1]
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(numeric):
            return None
        return numeric


__all__ = ["TrendFinder", "TrendInsight", "DEFAULT_TREND_UNIVERSE"]

