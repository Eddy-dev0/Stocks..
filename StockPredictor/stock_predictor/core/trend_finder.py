"""Utilities for discovering trending opportunities across a stock universe."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .config import PredictorConfig
from .modeling import StockPredictorAI
from .pipeline import MarketDataETL

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


_DEFAULT_WEIGHTING: dict[str, float] = {
    "prediction": 0.5,
    "sentiment": 0.25,
    "fundamental": 0.25,
}

_HORIZON_WEIGHTINGS: dict[int, dict[str, float]] = {
    1: {"prediction": 0.6, "sentiment": 0.25, "fundamental": 0.15},
    5: {"prediction": 0.55, "sentiment": 0.25, "fundamental": 0.2},
    21: {"prediction": 0.5, "sentiment": 0.2, "fundamental": 0.3},
    63: {"prediction": 0.45, "sentiment": 0.15, "fundamental": 0.4},
}


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
    confidence_rank: float | None = None
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
            "confidence_rank": self.confidence_rank,
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
        limit: int | None = None,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> list[TrendInsight]:
        """Return the strongest opportunities for ``horizon``."""

        if self.base_config is None:
            raise RuntimeError("TrendFinder requires an initial PredictorConfig")

        tickers = self._normalise_universe(universe or self._universe)
        if not tickers:
            return []

        self._prefetch_universe(tickers)

        try:
            resolved_horizon = self.base_config.resolve_horizon(
                None if horizon is None else horizon
            )
        except Exception as exc:
            LOGGER.debug("Failed to resolve horizon %s: %s", horizon, exc)
            resolved_horizon = self.base_config.prediction_horizons[0]

        insights: list[TrendInsight] = []
        total = len(tickers)
        for index, ticker in enumerate(tickers):
            progress_status = f"Scanning {ticker} ({index + 1}/{total})"
            try:
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

                ai_metadata = getattr(ai, "metadata", {}) or {}
                latest_features = ai_metadata.get("latest_features")
                if isinstance(latest_features, pd.DataFrame) and not latest_features.empty:
                    latest_row = latest_features.iloc[0]
                elif isinstance(features, pd.DataFrame) and not features.empty:
                    latest_row = features.iloc[-1]
                else:
                    continue

                category_map: Mapping[str, Any] = ai_metadata.get("feature_categories", {})
                technical_score = self._aggregate_score(
                    latest_row, category_map, _TECHNICAL_TOKENS
                )
                fundamental_score = self._aggregate_score(
                    latest_row, category_map, _FUNDAMENTAL_TOKENS
                )
                sentiment_score = self._aggregate_score(
                    latest_row, category_map, _SENTIMENT_TOKENS
                )

                try:
                    prediction_payload: Mapping[str, Any] | None = ai.predict(
                        horizon=resolved_horizon
                    )
                except Exception as exc:  # pragma: no cover - optional deps
                    LOGGER.debug(
                        "Unable to generate predictions for %s at horizon %s: %s",
                        ticker,
                        resolved_horizon,
                        exc,
                    )
                    prediction_payload = None

                prediction_score = self._prediction_score_from_payload(prediction_payload)
                confidence_details = self._confidence_details_from_payload(prediction_payload)
                confidence_rank = self._safe_float(
                    confidence_details.get("confidence_rank") if confidence_details else None
                )
                weights = self._resolve_horizon_weights(resolved_horizon)
                composite: float | None = None

                if prediction_score is not None:
                    weighted_components: dict[str, float] = {"prediction": prediction_score}
                    if fundamental_score is not None:
                        weighted_components["fundamental"] = fundamental_score
                    if sentiment_score is not None:
                        weighted_components["sentiment"] = sentiment_score
                    composite = self._weighted_composite(weighted_components, weights)
                else:
                    components = [
                        value
                        for value in (technical_score, fundamental_score, sentiment_score)
                        if value is not None
                    ]
                    if not components:
                        continue
                    composite = float(np.mean(components))

                if composite is None:
                    continue

                if isinstance(ai_metadata, Mapping):
                    insight_metadata: dict[str, Any] = dict(ai_metadata)
                else:
                    insight_metadata = {}
                previous_component_scores = insight_metadata.get("component_scores")
                if isinstance(previous_component_scores, Mapping):
                    component_scores = dict(previous_component_scores)
                else:
                    component_scores = {}
                component_scores.update(
                    {
                        "prediction": prediction_score,
                        "technical": technical_score,
                        "fundamental": fundamental_score,
                        "sentiment": sentiment_score,
                    }
                )
                insight_metadata["component_scores"] = component_scores
                insight_metadata["horizon_weights"] = dict(weights)
                if prediction_payload is not None:
                    insight_metadata["prediction_payload"] = prediction_payload
                else:
                    insight_metadata.pop("prediction_payload", None)
                if confidence_details:
                    insight_metadata["confidence"] = confidence_details
                else:
                    insight_metadata.pop("confidence", None)

                insights.append(
                    TrendInsight(
                        ticker=ticker,
                        horizon=resolved_horizon,
                        composite_score=float(composite),
                        confidence_rank=confidence_rank,
                        technical_score=technical_score,
                        fundamental_score=fundamental_score,
                        sentiment_score=sentiment_score,
                        metadata=insight_metadata,
                    )
                )
            finally:
                if progress_callback is not None:
                    try:
                        progress_callback(index + 1, total, progress_status)
                    except Exception:  # pragma: no cover - defensive guard
                        LOGGER.debug(
                            "Trend scan progress callback failed for %s", ticker, exc_info=True
                        )

        strong_insights = [
            insight for insight in insights if self._is_strong_confluence(insight)
        ]
        ranked_insights = strong_insights or insights
        ranked_insights.sort(
            key=lambda item: (
                self._safe_float(getattr(item, "confidence_rank", None))
                if getattr(item, "confidence_rank", None) is not None
                else -np.inf,
                item.composite_score,
            ),
            reverse=True,
        )

        effective_limit = 10
        if limit is not None:
            try:
                effective_limit = min(10, max(0, int(limit)))
            except (TypeError, ValueError):
                effective_limit = 10

        return ranked_insights[:effective_limit]

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

    def _prefetch_universe(self, tickers: Sequence[str]) -> None:
        for ticker in tickers:
            config = replace(self.base_config, ticker=ticker)
            etl = MarketDataETL(config)
            try:
                provider_sequence = [
                    name
                    for name in etl.preferred_price_providers()
                    if name != "yahoo_finance"
                ]
                fallback_candidates = list(provider_sequence)
                for preferred in ("alpha_vantage", "stooq"):
                    if preferred not in fallback_candidates:
                        if preferred == "alpha_vantage":
                            fallback_candidates.insert(0, preferred)
                        else:
                            fallback_candidates.append(preferred)
                deduped: list[str] = []
                for name in fallback_candidates:
                    if name not in deduped:
                        deduped.append(name)
                fallback = tuple(deduped[:3]) if deduped else None
                etl.refresh_prices(force=False, providers=fallback)
            except Exception as exc:  # pragma: no cover - defensive guard
                LOGGER.debug("Trend prefetch for %s failed: %s", ticker, exc)

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

    @staticmethod
    def _prediction_score_from_payload(
        payload: Mapping[str, Any] | None,
    ) -> float | None:
        if not isinstance(payload, Mapping):
            return None

        components: list[float] = []

        change_pct = TrendFinder._safe_float(payload.get("expected_change_pct"))
        if change_pct is not None:
            components.append(float(np.tanh(change_pct * 50.0)))

        predicted_return = TrendFinder._safe_float(payload.get("predicted_return"))
        if predicted_return is not None:
            components.append(float(np.tanh(predicted_return * 20.0)))

        up_prob = TrendFinder._safe_float(payload.get("direction_probability_up"))
        down_prob = TrendFinder._safe_float(payload.get("direction_probability_down"))
        if up_prob is not None and down_prob is not None:
            edge = up_prob - down_prob
            components.append(float(np.clip(edge, -1.0, 1.0)))
        elif up_prob is not None:
            components.append(float(np.clip(up_prob - 0.5, -1.0, 1.0)))
        elif down_prob is not None:
            components.append(float(np.clip(0.5 - down_prob, -1.0, 1.0)))

        predictions_block = payload.get("predictions")
        if isinstance(predictions_block, Mapping):
            predicted_close = TrendFinder._safe_float(predictions_block.get("close"))
            last_close = TrendFinder._safe_float(payload.get("last_close"))
            if predicted_close is not None and last_close is not None and last_close:
                pct_move = (predicted_close - last_close) / last_close
                components.append(float(np.tanh(pct_move * 50.0)))

        if not components:
            return None

        arr = np.asarray(components, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return None
        return float(np.mean(arr))

    @staticmethod
    def _confidence_details_from_payload(payload: Mapping[str, Any] | None) -> Mapping[str, Any]:
        if not isinstance(payload, Mapping):
            return {}

        details: dict[str, Any] = {}

        confluence_confidence = TrendFinder._safe_float(payload.get("confluence_confidence"))
        if confluence_confidence is not None:
            details["confluence_confidence"] = confluence_confidence

        confluence_block = payload.get("signal_confluence")
        confluence_score = None
        confluence_passed = None
        if isinstance(confluence_block, Mapping):
            confluence_score = TrendFinder._safe_float(confluence_block.get("score"))
            if confluence_score is not None:
                details["signal_confluence_score"] = confluence_score
            if "passed" in confluence_block:
                confluence_passed = bool(confluence_block.get("passed"))
                details["signal_confluence_passed"] = confluence_passed

        dir_prob_up = TrendFinder._safe_float(payload.get("direction_probability_up"))
        dir_prob_down = TrendFinder._safe_float(payload.get("direction_probability_down"))
        if dir_prob_up is not None:
            details["direction_probability_up"] = dir_prob_up
        if dir_prob_down is not None:
            details["direction_probability_down"] = dir_prob_down

        components: list[float] = []
        if confluence_confidence is not None:
            components.append(float(np.clip(confluence_confidence, 0.0, 1.0)))
        if confluence_score is not None:
            components.append(float(np.clip(confluence_score, 0.0, 1.0)))

        if dir_prob_up is not None or dir_prob_down is not None:
            direction_strength_candidates = [
                value
                for value in (dir_prob_up, dir_prob_down)
                if value is not None and np.isfinite(value)
            ]
            if direction_strength_candidates:
                direction_strength = float(max(direction_strength_candidates))
                details["direction_confidence"] = direction_strength
                components.append(float(np.clip(direction_strength, 0.0, 1.0)))

            if dir_prob_up is not None and dir_prob_down is not None:
                edge = float(np.clip(dir_prob_up - dir_prob_down, -1.0, 1.0))
                normalized_edge = 0.5 + 0.5 * edge
                details["direction_edge"] = edge
                components.append(float(np.clip(normalized_edge, 0.0, 1.0)))

        if confluence_passed is False:
            components = [value * 0.7 for value in components]

        if components:
            details["confidence_rank"] = float(np.mean(components))

        return details

    @staticmethod
    def _is_strong_confluence(insight: TrendInsight | Mapping[str, Any]) -> bool:
        if isinstance(insight, TrendInsight):
            metadata = insight.metadata
        elif isinstance(insight, Mapping):
            metadata = insight
        else:
            return False

        confidence_block = metadata.get("confidence") if isinstance(metadata, Mapping) else None
        if not isinstance(confidence_block, Mapping):
            prediction_payload = metadata.get("prediction_payload") if isinstance(metadata, Mapping) else None
            confidence_block = TrendFinder._confidence_details_from_payload(prediction_payload)

        if not confidence_block:
            return False

        threshold = 0.6
        rank = TrendFinder._safe_float(confidence_block.get("confidence_rank"))
        confluence_confidence = TrendFinder._safe_float(
            confidence_block.get("confluence_confidence")
        )
        confluence_score = TrendFinder._safe_float(
            confidence_block.get("signal_confluence_score")
        )
        passed = confidence_block.get("signal_confluence_passed")

        if isinstance(passed, bool) and passed:
            return True
        if confluence_confidence is not None and confluence_confidence >= threshold:
            return True
        if confluence_score is not None and confluence_score >= threshold:
            return True
        if rank is not None and rank >= threshold:
            return True

        return False

    @staticmethod
    def _resolve_horizon_weights(horizon: int) -> Mapping[str, float]:
        try:
            horizon_key = int(horizon)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            return dict(_DEFAULT_WEIGHTING)
        weights = _HORIZON_WEIGHTINGS.get(horizon_key, _DEFAULT_WEIGHTING)
        return dict(weights)

    @staticmethod
    def _weighted_composite(
        scores: Mapping[str, float], weights: Mapping[str, float]
    ) -> float | None:
        total_weight = 0.0
        weighted_sum = 0.0
        fallback_values: list[float] = []

        for key, score in scores.items():
            if score is None or not np.isfinite(score):
                continue
            try:
                weight_value = float(weights.get(key, 0.0))
            except (TypeError, ValueError):
                weight_value = 0.0
            if weight_value <= 0:
                fallback_values.append(float(score))
                continue
            weighted_sum += float(score) * weight_value
            total_weight += weight_value

        if total_weight > 0:
            return float(weighted_sum / total_weight)

        if fallback_values:
            return float(np.mean(fallback_values))

        return None


__all__ = ["TrendFinder", "TrendInsight", "DEFAULT_TREND_UNIVERSE"]

