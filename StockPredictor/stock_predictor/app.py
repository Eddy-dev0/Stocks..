"""Top-level application orchestration for the stock predictor platform."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterable

from stock_predictor.core import (
    PredictorConfig,
    StockPredictorAI,
    build_config,
    load_environment,
)

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class RunResult:
    """Wrapper used by the application to provide consistent responses."""

    status: str
    payload: dict[str, Any]


class StockPredictorApplication:
    """Coordinate data providers, modelling pipelines, and presentation layers."""

    def __init__(self, config: PredictorConfig) -> None:
        self.config = config
        self.pipeline = StockPredictorAI(config)

    @classmethod
    def from_environment(cls, **overrides: Any) -> "StockPredictorApplication":
        """Create an application instance using environment variables and overrides."""

        load_environment()
        config = build_config(**overrides)
        LOGGER.debug("Initialised configuration for ticker %s", config.ticker)
        return cls(config)

    # ------------------------------------------------------------------
    # High level orchestration helpers
    # ------------------------------------------------------------------
    def refresh_data(self, force: bool = False) -> dict[str, Any]:
        """Refresh cached datasets via the provider layer."""

        LOGGER.info("Refreshing market data (force=%s)", force)
        return self.pipeline.download_data(force=force)

    def train(self, *, targets: Iterable[str] | None = None, horizon: int | None = None) -> dict[str, Any]:
        """Train configured models and return evaluation metrics."""

        LOGGER.info("Training models for ticker %s", self.config.ticker)
        return self.pipeline.train_model(targets=targets, horizon=horizon)

    def predict(
        self,
        *,
        targets: Iterable[str] | None = None,
        refresh: bool = False,
        horizon: int | None = None,
    ) -> dict[str, Any]:
        """Generate predictions for the configured ticker."""

        LOGGER.info("Generating predictions (refresh=%s)", refresh)
        return self.pipeline.predict(targets=targets, refresh_data=refresh, horizon=horizon)

    def backtest(self, *, targets: Iterable[str] | None = None) -> dict[str, Any]:
        """Run historical simulations to evaluate the active models."""

        LOGGER.info("Running backtest for ticker %s", self.config.ticker)
        return self.pipeline.run_backtest(targets=targets)

    def feature_importance(self, target: str, *, horizon: int | None = None) -> dict[str, Any]:
        """Return feature importance scores for the specified target."""

        LOGGER.info("Computing feature importance for target %s", target)
        return self.pipeline.feature_importance(target=target, horizon=horizon)

    def list_models(self) -> dict[str, Any]:
        """List persisted models available to the application."""

        LOGGER.debug("Listing available models from %s", self.config.models_dir)
        return self.pipeline.list_available_models()

    def run(self, mode: str, **kwargs: Any) -> RunResult:
        """Dispatch execution based on the requested mode."""

        handlers = {
            "download-data": lambda: self.refresh_data(force=kwargs.get("force", False)),
            "train": lambda: self.train(targets=kwargs.get("targets"), horizon=kwargs.get("horizon")),
            "predict": lambda: self.predict(
                targets=kwargs.get("targets"),
                refresh=kwargs.get("refresh", False),
                horizon=kwargs.get("horizon"),
            ),
            "backtest": lambda: self.backtest(targets=kwargs.get("targets")),
            "importance": lambda: self.feature_importance(
                kwargs.get("target", "close"),
                horizon=kwargs.get("horizon"),
            ),
            "list-models": self.list_models,
        }

        if mode not in handlers:
            raise ValueError(f"Unsupported application mode: {mode}")

        payload = handlers[mode]()
        return RunResult(status="ok", payload={mode.replace("-", "_"): payload})


__all__ = ["StockPredictorApplication", "RunResult"]
