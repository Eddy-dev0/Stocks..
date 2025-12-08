"""Model lifecycle helpers for horizon-specific training and inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from stock_predictor.core import PredictorConfig, PredictionResult, StockPredictorAI
from stock_predictor.features.engineer import IndicatorFeatureEngineer


@dataclass
class HorizonModelTrainer:
    """Train and serve models for the configured ticker and horizon."""

    config: PredictorConfig
    feature_engineer: IndicatorFeatureEngineer

    def __post_init__(self) -> None:
        self.pipeline = StockPredictorAI(self.config)

    def train(self, *, targets: Iterable[str] | None = None, horizon: int | None = None) -> dict[str, Any]:
        """Train models for the requested targets and horizon."""

        return self.pipeline.train_model(targets=targets, horizon=horizon)

    def predict(
        self,
        *,
        targets: Iterable[str] | None = None,
        refresh: bool = False,
        horizon: int | None = None,
    ) -> PredictionResult:
        """Run inference using cached or refreshed features."""

        if refresh:
            self.feature_engineer.build_features(force_refresh=True)
        return self.pipeline.predict(targets=targets, refresh_data=refresh, horizon=horizon)

    def list_models(self) -> dict[str, Any]:
        """Enumerate stored models on disk."""

        return self.pipeline.list_available_models()


__all__ = ["HorizonModelTrainer"]
