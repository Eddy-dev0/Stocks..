"""Model lifecycle helpers for horizon-specific training and inference."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterable

from stock_predictor.core import PredictorConfig, PredictionResult, StockPredictorAI
from stock_predictor.features.engineer import IndicatorFeatureEngineer

LOGGER = logging.getLogger(__name__)


@dataclass
class HorizonModelTrainer:
    """Train and serve models for the configured ticker and horizon."""

    config: PredictorConfig
    feature_engineer: IndicatorFeatureEngineer

    def __post_init__(self) -> None:
        self.pipeline = StockPredictorAI(self.config)

    def train(self, *, targets: Iterable[str] | None = None, horizon: int | None = None) -> dict[str, Any]:
        """Train models for the requested targets and horizon."""

        self.pipeline._refresh_feature_assembler()
        report = self.pipeline.train_model(targets=targets, horizon=horizon)
        try:
            backtest_payload = self.pipeline.run_backtest(targets=targets, horizon=horizon)
            report["backtest_summary"] = backtest_payload.get("summary", {})
            report["backtest_success_rates"] = self.pipeline.metadata.get(
                "backtest_success_rates", {}
            )
        except Exception as exc:  # pragma: no cover - backtest depends on data availability
            LOGGER.warning("Backtest run failed after training: %s", exc)
        return report

    def predict(
        self,
        *,
        targets: Iterable[str] | None = None,
        refresh: bool = False,
        horizon: int | None = None,
    ) -> PredictionResult:
        """Run inference using cached or refreshed features."""

        self.pipeline._refresh_feature_assembler()
        if refresh:
            self.feature_engineer.build_features(force_refresh=True)
        return self.pipeline.predict(targets=targets, refresh_data=refresh, horizon=horizon)

    def list_models(self) -> dict[str, Any]:
        """Enumerate stored models on disk."""

        return self.pipeline.list_available_models()


__all__ = ["HorizonModelTrainer"]
