"""Backtesting utilities for the UI orchestration layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from stock_predictor.core import PredictorConfig
from stock_predictor.models.trainer import HorizonModelTrainer


@dataclass
class Backtester:
    """Delegate backtesting to the underlying modeling pipeline."""

    config: PredictorConfig
    trainer: HorizonModelTrainer

    def run(self, *, targets: Iterable[str] | None = None) -> dict[str, Any]:
        return self.trainer.pipeline.run_backtest(targets=targets)


__all__ = ["Backtester"]
