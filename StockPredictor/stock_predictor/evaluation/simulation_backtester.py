"""Monte Carlo-style backtesting for trained models."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
import logging
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

from stock_predictor.core import PredictorConfig
from stock_predictor.core.training_data import TrainingDataset, TrainingDatasetBuilder


LOGGER = logging.getLogger(__name__)

SCHEMA_VERSION = "simulation_backtest/1.0"


@dataclass(slots=True)
class SimulationBacktestConfig:
    """Configuration for simulation-heavy backtesting."""

    simulations: int = 900_000_000
    batch_size: int = 100_000
    window_size: int = 30
    horizon: int | None = None
    targets: tuple[str, ...] | None = None
    tolerance: float | None = None
    random_seed: int | None = None
    parallelism: int = 1
    output_dir: Path | None = None
    schema_version: str = SCHEMA_VERSION


@dataclass(slots=True)
class SimulationBacktestResult:
    """Structured representation of simulation backtest outcomes."""

    config: SimulationBacktestConfig
    results: dict[str, dict[str, float]]
    summary: dict[str, Any]
    outputs: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": self.config.schema_version,
            "config": self._serialise_config(),
            "results": self.results,
            "summary": self.summary,
            "outputs": self.outputs,
        }
        return payload

    def _serialise_config(self) -> dict[str, Any]:
        data = self.config.__dict__.copy()
        for key, value in list(data.items()):
            if isinstance(value, Path):
                data[key] = str(value)
        return data


class SimulationBacktester:
    """Perform repeated simulation backtests using historical feature snapshots."""

    def __init__(self, config: PredictorConfig, trainer: Any) -> None:
        self.config = config
        self.trainer = trainer
        self.dataset_builder = TrainingDatasetBuilder(config)

    def run(
        self,
        *,
        targets: Iterable[str] | None = None,
        horizon: int | None = None,
        sim_config: SimulationBacktestConfig | None = None,
    ) -> SimulationBacktestResult:
        cfg = self._resolve_config(sim_config, targets, horizon)
        dataset = self._prepare_dataset(cfg)
        results: dict[str, dict[str, float]] = {}
        totals: dict[str, int] = {}
        correct: dict[str, int] = {}

        for target in cfg.targets or ():
            outcome = self._simulate_target(target, dataset, cfg)
            if outcome is None:
                continue
            results[target] = outcome
            totals[target] = int(outcome["total_simulations"])
            correct[target] = int(outcome["correct_predictions"])

        total_simulations = int(sum(totals.values()))
        total_correct = int(sum(correct.values()))
        overall_success_rate = (
            float(total_correct) / float(total_simulations) if total_simulations else 0.0
        )

        summary = {
            "total_simulations": total_simulations,
            "total_correct": total_correct,
            "overall_success_rate": overall_success_rate,
        }

        result = SimulationBacktestResult(config=cfg, results=results, summary=summary)
        self._persist_outputs(result, cfg)
        return result

    def _resolve_config(
        self,
        sim_config: SimulationBacktestConfig | None,
        targets: Iterable[str] | None,
        horizon: int | None,
    ) -> SimulationBacktestConfig:
        resolved = sim_config or SimulationBacktestConfig()
        resolved_targets = tuple(targets) if targets is not None else resolved.targets
        if resolved_targets is None:
            resolved_targets = tuple(self.config.prediction_targets)
        resolved = replace(resolved, targets=tuple(resolved_targets))
        if resolved.horizon is None:
            resolved = replace(
                resolved,
                horizon=horizon
                if horizon is not None
                else int(self.config.prediction_horizons[0]),
            )
        if resolved.output_dir is None:
            resolved = replace(
                resolved,
                output_dir=(self.config.data_dir / "reports" / "simulation_backtests"),
            )
        return resolved

    def _prepare_dataset(self, cfg: SimulationBacktestConfig) -> TrainingDataset:
        force = not bool(getattr(self.config, "use_cached_training_data", True))
        return self.dataset_builder.build(force=force)

    def _simulate_target(
        self,
        target: str,
        dataset: TrainingDataset,
        cfg: SimulationBacktestConfig,
    ) -> dict[str, float] | None:
        horizon = int(cfg.horizon or self.config.prediction_horizons[0])
        target_series = dataset.targets.get(horizon, {}).get(target)
        if target_series is None:
            LOGGER.warning("Skipping simulation backtest for missing target %s.", target)
            return None

        features = dataset.features
        aligned_features, aligned_target = self._align_target(features, target_series)
        if aligned_target.empty:
            LOGGER.warning("No aligned target samples for %s; skipping.", target)
            return None

        pipeline = dataset.preprocessors.get(horizon)
        model = self.trainer.models.get((target, horizon)) if self.trainer is not None else None
        if model is None:
            model = self.trainer.load_model(target, horizon)
        if model is None:
            LOGGER.warning("Unable to load model for target %s horizon %s.", target, horizon)
            return None

        task = "classification" if target == "direction" else "regression"
        tolerance = cfg.tolerance
        if tolerance is None:
            tolerance = self.config.resolve_tolerance_band(horizon) or 0.0

        eligible_positions = self._eligible_positions(aligned_target, cfg.window_size)
        if eligible_positions.size == 0:
            LOGGER.warning("Insufficient samples for simulation backtest on %s.", target)
            return None

        simulations = max(0, int(cfg.simulations))
        batch_size = max(1, int(cfg.batch_size))
        rng = np.random.default_rng(cfg.random_seed)

        correct = 0
        total = 0

        if cfg.parallelism > 1:
            correct, total = self._run_parallel_batches(
                simulations,
                batch_size,
                cfg.parallelism,
                rng,
                eligible_positions,
                aligned_features,
                aligned_target,
                pipeline,
                model,
                task,
                tolerance,
            )
        else:
            remaining = simulations
            while remaining > 0:
                current = min(batch_size, remaining)
                positions = rng.choice(eligible_positions, size=current, replace=True)
                batch_correct, batch_total = self._score_batch(
                    positions,
                    aligned_features,
                    aligned_target,
                    pipeline,
                    model,
                    task,
                    tolerance,
                )
                correct += batch_correct
                total += batch_total
                remaining -= current

        success_rate = float(correct) / float(total) if total else 0.0
        return {
            "success_rate": success_rate,
            "correct_predictions": float(correct),
            "total_simulations": float(total),
            "tolerance": float(tolerance),
        }

    def _run_parallel_batches(
        self,
        simulations: int,
        batch_size: int,
        parallelism: int,
        rng: np.random.Generator,
        eligible_positions: np.ndarray,
        features: pd.DataFrame,
        target: pd.Series,
        pipeline: Any,
        model: Any,
        task: str,
        tolerance: float,
    ) -> tuple[int, int]:
        remaining = simulations
        correct = 0
        total = 0
        with ThreadPoolExecutor(max_workers=parallelism) as executor:
            futures = []
            while remaining > 0 or futures:
                while remaining > 0 and len(futures) < parallelism:
                    current = min(batch_size, remaining)
                    positions = rng.choice(eligible_positions, size=current, replace=True)
                    futures.append(
                        executor.submit(
                            self._score_batch,
                            positions,
                            features,
                            target,
                            pipeline,
                            model,
                            task,
                            tolerance,
                        )
                    )
                    remaining -= current
                if futures:
                    for future in as_completed(futures, timeout=None):
                        batch_correct, batch_total = future.result()
                        correct += batch_correct
                        total += batch_total
                        futures.remove(future)
                        break
        return correct, total

    @staticmethod
    def _eligible_positions(target: pd.Series, window_size: int) -> np.ndarray:
        usable = target.dropna()
        if usable.empty:
            return np.array([], dtype=int)
        start = max(int(window_size) - 1, 0)
        return np.arange(start, len(usable), dtype=int)

    @staticmethod
    def _align_target(
        features: pd.DataFrame, target_series: pd.Series
    ) -> tuple[pd.DataFrame, pd.Series]:
        sorted_index = pd.DatetimeIndex(features.index).sort_values()
        aligned_features = features.loc[sorted_index]
        aligned_target = target_series.reindex(sorted_index).dropna()
        aligned_features = aligned_features.loc[aligned_target.index]
        return aligned_features, aligned_target

    @staticmethod
    def _score_batch(
        positions: np.ndarray,
        features: pd.DataFrame,
        target: pd.Series,
        pipeline: Any,
        model: Any,
        task: str,
        tolerance: float,
    ) -> tuple[int, int]:
        X_batch = features.iloc[positions]
        y_batch = target.iloc[positions].to_numpy()
        if pipeline is not None:
            X_batch = pipeline.transform(X_batch)
        y_pred = model.predict(X_batch)
        if task == "classification":
            correct = int(np.sum(y_pred == y_batch))
        else:
            correct = int(np.sum(np.abs(y_pred - y_batch) <= tolerance))
        return correct, int(len(y_batch))

    def _persist_outputs(self, result: SimulationBacktestResult, cfg: SimulationBacktestConfig) -> None:
        output_dir = cfg.output_dir or (self.config.data_dir / "reports" / "simulation_backtests")
        versioned_dir = output_dir / cfg.schema_version.replace("/", "_")
        versioned_dir.mkdir(parents=True, exist_ok=True)
        json_path = versioned_dir / "simulation_backtest.json"

        with json_path.open("w", encoding="utf-8") as handle:
            json.dump(result.as_dict(), handle, indent=2, default=str)

        result.outputs["json"] = str(json_path)


__all__ = ["SimulationBacktester", "SimulationBacktestConfig", "SimulationBacktestResult", "SCHEMA_VERSION"]
