"""Evaluation utilities for the stock predictor."""

from .backtester import Backtester, BacktestConfig, BacktestResult, SCHEMA_VERSION as BACKTEST_SCHEMA_VERSION
from .simulation_backtester import (
    SimulationBacktester,
    SimulationBacktestConfig,
    SimulationBacktestResult,
    SCHEMA_VERSION as SIMULATION_SCHEMA_VERSION,
)

__all__ = [
    "Backtester",
    "BacktestConfig",
    "BacktestResult",
    "BACKTEST_SCHEMA_VERSION",
    "SimulationBacktester",
    "SimulationBacktestConfig",
    "SimulationBacktestResult",
    "SIMULATION_SCHEMA_VERSION",
]
