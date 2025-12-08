"""Core analytical components for the stock predictor application."""

from stock_predictor.core.backtesting import Backtester
from stock_predictor.core.config import (
    DEFAULT_PREDICTION_HORIZONS,
    DEFAULT_PREDICTION_TARGETS,
    PredictorConfig,
    build_config,
    load_environment,
)
from stock_predictor.core.buy_zone import BuyZoneAnalyzer, BuyZoneResult, IndicatorConfirmation
from stock_predictor.core.data_pipeline import AsyncDataPipeline
from stock_predictor.core.modeling import (
    FeatureUsageSummary,
    PredictionResult,
    StockPredictorAI,
    TargetSpec,
    make_volatility_label,
)
from stock_predictor.core.training_data import TrainingDatasetBuilder
from stock_predictor.core.trend_finder import (
    DEFAULT_TREND_UNIVERSE,
    TrendFinder,
    TrendInsight,
)
from stock_predictor.core.time_series import (
    ARIMAForecaster,
    ForecastResult,
    HoltWintersForecaster,
    ProphetForecaster,
    evaluate_time_series_baselines,
)

__all__ = [
    "AsyncDataPipeline",
    "Backtester",
    "PredictorConfig",
    "StockPredictorAI",
    "TargetSpec",
    "make_volatility_label",
    "TrainingDatasetBuilder",
    "FeatureUsageSummary",
    "PredictionResult",
    "BuyZoneAnalyzer",
    "BuyZoneResult",
    "IndicatorConfirmation",
    "DEFAULT_PREDICTION_TARGETS",
    "DEFAULT_PREDICTION_HORIZONS",
    "DEFAULT_TREND_UNIVERSE",
    "build_config",
    "load_environment",
    "TrendFinder",
    "TrendInsight",
    "ARIMAForecaster",
    "HoltWintersForecaster",
    "ProphetForecaster",
    "ForecastResult",
    "evaluate_time_series_baselines",
]

# Lazy accessors for optional imports
__all__.append("BacktestingJob")


def __getattr__(name: str):  # pragma: no cover - thin convenience wrapper
    if name == "BacktestingJob":
        from stock_predictor.core.backtesting_runner import BacktestingJob

        return BacktestingJob
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
