"""Modeling package exposing orchestration pipeline and ensemble tools."""

from .ensembles import EnsembleRegressor, create_default_regression_ensemble
from .main import StockPredictorAI, TargetSpec, make_volatility_label
from .exceptions import InsufficientSamplesError
from .multi_horizon_engine import MultiHorizonModelingEngine
from .prediction_result import FeatureUsageSummary, PredictionResult

__all__ = [
    "StockPredictorAI",
    "TargetSpec",
    "make_volatility_label",
    "PredictionResult",
    "FeatureUsageSummary",
    "EnsembleRegressor",
    "create_default_regression_ensemble",
    "MultiHorizonModelingEngine",
    "InsufficientSamplesError",
]
