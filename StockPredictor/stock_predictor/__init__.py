"""Modern, modular stock prediction toolkit."""

from stock_predictor.app import StockPredictorApplication
from stock_predictor.core import (
    PredictorConfig,
    StockPredictorAI,
    build_config,
    load_environment,
)
from stock_predictor.research import (
    WaveSegment,
    apply_wave_features,
    detect_elliott_waves,
)

__all__ = [
    "PredictorConfig",
    "StockPredictorAI",
    "StockPredictorApplication",
    "WaveSegment",
    "apply_wave_features",
    "detect_elliott_waves",
    "build_config",
    "load_environment",
]
