"""Stock Predictor package.

This package exposes the :class:`StockPredictorAI` class which provides a
complete pipeline for downloading data, preparing features, training a machine
learning model and generating predictions for stock prices.
"""

from .model import StockPredictorAI

__all__ = ["StockPredictorAI"]
