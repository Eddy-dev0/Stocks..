"""Stock Predictor package.

This package exposes the :class:`StockPredictorAI` class which provides a
complete pipeline for downloading data, preparing features, training a machine
learning model and generating predictions for stock prices.
"""

from .database import Database
from .etl import MarketDataETL
from .model import StockPredictorAI

__all__ = ["StockPredictorAI", "Database", "MarketDataETL"]
