"""Lightweight wrappers around classic univariate time series models."""
from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import pandas as pd

from .models import regression_metrics

LOGGER = logging.getLogger(__name__)


@dataclass
class ForecastResult:
    """Container for forecast outputs and evaluation metrics."""

    model: str
    predictions: pd.Series
    metrics: Dict[str, float]


class TimeSeriesForecaster:
    """Protocol-like base class for univariate forecasters."""

    def fit(self, series: pd.Series) -> "TimeSeriesForecaster":  # pragma: no cover - interface
        raise NotImplementedError

    def forecast(self, steps: int) -> pd.Series:  # pragma: no cover - interface
        raise NotImplementedError


class ARIMAForecaster(TimeSeriesForecaster):
    """ARIMA/SARIMA wrapper with optional seasonal configuration."""

    def __init__(
        self,
        *,
        order: Sequence[int] | None = None,
        seasonal_order: Sequence[int] | None = None,
        seasonal_periods: int | None = None,
    ) -> None:
        self.order = tuple(order) if order else (1, 1, 1)
        if seasonal_order is not None and len(seasonal_order) != 4:
            raise ValueError("seasonal_order must contain four integers (P, D, Q, m)")
        self.seasonal_order = tuple(seasonal_order) if seasonal_order else None
        self.seasonal_periods = seasonal_periods
        self._result = None

    def _build_model(self, series: pd.Series):
        sarimax_module = importlib.import_module("statsmodels.tsa.statespace.sarimax")
        if self.seasonal_order is not None:
            seasonal = tuple(self.seasonal_order)
        elif self.seasonal_periods:
            seasonal = (1, 0, 1, int(self.seasonal_periods))
        else:
            seasonal = (0, 0, 0, 0)
        return sarimax_module.SARIMAX(series, order=self.order, seasonal_order=seasonal)

    def fit(self, series: pd.Series) -> "ARIMAForecaster":
        model = self._build_model(series)
        self._result = model.fit(disp=False)
        return self

    def forecast(self, steps: int) -> pd.Series:
        if self._result is None:
            raise RuntimeError("Model must be fitted before forecasting.")
        forecast = self._result.forecast(steps)
        return pd.Series(forecast)


class HoltWintersForecaster(TimeSeriesForecaster):
    """Exponential smoothing with additive trend/seasonality."""

    def __init__(
        self,
        *,
        trend: str | None = "add", 
        seasonal: str | None = "add",
        seasonal_periods: int | None = None,
    ) -> None:
        self.trend = trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self._result = None

    def fit(self, series: pd.Series) -> "HoltWintersForecaster":
        hw_module = importlib.import_module("statsmodels.tsa.holtwinters")
        model = hw_module.ExponentialSmoothing(
            series,
            trend=self.trend,
            seasonal=self.seasonal if self.seasonal_periods else None,
            seasonal_periods=self.seasonal_periods,
        )
        self._result = model.fit()
        return self

    def forecast(self, steps: int) -> pd.Series:
        if self._result is None:
            raise RuntimeError("Model must be fitted before forecasting.")
        forecast = self._result.forecast(steps)
        return pd.Series(forecast)


class ProphetForecaster(TimeSeriesForecaster):
    """Wrapper around Prophet for compatibility with the baseline interface."""

    def __init__(
        self,
        *,
        seasonality_mode: str = "additive",
        seasonal_periods: int | None = None,
        fourier_order: int = 5,
    ) -> None:
        self.seasonality_mode = seasonality_mode
        self.seasonal_periods = seasonal_periods
        self.fourier_order = fourier_order
        self._model = None
        self._train_frequency: str | None = None

    def fit(self, series: pd.Series) -> "ProphetForecaster":
        prophet_module = importlib.import_module("prophet")
        Prophet = getattr(prophet_module, "Prophet")
        self._model = Prophet(seasonality_mode=self.seasonality_mode)
        if self.seasonal_periods:
            self._model.add_seasonality(
                name="custom_seasonality",
                period=float(self.seasonal_periods),
                fourier_order=int(self.fourier_order),
            )

        frame = series.to_frame(name="y")
        frame["ds"] = frame.index
        self._train_frequency = pd.infer_freq(frame["ds"]) or "D"
        self._model.fit(frame)
        return self

    def forecast(self, steps: int) -> pd.Series:
        if self._model is None or self._train_frequency is None:
            raise RuntimeError("Model must be fitted before forecasting.")
        future = self._model.make_future_dataframe(periods=int(steps), freq=self._train_frequency, include_history=False)
        forecast = self._model.predict(future)
        return forecast["yhat"]


def build_time_series_model(
    name: str,
    *,
    seasonal_periods: int | None = None,
    params: Mapping[str, Any] | None = None,
) -> TimeSeriesForecaster:
    """Instantiate a baseline forecaster by name."""

    lower_name = name.lower()
    params = dict(params or {})
    if lower_name == "arima":
        return ARIMAForecaster(
            order=params.get("order"),
            seasonal_order=params.get("seasonal_order"),
            seasonal_periods=params.get("seasonal_periods", seasonal_periods),
        )
    if lower_name in {"holt", "holt_winters", "holt-winters", "holtwinters"}:
        return HoltWintersForecaster(
            trend=params.get("trend", "add"),
            seasonal=params.get("seasonal", "add"),
            seasonal_periods=params.get("seasonal_periods", seasonal_periods),
        )
    if lower_name == "prophet":
        return ProphetForecaster(
            seasonality_mode=params.get("seasonality_mode", "additive"),
            seasonal_periods=params.get("seasonal_periods", seasonal_periods),
            fourier_order=int(params.get("fourier_order", 5)),
        )
    raise ValueError(f"Unknown time series model '{name}'.")


def evaluate_time_series_baselines(
    train_series: pd.Series,
    test_series: pd.Series,
    baselines: Sequence[str],
    *,
    seasonal_periods: int | None = None,
    model_params: Mapping[str, Mapping[str, Any]] | None = None,
) -> Dict[str, ForecastResult]:
    """Fit and evaluate configured baseline models on a holdout split."""

    results: Dict[str, ForecastResult] = {}
    params_mapping = {key.lower(): dict(value) for key, value in (model_params or {}).items()}
    for name in baselines:
        try:
            params = params_mapping.get(name.lower(), {})
            model = build_time_series_model(name, seasonal_periods=seasonal_periods, params=params)
            model.fit(train_series)
            forecast = model.forecast(len(test_series))
            metrics = regression_metrics(test_series.to_numpy(), np.asarray(forecast))
            predictions = pd.Series(forecast, index=test_series.index)
            results[name] = ForecastResult(model=name, predictions=predictions, metrics=metrics)
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.warning("Skipping baseline '%s' due to error: %s", name, exc)
    return results
