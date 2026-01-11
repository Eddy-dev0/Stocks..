"""Lightweight ML utilities for Fabio live trading recommendations.

This module provides decision-support outputs only; it does not execute trades
or offer financial advice.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from sklearn.linear_model import SGDClassifier
import joblib

from stock_predictor.fabio_indicators import (
    calculate_cvd,
    calculate_volume_profile,
    simulate_orderflow_signals,
)


@dataclass(slots=True)
class FabioAISignal:
    """Recommendation payload from the Fabio AI model."""

    long_probability: float
    short_probability: float
    stop_loss: float | None
    target: float | None


FEATURE_COLUMNS = (
    "current_price",
    "lvn_distance_frac",
    "poc_distance_frac",
    "bars_since_bos_frac",
    "cvd_slope",
    "orderflow_direction",
    "orderflow_strength",
    "minutes_since_open",
    "range_fraction",
)


def _safe_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric):
        return None
    return numeric


def _minutes_since_us_open(timestamp: Any) -> float:
    if timestamp is None:
        return 0.0
    try:
        ts = np.datetime64(timestamp)
        if np.isnat(ts):
            return 0.0
        python_ts = np.datetime64(ts).astype("datetime64[ms]").astype(object)
    except Exception:
        return 0.0

    try:
        from zoneinfo import ZoneInfo
        import pandas as pd

        ts_pd = pd.Timestamp(python_ts)
        est = ts_pd.tz_localize("UTC") if ts_pd.tzinfo is None else ts_pd.tz_convert("UTC")
        est = est.tz_convert(ZoneInfo("America/New_York"))
        market_open = est.normalize() + pd.Timedelta(hours=9, minutes=30)
        delta = est - market_open
        return max(delta.total_seconds() / 60.0, 0.0)
    except Exception:
        return 0.0


def extract_features(
    bars: Iterable[dict[str, Any]],
    lvns: Iterable[tuple[float, float]],
    structure_info: Mapping[str, Any],
) -> dict[str, float]:
    """Extract feature values for ML-based decision support."""

    bars_list = list(bars)
    if not bars_list:
        return {name: 0.0 for name in FEATURE_COLUMNS}

    last_bar = bars_list[-1]
    current_price = _safe_float(last_bar.get("close")) or 0.0
    volume_profile = calculate_volume_profile(bars_list, bin_count=40)
    poc_price = None
    if volume_profile:
        poc_price = max(volume_profile, key=lambda item: item[1])[0]

    lvn_distance = None
    lvn_zones = list(lvns)
    if lvn_zones:
        lvn_distance = min(
            (0.0 if low <= current_price <= high else min(abs(current_price - low), abs(current_price - high)))
            for low, high in lvn_zones
        )

    lvn_distance_frac = (lvn_distance / current_price) if current_price and lvn_distance is not None else 0.0
    poc_distance_frac = (
        (current_price - poc_price) / current_price
        if current_price and poc_price is not None
        else 0.0
    )

    bos_events = structure_info.get("bos_events") if isinstance(structure_info, Mapping) else None
    last_bos_index = None
    if isinstance(bos_events, list) and bos_events:
        last_bos_index = int(bos_events[-1].get("index", len(bars_list) - 1))
    bars_since_bos = (len(bars_list) - 1) - (last_bos_index or 0)
    bars_since_bos_frac = bars_since_bos / max(len(bars_list) - 1, 1)

    cvd_values = calculate_cvd(bars_list)
    cvd_slope = 0.0
    if len(cvd_values) > 1:
        cvd_slope = (cvd_values[-1] - cvd_values[0]) / max(len(cvd_values) - 1, 1)

    orderflow_signals = simulate_orderflow_signals(bars_list, lvn_zones)
    orderflow_direction = 0.0
    orderflow_strength = 0.0
    if orderflow_signals:
        last_signal = orderflow_signals[-1]
        orderflow_direction = float(last_signal.get("direction", 0))
        orderflow_strength = float(last_signal.get("strength", 0))

    minutes_since_open = _minutes_since_us_open(last_bar.get("timestamp"))

    highs = [_safe_float(bar.get("high")) or 0.0 for bar in bars_list]
    lows = [_safe_float(bar.get("low")) or 0.0 for bar in bars_list]
    range_fraction = 0.0
    if current_price and highs and lows:
        range_fraction = (max(highs) - min(lows)) / current_price

    return {
        "current_price": float(current_price),
        "lvn_distance_frac": float(lvn_distance_frac),
        "poc_distance_frac": float(poc_distance_frac),
        "bars_since_bos_frac": float(bars_since_bos_frac),
        "cvd_slope": float(cvd_slope),
        "orderflow_direction": float(orderflow_direction),
        "orderflow_strength": float(orderflow_strength),
        "minutes_since_open": float(minutes_since_open),
        "range_fraction": float(range_fraction),
    }


class FabioAIModel:
    """Lightweight ML model for entry/exit recommendations."""

    def __init__(self, model_path: Path | None = None) -> None:
        if model_path is None:
            model_path = (
                Path(__file__).resolve().parents[1] / "models" / "fabio_ai.pkl"
            )
        self.model_path = model_path
        self.model: SGDClassifier | None = None
        self._classes = np.array([0, 1, 2], dtype=int)
        self._load_model()

    def _load_model(self) -> None:
        if self.model_path.exists():
            try:
                self.model = joblib.load(self.model_path)
            except Exception:
                self.model = None

    def _save_model(self) -> None:
        if self.model is None:
            return
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, self.model_path)

    def _create_model(self) -> SGDClassifier:
        return SGDClassifier(loss="log_loss", max_iter=1000, tol=1e-3)

    def _vectorize_features(self, features: Mapping[str, Any]) -> np.ndarray:
        values = [float(_safe_float(features.get(name)) or 0.0) for name in FEATURE_COLUMNS]
        return np.array(values, dtype=float).reshape(1, -1)

    def _prepare_training_data(
        self, training_data: Iterable[Mapping[str, Any]]
    ) -> tuple[np.ndarray, np.ndarray]:
        rows: list[np.ndarray] = []
        labels: list[int] = []
        for entry in training_data:
            feature_block = entry.get("features") if isinstance(entry, Mapping) else None
            if not isinstance(feature_block, Mapping):
                continue
            label_value = entry.get("label") if isinstance(entry, Mapping) else None
            label = self._encode_label(label_value)
            if label is None:
                continue
            rows.append(self._vectorize_features(feature_block))
            labels.append(label)
        if not rows:
            return np.zeros((0, len(FEATURE_COLUMNS))), np.zeros((0,), dtype=int)
        matrix = np.vstack(rows)
        return matrix, np.array(labels, dtype=int)

    @staticmethod
    def _encode_label(label_value: Any) -> int | None:
        if label_value is None:
            return None
        if isinstance(label_value, str):
            lowered = label_value.strip().lower()
            if lowered in {"long", "buy"}:
                return 1
            if lowered in {"short", "sell"}:
                return 2
            if lowered in {"neutral", "flat"}:
                return 0
        if isinstance(label_value, (int, float)):
            if label_value > 0:
                return 1
            if label_value < 0:
                return 2
            return 0
        return None

    def train_model(self, training_data: Iterable[Mapping[str, Any]]) -> None:
        """Train the model on labeled entry/exit data."""

        features, labels = self._prepare_training_data(training_data)
        if features.size == 0:
            return
        self.model = self._create_model()
        self.model.fit(features, labels)
        self._save_model()

    def update_model(self, features: Mapping[str, Any], outcome: Any) -> None:
        """Incrementally update the model with a new labeled outcome."""

        label = self._encode_label(outcome if not isinstance(outcome, Mapping) else outcome.get("label"))
        if label is None:
            return
        if self.model is None:
            self.model = self._create_model()
            self.model.partial_fit(self._vectorize_features(features), [label], classes=self._classes)
        else:
            self.model.partial_fit(self._vectorize_features(features), [label])
        self._save_model()

    def predict_signals(self, features: Mapping[str, Any]) -> FabioAISignal:
        """Return probabilities and recommended levels for the given features."""

        long_prob = 0.33
        short_prob = 0.33
        if self.model is not None:
            try:
                probabilities = self.model.predict_proba(self._vectorize_features(features))[0]
                class_probs = dict(zip(self.model.classes_, probabilities))
                long_prob = float(class_probs.get(1, long_prob))
                short_prob = float(class_probs.get(2, short_prob))
            except Exception:
                long_prob = 0.33
                short_prob = 0.33

        current_price = _safe_float(features.get("current_price")) or 0.0
        range_fraction = abs(_safe_float(features.get("range_fraction")) or 0.0)
        risk_unit = max(current_price * max(range_fraction, 0.002), 0.01)
        direction = 1 if long_prob >= short_prob else -1
        stop_loss = None
        target = None
        if current_price > 0:
            stop_loss = current_price - risk_unit * direction
            target = current_price + risk_unit * direction * 1.5
        return FabioAISignal(
            long_probability=long_prob,
            short_probability=short_prob,
            stop_loss=stop_loss,
            target=target,
        )
