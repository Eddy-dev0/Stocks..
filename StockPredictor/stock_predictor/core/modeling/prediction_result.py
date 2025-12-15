from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterator, Literal, Mapping, MutableMapping

import pandas as pd


@dataclass
class FeatureUsageSummary:
    """Breakdown of how many signals were executed per feature group."""

    name: str
    count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PredictionResult(Mapping[str, Any]):
    """Container for structured prediction outputs and metadata."""

    predicted_close: float | None
    expected_low: float | None
    stop_loss: float | None
    probability_within_tolerance: float | None = None
    tolerance_band: float | None = None
    training_accuracy: Mapping[str, Any] | None = None
    status: str | None = None
    reason: str | None = None
    message: str | None = None
    sample_counts: Mapping[str, Any] | None = None
    missing_targets: Mapping[str, Any] | None = None
    feature_groups_used: list[str] = field(default_factory=list)
    indicators_used: list[str] = field(default_factory=list)
    feature_usage_summary: list[FeatureUsageSummary] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dictionary representation suitable for serialization."""

        feature_usage_summary = [entry.to_dict() for entry in self.feature_usage_summary]
        payload = {
            "predicted_close": self.predicted_close,
            "expected_low": self.expected_low,
            "stop_loss": self.stop_loss,
            "probability_within_tolerance": self.probability_within_tolerance,
            "tolerance_band": self.tolerance_band,
            "training_accuracy": (
                dict(self.training_accuracy) if self.training_accuracy is not None else None
            ),
            "status": self.status,
            "reason": self.reason,
            "message": self.message,
            "sample_counts": dict(self.sample_counts or {}),
            "missing_targets": dict(self.missing_targets or {}),
            "feature_groups_used": list(self.feature_groups_used),
            "indicators_used": list(self.indicators_used),
            "feature_usage_summary": feature_usage_summary,
        }
        merged = {**self.meta, **payload}
        return merged

    # Mapping protocol -------------------------------------------------
    def __getitem__(self, key: str) -> Any:  # pragma: no cover - trivial mapping wrapper
        return self.to_dict()[key]

    def __iter__(self) -> Iterator[str]:  # pragma: no cover - trivial mapping wrapper
        return iter(self.to_dict())

    def __len__(self) -> int:  # pragma: no cover - trivial mapping wrapper
        return len(self.to_dict())

    def get(self, key: str, default: Any | None = None) -> Any:
        """Fetch a value from the combined payload and metadata."""

        return self.to_dict().get(key, default)


@dataclass
class PredictionOutcome(Mapping[str, Any]):
    """Structured response produced by the multi-horizon prediction engine."""

    status: Literal["ok", "no_data", "error"]
    horizon: int
    symbol: str | None = None
    predictions: Mapping[str, Any] | None = None
    reason: str | None = None
    message: str | None = None
    sample_counts: Mapping[int, Any] | None = None
    missing_targets: Mapping[int, Any] | None = None
    checked_at: pd.Timestamp | str | None = None
    probabilities: Mapping[str, Any] | None = None
    quantile_forecasts: Mapping[str, Any] | None = None
    feature_columns: list[str] | None = None
    metrics: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dictionary representation suitable for serialization."""

        def _normalize_timestamp(value: pd.Timestamp | str | None) -> str | None:
            if value is None:
                return None
            try:
                ts = pd.to_datetime(value, errors="coerce")
            except Exception:  # pragma: no cover - defensive guard
                return None
            if pd.isna(ts):
                return None
            return pd.Timestamp(ts).isoformat()

        payload: MutableMapping[str, Any] = {
            "status": self.status,
            "horizon": self.horizon,
            "symbol": self.symbol,
            "predictions": dict(self.predictions or {}),
            "probabilities": dict(self.probabilities or {}),
            "quantile_forecasts": dict(self.quantile_forecasts or {}),
            "reason": self.reason,
            "message": self.message,
            "sample_counts": dict(self.sample_counts or {}),
            "missing_targets": dict(self.missing_targets or {}),
            "checked_at": _normalize_timestamp(self.checked_at),
            "feature_columns": list(self.feature_columns or []),
            "metrics": dict(self.metrics or {}),
        }
        return dict(payload)

    # Mapping protocol -------------------------------------------------
    def __getitem__(self, key: str) -> Any:  # pragma: no cover - trivial mapping wrapper
        return self.to_dict()[key]

    def __iter__(self) -> Iterator[str]:  # pragma: no cover - trivial mapping wrapper
        return iter(self.to_dict())

    def __len__(self) -> int:  # pragma: no cover - trivial mapping wrapper
        return len(self.to_dict())

    def get(self, key: str, default: Any | None = None) -> Any:
        """Fetch a value from the combined payload and metadata."""

        return self.to_dict().get(key, default)


__all__ = ["FeatureUsageSummary", "PredictionResult", "PredictionOutcome"]
