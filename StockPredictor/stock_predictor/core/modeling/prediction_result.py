from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterator, Mapping


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


__all__ = ["FeatureUsageSummary", "PredictionResult"]
