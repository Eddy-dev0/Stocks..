"""Custom exceptions for the modeling package."""

from __future__ import annotations

from typing import Iterable, Mapping


class InsufficientSamplesError(ValueError):
    """Raised when there are not enough samples to proceed with training."""

    def __init__(
        self,
        message: str | None = None,
        *,
        horizons: Iterable[int],
        targets: Iterable[str],
        sample_counts: Mapping[int, Mapping[str, int]] | None = None,
        missing_targets: Mapping[int, Mapping[str, int]] | None = None,
    ) -> None:
        self.horizons = tuple(int(h) for h in horizons)
        self.targets = tuple(targets)
        self.sample_counts = (
            {int(h): {t: int(c) for t, c in counts.items()} for h, counts in sample_counts.items()}
            if sample_counts
            else None
        )
        self.missing_targets = (
            {int(h): {t: int(c) for t, c in counts.items()} for h, counts in missing_targets.items()}
            if missing_targets
            else None
        )

        details: list[str] = [message or "Insufficient samples for requested training run."]
        if self.missing_targets:
            details.append(f"Missing targets: {self.missing_targets}")
        if self.sample_counts:
            details.append(f"Sample counts: {self.sample_counts}")

        super().__init__(" ".join(details))


__all__ = ["InsufficientSamplesError"]
