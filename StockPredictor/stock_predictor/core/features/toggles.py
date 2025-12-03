"""Dataclass helpers for feature toggle handling."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, MutableMapping
from dataclasses import asdict, dataclass, field
from typing import Any, Iterator


_FEATURE_FIELDS = (
    "elliott",
    "macro",
    "technical",
    "fundamental",
    "sentiment",
    "volume_liquidity",
)


def _normalise_key(key: str) -> str:
    name = str(key).strip().lower()
    if not name:
        raise KeyError("Feature toggle name cannot be empty.")
    return name


@dataclass(slots=True)
class FeatureToggles(MutableMapping[str, bool]):
    """Structured feature toggle container with mapping-like behaviour."""

    elliott: bool = True
    macro: bool = True
    technical: bool = True
    fundamental: bool = True
    sentiment: bool = True
    volume_liquidity: bool = True
    extras: dict[str, bool] = field(default_factory=dict)

    @classmethod
    def from_any(
        cls,
        toggles: FeatureToggles | Mapping[str, Any] | Iterable[str] | str | None,
        *,
        defaults: Mapping[str, bool] | None = None,
    ) -> "FeatureToggles":
        base_fields: dict[str, bool] = {name: True for name in _FEATURE_FIELDS}
        extras: dict[str, bool] = {}

        if defaults is not None:
            for key, value in defaults.items():
                name = str(key).strip().lower()
                if not name:
                    continue
                if name in base_fields:
                    base_fields[name] = bool(value)
                else:
                    extras[name] = bool(value)

        if toggles is None:
            return cls(**base_fields, extras=extras)

        if isinstance(toggles, FeatureToggles):
            merged_extras = {**extras, **toggles.extras}
            return cls(
                **{field: bool(getattr(toggles, field, base_fields[field])) for field in _FEATURE_FIELDS},
                extras=merged_extras,
            )

        if isinstance(toggles, Mapping):
            entries = toggles.items()
        else:
            if isinstance(toggles, str):
                entries = ((token, True) for token in toggles.split(","))
            else:
                entries = ((token, True) for token in toggles)

        resolved = dict(base_fields)
        extra_flags = dict(extras)
        for key, value in entries:
            name = str(key).strip().lower()
            if not name:
                continue
            if name in resolved:
                resolved[name] = bool(value)
            else:
                extra_flags[name] = bool(value)
        return cls(**resolved, extras=extra_flags)

    # Mapping interface -------------------------------------------------
    def __getitem__(self, key: str) -> bool:
        name = _normalise_key(key)
        if name in _FEATURE_FIELDS:
            return getattr(self, name)
        if name in self.extras:
            return self.extras[name]
        raise KeyError(name)

    def __setitem__(self, key: str, value: Any) -> None:
        name = _normalise_key(key)
        flag = bool(value)
        if name in _FEATURE_FIELDS:
            setattr(self, name, flag)
        else:
            self.extras[name] = flag

    def __delitem__(self, key: str) -> None:  # pragma: no cover - defensive
        name = _normalise_key(key)
        if name in _FEATURE_FIELDS:
            raise KeyError(f"Cannot delete required feature toggle '{name}'.")
        self.extras.pop(name, None)

    def __iter__(self) -> Iterator[str]:
        yield from _FEATURE_FIELDS
        yield from self.extras

    def __len__(self) -> int:
        return len(_FEATURE_FIELDS) + len(self.extras)

    # Convenience helpers ----------------------------------------------
    def asdict(self) -> dict[str, bool]:
        data = asdict(self)
        extras = data.pop("extras", {}) or {}
        merged: dict[str, bool] = {key: bool(value) for key, value in data.items()}
        merged.update({str(key): bool(value) for key, value in extras.items()})
        return merged

    def copy(self) -> "FeatureToggles":
        return FeatureToggles.from_any(self)

    def update(self, other: Mapping[str, Any] | Iterable[tuple[str, Any]] | None = None, **kwargs: Any) -> None:  # type: ignore[override]
        if other is not None:
            if isinstance(other, Mapping):
                items = other.items()
            else:
                items = other
            for key, value in items:
                self[key] = value
        if kwargs:
            for key, value in kwargs.items():
                self[key] = value

    def items(self):  # type: ignore[override]
        return self.asdict().items()


__all__ = ["FeatureToggles"]
