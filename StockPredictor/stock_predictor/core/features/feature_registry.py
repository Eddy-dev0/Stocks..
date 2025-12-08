"""Feature group registry definitions and helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Mapping, Sequence, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - type checking only
    import pandas as pd
    from . import FeatureBlock
from .toggles import FeatureToggles


FeatureGroupBuilder = Callable[["FeatureBuildContext"], "FeatureBuildOutput"]


@dataclass(slots=True)
class FeatureBuildContext:
    """Execution context provided to each feature group builder."""

    price_df: "pd.DataFrame"
    news_df: "pd.DataFrame | None"
    macro_df: "pd.DataFrame | None"
    sentiment_enabled: bool
    technical_indicator_config: Mapping[str, Mapping[str, object]] | None


@dataclass(slots=True)
class FeatureBuildOutput:
    """Result produced by a feature group builder."""

    blocks: list["FeatureBlock"] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)
    status: str = "executed"


@dataclass(frozen=True)
class FeatureGroupSpec:
    """Metadata describing a feature group within the registry."""

    name: str
    description: str
    builder: FeatureGroupBuilder
    dependencies: tuple[str, ...] = ()
    implemented: bool = True
    default_enabled: bool = True


class FeatureRegistryError(RuntimeError):
    """Base class for registry related errors."""


class FeatureDependencyError(FeatureRegistryError):
    """Raised when a feature group dependency is missing."""

    def __init__(self, group: str, missing: Sequence[str]) -> None:
        missing_list = ", ".join(sorted(missing))
        super().__init__(
            f"Feature group '{group}' requires the following groups to be enabled: {missing_list}."
        )
        self.group = group
        self.missing = tuple(missing)


class FeatureNotImplementedError(FeatureRegistryError):
    """Raised when an unimplemented feature group is requested."""

    def __init__(self, group: str) -> None:
        super().__init__(f"Feature group '{group}' is declared but not implemented yet.")
        self.group = group


REGISTRY_BLUEPRINT: Dict[str, dict[str, object]] = {
    "technical": {
        "description": "Classical price-driven technical indicators.",
        "dependencies": (),
        "default_enabled": True,
        "implemented": True,
    },
    "elliott": {
        "description": "Heuristic Elliott wave descriptors and swing structure markers.",
        "dependencies": (),
        "default_enabled": True,
        "implemented": True,
    },
    "macro": {
        "description": "Macro context derived from price volatility, trends, and benchmarks.",
        "dependencies": (),
        "default_enabled": True,
        "implemented": True,
    },
    "sentiment": {
        "description": "News sentiment aggregates and rolling trend indicators.",
        "dependencies": (),
        "default_enabled": True,
        "implemented": True,
    },
    "identification": {
        "description": "Pattern identification and market regime labeling (planned).",
        "dependencies": ("technical",),
        "default_enabled": False,
        "implemented": False,
    },
    "volume_liquidity": {
        "description": "Advanced volume and liquidity analytics (planned).",
        "dependencies": ("technical",),
        "default_enabled": True,
        "implemented": True,
    },
    "options": {
        "description": "Options market derived features and Greeks (planned).",
        "dependencies": ("identification",),
        "default_enabled": False,
        "implemented": False,
    },
    "esg": {
        "description": "Environmental, social, and governance signals (planned).",
        "dependencies": (),
        "default_enabled": False,
        "implemented": False,
    },
}


def build_feature_registry(**builders: FeatureGroupBuilder) -> dict[str, FeatureGroupSpec]:
    """Construct the feature registry using supplied builder callables."""

    registry: dict[str, FeatureGroupSpec] = {}
    for name, blueprint in REGISTRY_BLUEPRINT.items():
        builder = builders.get(name)
        if builder is None:
            raise KeyError(f"Missing builder function for feature group '{name}'.")
        registry[name] = FeatureGroupSpec(
            name=name,
            description=str(blueprint.get("description", "")),
            builder=builder,
            dependencies=tuple(blueprint.get("dependencies", ())),
            implemented=bool(blueprint.get("implemented", True)),
            default_enabled=bool(blueprint.get("default_enabled", False)),
        )
    return registry


def default_feature_toggles(
    registry: Mapping[str, FeatureGroupSpec] | None = None,
) -> FeatureToggles:
    """Return the default toggle map for all known feature groups."""

    defaults: dict[str, bool]
    if registry is not None:
        defaults = {name: spec.default_enabled for name, spec in registry.items()}
    else:
        defaults = {
            name: bool(blueprint.get("default_enabled", False))
            for name, blueprint in REGISTRY_BLUEPRINT.items()
        }
    return FeatureToggles.from_any(defaults)


__all__ = [
    "FeatureBuildContext",
    "FeatureBuildOutput",
    "FeatureDependencyError",
    "FeatureGroupSpec",
    "FeatureNotImplementedError",
    "FeatureRegistryError",
    "FeatureGroupBuilder",
    "REGISTRY_BLUEPRINT",
    "build_feature_registry",
    "default_feature_toggles",
]

