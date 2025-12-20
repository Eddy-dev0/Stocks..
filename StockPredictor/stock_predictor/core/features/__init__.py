"""Feature engineering utilities for the stock predictor."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Dict, Iterable, Iterator, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from ..indicator_bundle import (
    DEFAULT_INDICATOR_CONFIG,
    compute_indicators,
    compute_multi_timeframe_trends,
)
from ..fear_greed import compute_fear_greed_features
from ..sentiment import aggregate_daily_sentiment, attach_sentiment
from .feature_registry import (
    FeatureBuildContext,
    FeatureBuildOutput,
    FeatureDependencyError,
    FeatureGroupSpec,
    FeatureNotImplementedError,
    build_feature_registry,
    default_feature_toggles,
)
from .toggles import FeatureToggles


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class FeatureResult:
    """Container holding processed features and metadata."""

    features: pd.DataFrame
    targets: Dict[int, Dict[str, pd.Series]]
    metadata: Dict[str, object]


@dataclass(slots=True)
class FeatureBlock:
    """Bundle of features that belong to the same semantic category."""

    frame: pd.DataFrame
    category: str
    column_categories: Optional[Dict[str, str]] = None

    def iter_categories(self) -> Iterator[tuple[str, str]]:
        mapping = self.column_categories
        if mapping is None:
            mapping = {col: self.category for col in self.frame.columns if col != "Date"}
        return ((column, category) for column, category in mapping.items() if column != "Date")


def _not_implemented_group_builder(name: str):
    def _builder(_: FeatureBuildContext) -> FeatureBuildOutput:
        raise NotImplementedError(f"Feature group '{name}' is not implemented yet.")

    return _builder


def _technical_group_builder(context: FeatureBuildContext) -> FeatureBuildOutput:
    block, metadata = _build_technical_features(
        context.price_df,
        indicator_config=context.technical_indicator_config,
    )
    blocks = [block] if block is not None else []
    status = "executed" if blocks else "skipped_no_data"
    return FeatureBuildOutput(blocks=blocks, metadata=metadata, status=status)


def _elliott_group_builder(context: FeatureBuildContext) -> FeatureBuildOutput:
    block = _build_elliott_wave_descriptors(context.price_df)
    blocks = [block] if block is not None else []
    status = "executed" if blocks else "skipped_no_data"
    return FeatureBuildOutput(blocks=blocks, status=status)

def _macro_group_builder(context: FeatureBuildContext) -> FeatureBuildOutput:
    blocks = [
        block
        for block in _build_macro_blocks(
            context.price_df,
            macro_df=context.macro_df,
            event_params=context.event_params,
        )
        if block is not None
    ]
    status = "executed" if blocks else "skipped_no_data"
    return FeatureBuildOutput(blocks=blocks, status=status)


def _regime_group_builder(context: FeatureBuildContext) -> FeatureBuildOutput:
    block = _build_regime_block(
        context.price_df,
        regime_params=context.regime_params,
    )
    blocks = [block] if block is not None else []
    status = "executed" if blocks else "skipped_no_data"
    return FeatureBuildOutput(blocks=blocks, status=status)


def _volume_liquidity_group_builder(context: FeatureBuildContext) -> FeatureBuildOutput:
    block = _build_volume_liquidity_block(context.price_df)
    if block is None:
        return FeatureBuildOutput(blocks=[], status="missing_data")
    return FeatureBuildOutput(blocks=[block], status="executed")


def _sentiment_group_builder(context: FeatureBuildContext) -> FeatureBuildOutput:
    metadata: Dict[str, object] = {}
    if not context.sentiment_enabled:
        metadata["sentiment_daily"] = _empty_sentiment_frame()
        return FeatureBuildOutput(blocks=[], metadata=metadata, status="global_disabled")

    news_df = context.news_df
    if news_df is None or news_df.empty:
        metadata["sentiment_daily"] = _empty_sentiment_frame()
        return FeatureBuildOutput(blocks=[], metadata=metadata, status="missing_data")

    block = _build_sentiment_features(news_df)
    if block is None:
        metadata["sentiment_daily"] = _empty_sentiment_frame()
        return FeatureBuildOutput(blocks=[], metadata=metadata, status="missing_data")

    metadata["sentiment_daily"] = block.frame
    return FeatureBuildOutput(blocks=[block], metadata=metadata, status="executed")


FEATURE_REGISTRY: Dict[str, FeatureGroupSpec] = build_feature_registry(
    technical=_technical_group_builder,
    elliott=_elliott_group_builder,
    macro=_macro_group_builder,
    regime=_regime_group_builder,
    sentiment=_sentiment_group_builder,
    identification=_not_implemented_group_builder("identification"),
    volume_liquidity=_volume_liquidity_group_builder,
    options=_not_implemented_group_builder("options"),
    esg=_not_implemented_group_builder("esg"),
)


class FeatureAssembler:
    """Build enriched feature sets driven by the feature registry."""

    def __init__(
        self,
        feature_toggles: FeatureToggles | Mapping[str, bool] | Iterable[str] | None,
        horizons: Iterable[int] | None = None,
        *,
        registry: Mapping[str, FeatureGroupSpec] | None = None,
        technical_indicator_config: Mapping[str, Mapping[str, object]] | None = None,
        regime_params: Mapping[str, float] | None = None,
        event_params: Mapping[str, object] | None = None,
        direction_params: Mapping[str, float] | None = None,
    ) -> None:
        self.registry: Dict[str, FeatureGroupSpec] = dict(registry or FEATURE_REGISTRY)
        if not self.registry:
            raise ValueError("Feature registry cannot be empty.")

        self.technical_indicator_config = self._build_indicator_config(technical_indicator_config)

        self.feature_toggles = self._normalise_toggles(feature_toggles)
        self.enabled_groups = [name for name, enabled in self.feature_toggles.items() if enabled]
        if not self.enabled_groups:
            if "technical" in self.feature_toggles:
                self.feature_toggles["technical"] = True
                self.enabled_groups = ["technical"]
            else:
                first = next(iter(self.registry))
                self.feature_toggles[first] = True
                self.enabled_groups = [first]

        if horizons is None:
            horizons = (1,)
        self.horizons = _normalise_horizons(horizons)
        self.regime_params = dict(regime_params or {})
        self.event_params = dict(event_params or {})
        self.direction_params = dict(direction_params or {})

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def build(
        self,
        price_df: pd.DataFrame,
        news_df: pd.DataFrame | None,
        sentiment_enabled: bool,
        *,
        macro_df: pd.DataFrame | None = None,
    ) -> FeatureResult:
        if price_df.empty:
            raise ValueError("Price dataframe cannot be empty when building features.")

        processed = _ensure_datetime_index(price_df)
        context_news = None
        if news_df is not None:
            context_news = news_df.copy()
        context_macro = None
        if macro_df is not None:
            context_macro = macro_df.copy()

        context = FeatureBuildContext(
            price_df=processed,
            news_df=context_news,
            macro_df=context_macro,
            sentiment_enabled=sentiment_enabled,
            technical_indicator_config=self.technical_indicator_config,
            regime_params=self.regime_params,
            event_params=self.event_params,
        )

        feature_blocks: list[FeatureBlock] = []
        metadata: Dict[str, object] = {}
        feature_categories: Dict[str, str] = {}

        metadata["feature_toggles"] = self.feature_toggles.asdict()
        metadata["enabled_feature_groups"] = list(self.enabled_groups)

        group_metadata: dict[str, dict[str, object]] = {
            name: {
                "configured": bool(self.feature_toggles.get(name, False)),
                "executed": False,
                "dependencies": list(spec.dependencies),
                "implemented": spec.implemented,
                "description": spec.description,
                "columns": [],
                "categories": set(),
                "status": "configured" if self.feature_toggles.get(name, False) else "disabled",
            }
            for name, spec in self.registry.items()
        }

        execution_plan = self._resolve_execution_plan()

        for spec in execution_plan:
            summary = group_metadata[spec.name]
            if not spec.implemented:
                message = (
                    f"Feature group '{spec.name}' is declared but not implemented. Skipping."
                )
                logger.warning(message)
                summary["status"] = "unimplemented"
                summary["executed"] = False
                metadata.setdefault("warnings", []).append(message)
                continue

            if spec.name == "sentiment" and not sentiment_enabled:
                output = FeatureBuildOutput(
                    blocks=[],
                    metadata={"sentiment_daily": _empty_sentiment_frame()},
                    status="global_disabled",
                )
            else:
                try:
                    output = spec.builder(context)
                except NotImplementedError as exc:  # pragma: no cover - defensive
                    raise FeatureNotImplementedError(spec.name) from exc

            if output is None:
                output = FeatureBuildOutput(blocks=[], status="skipped_no_data")

            blocks = [block for block in output.blocks if block is not None]

            if spec.name == "sentiment" and "sentiment_daily" not in output.metadata:
                if blocks:
                    output.metadata["sentiment_daily"] = blocks[0].frame
                else:
                    output.metadata["sentiment_daily"] = _empty_sentiment_frame()

            for block in blocks:
                feature_blocks.append(block)
                pairs = list(block.iter_categories())
                summary["columns"].extend([column for column, _ in pairs])
                summary["categories"].update(category for _, category in pairs)
                for column, category in pairs:
                    feature_categories[column] = category

            summary["status"] = output.status
            summary["executed"] = bool(blocks)
            metadata.update(output.metadata)

        if "sentiment_daily" not in metadata:
            metadata["sentiment_daily"] = _empty_sentiment_frame()

        if not feature_blocks:
            raise RuntimeError("No feature blocks were generated. Check configuration.")

        merged = processed.reset_index(drop=True)[["Date", "Close"]].copy()
        for block in feature_blocks:
            merged = merged.merge(block.frame, on="Date", how="left")

        merged = merged.reset_index(drop=True).sort_values("Date").reset_index(drop=True)
        merged = merged.replace([np.inf, -np.inf], np.nan)
        merged = merged.ffill().bfill()
        numeric_cols = merged.select_dtypes(include=[np.number]).columns

        beta_columns = [
            column
            for column in numeric_cols
            if feature_categories.get(column) == "macro_beta" or column.startswith("Beta_")
        ]
        fill_columns = [column for column in numeric_cols if column not in beta_columns]
        merged[fill_columns] = merged[fill_columns].fillna(0.0)

        merged = merged.assign(Close_Current=merged["Close"])

        targets = _generate_targets(
            merged,
            self.horizons,
            direction_neutral_threshold=self.direction_params.get("neutral_threshold"),
        )
        metadata["target_validation"] = _validate_target_alignment(
            merged, targets, self.horizons
        )

        feature_columns = [col for col in merged.columns if col not in {"Date", "Close"}]
        metadata.update(
            {
                "feature_columns": feature_columns,
                "latest_features": merged.iloc[[-1]][feature_columns],
                "latest_close": float(merged.iloc[-1]["Close"]),
                "latest_date": pd.to_datetime(merged.iloc[-1]["Date"]),
                "horizons": self.horizons,
                "target_dates": _estimate_target_dates(merged["Date"], self.horizons),
                "feature_categories": feature_categories,
                "target_kind": "pct_return",
                "target_variants": ("return", "log_return"),
            }
        )

        for summary in group_metadata.values():
            columns = summary["columns"]
            summary["columns"] = sorted(dict.fromkeys(columns))
            categories_set = summary.pop("categories")
            summary["categories"] = sorted(categories_set)
            if not summary["configured"]:
                summary["status"] = "disabled"
            elif not summary["executed"] and summary["status"] in {"configured", "executed"}:
                summary["status"] = "skipped_no_data"

        metadata["feature_groups"] = {
            name: summary for name, summary in group_metadata.items()
        }
        metadata["executed_feature_groups"] = [
            name for name, summary in group_metadata.items() if summary.get("executed")
        ]

        features = merged[feature_columns]
        return FeatureResult(features=features, targets=targets, metadata=metadata)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _normalise_toggles(
        self,
        toggles: FeatureToggles | Mapping[str, bool] | Iterable[str] | None,
    ) -> FeatureToggles:
        defaults = default_feature_toggles(self.registry)
        return FeatureToggles.from_any(toggles, defaults=defaults.asdict())

    @staticmethod
    def _build_indicator_config(
        overrides: Mapping[str, Mapping[str, object]] | None,
    ) -> dict[str, dict[str, object]]:
        config = {
            name: dict(DEFAULT_INDICATOR_CONFIG[name]) for name in DEFAULT_INDICATOR_CONFIG
        }
        if overrides:
            for name, params in overrides.items():
                if name not in config:
                    continue
                config[name].update(params)
        return config

    def _resolve_execution_plan(self) -> list[FeatureGroupSpec]:
        plan: list[FeatureGroupSpec] = []
        added: set[str] = set()
        visiting: set[str] = set()

        def visit(name: str) -> None:
            if name not in self.registry:
                raise KeyError(f"Unknown feature group '{name}'.")
            if name in added:
                return
            if name in visiting:
                raise RuntimeError(
                    f"Circular dependency detected for feature group '{name}'."
                )

            spec = self.registry[name]
            visiting.add(name)
            missing = [dep for dep in spec.dependencies if not self.feature_toggles.get(dep, False)]
            if missing:
                raise FeatureDependencyError(spec.name, missing)
            for dependency in spec.dependencies:
                visit(dependency)
            visiting.remove(name)

            if self.feature_toggles.get(spec.name, False):
                added.add(spec.name)
                plan.append(spec)

        for name in self.enabled_groups:
            visit(name)

        ordered: list[FeatureGroupSpec] = []
        seen: set[str] = set()
        for spec in plan:
            if spec.name not in seen:
                ordered.append(spec)
                seen.add(spec.name)
        return ordered


# ----------------------------------------------------------------------
# Feature blocks
# ----------------------------------------------------------------------

def _ensure_datetime_index(price_df: pd.DataFrame) -> pd.DataFrame:
    df = price_df.copy()
    initial_rows = len(df)

    candidate_columns = [
        column
        for column in df.columns
        if column.lower() in {"date", "datetime"}
    ]

    datetime_column: str | None = None
    for column in candidate_columns:
        if df[column].notna().any():
            datetime_column = column
            break

    if datetime_column is not None:
        df = df.reset_index(drop=True)
        if datetime_column != "Date":
            df = df.rename(columns={datetime_column: "Date"})
        unused_sources = [col for col in candidate_columns if col != datetime_column]
        if unused_sources:
            df = df.drop(columns=unused_sources)
        datetime_source = "column"
    else:
        df = df.reset_index()
        index_column = df.columns[0]
        df = df.rename(columns={index_column: "Date"})
        if candidate_columns:
            df = df.drop(columns=candidate_columns, errors="ignore")
        datetime_source = "index"

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", utc=True)
    df = df.dropna(subset=["Date"])

    numeric_candidates = [
        "Open",
        "High",
        "Low",
        "Close",
        "Adj Close",
        "Volume",
    ]
    for column in numeric_candidates:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=["Close"])
    df = df.sort_values("Date")
    df = df.drop_duplicates(subset=["Date"], keep="last")
    df = df.set_index("Date")
    df.index.name = "Date"
    df = df.sort_index()

    # Preserve the timestamp information as an explicit column because many
    # downstream feature builders expect ``Date`` both as an index and as a
    # column when joining or exporting feature frames.
    if "Date" not in df.columns:
        df = df.copy()
        df["Date"] = df.index

    final_rows = len(df)
    logger.info(
        "Normalized datetime from %s; rows before=%d after=%d",
        datetime_source,
        initial_rows,
        final_rows,
    )

    if df.empty:
        raise ValueError("Input price data must contain at least one valid datetime row.")

    return df


def _build_technical_features(
    price_df: pd.DataFrame,
    *,
    indicator_config: Mapping[str, Mapping[str, object]] | None,
) -> tuple[FeatureBlock | None, dict[str, object]]:
    metadata: dict[str, object] = {}
    if price_df.empty:
        return None, metadata

    df = price_df.reset_index(drop=True).sort_values("Date").reset_index(drop=True)
    close = pd.to_numeric(df["Close"], errors="coerce")
    volume = _get_numeric_series(df, "Volume", default=np.nan)
    high = _get_numeric_series(df, "High", default=np.nan).fillna(close)
    low = _get_numeric_series(df, "Low", default=np.nan).fillna(close)

    indicator_result = compute_indicators(df, indicator_config)
    indicator_frame = indicator_result.dataframe.reset_index(drop=True)
    metadata["indicator_columns"] = list(indicator_result.columns)

    fear_greed_frame = compute_fear_greed_features(df).reset_index(drop=True)
    fear_greed_frame.insert(0, "Date", df["Date"].reset_index(drop=True))

    trend_summary = compute_multi_timeframe_trends(df)
    if trend_summary:
        metadata["trend_summary"] = trend_summary
        higher_timeframes = {
            key: value
            for key, value in trend_summary.get("timeframes", {}).items()
            if key != trend_summary.get("base_timeframe")
        }
        metadata["higher_timeframe_trends"] = higher_timeframes

    feature_frame = pd.DataFrame({"Date": df["Date"].reset_index(drop=True)})
    feature_frame["Return_1d"] = close.pct_change(fill_method=None)
    feature_frame["LogReturn_1d"] = np.log(close.replace(0, np.nan)).diff()

    for window in (5, 10):
        sma = close.rolling(window=window, min_periods=1).mean()
        feature_frame[f"SMA_{window}"] = sma

    for span in (10, 20, 50, 100, 200):
        ema = close.ewm(span=span, adjust=False, min_periods=1).mean()
        feature_frame[f"EMA_{span}"] = ema

    for window in (5, 10, 20, 63, 126):
        feature_frame[f"ROC_{window}"] = close.pct_change(periods=window, fill_method=None)

    volume_change = volume.pct_change(fill_method=None)
    feature_frame["Volume_Change"] = volume_change
    if not volume.isna().all():
        for window in (10, 20, 50):
            rolling_vol = volume.rolling(window=window, min_periods=1).mean()
            rolling_std = volume.rolling(window=window, min_periods=1).std()
            feature_frame[f"Volume_SMA_{window}"] = rolling_vol
            feature_frame[f"Volume_EMA_{window}"] = volume.ewm(span=window, adjust=False, min_periods=1).mean()
            feature_frame[f"Volume_ZScore_{window}"] = (volume - rolling_vol) / rolling_std.replace(0, np.nan)
        feature_frame["Volume_to_Price"] = _safe_divide(volume, close)

    combined = pd.concat([feature_frame, indicator_frame, fear_greed_frame], axis=1)
    combined = combined.loc[:, ~combined.columns.duplicated()]

    column_categories: dict[str, str] = {
        column: "technical"
        for column in combined.columns
        if column != "Date"
    }
    column_categories.update(
        {
            column: "technical_indicator"
            for column in indicator_frame.columns
            if column != "Date"
        }
    )
    column_categories.update(
        {column: "fear_greed" for column in fear_greed_frame.columns if column != "Date"}
    )

    for column in sorted(col for col in combined.columns if col.startswith("SMA_")):
        combined[f"Price_to_{column}"] = _safe_divide(close, combined[column])

    for column in sorted(col for col in combined.columns if col.startswith("EMA_")):
        combined[f"Price_to_{column}"] = _safe_divide(close, combined[column])

    return FeatureBlock(
        combined, category="technical", column_categories=column_categories
    ), metadata


def _get_numeric_series(df: pd.DataFrame, column: str, default: float = np.nan) -> pd.Series:
    if column in df:
        return pd.to_numeric(df[column], errors="coerce")
    return pd.Series(default, index=df.index, dtype="float64")


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0, np.nan)


def _event_distance(flags: np.ndarray) -> tuple[pd.Series, pd.Series]:
    if flags.size == 0:
        return pd.Series(dtype="float64"), pd.Series(dtype="float64")
    last_gap = np.full(flags.shape[0], np.nan, dtype="float64")
    next_gap = np.full(flags.shape[0], np.nan, dtype="float64")

    last_idx: int | None = None
    for idx, value in enumerate(flags):
        if value > 0:
            last_idx = idx
        if last_idx is not None:
            last_gap[idx] = float(idx - last_idx)

    next_idx: int | None = None
    for idx in range(flags.shape[0] - 1, -1, -1):
        if flags[idx] > 0:
            next_idx = idx
        if next_idx is not None:
            next_gap[idx] = float(next_idx - idx)

    return pd.Series(last_gap), pd.Series(next_gap)


def _normalize_event_label(label: str) -> str:
    safe = "".join(ch if ch.isalnum() else "_" for ch in str(label).strip())
    safe = safe.strip("_")
    if not safe:
        return "Event"
    return safe.upper()


def _build_volume_liquidity_block(price_df: pd.DataFrame) -> FeatureBlock | None:
    if price_df.empty or "Volume" not in price_df:
        return None

    df = price_df.reset_index(drop=True).sort_values("Date").reset_index(drop=True)
    raw_volume = pd.to_numeric(df["Volume"], errors="coerce")
    if raw_volume.isna().all():
        return None
    volume = raw_volume.fillna(0.0)
    if np.allclose(volume, 0.0):
        return None

    close = _get_numeric_series(df, "Close", default=np.nan)
    high = _get_numeric_series(df, "High", default=np.nan).fillna(close)
    low = _get_numeric_series(df, "Low", default=np.nan).fillna(close)
    open_price = _get_numeric_series(df, "Open", default=np.nan).fillna(close)
    adj_close = _get_numeric_series(df, "Adj Close", default=np.nan)

    typical_price = (high + low + close) / 3.0
    price_change = close.diff().fillna(0.0)

    obv_direction = np.sign(price_change)
    obv_direction.iloc[0] = 0.0
    on_balance_volume = (obv_direction * volume).cumsum()

    pct_change = close.pct_change(fill_method=None).fillna(0.0)
    volume_price_trend = (pct_change * volume).cumsum()

    raw_money_flow = typical_price * volume
    price_delta = typical_price.diff().fillna(0.0)
    positive_flow = raw_money_flow.where(price_delta > 0, 0.0)
    negative_flow = raw_money_flow.where(price_delta < 0, 0.0).abs()
    money_ratio = positive_flow.rolling(window=14, min_periods=1).sum() / (
        negative_flow.rolling(window=14, min_periods=1).sum().replace(0, np.nan)
    )
    money_flow_index = 100 - (100 / (1 + money_ratio))

    turnover_value = close.fillna(0.0) * volume
    if adj_close.notna().any():
        turnover_value = adj_close.fillna(close).fillna(0.0) * volume

    liquidity_frame = pd.DataFrame({"Date": df["Date"].reset_index(drop=True)})
    liquidity_frame["OBV"] = on_balance_volume
    liquidity_frame["VPT"] = volume_price_trend
    liquidity_frame["MoneyFlowIndex_14"] = money_flow_index
    liquidity_frame["Volume_RollingMean_10"] = volume.rolling(window=10, min_periods=1).mean()
    liquidity_frame["Volume_RollingMean_20"] = volume.rolling(window=20, min_periods=1).mean()
    liquidity_frame["Volume_RollingStd_20"] = volume.rolling(window=20, min_periods=2).std()
    liquidity_frame["Turnover_Value"] = turnover_value
    liquidity_frame["Volume_to_Avg20"] = _safe_divide(
        volume, volume.rolling(window=20, min_periods=1).mean()
    )
    liquidity_frame["Volume_Price_Elasticity"] = _safe_divide(volume.diff(), price_change.replace(0, np.nan))
    range_value = (high - low).replace(0, np.nan)
    close_location = _safe_divide((2 * close - high - low), range_value)
    body = close - open_price
    body_ratio = _safe_divide(body, range_value)
    up_volume = volume.where(body > 0, 0.0)
    down_volume = volume.where(body < 0, 0.0)
    volume_imbalance = _safe_divide(up_volume - down_volume, volume)
    vwap = _safe_divide((typical_price * volume).cumsum(), volume.cumsum())
    liquidity_frame["Range"] = range_value
    liquidity_frame["Close_Location_Value"] = close_location
    liquidity_frame["Body_to_Range"] = body_ratio
    liquidity_frame["Volume_Imbalance"] = volume_imbalance
    liquidity_frame["VWAP_Deviation"] = _safe_divide(close, vwap) - 1.0
    liquidity_frame["Range_to_Close"] = _safe_divide(range_value, close)

    column_categories = {
        "OBV": "volume_trend",
        "VPT": "volume_trend",
        "MoneyFlowIndex_14": "liquidity_oscillator",
        "Volume_RollingMean_10": "volume_level",
        "Volume_RollingMean_20": "volume_level",
        "Volume_RollingStd_20": "volume_volatility",
        "Turnover_Value": "liquidity_turnover",
        "Volume_to_Avg20": "volume_relative",
        "Volume_Price_Elasticity": "liquidity_sensitivity",
        "Range": "microstructure_range",
        "Close_Location_Value": "microstructure_price_action",
        "Body_to_Range": "microstructure_price_action",
        "Volume_Imbalance": "microstructure_volume",
        "VWAP_Deviation": "microstructure_price_action",
        "Range_to_Close": "microstructure_range",
    }

    liquidity_frame = liquidity_frame.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return FeatureBlock(
        liquidity_frame,
        category="volume_liquidity",
        column_categories=column_categories,
    )


def _build_elliott_wave_descriptors(price_df: pd.DataFrame) -> FeatureBlock | None:
    if "Close" not in price_df:
        return None

    df = price_df.copy()

    # Avoid duplicating the Date column when it already exists alongside a Date
    # index (pandas would otherwise raise "cannot insert Date, already exists").
    if "Date" in df.columns:
        df = df.reset_index(drop=True)
    else:
        df = df.reset_index()
        if "Date" not in df.columns:
            df = df.rename(columns={df.columns[0]: "Date"})

    df = df[["Date", "Close"]].copy()
    df = df.sort_values("Date").reset_index(drop=True)
    close = pd.to_numeric(df["Close"], errors="coerce")

    df["Swing_High"] = close.rolling(window=5, center=True, min_periods=1).max()
    df["Swing_Low"] = close.rolling(window=5, center=True, min_periods=1).min()
    df["Wave_Strength"] = _safe_divide(df["Swing_High"] - df["Swing_Low"], close.abs())
    df["Impulse"] = close.diff(3)
    df["Corrective"] = close.diff().rolling(window=3, min_periods=1).sum()
    df["Wave_Oscillator"] = df["Impulse"].rolling(window=5, min_periods=1).mean()

    df["Wave_Phase_Code"] = np.select(
        [df["Impulse"] > 0, df["Impulse"] < 0],
        [1.0, -1.0],
        default=0.0,
    )
    try:
        df["Wave_Strength_Bucket"] = pd.qcut(
            df["Wave_Strength"].rank(method="first"),
            q=4,
            labels=False,
        ).astype(float)
    except ValueError:
        df["Wave_Strength_Bucket"] = np.nan

    df = df.fillna(0.0)

    columns = [
        "Date",
        "Swing_High",
        "Swing_Low",
        "Wave_Strength",
        "Impulse",
        "Corrective",
        "Wave_Oscillator",
        "Wave_Phase_Code",
        "Wave_Strength_Bucket",
    ]

    column_categories = {
        "Swing_High": "elliott",
        "Swing_Low": "elliott",
        "Wave_Strength": "elliott",
        "Impulse": "elliott",
        "Corrective": "elliott",
        "Wave_Oscillator": "elliott",
        "Wave_Phase_Code": "elliott_categorical",
        "Wave_Strength_Bucket": "elliott_categorical",
    }

    return FeatureBlock(df[columns], category="elliott", column_categories=column_categories)


def _build_macro_blocks(
    price_df: pd.DataFrame,
    *,
    macro_df: pd.DataFrame | None = None,
    event_params: Mapping[str, object] | None = None,
) -> list[FeatureBlock | None]:
    return [
        _build_macro_context(price_df),
        _build_macro_benchmarks(price_df, macro_df=macro_df),
        _build_cross_sectional_betas(price_df, macro_df=macro_df),
        _build_event_aware_features(price_df, macro_df=macro_df, event_params=event_params),
    ]


def _build_macro_context(price_df: pd.DataFrame) -> FeatureBlock | None:
    if "Close" not in price_df:
        return None

    df = price_df.reset_index(drop=True).sort_values("Date").reset_index(drop=True)
    returns = pd.to_numeric(df["Close"], errors="coerce").pct_change(fill_method=None)

    macro = pd.DataFrame({"Date": df["Date"]})
    macro["Volatility_21"] = returns.rolling(window=21, min_periods=5).std()
    macro["Volatility_63"] = returns.rolling(window=63, min_periods=10).std()
    macro["Trend_Slope_21"] = _rolling_linear_trend(df["Close"], window=21)
    macro["Trend_Curvature_63"] = _rolling_linear_trend(df["Close"], window=63)
    macro["Return_Corr_63"] = returns.rolling(window=63, min_periods=10).corr(returns.shift(1))
    macro = macro.fillna(0.0)

    column_categories = {
        "Volatility_21": "macro_volatility",
        "Volatility_63": "macro_volatility",
        "Trend_Slope_21": "macro_trend",
        "Trend_Curvature_63": "macro_trend",
        "Return_Corr_63": "macro_correlation",
    }

    return FeatureBlock(macro, category="macro", column_categories=column_categories)


def _build_regime_block(
    price_df: pd.DataFrame, *, regime_params: Mapping[str, float] | None = None
) -> FeatureBlock | None:
    if "Close" not in price_df:
        return None

    params = dict(regime_params or {})
    trend_window = int(params.get("trend_window", 63))
    mean_reversion_window = int(params.get("mean_reversion_window", 21))
    vol_short_window = int(params.get("vol_short_window", 21))
    vol_long_window = int(params.get("vol_long_window", 63))
    trend_threshold = float(params.get("trend_threshold", 0.003))
    mean_reversion_threshold = float(params.get("mean_reversion_threshold", 0.15))
    vol_high_threshold = float(params.get("vol_high_threshold", 1.25))
    vol_low_threshold = float(params.get("vol_low_threshold", 0.8))

    df = price_df.reset_index(drop=True).sort_values("Date").reset_index(drop=True)
    close = pd.to_numeric(df["Close"], errors="coerce")
    returns = close.pct_change(fill_method=None)

    trend_slope = _rolling_linear_trend(close, window=trend_window)
    trend_strength = _safe_divide(trend_slope.abs(), close.rolling(window=trend_window, min_periods=5).mean())
    mean_reversion = -returns.rolling(window=mean_reversion_window, min_periods=5).corr(
        returns.shift(1)
    )
    vol_short = returns.rolling(window=vol_short_window, min_periods=5).std()
    vol_long = returns.rolling(window=vol_long_window, min_periods=10).std()
    vol_ratio = _safe_divide(vol_short, vol_long)

    trend_flag = (trend_strength > trend_threshold).astype(float)
    mean_reversion_flag = (mean_reversion > mean_reversion_threshold).astype(float)
    vol_high = (vol_ratio > vol_high_threshold).astype(float)
    vol_low = (vol_ratio < vol_low_threshold).astype(float)

    regime_label = np.where(trend_flag > 0, 1.0, 0.0)

    frame = pd.DataFrame(
        {
            "Date": df["Date"],
            "Regime_TrendStrength": trend_strength,
            "Regime_MeanReversion": mean_reversion,
            "Regime_VolatilityRatio": vol_ratio,
            "Regime_TrendFlag": trend_flag,
            "Regime_MeanReversionFlag": mean_reversion_flag,
            "Regime_VolHigh": vol_high,
            "Regime_VolLow": vol_low,
            "Regime_Label": regime_label,
        }
    ).fillna(0.0)

    column_categories = {
        "Regime_TrendStrength": "regime_trend",
        "Regime_MeanReversion": "regime_mean_reversion",
        "Regime_VolatilityRatio": "regime_volatility",
        "Regime_TrendFlag": "regime_label",
        "Regime_MeanReversionFlag": "regime_label",
        "Regime_VolHigh": "regime_volatility",
        "Regime_VolLow": "regime_volatility",
        "Regime_Label": "regime_label",
    }

    return FeatureBlock(frame, category="regime", column_categories=column_categories)


def _build_event_aware_features(
    price_df: pd.DataFrame,
    *,
    macro_df: pd.DataFrame | None = None,
    event_params: Mapping[str, object] | None = None,
) -> FeatureBlock | None:
    if macro_df is None or macro_df.empty or "Date" not in macro_df.columns:
        return None
    if "Close" not in price_df.columns:
        return None

    event_params = dict(event_params or {})
    date_series = pd.to_datetime(price_df["Date"], errors="coerce")
    macro = macro_df.copy()
    macro["Date"] = pd.to_datetime(macro["Date"], errors="coerce")
    macro = macro.dropna(subset=["Date"]).sort_values("Date").set_index("Date")

    keyword_map = tuple(
        event_params.get(
            "keywords",
            (
                "earnings",
                "fomc",
                "cpi",
                "pce",
                "gdp",
                "ppi",
                "jobs",
                "nfp",
                "inflation",
                "rate",
                "fed",
                "pmi",
                "ism",
            ),
        )
    )
    event_columns = [
        col
        for col in macro.columns
        if any(keyword in col.lower() for keyword in keyword_map)
    ]
    if not event_columns:
        return None

    close = pd.to_numeric(price_df["Close"], errors="coerce")
    returns = close.pct_change(fill_method=None)

    event_window = int(event_params.get("event_window", 5))
    frame = pd.DataFrame({"Date": date_series})
    column_categories: dict[str, str] = {}

    for col in event_columns:
        raw = pd.to_numeric(macro[col], errors="coerce").fillna(0.0)
        flag = raw.where(raw == 0, 1.0)
        aligned = flag.reindex(pd.DatetimeIndex(date_series)).fillna(0.0)
        aligned = pd.Series(aligned.to_numpy(), index=price_df.index)
        last_gap, next_gap = _event_distance(aligned.to_numpy())
        event_return = returns.rolling(window=event_window, min_periods=1).sum()
        event_impact = event_return.where(aligned > 0, 0.0)
        label = _normalize_event_label(col)

        frame[f"{label}_Flag"] = aligned.reset_index(drop=True)
        frame[f"{label}_Days_Since"] = last_gap
        frame[f"{label}_Days_Until"] = next_gap
        frame[f"{label}_Impact_{event_window}d"] = event_impact.reset_index(drop=True)
        frame[f"{label}_Upcoming_Score"] = _safe_divide(1.0, 1.0 + next_gap)

        column_categories[f"{label}_Flag"] = "event_flag"
        column_categories[f"{label}_Days_Since"] = "event_timing"
        column_categories[f"{label}_Days_Until"] = "event_timing"
        column_categories[f"{label}_Impact_{event_window}d"] = "event_impact"
        column_categories[f"{label}_Upcoming_Score"] = "event_timing"

    frame = frame.fillna(0.0)
    return FeatureBlock(frame, category="macro_event", column_categories=column_categories)


def _rolling_linear_trend(series: pd.Series, window: int) -> pd.Series:
    if window <= 1:
        return pd.Series(0.0, index=series.index)
    slopes = []
    x = np.arange(window)
    for i in range(len(series)):
        if i + 1 < window:
            slopes.append(np.nan)
            continue
        window_values = series.iloc[i + 1 - window : i + 1].values
        if np.isnan(window_values).all():
            slopes.append(np.nan)
            continue
        coeffs = np.polyfit(x, window_values, deg=2)
        slopes.append(coeffs[1])
    return pd.Series(slopes, index=series.index)


def _build_macro_benchmarks(
    price_df: pd.DataFrame, *, macro_df: pd.DataFrame | None = None
) -> FeatureBlock | None:
    base_close_raw = price_df.get("Close")
    if base_close_raw is None:
        return None
    base_close = pd.to_numeric(base_close_raw, errors="coerce")

    benchmarks = {
        "^GSPC": "sp500",
        "^VIX": "vix",
    }

    features: Dict[str, pd.Series] = {}
    date_series = price_df.get("Date")
    for symbol, name in benchmarks.items():
        benchmark = _extract_benchmark_series(price_df, symbol, macro_df=macro_df)
        if benchmark is None:
            continue
        benchmark_returns = benchmark.pct_change(fill_method=None)
        symbol_prefix = name.upper()
        features[f"{symbol_prefix}_Return"] = benchmark_returns
        features[f"{symbol_prefix}_RollingCorr_21"] = (
            base_close.pct_change(fill_method=None)
            .rolling(window=21, min_periods=5)
            .corr(benchmark_returns)
        )
        features[f"{symbol_prefix}_Relative_Return"] = base_close.pct_change(fill_method=None) - benchmark_returns
        features[f"{symbol_prefix}_Price_Ratio"] = _safe_divide(base_close, benchmark)

    if not features:
        return None

    feature_frame = pd.DataFrame({"Date": date_series.reset_index(drop=True)})
    for name, series in features.items():
        feature_frame[name] = series.reset_index(drop=True)

    column_categories = {col: "macro_benchmark" for col in feature_frame.columns if col != "Date"}
    return FeatureBlock(feature_frame.fillna(0.0), category="macro", column_categories=column_categories)


def _build_cross_sectional_betas(
    price_df: pd.DataFrame, *, macro_df: pd.DataFrame | None = None
) -> FeatureBlock | None:
    base_close_raw = price_df.get("Close")
    if base_close_raw is None:
        return None
    base_close = pd.to_numeric(base_close_raw, errors="coerce")

    base_returns = base_close.pct_change(fill_method=None)
    benchmarks = {
        "^GSPC": "Beta_SP500",
        "^VIX": "Beta_VIX",
    }

    beta_features: Dict[str, pd.Series] = {}
    placeholder_index = price_df.index
    for symbol, label in benchmarks.items():
        benchmark = _extract_benchmark_series(price_df, symbol, macro_df=macro_df)
        if benchmark is None:
            # Preserve a stable feature schema even when benchmark data is missing
            # by emitting NaN-filled columns for each rolling window. This prevents
            # downstream preprocessors (e.g., imputers) from erroring out when a
            # previously seen feature suddenly disappears because the benchmark
            # series could not be located in the current dataset.
            for window in (21, 63, 126):
                beta_features[f"{label}_{window}"] = pd.Series(
                    np.nan, index=placeholder_index
                )
            continue
        benchmark_returns = benchmark.pct_change(fill_method=None)
        for window in (21, 63, 126):
            cov = base_returns.rolling(window=window, min_periods=5).cov(benchmark_returns)
            var = benchmark_returns.rolling(window=window, min_periods=5).var()
            beta_features[f"{label}_{window}"] = _safe_divide(cov, var)

    if not beta_features:
        return None

    frame = pd.DataFrame({"Date": price_df["Date"].reset_index(drop=True)})
    for name, series in beta_features.items():
        frame[name] = series.reset_index(drop=True)

    column_categories = {col: "macro_beta" for col in frame.columns if col != "Date"}
    return FeatureBlock(frame, category="macro", column_categories=column_categories)


def _extract_benchmark_series(
    price_df: pd.DataFrame, symbol: str, *, macro_df: pd.DataFrame | None = None
) -> pd.Series | None:
    normalized_symbol = symbol.replace("^", "")
    possible_names = [
        f"Close_{symbol}",
        f"Close_{normalized_symbol}",
        f"{symbol}_Close",
        f"{normalized_symbol}_Close",
        f"Adj Close_{symbol}",
        f"Adj Close_{normalized_symbol}",
        f"{symbol}_Adj Close",
        f"{normalized_symbol}_Adj Close",
    ]

    frames = [price_df]
    if macro_df is not None:
        frames.append(macro_df)

    for frame in frames:
        candidates: list[str] = []
        for name in possible_names:
            if name in frame.columns:
                candidates.append(name)
        if not candidates:
            for column in frame.columns:
                if normalized_symbol in column and "close" in column.lower():
                    candidates.append(column)
        if candidates:
            return pd.to_numeric(frame[candidates[0]], errors="coerce")

    return None


def _build_sentiment_features(news_df: pd.DataFrame) -> FeatureBlock | None:
    scored = attach_sentiment(news_df)
    aggregated = aggregate_daily_sentiment(scored)
    if aggregated.empty:
        return None

    aggregated = aggregated.rename(columns={"sentiment": "Sentiment_Avg"})
    aggregated["Sentiment_Change"] = aggregated["Sentiment_Avg"].diff()
    aggregated["Sentiment_7d"] = aggregated["Sentiment_Avg"].rolling(window=7, min_periods=1).mean()
    aggregated["Sentiment_30d"] = aggregated["Sentiment_Avg"].rolling(window=30, min_periods=1).mean()
    aggregated["Sentiment_ZScore_30"] = (
        aggregated["Sentiment_Avg"] - aggregated["Sentiment_Avg"].rolling(window=30, min_periods=5).mean()
    ) / aggregated["Sentiment_Avg"].rolling(window=30, min_periods=5).std().replace(0, np.nan)
    aggregated = aggregated.fillna(0.0)

    column_categories = {
        "Sentiment_Avg": "sentiment",
        "Sentiment_Change": "sentiment_trend",
        "Sentiment_7d": "sentiment_trend",
        "Sentiment_30d": "sentiment_trend",
        "Sentiment_ZScore_30": "sentiment_trend",
    }

    return FeatureBlock(aggregated, category="sentiment", column_categories=column_categories)


def _empty_sentiment_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "Date",
            "Sentiment_Avg",
            "Sentiment_Change",
            "Sentiment_7d",
            "Sentiment_30d",
            "Sentiment_ZScore_30",
        ]
    )


# ----------------------------------------------------------------------
# Target generation
# ----------------------------------------------------------------------

def _generate_targets(
    merged: pd.DataFrame,
    horizons: Iterable[int],
    *,
    direction_neutral_threshold: float | None = None,
) -> Dict[int, Dict[str, pd.Series]]:
    merged = merged.copy()
    pct_returns = merged["Close"].pct_change(fill_method=None)
    merged["Daily_Return"] = pct_returns

    targets_by_horizon: Dict[int, Dict[str, pd.Series]] = {}
    for horizon in horizons:
        future_close = merged["Close"].shift(-horizon)
        cumulative_return = _safe_divide(future_close, merged["Close"]) - 1.0
        log_return = np.log1p(cumulative_return)
        if direction_neutral_threshold is None:
            direction = (cumulative_return > 0).astype(float)
            direction[cumulative_return.isna()] = np.nan
        else:
            threshold = float(abs(direction_neutral_threshold))
            direction = pd.Series(np.nan, index=cumulative_return.index, dtype="float64")
            direction[cumulative_return > threshold] = 1.0
            direction[cumulative_return < -threshold] = -1.0
            direction[(cumulative_return <= threshold) & (cumulative_return >= -threshold)] = 0.0
        volatility = (
            merged["Daily_Return"].rolling(window=horizon, min_periods=1).std().shift(-horizon)
        )

        horizon_targets: Dict[str, pd.Series] = {
            "close": future_close,
            "return": cumulative_return,
            "log_return": log_return,
            "direction": direction,
            "volatility": volatility,
        }

        valid_targets: Dict[str, pd.Series] = {}
        for name, series in horizon_targets.items():
            cleaned = series.dropna()
            if cleaned.empty:
                continue
            valid_targets[name] = series
        if valid_targets:
            targets_by_horizon[horizon] = valid_targets

    return targets_by_horizon


def _validate_target_alignment(
    merged: pd.DataFrame, targets: Mapping[int, Mapping[str, pd.Series]], horizons: Iterable[int]
) -> Dict[int, Dict[str, object]]:
    """Validate that generated targets line up with expected future shifts.

    For each requested horizon this checks whether the close and return targets
    resemble a simple shift of the underlying price series. The resulting
    summary is stored in metadata so downstream components (UI, tests) can
    confirm that the model trains on the intended labels.
    """

    base_close = pd.to_numeric(merged["Close"], errors="coerce")
    summary: Dict[int, Dict[str, object]] = {}

    for horizon in horizons:
        horizon_int = int(horizon)
        record: Dict[str, object] = {"horizon": horizon_int}
        block = targets.get(horizon_int, {}) if isinstance(targets, Mapping) else {}

        shifted_close = base_close.shift(-horizon_int)

        close_series = block.get("close") if isinstance(block, Mapping) else None
        if close_series is not None:
            aligned = pd.to_numeric(close_series, errors="coerce")
            diff = (aligned - shifted_close).abs().dropna()
            max_error = float(diff.max()) if not diff.empty else None
            record["close_alignment_error"] = max_error
            record["close_aligned"] = bool(max_error is not None and max_error < 1e-6)

        return_series = block.get("return") if isinstance(block, Mapping) else None
        if return_series is not None:
            expected_return = _safe_divide(shifted_close, base_close) - 1.0
            diff = (pd.to_numeric(return_series, errors="coerce") - expected_return).abs().dropna()
            max_error = float(diff.max()) if not diff.empty else None
            record["return_alignment_error"] = max_error
            record["return_aligned"] = bool(max_error is not None and max_error < 1e-6)

        summary[horizon_int] = record

    return summary


def _estimate_target_dates(dates: pd.Series, horizons: Iterable[int]) -> Dict[int, pd.Timestamp]:
    timestamps = pd.to_datetime(dates).dropna().sort_values()
    if timestamps.empty:
        return {}
    latest = timestamps.iloc[-1]
    result: Dict[int, pd.Timestamp] = {}
    for horizon in horizons:
        try:
            offset = pd.tseries.offsets.BDay(int(horizon))
        except Exception:  # pragma: no cover - defensive
            offset = pd.Timedelta(days=int(horizon))
        result[int(horizon)] = latest + offset
    return result


def _normalise_horizons(horizons: Iterable[int]) -> tuple[int, ...]:
    unique: list[int] = []
    for raw in horizons:
        try:
            value = int(raw)
        except (TypeError, ValueError):
            continue
        if value <= 0:
            continue
        if value not in unique:
            unique.append(value)
    if not unique:
        unique = [1]
    unique.sort()
    return tuple(unique)
