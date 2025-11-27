"""Feature engineering utilities for the stock predictor."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import re
from typing import Dict, Iterable, Iterator, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from ..indicator_bundle import (
    DEFAULT_INDICATOR_CONFIG,
    compute_indicators,
    compute_multi_timeframe_trends,
)
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


def _fundamental_group_builder(context: FeatureBuildContext) -> FeatureBuildOutput:
    blocks = [
        block
        for block in _build_fundamental_blocks(context.price_df, context.fundamentals_df)
        if block is not None
    ]
    status = "executed" if blocks else "skipped_no_data"
    return FeatureBuildOutput(blocks=blocks, status=status)


def _macro_group_builder(context: FeatureBuildContext) -> FeatureBuildOutput:
    blocks = [
        block
        for block in _build_macro_blocks(context.price_df, macro_df=context.macro_df)
        if block is not None
    ]
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
    fundamental=_fundamental_group_builder,
    macro=_macro_group_builder,
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
        feature_toggles: Mapping[str, bool] | Iterable[str] | None,
        horizons: Iterable[int] | None = None,
        *,
        registry: Mapping[str, FeatureGroupSpec] | None = None,
        technical_indicator_config: Mapping[str, Mapping[str, object]] | None = None,
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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def build(
        self,
        price_df: pd.DataFrame,
        news_df: pd.DataFrame | None,
        sentiment_enabled: bool,
        *,
        fundamentals_df: pd.DataFrame | None = None,
        macro_df: pd.DataFrame | None = None,
    ) -> FeatureResult:
        if price_df.empty:
            raise ValueError("Price dataframe cannot be empty when building features.")

        processed = _ensure_datetime_index(price_df)
        context_news = None
        if news_df is not None:
            context_news = news_df.copy()
        context_fundamentals = None
        if fundamentals_df is not None:
            context_fundamentals = fundamentals_df.copy()
        context_macro = None
        if macro_df is not None:
            context_macro = macro_df.copy()

        context = FeatureBuildContext(
            price_df=processed,
            news_df=context_news,
            fundamentals_df=context_fundamentals,
            macro_df=context_macro,
            sentiment_enabled=sentiment_enabled,
            technical_indicator_config=self.technical_indicator_config,
        )

        feature_blocks: list[FeatureBlock] = []
        metadata: Dict[str, object] = {}
        feature_categories: Dict[str, str] = {}

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

        merged = processed[["Date", "Close"]].copy()
        for block in feature_blocks:
            merged = merged.merge(block.frame, on="Date", how="left")

        merged = merged.sort_values("Date").reset_index(drop=True)
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

        targets = _generate_targets(merged, self.horizons)

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

        features = merged[feature_columns]
        return FeatureResult(features=features, targets=targets, metadata=metadata)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _normalise_toggles(
        self,
        toggles: Mapping[str, bool] | Iterable[str] | None,
    ) -> dict[str, bool]:
        defaults = default_feature_toggles(self.registry)
        if toggles is None:
            return defaults

        normalised: dict[str, bool] = {}
        if isinstance(toggles, Mapping):
            for key, value in toggles.items():
                name = str(key).strip().lower()
                if name in defaults:
                    normalised[name] = bool(value)
        else:
            if isinstance(toggles, str):
                tokens = [part.strip() for part in toggles.split(",")]
            else:
                tokens = [str(item).strip() for item in toggles]
            for token in tokens:
                if not token:
                    continue
                name = token.lower()
                if name in defaults:
                    normalised[name] = True

        defaults.update(normalised)
        return defaults

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
    if "Date" not in df.columns:
        raise ValueError("Input price data must contain a 'Date' column.")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
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
    return df


def _build_technical_features(
    price_df: pd.DataFrame,
    *,
    indicator_config: Mapping[str, Mapping[str, object]] | None,
) -> tuple[FeatureBlock | None, dict[str, object]]:
    metadata: dict[str, object] = {}
    if price_df.empty:
        return None, metadata

    df = price_df.sort_values("Date").reset_index(drop=True)
    close = pd.to_numeric(df["Close"], errors="coerce")
    volume = _get_numeric_series(df, "Volume", default=np.nan)
    high = _get_numeric_series(df, "High", default=np.nan).fillna(close)
    low = _get_numeric_series(df, "Low", default=np.nan).fillna(close)

    indicator_result = compute_indicators(df, indicator_config)
    indicator_frame = indicator_result.dataframe.reset_index(drop=True)
    metadata["indicator_columns"] = list(indicator_result.columns)

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

    combined = pd.concat([feature_frame, indicator_frame], axis=1)
    combined = combined.loc[:, ~combined.columns.duplicated()]

    for column in sorted(col for col in combined.columns if col.startswith("SMA_")):
        combined[f"Price_to_{column}"] = _safe_divide(close, combined[column])

    for column in sorted(col for col in combined.columns if col.startswith("EMA_")):
        combined[f"Price_to_{column}"] = _safe_divide(close, combined[column])

    return FeatureBlock(combined, category="technical"), metadata


def _get_numeric_series(df: pd.DataFrame, column: str, default: float = np.nan) -> pd.Series:
    if column in df:
        return pd.to_numeric(df[column], errors="coerce")
    return pd.Series(default, index=df.index, dtype="float64")


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0, np.nan)


def _build_volume_liquidity_block(price_df: pd.DataFrame) -> FeatureBlock | None:
    if price_df.empty or "Volume" not in price_df:
        return None

    df = price_df.sort_values("Date").reset_index(drop=True)
    raw_volume = pd.to_numeric(df["Volume"], errors="coerce")
    if raw_volume.isna().all():
        return None
    volume = raw_volume.fillna(0.0)
    if np.allclose(volume, 0.0):
        return None

    close = _get_numeric_series(df, "Close", default=np.nan)
    high = _get_numeric_series(df, "High", default=np.nan).fillna(close)
    low = _get_numeric_series(df, "Low", default=np.nan).fillna(close)
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

    df = price_df[["Date", "Close"]].copy()
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


def _build_fundamental_blocks(
    price_df: pd.DataFrame, fundamentals_df: pd.DataFrame | None
) -> list[FeatureBlock | None]:
    return [
        _build_fundamental_metrics(price_df, fundamentals_df),
    ]


def _build_fundamental_metrics(
    price_df: pd.DataFrame, fundamentals_df: pd.DataFrame | None
) -> FeatureBlock | None:
    if fundamentals_df is None or fundamentals_df.empty:
        return None

    aligned = _align_fundamentals_to_price(price_df, fundamentals_df)
    if aligned is None or aligned.empty:
        return None

    column_data: Dict[str, pd.Series] = {"Date": aligned["Date"]}
    column_categories: Dict[str, str] = {}

    value_columns = [column for column in aligned.columns if column != "Date"]
    for column in value_columns:
        series = pd.to_numeric(aligned[column], errors="coerce")
        if series.dropna().empty:
            continue
        base_name = _sanitize_metric_name(column)
        latest_col = f"Fundamental_{base_name}_Latest"
        filled = series.ffill()
        column_data[latest_col] = filled
        column_categories[latest_col] = "fundamental"

        pct_change_col = f"Fundamental_{base_name}_PctChange_63"
        column_data[pct_change_col] = filled.pct_change(periods=63, fill_method=None)
        column_categories[pct_change_col] = "fundamental_trend"

        zscore_col = f"Fundamental_{base_name}_ZScore_252"
        rolling_mean = filled.rolling(window=252, min_periods=5).mean()
        rolling_std = filled.rolling(window=252, min_periods=5).std()
        column_data[zscore_col] = (filled - rolling_mean) / rolling_std.replace(0, np.nan)
        column_categories[zscore_col] = "fundamental_trend"

    metrics = pd.DataFrame(column_data).fillna(0.0)
    if len(metrics.columns) <= 1:
        return None
    return FeatureBlock(metrics, category="fundamental", column_categories=column_categories)


def _align_fundamentals_to_price(
    price_df: pd.DataFrame, fundamentals_df: pd.DataFrame
) -> pd.DataFrame | None:
    prepared = _normalize_fundamentals_frame(fundamentals_df)
    if prepared.empty:
        return None

    pivot = (
        prepared.pivot_table(index="AsOf", columns="MetricName", values="Value", aggfunc="last")
        .sort_index()
        .ffill()
    )
    if pivot.empty:
        return None
    pivot = pivot.reset_index().rename(columns={"AsOf": "Date"})

    price_dates = pd.DataFrame({"Date": pd.to_datetime(price_df["Date"], errors="coerce")})
    price_dates = price_dates.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    pivot["Date"] = pd.to_datetime(pivot["Date"], errors="coerce")
    pivot = pivot.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    if price_dates.empty or pivot.empty:
        return None

    aligned = pd.merge_asof(
        price_dates,
        pivot,
        on="Date",
        direction="backward",
        allow_exact_matches=True,
    )
    aligned = aligned.ffill()
    return aligned


def _normalize_fundamentals_frame(fundamentals_df: pd.DataFrame) -> pd.DataFrame:
    df = fundamentals_df.copy()
    date_column = None
    for candidate in ("AsOf", "Date", "date", "as_of"):
        if candidate in df.columns:
            date_column = candidate
            break
    if date_column is None:
        return pd.DataFrame(columns=["AsOf", "MetricName", "Value"])
    metric_column = None
    for candidate in ("Metric", "metric", "Name", "name"):
        if candidate in df.columns:
            metric_column = candidate
            break
    if metric_column is None:
        return pd.DataFrame(columns=["AsOf", "MetricName", "Value"])
    value_column = None
    for candidate in ("Value", "value"):
        if candidate in df.columns:
            value_column = candidate
            break
    if value_column is None:
        return pd.DataFrame(columns=["AsOf", "MetricName", "Value"])

    df["AsOf"] = pd.to_datetime(df[date_column], errors="coerce")
    df = df.dropna(subset=["AsOf"])
    df["MetricName"] = df.apply(_compose_metric_label(metric_column), axis=1)
    df["MetricName"] = df["MetricName"].astype(str).str.strip()
    df["Value"] = pd.to_numeric(df[value_column], errors="coerce")
    df = df.dropna(subset=["MetricName", "Value"])
    return df[["AsOf", "MetricName", "Value"]]


def _compose_metric_label(metric_column: str):
    def _builder(row: pd.Series) -> str:
        metric_value = str(row.get(metric_column, "")).strip()
        statement = str(row.get("Statement", "")).strip()
        period = str(row.get("Period", "")).strip()
        parts = [part for part in (statement, period, metric_value) if part]
        if not parts:
            return "Metric"
        return "_".join(parts)

    return _builder


def _sanitize_metric_name(metric: str) -> str:
    token = re.sub(r"[^0-9a-zA-Z]+", "_", str(metric).strip())
    token = token.strip("_")
    if not token:
        return "Metric"
    if token[0].isdigit():
        token = f"M{token}"
    return token


def _build_macro_blocks(
    price_df: pd.DataFrame, *, macro_df: pd.DataFrame | None = None
) -> list[FeatureBlock | None]:
    return [
        _build_macro_context(price_df),
        _build_macro_benchmarks(price_df, macro_df=macro_df),
        _build_cross_sectional_betas(price_df, macro_df=macro_df),
    ]


def _build_macro_context(price_df: pd.DataFrame) -> FeatureBlock | None:
    if "Close" not in price_df:
        return None

    df = price_df.sort_values("Date").reset_index(drop=True)
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

    feature_frame = pd.DataFrame({"Date": date_series})
    for name, series in features.items():
        feature_frame[name] = series

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
    for symbol, label in benchmarks.items():
        benchmark = _extract_benchmark_series(price_df, symbol, macro_df=macro_df)
        if benchmark is None:
            continue
        benchmark_returns = benchmark.pct_change(fill_method=None)
        for window in (21, 63, 126):
            cov = base_returns.rolling(window=window, min_periods=5).cov(benchmark_returns)
            var = benchmark_returns.rolling(window=window, min_periods=5).var()
            beta_features[f"{label}_{window}"] = _safe_divide(cov, var)

    if not beta_features:
        return None

    frame = pd.DataFrame({"Date": price_df["Date"]})
    for name, series in beta_features.items():
        frame[name] = series

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

def _generate_targets(merged: pd.DataFrame, horizons: Iterable[int]) -> Dict[int, Dict[str, pd.Series]]:
    merged = merged.copy()
    merged["Daily_Return"] = merged["Close"].pct_change(fill_method=None)

    targets_by_horizon: Dict[int, Dict[str, pd.Series]] = {}
    for horizon in horizons:
        future_close = merged["Close"].shift(-horizon)
        future_return = (future_close - merged["Close"]) / merged["Close"]
        direction = (future_return > 0).astype(float)
        direction[future_return.isna()] = np.nan
        volatility = (
            merged["Daily_Return"].rolling(window=horizon, min_periods=1).std().shift(-horizon)
        )

        horizon_targets: Dict[str, pd.Series] = {
            "close": future_close,
            "return": future_return,
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
