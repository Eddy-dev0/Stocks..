"""Data processing helpers for feature engineering."""

from __future__ import annotations

import logging
from typing import Dict, Iterable, List, Mapping, Tuple

import numpy as np
import pandas as pd

from .fear_greed import compute_fear_greed_features
from .features.toggles import FeatureToggles
from .indicator_bundle import compute_indicators
from .sentiment import aggregate_daily_sentiment, attach_sentiment

LOGGER = logging.getLogger(__name__)


PRICE_BASE_COLUMNS = {"Open", "High", "Low", "Close", "Adj Close"}
PRICE_PREFIXES = ("SMA_", "EMA_")
PRICE_EXACT_COLUMNS = {
    "MACD",
    "MACD_Signal",
    "MACD_Hist",
    "BB_Middle_20",
    "BB_Upper_20",
    "BB_Lower_20",
}


LEGACY_INDICATOR_COLUMNS = [
    "Return_1d",
    "LogReturn_1d",
    "SMA_5",
    "SMA_10",
    "EMA_9",
    "Volatility_5",
    "Volume_Change",
]


def _identify_price_columns(df: pd.DataFrame) -> list[str]:
    price_columns: list[str] = []
    for column in df.columns:
        if column in PRICE_BASE_COLUMNS or column in PRICE_EXACT_COLUMNS:
            price_columns.append(column)
            continue
        if column.startswith(PRICE_PREFIXES):
            price_columns.append(column)
    return price_columns


PRICE_FEATURE_DEFAULTS: dict[str, bool] = {
    "obv": True,
    "volume_weighted_momentum": True,
    "orderflow": True,
    "macro_merge": True,
}


# Map higher-level feature groups to the concrete price feature toggles they
# control so UI and configuration changes can stay in sync.
FEATURE_GROUP_PRICE_MAPPING: dict[str, tuple[str, ...]] = {
    "volume_liquidity": ("obv", "volume_weighted_momentum", "orderflow"),
    "macro": ("macro_merge",),
}


def default_price_feature_toggles() -> dict[str, bool]:
    return dict(PRICE_FEATURE_DEFAULTS)


def _normalise_toggle_entries(
    toggles: FeatureToggles | Mapping[str, object] | Iterable[str] | None,
) -> list[tuple[str, object]]:
    if toggles is None:
        return []

    if isinstance(toggles, Mapping):
        source = toggles.items()
    elif isinstance(toggles, str):
        source = ((token, True) for token in toggles.split(","))
    else:
        source = ((token, True) for token in toggles)

    entries: list[tuple[str, object]] = []
    for key, value in source:
        name = str(key).strip().lower()
        if not name:
            continue
        entries.append((name, value))
    return entries


def derive_price_feature_toggles(
    toggles: FeatureToggles | Mapping[str, object] | Iterable[str] | None,
) -> dict[str, bool]:
    """Expand group-level feature toggles into concrete price feature flags."""

    defaults = default_price_feature_toggles()
    entries = _normalise_toggle_entries(toggles)

    for name, enabled in entries:
        if name in FEATURE_GROUP_PRICE_MAPPING:
            for feature_name in FEATURE_GROUP_PRICE_MAPPING[name]:
                defaults[feature_name] = bool(enabled)

    for name, enabled in entries:
        if name in defaults:
            defaults[name] = bool(enabled)

    return defaults


def _coerce_price_feature_toggles(
    toggles: FeatureToggles | Mapping[str, object] | Iterable[str] | None,
) -> dict[str, bool]:
    return derive_price_feature_toggles(toggles)


def _normalize_dates(frame: pd.DataFrame, column: str = "Date") -> pd.DataFrame:
    if frame.empty or column not in frame.columns:
        return frame
    normalized = frame.copy()
    normalized[column] = pd.to_datetime(normalized[column], errors="coerce")
    normalized = normalized.dropna(subset=[column])
    normalized[column] = normalized[column].dt.normalize()
    return normalized


def compute_price_features(
    price_df: pd.DataFrame,
    *,
    feature_toggles: FeatureToggles | Mapping[str, object] | Iterable[str] | None = None,
    macro_df: pd.DataFrame | None = None,
    macro_symbols: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Compute technical indicators, lag features, and optional contextual series."""

    if price_df.empty:
        raise ValueError("Price dataframe is empty.")

    df = price_df.copy()
    if "Date" not in df.columns:
        raise ValueError("Expected a 'Date' column in price data.")

    # Ensure the Date column is a datetime so that sorting behaves predictably.
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    if df["Date"].isna().all():
        raise ValueError("Unable to parse any valid dates from the price data.")

    # Remove rows where we could not parse the date; they cannot be used in time
    # series calculations and would otherwise propagate NaT values.
    df = df.dropna(subset=["Date"])
    df["Date"] = df["Date"].dt.normalize()

    # Coerce numeric columns to floats/ints so downstream math works as expected.
    numeric_columns = {
        "Open",
        "High",
        "Low",
        "Close",
        "Adj Close",
        "Volume",
    }.intersection(df.columns)
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # Drop rows where the close price is missing; without a close value we cannot
    # build any of the derived price-based features.
    if "Close" not in df.columns:
        raise ValueError("Expected a 'Close' column in price data.")

    df = df.dropna(subset=[col for col in ["Close", "Volume"] if col in df.columns])
    if df.empty:
        raise ValueError("Price dataframe has no valid rows after cleaning.")

    toggles = _coerce_price_feature_toggles(feature_toggles)

    df = df.sort_values("Date").drop_duplicates(subset=["Date"], keep="last").reset_index(
        drop=True
    )
    df["Return_1d"] = df["Close"].pct_change()
    df["LogReturn_1d"] = np.log(df["Close"]).diff()
    df["SMA_5"] = df["Close"].rolling(window=5, min_periods=1).mean()
    df["SMA_10"] = df["Close"].rolling(window=10, min_periods=1).mean()
    df["EMA_9"] = df["Close"].ewm(span=9, adjust=False).mean()
    df["Volatility_5"] = df["Return_1d"].rolling(window=5, min_periods=1).std()
    df["Volume_Change"] = df["Volume"].pct_change()

    volume_feature_columns: list[str] = []
    if toggles.get("obv", False) and "Volume" in df.columns:
        direction = np.sign(df["Close"].diff().fillna(0.0))
        df["Volume_OBV"] = (direction * df["Volume"]).fillna(0.0).cumsum()
        volume_feature_columns.append("Volume_OBV")

    if toggles.get("volume_weighted_momentum", False) and "Volume" in df.columns:
        weighted_returns = df["Close"].pct_change(fill_method=None) * df["Volume"]
        for window in (5, 10):
            column = f"VWMomentum_{window}"
            df[column] = weighted_returns.rolling(window=window, min_periods=1).mean()
            volume_feature_columns.append(column)

    if toggles.get("orderflow", False) and "Volume" in df.columns:
        signed_volume = df["Volume"] * np.sign(df["Close"].diff().fillna(0.0))
        for window in (5, 20):
            imbalance_col = f"Orderflow_Imbalance_{window}"
            tick_col = f"Tick_Imbalance_{window}"
            df[imbalance_col] = signed_volume.rolling(window=window, min_periods=1).sum()
            df[tick_col] = (
                np.sign(df["Close"].diff().fillna(0.0))
                .rolling(window=window, min_periods=1)
                .sum()
            )
            volume_feature_columns.extend([imbalance_col, tick_col])

    if toggles.get("macro_merge", False) and macro_df is not None and not macro_df.empty:
        symbols = list(macro_symbols) if macro_symbols is not None else ["^VIX", "DXY", "^TNX"]
        merged = _merge_macro_series(df, macro_df, symbols)
        df = merged
        volume_feature_columns.extend(
            [col for col in df.columns if col.startswith("Macro_")]
        )

    indicator_result = compute_indicators(df)
    fear_greed = compute_fear_greed_features(df)
    df = pd.concat([df, indicator_result.dataframe, fear_greed], axis=1)

    df = df.ffill().bfill()
    indicator_columns = []
    seen: set[str] = set()
    for column in [
        *LEGACY_INDICATOR_COLUMNS,
        *indicator_result.columns,
        *fear_greed.columns.tolist(),
        *volume_feature_columns,
    ]:
        if column not in df.columns:
            continue
        if column in seen:
            continue
        indicator_columns.append(column)
        seen.add(column)
    df.attrs["indicator_columns"] = indicator_columns
    df.attrs["price_columns"] = _identify_price_columns(df)
    return df


def _merge_macro_series(
    base_df: pd.DataFrame, macro_df: pd.DataFrame, symbols: list[str]
) -> pd.DataFrame:
    macro_df = macro_df.copy()
    if "Date" not in macro_df.columns:
        return base_df
    macro_df["Date"] = pd.to_datetime(macro_df["Date"], errors="coerce")
    macro_df = macro_df.dropna(subset=["Date"])
    macro_df["Date"] = macro_df["Date"].dt.normalize()
    macro_df = macro_df.sort_values("Date")
    macro_df = macro_df[macro_df["Date"].isin(base_df["Date"])]

    macro_features: dict[str, pd.Series] = {"Date": base_df["Date"]}
    for symbol in symbols:
        series = _extract_macro_series(macro_df, symbol)
        if series is None:
            continue
        normalized = symbol.replace("^", "")
        close_col = f"Macro_{normalized}_Close"
        return_col = f"Macro_{normalized}_Return"
        macro_features[close_col] = series
        macro_features[return_col] = pd.to_numeric(series, errors="coerce").pct_change(
            fill_method=None
        )

    if len(macro_features) == 1:
        return base_df

    feature_frame = pd.DataFrame(macro_features)
    merged = base_df.merge(feature_frame, on="Date", how="left")
    return merged


def _extract_macro_series(macro_df: pd.DataFrame, symbol: str) -> pd.Series | None:
    normalized_symbol = symbol.replace("^", "")
    candidates = [
        f"Close_{symbol}",
        f"Close_{normalized_symbol}",
        f"{symbol}_Close",
        f"{normalized_symbol}_Close",
        f"Adj Close_{symbol}",
        f"Adj Close_{normalized_symbol}",
        f"{symbol}_Adj Close",
        f"{normalized_symbol}_Adj Close",
        symbol,
        normalized_symbol,
    ]
    for column in macro_df.columns:
        if column in candidates:
            return pd.to_numeric(macro_df[column], errors="coerce")
        if normalized_symbol in column and "close" in column.lower():
            return pd.to_numeric(macro_df[column], errors="coerce")
    return None


def merge_with_sentiment(
    price_df: pd.DataFrame, news_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Attach aggregated sentiment information to the price dataframe."""

    if news_df.empty:
        LOGGER.info("No news data available; skipping sentiment merge.")
        return price_df, pd.DataFrame()

    scored = attach_sentiment(news_df)
    aggregated = aggregate_daily_sentiment(scored)
    if not aggregated.empty:
        aggregated = _normalize_dates(aggregated, "Date")
        aggregated = aggregated.drop_duplicates(subset=["Date"], keep="last")
    aligned_prices = _normalize_dates(price_df, "Date")
    if not aggregated.empty:
        aggregated = aggregated[aggregated["Date"].isin(aligned_prices["Date"])]
    merged = aligned_prices.merge(aggregated, on="Date", how="left")
    merged["sentiment"] = merged["sentiment"].fillna(0.0)
    return merged, aggregated


def build_supervised_dataset(
    price_df: pd.DataFrame,
    sentiment_df: pd.DataFrame | None = None,
    *,
    use_log_returns: bool = False,
    feature_toggles: Mapping[str, object] | Iterable[str] | None = None,
    macro_df: pd.DataFrame | None = None,
    macro_symbols: Iterable[str] | None = None,
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, object]]:
    """Prepare the feature matrix and target vector."""

    price_features = compute_price_features(
        price_df,
        feature_toggles=feature_toggles,
        macro_df=macro_df,
        macro_symbols=macro_symbols,
    )
    sentiment_df = sentiment_df if sentiment_df is not None else pd.DataFrame()
    indicator_columns = price_features.attrs.get("indicator_columns", [])
    price_columns_attr = price_features.attrs.get("price_columns", [])
    merged, aggregated = merge_with_sentiment(price_features, sentiment_df)

    dataset = merged.copy()
    if use_log_returns:
        returns = np.log(dataset["Close"]).diff()
        target_kind = "log_return"
    else:
        returns = dataset["Close"].pct_change(fill_method=None)
        target_kind = "pct_return"

    target = returns.shift(-1)
    dataset = dataset.assign(Target=target)

    if "Target" not in dataset.columns:
        raise KeyError("Failed to create the 'Target' column for the supervised dataset.")

    dataset = dataset.loc[target.notna()].reset_index(drop=True)

    feature_columns = [
        col
        for col in dataset.columns
        if col not in {"Date", "Target"}
    ]
    X = dataset[feature_columns]
    y = dataset["Target"]

    latest_row = merged.iloc[[-1]][feature_columns].ffill().bfill()

    metadata = {
        "feature_columns": feature_columns,
        "latest_features": latest_row,
        "latest_close": float(price_df.iloc[-1]["Close"]),
        "latest_date": pd.to_datetime(price_df.iloc[-1]["Date"]),
        "indicator_columns": indicator_columns,
        "price_columns": price_columns_attr,
        "target_kind": target_kind,
        "target_variants": ("pct_return", "log_return"),
        "target_horizon": 1,
    }
    return X, y, metadata
