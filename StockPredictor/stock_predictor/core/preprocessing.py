"""Data processing helpers for feature engineering."""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

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


def _identify_price_columns(df: pd.DataFrame) -> list[str]:
    price_columns: list[str] = []
    for column in df.columns:
        if column in PRICE_BASE_COLUMNS or column in PRICE_EXACT_COLUMNS:
            price_columns.append(column)
            continue
        if column.startswith(PRICE_PREFIXES):
            price_columns.append(column)
    return price_columns


def compute_price_features(price_df: pd.DataFrame) -> pd.DataFrame:
    """Compute technical indicators and lag features from price data."""

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

    df = df.sort_values("Date").reset_index(drop=True)
    df["Return_1d"] = df["Close"].pct_change()
    df["LogReturn_1d"] = np.log(df["Close"]).diff()
    df["SMA_5"] = df["Close"].rolling(window=5, min_periods=1).mean()
    df["SMA_10"] = df["Close"].rolling(window=10, min_periods=1).mean()
    df["EMA_9"] = df["Close"].ewm(span=9, adjust=False).mean()
    df["Volatility_5"] = df["Return_1d"].rolling(window=5, min_periods=1).std()
    df["Volume_Change"] = df["Volume"].pct_change()

    indicator_result = compute_indicators(df)
    df = pd.concat([df, indicator_result.dataframe], axis=1)

    df = df.ffill().bfill()
    df.attrs["indicator_columns"] = list(indicator_result.columns)
    df.attrs["price_columns"] = _identify_price_columns(df)
    return df


def merge_with_sentiment(
    price_df: pd.DataFrame, news_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Attach aggregated sentiment information to the price dataframe."""

    if news_df.empty:
        LOGGER.info("No news data available; skipping sentiment merge.")
        return price_df, pd.DataFrame()

    scored = attach_sentiment(news_df)
    aggregated = aggregate_daily_sentiment(scored)
    merged = price_df.merge(aggregated, on="Date", how="left")
    merged["sentiment"] = merged["sentiment"].fillna(0.0)
    return merged, aggregated


def build_supervised_dataset(
    price_df: pd.DataFrame, sentiment_df: pd.DataFrame | None = None
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, object]]:
    """Prepare the feature matrix and target vector."""

    price_features, waves = compute_price_features(price_df)
    sentiment_df = sentiment_df if sentiment_df is not None else pd.DataFrame()
    indicator_columns = price_features.attrs.get("indicator_columns", [])
    price_columns_attr = price_features.attrs.get("price_columns", [])
    merged, aggregated = merge_with_sentiment(price_features, sentiment_df)

    dataset = merged.copy()
    target = dataset["Close"].shift(-1)
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
    }
    return X, y, metadata
