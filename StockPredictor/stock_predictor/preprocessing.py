"""Data processing helpers for feature engineering."""

from __future__ import annotations

import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .sentiment import aggregate_daily_sentiment, attach_sentiment

LOGGER = logging.getLogger(__name__)


def compute_price_features(price_df: pd.DataFrame) -> pd.DataFrame:
    """Compute technical indicators and lag features from price data."""

    if price_df.empty:
        raise ValueError("Price dataframe is empty.")

    df = price_df.copy()
    if "Date" not in df.columns:
        raise ValueError("Expected a 'Date' column in price data.")

    df = df.sort_values("Date")
    df["Return_1d"] = df["Close"].pct_change()
    df["LogReturn_1d"] = np.log(df["Close"]).diff()
    df["SMA_5"] = df["Close"].rolling(window=5, min_periods=1).mean()
    df["SMA_10"] = df["Close"].rolling(window=10, min_periods=1).mean()
    df["EMA_9"] = df["Close"].ewm(span=9, adjust=False).mean()
    df["Volatility_5"] = df["Return_1d"].rolling(window=5, min_periods=1).std()
    df["Volume_Change"] = df["Volume"].pct_change()

    df = df.ffill().bfill()
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

    price_features = compute_price_features(price_df)
    sentiment_df = sentiment_df if sentiment_df is not None else pd.DataFrame()
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
    }
    return X, y, metadata
