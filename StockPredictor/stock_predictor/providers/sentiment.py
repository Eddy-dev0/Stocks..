"""Sentiment analysis helpers for financial news text."""

from __future__ import annotations

import logging
from typing import Iterable

import pandas as pd
from pandas.api.types import is_string_dtype
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

LOGGER = logging.getLogger(__name__)
_ANALYZER: SentimentIntensityAnalyzer | None = None


def _get_analyzer() -> SentimentIntensityAnalyzer:
    global _ANALYZER
    if _ANALYZER is None:
        _ANALYZER = SentimentIntensityAnalyzer()
    return _ANALYZER


def score_sentiment(texts: Iterable[str]) -> list[float]:
    """Return compound sentiment scores for the provided texts."""

    analyzer = _get_analyzer()
    scores: list[float] = []
    for text in texts:
        if not isinstance(text, str) or not text.strip():
            scores.append(0.0)
            continue
        scores.append(analyzer.polarity_scores(text).get("compound", 0.0))
    return scores


def attach_sentiment(news_df: pd.DataFrame) -> pd.DataFrame:
    """Add a ``sentiment`` column to the given news dataframe."""

    if news_df.empty:
        return news_df

    normalized_columns = {col.lower(): col for col in news_df.columns}
    text_series: pd.Series | None = None

    for candidate in ("summary", "content", "description", "text", "title"):
        source_column = normalized_columns.get(candidate)
        if source_column is not None:
            text_series = news_df[source_column].fillna("").astype(str)
            break

    if text_series is None:
        string_columns = [
            column
            for column in news_df.columns
            if is_string_dtype(news_df[column])
        ]

        if string_columns:
            combined = news_df[string_columns].apply(
                lambda row: " ".join(
                    value.strip()
                    for value in row
                    if isinstance(value, str) and value.strip()
                ),
                axis=1,
            )
            text_series = combined.str.replace(r"\s+", " ", regex=True).str.strip()

    if text_series is None:
        LOGGER.warning("No textual column found for sentiment analysis.")
        news_df["sentiment"] = 0.0
        return news_df

    news_df = news_df.copy()
    news_df["sentiment"] = score_sentiment(text_series.fillna(""))
    return news_df


def aggregate_daily_sentiment(news_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate sentiment scores by publication date."""

    if news_df.empty or "publishedDate" not in news_df.columns:
        return pd.DataFrame(columns=["Date", "sentiment"])

    df = news_df.copy()
    df["Date"] = pd.to_datetime(df["publishedDate"]).dt.date
    grouped = df.groupby("Date")["sentiment"].mean().reset_index()
    grouped["Date"] = pd.to_datetime(grouped["Date"])
    return grouped
