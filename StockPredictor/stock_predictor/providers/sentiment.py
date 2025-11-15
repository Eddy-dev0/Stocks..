"""Sentiment analysis helpers for financial news text."""

from __future__ import annotations

import logging
from typing import Iterable

import pandas as pd
from pandas.api.types import is_string_dtype
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

try:  # Optional dependency for FinBERT-style sentiment classification
    from transformers import pipeline as hf_pipeline
except Exception:  # pragma: no cover - dependency is optional
    hf_pipeline = None  # type: ignore[assignment]

import numpy as np

LOGGER = logging.getLogger(__name__)
_ANALYZER: SentimentIntensityAnalyzer | None = None
_FINBERT_PIPELINE = None
_FINBERT_ATTEMPTED = False


def _get_analyzer() -> SentimentIntensityAnalyzer:
    global _ANALYZER
    if _ANALYZER is None:
        _ANALYZER = SentimentIntensityAnalyzer()
    return _ANALYZER


def _get_finbert_pipeline():
    """Return a cached HuggingFace pipeline when the dependency is available."""

    global _FINBERT_PIPELINE, _FINBERT_ATTEMPTED
    if _FINBERT_PIPELINE is not None or _FINBERT_ATTEMPTED:
        return _FINBERT_PIPELINE
    _FINBERT_ATTEMPTED = True
    if hf_pipeline is None:  # pragma: no cover - optional dependency
        return None
    try:
        _FINBERT_PIPELINE = hf_pipeline(
            "text-classification",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert",
            return_all_scores=True,
        )
    except Exception as exc:  # pragma: no cover - dependency may be unavailable
        LOGGER.debug("FinBERT sentiment pipeline unavailable: %s", exc)
        _FINBERT_PIPELINE = None
    return _FINBERT_PIPELINE


def score_sentiment(texts: Iterable[str]) -> list[float]:
    """Return compound sentiment scores for the provided texts."""

    analyzer = _get_analyzer()
    scores: list[float] = []
    cleaned: list[str] = []
    for text in texts:
        if not isinstance(text, str):
            cleaned.append("")
        else:
            cleaned.append(text.strip())

    finbert = _get_finbert_pipeline()
    if finbert is not None:
        try:
            results = finbert([text or "" for text in cleaned])
            for result in results:
                label_scores = {entry["label"].lower(): float(entry["score"]) for entry in result}
                positive = label_scores.get("positive", 0.0)
                negative = label_scores.get("negative", 0.0)
                neutral = label_scores.get("neutral", 0.0)
                compound = positive - negative
                if neutral > 0.8 and abs(compound) < 0.05:
                    compound = 0.0
                scores.append(compound)
        except Exception as exc:  # pragma: no cover - fallback to VADER
            LOGGER.debug("FinBERT scoring failed; falling back to VADER: %s", exc)
            scores.clear()

    if not scores or len(scores) != len(cleaned):
        scores = []
        for text in cleaned:
            if not text:
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
    scores = score_sentiment(text_series.fillna(""))
    news_df["sentiment"] = scores
    news_df["sentiment_magnitude"] = np.abs(news_df["sentiment"])
    news_df["sentiment_label"] = np.select(
        [
            news_df["sentiment"] >= 0.15,
            news_df["sentiment"] <= -0.15,
        ],
        ["bullish", "bearish"],
        default="neutral",
    )
    return news_df


def aggregate_daily_sentiment(news_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate sentiment scores by publication date."""

    if news_df.empty or "publishedDate" not in news_df.columns:
        return pd.DataFrame(columns=["Date", "sentiment"])

    df = news_df.copy()
    df["Date"] = pd.to_datetime(df["publishedDate"]).dt.date
    grouped = (
        df.groupby("Date")
        .agg(
            sentiment=("sentiment", "mean"),
            sentiment_magnitude=("sentiment_magnitude", "mean"),
            bullish_ratio=("sentiment_label", lambda s: float((s == "bullish").mean())),
            bearish_ratio=("sentiment_label", lambda s: float((s == "bearish").mean())),
        )
        .reset_index()
    )
    grouped["Date"] = pd.to_datetime(grouped["Date"])
    return grouped
