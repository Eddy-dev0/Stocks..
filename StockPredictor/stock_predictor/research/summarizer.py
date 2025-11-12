"""Summarisation and sentiment helpers for research artefacts."""

from __future__ import annotations

import re
from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


SENTENCE_SPLIT_REGEX = re.compile(r"(?<=[.!?])\s+")


def _normalise_sentences(text: str) -> list[str]:
    sentences = [segment.strip() for segment in SENTENCE_SPLIT_REGEX.split(text) if segment.strip()]
    if not sentences and text.strip():
        sentences = [text.strip()]
    return sentences


@dataclass(slots=True)
class ResearchSummary:
    extractive: str
    abstractive: str
    sentiment_label: str
    sentiment_score: float


class ResearchSummarizer:
    """Combine extractive and abstractive summaries with sentiment tagging."""

    def __init__(self) -> None:
        self._sentiment = SentimentIntensityAnalyzer()

    def summarize(self, text: str, *, max_sentences: int = 3) -> ResearchSummary:
        clean_text = text.strip()
        if not clean_text:
            return ResearchSummary("", "", "neutral", 0.0)

        extractive = self._extractive_summary(clean_text, max_sentences=max_sentences)
        abstractive = self._abstractive_summary(clean_text)
        sentiment_score, sentiment_label = self._sentiment_label(clean_text)
        return ResearchSummary(extractive, abstractive, sentiment_label, sentiment_score)

    def _extractive_summary(self, text: str, *, max_sentences: int) -> str:
        sentences = _normalise_sentences(text)
        if not sentences:
            return ""
        if len(sentences) <= max_sentences:
            return " ".join(sentences)
        vectorizer = TfidfVectorizer(stop_words="english")
        try:
            matrix = vectorizer.fit_transform(sentences)
        except ValueError:
            return " ".join(sentences[:max_sentences])
        scores = matrix.sum(axis=1)
        ranked = scores.A.ravel().argsort()[::-1]
        top_indices = sorted(ranked[:max_sentences])
        return " ".join(sentences[index] for index in top_indices)

    def _abstractive_summary(self, text: str, top_terms: int = 5) -> str:
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        try:
            matrix = vectorizer.fit_transform([text])
        except ValueError:
            first_sentence = _normalise_sentences(text)
            return first_sentence[0] if first_sentence else text[:160]
        features = vectorizer.get_feature_names_out()
        scores = matrix.toarray()[0]
        ranked = scores.argsort()[::-1]
        keywords: list[str] = []
        for index in ranked:
            token = features[index].strip()
            if not token or token.isdigit():
                continue
            if token in keywords:
                continue
            keywords.append(token)
            if len(keywords) >= top_terms:
                break
        if not keywords:
            snippet = text[:160].strip()
            return snippet
        focus = ", ".join(keywords[: min(3, len(keywords))])
        emphasis_index = min(len(keywords) - 1, 3)
        emphasis = keywords[emphasis_index]
        return f"The document highlights {focus} with emphasis on {emphasis}."

    def _sentiment_label(self, text: str) -> tuple[float, str]:
        scores = self._sentiment.polarity_scores(text)
        compound = scores.get("compound", 0.0)
        label = "neutral"
        if compound >= 0.05:
            label = "positive"
        elif compound <= -0.05:
            label = "negative"
        return (float(compound), label)


__all__ = ["ResearchSummarizer", "ResearchSummary"]

