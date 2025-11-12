"""Research orchestration services for crawling and manual ingestion."""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable
from urllib.parse import urlsplit

from bs4 import BeautifulSoup

from stock_predictor.core.config import PredictorConfig
from stock_predictor.providers.database import Database

from .crawler import AsyncCrawler
from .summarizer import ResearchSummarizer


def _extract_text(html: str | None) -> str:
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    return " ".join(segment.strip() for segment in soup.stripped_strings)


def _normalise_allow_list(values: Iterable[str] | None) -> set[str]:
    allow_set: set[str] = set()
    if not values:
        return allow_set
    for value in values:
        token = str(value or "").strip().lower()
        if not token:
            continue
        allow_set.add(token)
    return allow_set


@dataclass
class ResearchService:
    """High-level interface for orchestrating research ingestion."""

    config: PredictorConfig
    database: Database | None = None
    summarizer: ResearchSummarizer | None = None
    crawler: AsyncCrawler | None = None
    _allow_list: set[str] = field(default_factory=set, init=False)
    _api_keys: set[str] = field(default_factory=set, init=False)

    def __post_init__(self) -> None:
        self.database = self.database or Database(self.config.database_url)
        self.summarizer = self.summarizer or ResearchSummarizer()
        self.crawler = self.crawler or AsyncCrawler()
        self._allow_list = _normalise_allow_list(getattr(self.config, "research_allow_list", None))
        keys = getattr(self.config, "research_api_keys", None) or ()
        self._api_keys = {str(key).strip() for key in keys if str(key).strip()}

    async def aclose(self) -> None:
        if self.crawler:
            await self.crawler.aclose()

    # ------------------------------------------------------------------
    # API key helpers
    # ------------------------------------------------------------------
    def is_authorised(self, api_key: str | None) -> bool:
        if not self._api_keys:
            return True
        if not api_key:
            return False
        return api_key in self._api_keys

    # ------------------------------------------------------------------
    # Public ingestion endpoints
    # ------------------------------------------------------------------
    async def crawl(self, url: str, *, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        self._validate_url(url)
        if self.crawler is None:
            raise RuntimeError("Crawler service is not initialised")
        html = await self.crawler.fetch(url)
        text = _extract_text(html)
        return await self._persist(url, text, source="crawler", metadata=metadata or {})

    async def ingest_snippet(
        self,
        url: str,
        *,
        text: str | None = None,
        html: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self._validate_url(url)
        body = (text or "").strip()
        if not body and html:
            body = _extract_text(html)
        if not body:
            raise ValueError("Snippet ingestion requires either plain text or HTML content")
        meta = dict(metadata or {})
        if html:
            meta.setdefault("snippet_length", len(html))
        return await self._persist(url, body, source="snippet", metadata=meta)

    async def get_feed(self, *, limit: int = 50) -> list[dict[str, Any]]:
        if self.database is None:
            raise RuntimeError("Database instance is not available")
        records = await asyncio.to_thread(self.database.get_research_artifacts, limit)
        for record in records:
            captured_at = record.get("captured_at")
            if isinstance(captured_at, datetime):
                record["captured_at"] = captured_at.isoformat()
        return records

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    async def _persist(
        self,
        url: str,
        text: str,
        *,
        source: str,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        if self.database is None or self.summarizer is None:
            raise RuntimeError("Research service is not fully initialised")
        cleaned = text.strip()
        if not cleaned:
            raise ValueError("Content must not be empty after cleaning")
        summary = await asyncio.to_thread(self.summarizer.summarize, cleaned)
        content_hash = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()
        payload = {
            "url": url,
            "content_hash": content_hash,
            "captured_at": datetime.utcnow(),
            "source": source,
            "raw_content": cleaned,
            "extractive_summary": summary.extractive,
            "abstractive_summary": summary.abstractive,
            "sentiment_label": summary.sentiment_label,
            "sentiment_score": summary.sentiment_score,
            "metadata": metadata,
        }
        stored = await asyncio.to_thread(self.database.upsert_research_artifact, payload)
        captured_at = stored.get("captured_at")
        if isinstance(captured_at, datetime):
            stored["captured_at"] = captured_at.isoformat()
        return stored

    def _validate_url(self, url: str) -> None:
        parts = urlsplit(url)
        if not parts.scheme or not parts.netloc:
            raise ValueError("A fully qualified URL is required")
        if not self._allow_list or "*" in self._allow_list:
            return
        hostname = parts.hostname or ""
        if hostname.lower() not in self._allow_list:
            raise PermissionError(f"URL host '{hostname}' is not in the allow list")


__all__ = ["ResearchService"]

