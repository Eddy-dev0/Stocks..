"""Asynchronous crawling utilities that respect robots.txt policies."""

from __future__ import annotations

import asyncio
import logging
from typing import Dict
from urllib.parse import urljoin, urlsplit
from urllib.robotparser import RobotFileParser

import httpx

LOGGER = logging.getLogger(__name__)


class AsyncCrawler:
    """Fetch remote documents asynchronously while respecting ``robots.txt``."""

    def __init__(
        self,
        *,
        user_agent: str = "StockPredictorResearchBot/1.0",
        timeout: float = 10.0,
    ) -> None:
        self.user_agent = user_agent
        self.timeout = timeout
        self._client = httpx.AsyncClient(timeout=timeout, headers={"User-Agent": user_agent})
        self._parsers: Dict[str, RobotFileParser] = {}
        self._lock = asyncio.Lock()

    async def aclose(self) -> None:
        """Close the underlying HTTP client."""

        await self._client.aclose()

    async def fetch(self, url: str) -> str:
        """Download the document at ``url`` if permitted."""

        if not url:
            raise ValueError("url must be a non-empty string")
        parser = await self._get_robot_parser(url)
        if parser and not parser.can_fetch(self.user_agent, url):
            raise PermissionError(f"Robots policy forbids crawling {url}")
        response = await self._client.get(url)
        response.raise_for_status()
        LOGGER.debug("Fetched %s with status %s", url, response.status_code)
        return response.text

    async def _get_robot_parser(self, url: str) -> RobotFileParser | None:
        parts = urlsplit(url)
        if not parts.scheme or not parts.netloc:
            raise ValueError(f"Invalid URL provided: {url}")
        base = f"{parts.scheme}://{parts.netloc}"
        async with self._lock:
            parser = self._parsers.get(base)
            if parser is not None:
                return parser
            parser = RobotFileParser()
            robots_url = urljoin(base, "/robots.txt")
            try:
                response = await self._client.get(robots_url)
            except httpx.HTTPError as exc:  # pragma: no cover - network guard
                LOGGER.debug("Unable to download robots.txt from %s: %s", robots_url, exc)
                self._parsers[base] = parser
                parser.parse([])
                return parser
            if response.status_code == httpx.codes.OK:
                parser.parse(response.text.splitlines())
            else:
                parser.parse([])
            self._parsers[base] = parser
            return parser


__all__ = ["AsyncCrawler"]

