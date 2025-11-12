"""Asynchronous data pipeline orchestrating provider fetches and persistence."""

from __future__ import annotations

import asyncio
import logging
from collections import Counter
from typing import Iterable, Mapping, Sequence

import pandas as pd

from stock_predictor.core.config import PredictorConfig
from stock_predictor.providers.base import (
    DatasetType,
    EconomicIndicator,
    NewsArticle,
    PriceBar,
    ProviderRegistry,
    ProviderRequest,
    ProviderResult,
    SentimentSignal,
    build_default_registry,
)
from stock_predictor.providers.database import Database

LOGGER = logging.getLogger(__name__)


class PipelineResult:
    """Container summarising a dataset refresh."""

    def __init__(
        self,
        dataset_type: DatasetType,
        frame: pd.DataFrame | None,
        persisted: int,
        sources: Mapping[str, int],
    ) -> None:
        self.dataset_type = dataset_type
        self.frame = frame
        self.persisted = persisted
        self.sources = dict(sources)

    def to_dict(self) -> dict[str, object]:
        return {
            "dataset": self.dataset_type,
            "persisted": self.persisted,
            "sources": self.sources,
            "rows": 0 if self.frame is None else int(self.frame.shape[0]),
        }


class AsyncDataPipeline:
    """Pipeline coordinating asynchronous provider fetches."""

    def __init__(
        self,
        config: PredictorConfig,
        *,
        registry: ProviderRegistry | None = None,
        database: Database | None = None,
    ) -> None:
        self.config = config
        self.registry = registry or build_default_registry()
        self.database = database or Database(config.database_url)
        self._closed = False

    async def aclose(self) -> None:
        if self._closed:
            return
        await self.registry.aclose()
        self._closed = True

    async def __aenter__(self) -> "AsyncDataPipeline":
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    async def refresh(
        self,
        datasets: Sequence[DatasetType] | None = None,
        *,
        persist: bool = True,
        request_overrides: Mapping[DatasetType, Mapping[str, object]] | None = None,
    ) -> dict[DatasetType, PipelineResult]:
        """Fetch and optionally persist datasets from all registered providers."""

        target_datasets = list(datasets or (DatasetType.PRICES, DatasetType.NEWS))
        tasks = []
        for dataset in target_datasets:
            params = dict(self._default_params(dataset))
            if request_overrides and dataset in request_overrides:
                params.update(request_overrides[dataset])
            request = ProviderRequest(dataset_type=dataset, symbol=self.config.ticker, params=params)
            tasks.append(self.registry.fetch_all(request))

        provider_results = await asyncio.gather(*tasks)
        summary: dict[DatasetType, PipelineResult] = {}
        for dataset, results in zip(target_datasets, provider_results):
            frame, persisted, counts = self._coalesce(dataset, results, persist=persist)
            summary[dataset] = PipelineResult(dataset, frame, persisted, counts)
        return summary

    def _default_params(self, dataset: DatasetType) -> Mapping[str, object]:
        if dataset == DatasetType.PRICES:
            params: dict[str, object] = {
                "interval": self.config.interval,
            }
            if self.config.start_date:
                params["start"] = self.config.start_date.isoformat()
            if self.config.end_date:
                params["end"] = self.config.end_date.isoformat()
            return params
        if dataset == DatasetType.NEWS:
            return {"query": self.config.ticker, "limit": 50}
        if dataset == DatasetType.MACRO:
            return {"series_id": self.config.ticker}
        return {}

    def _coalesce(
        self,
        dataset: DatasetType,
        results: Sequence[ProviderResult],
        *,
        persist: bool,
    ) -> tuple[pd.DataFrame | None, int, Mapping[str, int]]:
        records: list[object] = []
        source_counts: Counter[str] = Counter()
        for result in results:
            records.extend(result.records)
            source_counts[result.source] += len(result.records)
        if dataset == DatasetType.PRICES:
            frame = self._dedupe_prices(records)
            persisted = self._persist_prices(frame) if persist and frame is not None else 0
            return frame, persisted, source_counts
        if dataset == DatasetType.NEWS:
            frame = self._dedupe_news(records)
            persisted = self._persist_news(frame) if persist and frame is not None else 0
            return frame, persisted, source_counts
        if dataset == DatasetType.MACRO:
            frame = self._dedupe_indicators(records)
            persisted = self._persist_indicators(frame) if persist and frame is not None else 0
            return frame, persisted, source_counts
        if dataset == DatasetType.SENTIMENT:
            frame = self._dedupe_sentiment(records)
            persisted = self._persist_sentiment(frame) if persist and frame is not None else 0
            return frame, persisted, source_counts
        LOGGER.info("No persistence strategy for dataset %s", dataset)
        return None, 0, source_counts

    def _dedupe_prices(self, records: Iterable[object]) -> pd.DataFrame | None:
        bars = [record for record in records if isinstance(record, PriceBar)]
        if not bars:
            return None
        frame = pd.DataFrame([bar.as_frame_row() for bar in bars])
        frame = frame.dropna(how="all")
        if frame.empty:
            return None
        frame = frame.sort_values("Date").drop_duplicates(subset=["Date"], keep="last")
        return frame.reset_index(drop=True)

    def _dedupe_news(self, records: Iterable[object]) -> pd.DataFrame | None:
        articles = [record for record in records if isinstance(record, NewsArticle)]
        if not articles:
            return None
        frame = pd.DataFrame([article.as_record() for article in articles])
        if frame.empty:
            return None
        frame = frame.sort_values("PublishedAt").drop_duplicates(subset=["Url", "Title"], keep="last")
        return frame.reset_index(drop=True)

    def _dedupe_indicators(self, records: Iterable[object]) -> pd.DataFrame | None:
        indicators = [record for record in records if isinstance(record, EconomicIndicator)]
        if not indicators:
            return None
        frame = pd.DataFrame([indicator.as_record() for indicator in indicators])
        if frame.empty:
            return None
        frame = frame.sort_values("Date").drop_duplicates(
            subset=["Indicator", "Date", "Ticker"], keep="last"
        )
        return frame.reset_index(drop=True)

    def _dedupe_sentiment(self, records: Iterable[object]) -> pd.DataFrame | None:
        signals = [record for record in records if isinstance(record, SentimentSignal)]
        if not signals:
            return None
        frame = pd.DataFrame([signal.as_record() for signal in signals])
        if frame.empty:
            return None
        frame = frame.sort_values("AsOf").drop_duplicates(
            subset=["Provider", "SignalType", "AsOf"], keep="last"
        )
        return frame.reset_index(drop=True)

    def _persist_prices(self, frame: pd.DataFrame | None) -> int:
        if frame is None or frame.empty:
            return 0
        return self.database.upsert_prices(self.config.ticker, self.config.interval, frame)

    def _persist_news(self, frame: pd.DataFrame | None) -> int:
        if frame is None or frame.empty:
            return 0
        records = frame.to_dict(orient="records")
        return self.database.upsert_news(records)

    def _persist_indicators(self, frame: pd.DataFrame | None) -> int:
        if frame is None or frame.empty:
            return 0
        records = frame.rename(columns={"Indicator": "Indicator", "Value": "Value"})
        return self.database.upsert_indicators(
            ticker=self.config.ticker,
            interval=self.config.interval,
            records=records.to_dict(orient="records"),
        )

    def _persist_sentiment(self, frame: pd.DataFrame | None) -> int:
        if frame is None or frame.empty:
            return 0
        records = frame.to_dict(orient="records")
        return self.database.upsert_sentiment_signals(records)


__all__ = ["AsyncDataPipeline", "PipelineResult"]
