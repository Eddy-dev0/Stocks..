from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
import sys

from typing import Sequence

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stock_predictor.core.config import PredictorConfig
from stock_predictor.core.trend_finder import TrendFinder
from stock_predictor.core.pipeline import MarketDataETL
from stock_predictor.providers.base import (
    DatasetType,
    PriceBar,
    ProviderFailure,
    ProviderFetchSummary,
    ProviderResult,
)
from stock_predictor.providers.data import DataFetcher


class FakeAI:
    """Minimal AI stub that pulls price data through the DataFetcher."""

    def __init__(self, config: PredictorConfig, *, horizon: int | None = None) -> None:
        self.config = config
        self.horizon = self.config.resolve_horizon(horizon)
        self.fetcher = DataFetcher(config)
        self.metadata: dict[str, object] = {}

    def prepare_features(self):
        price_df = self.fetcher.fetch_price_data()
        if price_df.empty:
            raise RuntimeError("Expected cached price data for TrendFinder test.")
        latest_close = float(price_df.iloc[-1]["Close"])
        features = pd.DataFrame(
            [
                {
                    "technical_score": 8.0,
                    "macro_score": 6.0,
                    "sentiment_score": 7.0,
                }
            ]
        )
        self.metadata = {
            "latest_features": features,
            "feature_categories": {
                "technical_score": "technical indicator",
                "macro_score": "macro trend",
                "sentiment_score": "sentiment gauge",
            },
        }
        return features, {}, {}

    def predict(self, horizon: int | None = None):
        return {
            "expected_change_pct": 0.02,
            "predicted_return": 0.02,
            "direction_probability_up": 0.98,
            "direction_probability_down": 0.02,
            "confluence_confidence": 0.97,
            "signal_confluence": {"score": 0.97, "passed": True},
        }


def test_trend_finder_reuses_cached_prices_when_rate_limited(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """TrendFinder should fall back to cached price data when providers rate-limit."""

    MarketDataETL._memory_cache.clear()
    call_counts: defaultdict[tuple[str, DatasetType], int] = defaultdict(int)
    providers_seen: list[tuple[str, ...] | None] = []

    def fake_fetch(
        self: MarketDataETL,
        dataset: DatasetType,
        params: dict[str, object] | None = None,
        *,
        providers: Sequence[str] | None = None,
    ) -> ProviderFetchSummary:
        assert dataset == DatasetType.PRICES
        providers_seen.append(tuple(providers) if providers is not None else None)
        key = (self.config.ticker, dataset)
        call_counts[key] += 1
        if call_counts[key] == 1:
            bar = PriceBar(
                symbol=self.config.ticker,
                timestamp=datetime(2024, 1, 2, 16, 0, tzinfo=timezone.utc),
                open=100.0,
                high=101.0,
                low=99.5,
                close=100.5,
                volume=1_000_000,
            )
            return ProviderFetchSummary(
                results=[
                    ProviderResult(
                        dataset_type=dataset,
                        source="yahoo_finance",
                        records=[bar],
                        metadata={"symbol": self.config.ticker},
                    )
                ],
                failures=[],
            )
        failure = ProviderFailure(
            provider="yahoo_finance",
            error="Rate limited",
            status_code=429,
            retry_after=45.0,
            is_rate_limited=True,
        )
        return ProviderFetchSummary(results=[], failures=[failure])

    monkeypatch.setattr(MarketDataETL, "_fetch_dataset", fake_fetch, raising=False)

    database_path = tmp_path / "prices.db"
    config = PredictorConfig(
        ticker="AAPL",
        start_date=date(2024, 1, 1),
        interval="1d",
        data_dir=tmp_path,
        database_url=f"sqlite:///{database_path}",
        sentiment=False,
        prediction_horizons=(5,),
    )
    config.ensure_directories()

    trend = TrendFinder(config, ai_factory=FakeAI)
    insights = trend.scan(universe=["AAPL"], limit=1)

    assert insights, "TrendFinder should return insights using cached data."
    assert providers_seen[0][0] == "alpha_vantage"
    assert "stooq" in providers_seen[0]
    assert providers_seen[-1] is None
    assert call_counts[("AAPL", DatasetType.PRICES)] == 2
