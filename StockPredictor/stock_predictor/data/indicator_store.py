"""Indicator-level data access and caching helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from stock_predictor.core import PredictorConfig
from stock_predictor.core.data_fetcher import DataFetcher


@dataclass
class IndicatorDataStore:
    """Fetch and persist indicator-oriented datasets for downstream modules."""

    config: PredictorConfig
    cache_name: str = "indicators.parquet"

    def __post_init__(self) -> None:
        self.fetcher = DataFetcher(self.config)
        self.cache_path: Path = self.config.data_dir / self.cache_name

    def fetch(self, *, force: bool = False) -> pd.DataFrame:
        """Retrieve the indicator dataset, refreshing the cache when requested."""

        if self.cache_path.exists() and not force:
            return pd.read_parquet(self.cache_path)

        indicator_df = self.fetcher.fetch_indicator_data()
        if not indicator_df.empty:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            indicator_df.to_parquet(self.cache_path)
        return indicator_df

    def last_price(self, indicator_df: Optional[pd.DataFrame] = None) -> float | None:
        """Extract the most recent price from the indicator dataset or live feed."""

        frame = indicator_df if indicator_df is not None else self.fetch(force=False)
        if frame is None or frame.empty:
            frame = pd.DataFrame()
        price_col = None
        for candidate in ("Close", "close", "close_price", "last"):
            if candidate in frame.columns:
                price_col = candidate
                break
        if price_col:
            sorted_frame = frame.sort_values(frame.columns[0]).dropna(subset=[price_col])
            if not sorted_frame.empty:
                latest_row = sorted_frame.iloc[-1]
                value = latest_row.get(price_col)
                return float(value) if pd.notna(value) else None

        try:
            live_price, _ = self.fetcher.fetch_live_price(force=False)
            if live_price is not None:
                return float(live_price)
        except Exception:
            return None
        return None


__all__ = ["IndicatorDataStore"]
