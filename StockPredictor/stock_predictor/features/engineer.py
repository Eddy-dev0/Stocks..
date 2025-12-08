"""Feature engineering on indicator-oriented datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import pandas as pd

from stock_predictor.data.indicator_store import IndicatorDataStore


@dataclass
class IndicatorFeatureEngineer:
    """Prepare model-ready features derived from indicator datasets."""

    indicator_store: IndicatorDataStore

    def build_features(self, *, force_refresh: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Return feature matrix and placeholder targets for downstream training."""

        indicator_df = self.indicator_store.fetch(force=force_refresh)
        if indicator_df is None:
            indicator_df = pd.DataFrame()

        feature_df = indicator_df.copy()
        feature_df = feature_df.replace({"": pd.NA}).dropna(how="all")
        feature_df = feature_df.sort_index()

        targets = pd.DataFrame(index=feature_df.index)
        return feature_df, targets


__all__ = ["IndicatorFeatureEngineer"]
