"""Tests for the feature registry driven assembler."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from stock_predictor.features import FeatureAssembler, default_feature_toggles
from stock_predictor.features.feature_registry import (
    FeatureDependencyError,
    FeatureNotImplementedError,
)


class FeatureAssemblerRegistryTests(unittest.TestCase):
    """Verify registry integration and dependency handling."""

    def setUp(self) -> None:
        dates = pd.date_range("2023-01-02", periods=60, freq="B")
        close = np.linspace(100.0, 120.0, len(dates))
        self.price_df = pd.DataFrame(
            {
                "Date": dates,
                "Open": close,
                "High": close + 1.5,
                "Low": close - 1.5,
                "Close": close,
                "Volume": np.linspace(1_000_000, 2_000_000, len(dates)),
            }
        )

    def test_technical_group_enabled(self) -> None:
        toggles = default_feature_toggles()
        toggles["sentiment"] = False
        assembler = FeatureAssembler(toggles, horizons=(1,))

        result = assembler.build(self.price_df, None, sentiment_enabled=False)

        self.assertIn("Return_1d", result.features.columns)
        groups = result.metadata["feature_groups"]
        self.assertTrue(groups["technical"]["executed"])
        self.assertIn("Return_1d", groups["technical"]["columns"])
        self.assertEqual(groups["technical"]["status"], "executed")

    def test_disabling_technical_allows_macro_only(self) -> None:
        toggles = default_feature_toggles()
        toggles.update({"technical": False, "macro": True, "sentiment": False})
        assembler = FeatureAssembler(toggles, horizons=(1,))

        result = assembler.build(self.price_df, None, sentiment_enabled=False)

        self.assertNotIn("Return_1d", result.features.columns)
        self.assertIn("Volatility_21", result.features.columns)
        groups = result.metadata["feature_groups"]
        self.assertEqual(groups["technical"]["status"], "disabled")
        self.assertTrue(groups["macro"]["executed"])

    def test_dependency_failure_for_options_group(self) -> None:
        toggles = default_feature_toggles()
        toggles.update({"options": True, "sentiment": False})
        toggles["identification"] = False
        assembler = FeatureAssembler(toggles, horizons=(1,))

        with self.assertRaises(FeatureDependencyError):
            assembler.build(self.price_df, None, sentiment_enabled=False)

    def test_unimplemented_group_raises(self) -> None:
        toggles = default_feature_toggles()
        toggles.update({"identification": True, "sentiment": False})
        assembler = FeatureAssembler(toggles, horizons=(1,))

        with self.assertRaises(FeatureNotImplementedError):
            assembler.build(self.price_df, None, sentiment_enabled=False)


if __name__ == "__main__":  # pragma: no cover - test harness
    unittest.main()

