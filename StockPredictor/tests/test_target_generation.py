"""Tests for target generation across different prediction horizons."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from stock_predictor.core.features import _generate_targets


class TargetGenerationTests(unittest.TestCase):
    """Validate that generated targets reflect cumulative moves."""

    def test_multi_day_horizon_uses_cumulative_return(self) -> None:
        dates = pd.date_range("2024-01-02", periods=4, freq="B")
        close = pd.Series([100.0, 80.0, 90.0, 99.0])
        merged = pd.DataFrame({"Date": dates, "Close": close})

        targets = _generate_targets(merged, horizons=(2,))

        self.assertIn(2, targets)
        horizon_targets = targets[2]
        cumulative_return = (close.shift(-2) / close) - 1.0

        pd.testing.assert_series_equal(
            horizon_targets["return"], cumulative_return, check_names=False
        )
        self.assertAlmostEqual(horizon_targets["return"].iloc[0], -0.1)
        self.assertAlmostEqual(horizon_targets["log_return"].iloc[0], np.log(0.9))
        self.assertEqual(horizon_targets["direction"].iloc[0], 0.0)

    def test_direction_reflects_multi_day_move(self) -> None:
        dates = pd.date_range("2024-02-01", periods=3, freq="B")
        close = pd.Series([100.0, 80.0, 90.0])
        merged = pd.DataFrame({"Date": dates, "Close": close})

        targets = _generate_targets(merged, horizons=(2,))

        horizon_targets = targets[2]
        expected_cumulative = (close.shift(-2) / close) - 1.0
        self.assertLess(expected_cumulative.iloc[0], 0)
        self.assertEqual(horizon_targets["direction"].iloc[0], 0.0)


if __name__ == "__main__":  # pragma: no cover - test harness
    unittest.main()

