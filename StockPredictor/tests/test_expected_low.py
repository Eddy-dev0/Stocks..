from __future__ import annotations

"""Unit tests covering expected low calculations across modules."""

from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stock_predictor.core import StockPredictorAI  # noqa: E402  pylint: disable=wrong-import-position
from stock_predictor.ui_app import StockPredictorDesktopApp  # noqa: E402  pylint: disable=wrong-import-position


class _ConfigStub:
    expected_low_sigma = 1.0


@pytest.mark.parametrize("volatility", [0.02, -0.02])
def test_model_expected_low_treats_volatility_as_percentage(volatility: float) -> None:
    predictor = StockPredictorAI.__new__(StockPredictorAI)
    predictor.config = _ConfigStub()

    result = predictor._compute_expected_low(  # type: ignore[attr-defined]
        100.0,
        volatility,
        quantile_forecasts=None,
        prediction_intervals=None,
    )

    assert result == pytest.approx(98.0)


def test_model_expected_low_is_clamped_at_zero() -> None:
    predictor = StockPredictorAI.__new__(StockPredictorAI)
    predictor.config = _ConfigStub()

    result = predictor._compute_expected_low(  # type: ignore[attr-defined]
        10.0,
        1.0,
        quantile_forecasts=None,
        prediction_intervals=None,
    )

    assert result == pytest.approx(0.0)


def test_ui_expected_low_matches_model_logic() -> None:
    app = StockPredictorDesktopApp.__new__(StockPredictorDesktopApp)
    app.expected_low_multiplier = 1.0

    expected_low = app._compute_expected_low(  # type: ignore[attr-defined]
        {
            "predicted_close": 100.0,
            "predicted_volatility": 0.02,
        },
        multiplier=1.0,
    )

    assert expected_low == pytest.approx(98.0)
