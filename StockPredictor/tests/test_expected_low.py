from __future__ import annotations

"""Unit tests covering expected low calculations across modules."""

from pathlib import Path
import sys

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stock_predictor.core import StockPredictorAI  # noqa: E402  pylint: disable=wrong-import-position
from stock_predictor.ui_app import StockPredictorDesktopApp  # noqa: E402  pylint: disable=wrong-import-position


class _ConfigStub:
    expected_low_sigma = 1.0
    k_stop = 1.0
    expected_low_max_volatility = 1.0
    expected_low_floor_window = 20


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
    app.expected_low_max_volatility = 1.0
    app.expected_low_floor_window = 20

    expected_low = app._compute_expected_low(  # type: ignore[attr-defined]
        {
            "predicted_close": 100.0,
            "predicted_volatility": 0.02,
        },
        multiplier=1.0,
    )

    assert expected_low == pytest.approx(98.0)


def test_model_expected_low_prefers_indicator_floor() -> None:
    predictor = StockPredictorAI.__new__(StockPredictorAI)
    predictor.config = _ConfigStub()
    predictor.metadata = {
        "latest_features": pd.DataFrame(
            {
                "BB_Lower_20": [96.0],
                "Support_1": [94.5],
                "Supertrend_7": [95.5],
                "Supertrend_Direction_7": [1],
            }
        )
    }

    indicator_floor, components = predictor._indicator_support_floor()  # type: ignore[attr-defined]
    assert indicator_floor == pytest.approx(94.5)
    assert "Support_1" in components

    expected_low = predictor._compute_expected_low(  # type: ignore[attr-defined]
        100.0,
        0.1,
        quantile_forecasts=None,
        prediction_intervals=None,
        indicator_floor=indicator_floor,
    )

    assert expected_low == pytest.approx(94.5)


def test_model_stop_loss_uses_percentage_formula() -> None:
    predictor = StockPredictorAI.__new__(StockPredictorAI)
    predictor.config = _ConfigStub()

    stop_loss = predictor._compute_stop_loss(  # type: ignore[attr-defined]
        100.0,
        0.02,
    )

    assert stop_loss == pytest.approx(98.0)


def test_model_stop_loss_is_clamped_between_zero_and_close() -> None:
    predictor = StockPredictorAI.__new__(StockPredictorAI)
    predictor.config = _ConfigStub()

    stop_loss = predictor._compute_stop_loss(  # type: ignore[attr-defined]
        10.0,
        5.0,
    )

    assert stop_loss == pytest.approx(0.0)


def test_model_stop_loss_falls_back_to_expected_low() -> None:
    predictor = StockPredictorAI.__new__(StockPredictorAI)
    predictor.config = _ConfigStub()

    stop_loss = predictor._compute_stop_loss(  # type: ignore[attr-defined]
        100.0,
        None,
        expected_low=92.0,
    )

    assert stop_loss == pytest.approx(92.0)


def test_ui_expected_low_uses_indicator_snapshot() -> None:
    app = StockPredictorDesktopApp.__new__(StockPredictorDesktopApp)
    app.expected_low_multiplier = 1.0
    app.expected_low_max_volatility = 1.0
    app.expected_low_floor_window = 20
    app.feature_snapshot = pd.DataFrame({
        "BB_Lower_20": [97.0],
        "Support_1": [93.0],
        "Supertrend_7": [95.0],
    })
    app.indicator_history = None

    expected_low = app._compute_expected_low(  # type: ignore[attr-defined]
        {
            "predicted_close": 100.0,
            "predicted_volatility": 0.1,
        },
        multiplier=1.0,
    )

    assert expected_low == pytest.approx(93.0)


def test_model_expected_low_interprets_percentage_scale() -> None:
    predictor = StockPredictorAI.__new__(StockPredictorAI)
    predictor.config = _ConfigStub()
    predictor.metadata = {}

    expected_low = predictor._compute_expected_low(  # type: ignore[attr-defined]
        100.0,
        2.0,
        quantile_forecasts=None,
        prediction_intervals=None,
    )

    assert expected_low == pytest.approx(98.0)


def test_model_expected_low_clips_to_drawdown_cap() -> None:
    predictor = StockPredictorAI.__new__(StockPredictorAI)
    predictor.config = _ConfigStub()
    predictor.config.expected_low_max_volatility = 0.5
    predictor.metadata = {"max_drawdown_fraction": 0.03}

    expected_low = predictor._compute_expected_low(  # type: ignore[attr-defined]
        100.0,
        0.5,
        quantile_forecasts=None,
        prediction_intervals=None,
    )

    assert expected_low == pytest.approx(97.0)
