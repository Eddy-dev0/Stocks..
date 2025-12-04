"""Regression tests for plausibility guards on forecast outputs."""

import pytest

from stock_predictor.core import StockPredictorAI


class _ConfigStub:
    expected_low_sigma = 1.0
    k_stop = 1.0
    expected_low_max_volatility = 1.0
    expected_low_floor_window = 20


def test_plausibility_validation_allows_reasonable_move() -> None:
    predictor = StockPredictorAI.__new__(StockPredictorAI)
    predictor.config = _ConfigStub()

    predictor._validate_prediction_ranges(  # type: ignore[attr-defined]
        predicted_return=0.1,
        expected_change_pct=0.1,
        predicted_close=110.0,
        expected_low=102.0,
        stop_loss=101.0,
        anchor_price=100.0,
        horizon=1,
    )


def test_plausibility_validation_blocks_absurd_move() -> None:
    predictor = StockPredictorAI.__new__(StockPredictorAI)
    predictor.config = _ConfigStub()

    with pytest.raises(ValueError):
        predictor._validate_prediction_ranges(  # type: ignore[attr-defined]
            predicted_return=2.5,
            expected_change_pct=2.5,
            predicted_close=50.0,
            expected_low=60.0,
            stop_loss=55.0,
            anchor_price=100.0,
            horizon=1,
        )
