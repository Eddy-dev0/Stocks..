"""Regression tests for plausibility guards on forecast outputs."""

import pytest

from stock_predictor.core import StockPredictorAI


class _ConfigStub:
    expected_low_sigma = 1.0
    k_stop = 1.0
    expected_low_max_volatility = 1.0
    expected_low_floor_window = 20
    plausibility_sigma_multiplier = 3.0


def test_plausibility_validation_allows_reasonable_move() -> None:
    predictor = StockPredictorAI.__new__(StockPredictorAI)
    predictor.config = _ConfigStub()
    predictor.metadata = {}
    warnings: list[str] = []

    result = predictor._validate_prediction_ranges(  # type: ignore[attr-defined]
        predicted_return=0.1,
        expected_change_pct=0.1,
        predicted_close=110.0,
        expected_low=102.0,
        stop_loss=101.0,
        anchor_price=100.0,
        horizon=1,
        prediction_warnings=warnings,
    )

    assert result == pytest.approx((0.1, 0.1, 110.0, 102.0, 101.0))
    assert warnings == []


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


def test_dynamic_band_clamps_stable_ticker() -> None:
    predictor = StockPredictorAI.__new__(StockPredictorAI)
    predictor.config = _ConfigStub()
    predictor.metadata = {"recent_return_std": 0.01}
    warnings: list[str] = []

    result = predictor._validate_prediction_ranges(  # type: ignore[attr-defined]
        predicted_return=0.2,
        expected_change_pct=0.2,
        predicted_close=120.0,
        expected_low=90.0,
        stop_loss=88.0,
        anchor_price=100.0,
        horizon=1,
        prediction_warnings=warnings,
    )

    max_move = pytest.approx(0.03)
    assert result[0] == max_move
    assert result[1] == max_move
    assert result[2] == pytest.approx(103.0)
    assert warnings and "exceeds volatility band" in warnings[0]


def test_dynamic_band_accepts_volatile_ticker() -> None:
    predictor = StockPredictorAI.__new__(StockPredictorAI)
    predictor.config = _ConfigStub()
    predictor.metadata = {"recent_return_std": 0.08}
    warnings: list[str] = []

    result = predictor._validate_prediction_ranges(  # type: ignore[attr-defined]
        predicted_return=0.15,
        expected_change_pct=0.15,
        predicted_close=115.0,
        expected_low=100.0,
        stop_loss=98.0,
        anchor_price=100.0,
        horizon=1,
        prediction_warnings=warnings,
    )

    assert result == pytest.approx((0.15, 0.15, 115.0, 100.0, 98.0))
    assert warnings == []
