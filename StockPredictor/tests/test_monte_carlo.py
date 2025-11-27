import numpy as np

from stock_predictor.core.modeling.simulation import run_monte_carlo


def test_monte_carlo_hits_initial_price_when_equal() -> None:
    prob = run_monte_carlo(
        current_price=100,
        target_price=100,
        drift=0.0,
        volatility=0.0,
        horizon=5,
        paths=500,
    )
    assert prob == 1.0


def test_monte_carlo_no_volatility_below_target() -> None:
    prob = run_monte_carlo(
        current_price=100,
        target_price=110,
        drift=0.0,
        volatility=0.0,
        horizon=10,
        paths=500,
    )
    assert prob == 0.0


def test_monte_carlo_high_positive_drift_hits_target() -> None:
    prob = run_monte_carlo(
        current_price=100,
        target_price=120,
        drift=0.2,
        volatility=0.01,
        horizon=10,
        paths=2000,
        random_state=0,
    )
    assert prob > 0.95
