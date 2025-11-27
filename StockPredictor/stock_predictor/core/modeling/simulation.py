"""Simulation helpers for Monte Carlo-based scenario analysis."""

from __future__ import annotations

import numpy as np


def run_monte_carlo(
    *,
    current_price: float,
    target_price: float,
    drift: float,
    volatility: float,
    horizon: int,
    paths: int = 10_000,
    random_state: int | None = None,
) -> float:
    """Estimate the probability of hitting ``target_price`` within ``horizon`` steps.

    The simulation uses a vectorized geometric Brownian motion with constant drift
    and volatility. The output is the fraction of simulated price paths that touch or
    exceed the target at any point, including the initial price.
    """

    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if paths <= 0:
        raise ValueError("paths must be positive")

    current = float(current_price)
    target = float(target_price)
    mu = float(drift)
    sigma = float(volatility)

    if not np.isfinite(current) or not np.isfinite(target):
        raise ValueError("current_price and target_price must be finite numbers")

    steps = int(horizon)
    rng = np.random.default_rng(random_state)
    dt = 1.0

    deterministic_increment = (mu - 0.5 * sigma**2) * dt
    if sigma == 0.0:
        increments = np.full((paths, steps), deterministic_increment, dtype=float)
    else:
        noise = rng.standard_normal((paths, steps)) * np.sqrt(dt)
        increments = deterministic_increment + sigma * noise

    log_paths = np.cumsum(increments, axis=1)
    prices = current * np.exp(log_paths)
    prices_with_start = np.concatenate([
        np.full((paths, 1), current, dtype=float),
        prices,
    ], axis=1)

    hits = (prices_with_start >= target).any(axis=1)
    return float(np.mean(hits))
