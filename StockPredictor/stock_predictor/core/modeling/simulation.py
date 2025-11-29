"""Simulation helpers for Monte Carlo-based scenario analysis."""

from __future__ import annotations

import numpy as np


def _simulate_gbm_hits(
    *,
    current_price: float,
    target_price: float,
    drift: float,
    volatility: float,
    horizon: int,
    paths: int,
    rng: np.random.Generator,
) -> int:
    """Return number of simulated paths that reach ``target_price``.

    The helper performs a vectorised geometric Brownian motion rollout and counts
    how many trajectories touch or exceed the target. Returning hits instead of a
    probability enables incremental aggregation for adaptive Monte Carlo runs.
    """

    current = float(current_price)
    target = float(target_price)
    mu = float(drift)
    sigma = float(volatility)

    steps = int(horizon)
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
    return int(np.count_nonzero(hits))


def run_monte_carlo(
    *,
    current_price: float,
    target_price: float,
    drift: float,
    volatility: float,
    horizon: int,
    paths: int = 500_000,
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

    hits = _simulate_gbm_hits(
        current_price=current,
        target_price=target,
        drift=mu,
        volatility=sigma,
        horizon=steps,
        paths=paths,
        rng=rng,
    )
    return float(hits) / float(paths)


def run_monte_carlo_adaptive(
    *,
    current_price: float,
    target_price: float,
    drift: float,
    volatility: float,
    horizon: int,
    initial_paths: int = 500_000,
    max_paths: int = 2_000_000,
    precision_target: float = 0.001,
    random_state: int | None = None,
) -> tuple[float, int, float]:
    """Run Monte Carlo batches until the standard error drops below the target.

    The function aggregates hit counts incrementally to avoid exhausting memory
    while still converging toward a more precise probability estimate. A maximum
    path budget guards against unbounded computation.
    """

    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if initial_paths <= 0:
        raise ValueError("initial_paths must be positive")
    if max_paths <= 0:
        raise ValueError("max_paths must be positive")
    if not np.isfinite(precision_target) or precision_target <= 0:
        raise ValueError("precision_target must be a positive finite number")

    current = float(current_price)
    target = float(target_price)
    mu = float(drift)
    sigma = float(volatility)

    if not np.isfinite(current) or not np.isfinite(target):
        raise ValueError("current_price and target_price must be finite numbers")

    rng = np.random.default_rng(random_state)
    steps = int(horizon)

    total_paths = 0
    total_hits = 0
    batch_size = int(initial_paths)

    while total_paths < max_paths:
        remaining_budget = max_paths - total_paths
        current_batch = min(batch_size, remaining_budget)
        hits = _simulate_gbm_hits(
            current_price=current,
            target_price=target,
            drift=mu,
            volatility=sigma,
            horizon=steps,
            paths=current_batch,
            rng=rng,
        )
        total_hits += hits
        total_paths += current_batch

        probability = float(total_hits) / float(total_paths)
        standard_error = float(np.sqrt(probability * (1 - probability) / total_paths))

        if standard_error <= precision_target:
            return probability, total_paths, standard_error

        batch_size *= 2

    probability = float(total_hits) / float(total_paths)
    standard_error = float(np.sqrt(probability * (1 - probability) / total_paths))
    return probability, total_paths, standard_error
