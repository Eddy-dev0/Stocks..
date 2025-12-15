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
    tolerance_center: float | None = None,
    tolerance_fraction: float | None = None,
) -> tuple[int, int | None]:
    """Return hit counts for target price and optional tolerance band.

    The helper performs a vectorised geometric Brownian motion rollout and counts
    how many trajectories touch or exceed the target. Returning hits instead of a
    probability enables incremental aggregation for adaptive Monte Carlo runs.

    When ``tolerance_center`` and ``tolerance_fraction`` are provided, the
    function also counts how many terminal prices end within the symmetric band
    defined around ``tolerance_center``.
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

    tolerance_hits: int | None = None
    if tolerance_center is not None and tolerance_fraction is not None and tolerance_fraction > 0:
        band = abs(float(tolerance_center)) * float(tolerance_fraction)
        lower = float(tolerance_center) - band
        upper = float(tolerance_center) + band
        terminal = prices_with_start[:, -1]
        tolerance_hits = int(np.count_nonzero((terminal >= lower) & (terminal <= upper)))

    return int(np.count_nonzero(hits)), tolerance_hits


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

    hits, _ = _simulate_gbm_hits(
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
    tolerance_center: float | None = None,
    tolerance_fraction: float | None = None,
) -> tuple[float, int, float, float | None]:
    """Run Monte Carlo batches until the standard error drops below the target.

    The function aggregates hit counts incrementally to avoid exhausting memory
    while still converging toward a more precise probability estimate. A maximum
    path budget guards against unbounded computation. When tolerance parameters
    are provided, the final element of the return tuple reports the probability
    of landing within the specified band at the terminal step.
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
    total_tolerance_hits: int | None = 0 if (
        tolerance_center is not None and tolerance_fraction is not None and tolerance_fraction > 0
    ) else None
    batch_size = int(initial_paths)

    while total_paths < max_paths:
        remaining_budget = max_paths - total_paths
        current_batch = min(batch_size, remaining_budget)
        hits, tolerance_hits = _simulate_gbm_hits(
            current_price=current,
            target_price=target,
            drift=mu,
            volatility=sigma,
            horizon=steps,
            paths=current_batch,
            rng=rng,
            tolerance_center=tolerance_center,
            tolerance_fraction=tolerance_fraction,
        )
        total_hits += hits
        total_paths += current_batch
        if total_tolerance_hits is not None and tolerance_hits is not None:
            total_tolerance_hits += tolerance_hits

        probability = float(total_hits) / float(total_paths)
        standard_error = float(np.sqrt(probability * (1 - probability) / total_paths))

        if standard_error <= precision_target:
            tolerance_probability = (
                float(total_tolerance_hits) / float(total_paths)
                if total_tolerance_hits is not None
                else None
            )
            return probability, total_paths, standard_error, tolerance_probability

        batch_size *= 2

    probability = float(total_hits) / float(total_paths)
    standard_error = float(np.sqrt(probability * (1 - probability) / total_paths))
    tolerance_probability = (
        float(total_tolerance_hits) / float(total_paths)
        if total_tolerance_hits is not None
        else None
    )
    return probability, total_paths, standard_error, tolerance_probability
