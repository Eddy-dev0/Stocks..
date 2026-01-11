"""Fabio-specific indicator helpers for live charting."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np


def _safe_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric):
        return None
    return numeric


def _extract_values(bars: Iterable[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for bar in bars:
        numeric = _safe_float(bar.get(key))
        if numeric is not None:
            values.append(numeric)
    return values


def calculate_volume_profile(
    bars: Iterable[dict[str, Any]], bin_count: int = 50
) -> list[tuple[float, float]]:
    """Return a list of (price, volume) pairs for the given bars."""

    bars_list = list(bars)
    if not bars_list or bin_count <= 0:
        return []

    lows = _extract_values(bars_list, "low")
    highs = _extract_values(bars_list, "high")
    if not lows or not highs:
        return []

    min_price = min(lows)
    max_price = max(highs)
    if np.isclose(min_price, max_price):
        max_price = min_price + 1e-6

    bins = np.linspace(min_price, max_price, bin_count + 1)
    volumes = np.zeros(bin_count, dtype=float)

    for bar in bars_list:
        high = _safe_float(bar.get("high"))
        low = _safe_float(bar.get("low"))
        close = _safe_float(bar.get("close"))
        if high is None or low is None or close is None:
            continue
        price = (high + low + close) / 3.0
        volume = _safe_float(bar.get("volume")) or 0.0
        idx = int(np.clip(np.searchsorted(bins, price, side="right") - 1, 0, bin_count - 1))
        volumes[idx] += volume

    centers = (bins[:-1] + bins[1:]) / 2.0
    return list(zip(centers.tolist(), volumes.tolist()))


def identify_lvn_zones(
    volume_profile: Iterable[tuple[float, float]],
    *,
    neighbor_window: int = 1,
    relative_threshold: float = 0.55,
) -> list[tuple[float, float]]:
    """Return low-volume zones based on volume profile bins."""

    profile = list(volume_profile)
    if len(profile) < 3:
        return []
    centers = np.array([price for price, _ in profile], dtype=float)
    volumes = np.array([volume for _, volume in profile], dtype=float)
    if len(centers) < 2:
        return []

    bin_width = float(np.median(np.diff(centers)))
    zones: list[tuple[float, float]] = []
    max_index = len(volumes) - neighbor_window
    for idx in range(neighbor_window, max_index):
        neighbor = volumes[idx - neighbor_window : idx + neighbor_window + 1]
        neighbor_avg = (neighbor.sum() - volumes[idx]) / max(len(neighbor) - 1, 1)
        if neighbor_avg <= 0:
            continue
        if volumes[idx] < neighbor_avg * relative_threshold:
            center = centers[idx]
            low = center - bin_width / 2.0
            high = center + bin_width / 2.0
            zones.append((low, high))

    return _merge_zones(zones)


def _merge_zones(zones: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    sorted_zones = sorted((min(zone), max(zone)) for zone in zones)
    merged: list[tuple[float, float]] = []
    for low, high in sorted_zones:
        if not merged:
            merged.append((low, high))
            continue
        last_low, last_high = merged[-1]
        if low <= last_high:
            merged[-1] = (last_low, max(last_high, high))
        else:
            merged.append((low, high))
    return merged


def detect_market_structure(
    bars: Iterable[dict[str, Any]],
    *,
    balance_window: int = 8,
    balance_threshold: float = 0.004,
    bos_threshold: float = 0.002,
) -> dict[str, list[dict[str, float | int]]]:
    """Detect balance zones and breaks of structure."""

    bars_list = list(bars)
    balance_flags: list[bool] = []
    for idx, bar in enumerate(bars_list):
        if idx + 1 < balance_window:
            balance_flags.append(False)
            continue
        window = bars_list[idx - balance_window + 1 : idx + 1]
        highs = _extract_values(window, "high")
        lows = _extract_values(window, "low")
        close = _safe_float(bar.get("close"))
        if not highs or not lows or close is None or close == 0:
            balance_flags.append(False)
            continue
        range_pct = (max(highs) - min(lows)) / abs(close)
        balance_flags.append(range_pct <= balance_threshold)

    balance_zones: list[dict[str, float | int]] = []
    idx = 0
    while idx < len(balance_flags):
        if not balance_flags[idx]:
            idx += 1
            continue
        start = idx
        while idx < len(balance_flags) and balance_flags[idx]:
            idx += 1
        end = idx - 1
        segment = bars_list[start : end + 1]
        highs = _extract_values(segment, "high")
        lows = _extract_values(segment, "low")
        if highs and lows:
            balance_zones.append(
                {"start": start, "end": end, "high": max(highs), "low": min(lows)}
            )

    bos_events: list[dict[str, float | int]] = []
    breakout_zones: list[dict[str, float | int]] = []
    for zone in balance_zones:
        start = int(zone["end"]) + 1
        high = float(zone["high"])
        low = float(zone["low"])
        for idx in range(start, len(bars_list)):
            close = _safe_float(bars_list[idx].get("close"))
            if close is None:
                continue
            if close > high * (1 + bos_threshold):
                bos_events.append({"index": idx, "direction": 1, "level": high})
                breakout_zones.append(
                    {
                        "start": idx,
                        "end": min(idx + 3, len(bars_list) - 1),
                        "high": max(high, close),
                        "low": low,
                    }
                )
                break
            if close < low * (1 - bos_threshold):
                bos_events.append({"index": idx, "direction": -1, "level": low})
                breakout_zones.append(
                    {
                        "start": idx,
                        "end": min(idx + 3, len(bars_list) - 1),
                        "high": high,
                        "low": min(low, close),
                    }
                )
                break

    return {
        "balance_zones": balance_zones,
        "bos_events": bos_events,
        "breakout_zones": breakout_zones,
    }


def calculate_cvd(bars: Iterable[dict[str, Any]]) -> list[float]:
    """Calculate the cumulative volume delta."""

    cvd_values: list[float] = []
    total = 0.0
    for bar in bars:
        open_ = _safe_float(bar.get("open"))
        close = _safe_float(bar.get("close"))
        volume = _safe_float(bar.get("volume"))
        if open_ is None or close is None:
            delta = 0.0
        elif volume is None:
            delta = close - open_
        else:
            delta = volume if close >= open_ else -volume
        total += delta
        cvd_values.append(total)
    return cvd_values


def simulate_orderflow_signals(
    bars: Iterable[dict[str, Any]],
    lvns: Iterable[tuple[float, float]],
) -> list[dict[str, float | int]]:
    """Identify simulated aggressive orderflow points."""

    bars_list = list(bars)
    volumes = np.array(_extract_values(bars_list, "volume"), dtype=float)
    bodies = np.array(
        [
            abs((_safe_float(bar.get("close")) or 0.0) - (_safe_float(bar.get("open")) or 0.0))
            for bar in bars_list
        ],
        dtype=float,
    )
    if volumes.size == 0 or bodies.size == 0:
        return []
    vol_threshold = volumes.mean() + volumes.std() * 1.5
    body_threshold = bodies.mean() * 1.5 if bodies.mean() > 0 else 0
    lvn_zones = list(lvns)

    signals: list[dict[str, float | int]] = []
    for idx, bar in enumerate(bars_list):
        volume = _safe_float(bar.get("volume")) or 0.0
        open_ = _safe_float(bar.get("open"))
        close = _safe_float(bar.get("close"))
        if open_ is None or close is None:
            continue
        body = abs(close - open_)
        if volume >= vol_threshold and body >= body_threshold:
            price = close
            in_lvn = any(low <= price <= high for low, high in lvn_zones)
            signals.append(
                {
                    "index": idx,
                    "price": price,
                    "direction": 1 if close >= open_ else -1,
                    "strength": 2 if in_lvn else 1,
                }
            )
    return signals
