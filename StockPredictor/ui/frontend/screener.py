"""Pattern screener utilities for the Streamlit frontend."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Callable, Dict, Mapping

import numpy as np
import pandas as pd
import streamlit as st

PATTERN_CHOICES: tuple[str, ...] = (
    "Double Bottom",
    "Double Top",
    "Triple Bottom",
    "Triple Top",
    "Head and Shoulders",
    "Inverted Head and Shoulders",
    "Ascending Triangle",
    "Descending Triangle",
    "Pennant",
    "Flag",
    "Bearish Flag",
    "Channel",
    "Channel Up",
    "Channel Down",
    "Cup and Handle",
    "Diamond",
)

DEFAULT_PRODUCTS: tuple[str, ...] = (
    "AAPL", "MSFT", "JPM", "XOM", "WMT", "KO", "DIS", "NKE", "BA", "GS",
    "NVDA", "AMD", "TSLA", "F", "GE", "CAT", "IBM", "BAC", "PFE", "T",
    "ES=F", "NQ=F", "YM=F", "RTY=F", "CL=F", "GC=F", "SI=F", "HG=F", "NG=F", "ZB=F",
)


@dataclass
class PatternSignal:
    idx: int
    clean_score: float
    direction: str


def _coerce_dataframe(payload: Any) -> pd.DataFrame | None:
    if payload is None:
        return None
    if isinstance(payload, pd.DataFrame):
        return payload.copy()
    if isinstance(payload, dict):
        for key in ("prices", "data", "values", "records", "download_data"):
            if key in payload:
                frame = _coerce_dataframe(payload[key])
                if frame is not None and not frame.empty:
                    return frame
        if payload and all(isinstance(value, (list, tuple)) for value in payload.values()):
            try:
                return pd.DataFrame(payload)
            except ValueError:
                return None
    if isinstance(payload, list) and payload and isinstance(payload[0], Mapping):
        return pd.DataFrame(payload)
    return None


def _normalise_ohlc(frame: pd.DataFrame) -> pd.DataFrame | None:
    rename_map = {}
    for col in frame.columns:
        lower = str(col).lower()
        if lower in {"date", "datetime", "timestamp", "time"}:
            rename_map[col] = "time"
        elif lower in {"open", "o"}:
            rename_map[col] = "open"
        elif lower in {"high", "h"}:
            rename_map[col] = "high"
        elif lower in {"low", "l"}:
            rename_map[col] = "low"
        elif lower in {"close", "adj close", "adj_close", "c"}:
            rename_map[col] = "close"
    norm = frame.rename(columns=rename_map)
    required = {"high", "low", "close"}
    if not required.issubset(norm.columns):
        return None
    if "time" in norm.columns:
        norm["time"] = pd.to_datetime(norm["time"], errors="coerce")
        norm = norm.sort_values("time")
    norm = norm[[c for c in ["time", "open", "high", "low", "close"] if c in norm.columns]].copy()
    for c in ("open", "high", "low", "close"):
        if c in norm.columns:
            norm[c] = pd.to_numeric(norm[c], errors="coerce")
    norm = norm.dropna(subset=["high", "low", "close"])
    return norm.reset_index(drop=True)


def _extrema(series: pd.Series, window: int = 3) -> tuple[np.ndarray, np.ndarray]:
    values = series.to_numpy(dtype=float)
    if len(values) < window * 2 + 1:
        return np.array([], dtype=int), np.array([], dtype=int)
    mins: list[int] = []
    maxs: list[int] = []
    for i in range(window, len(values) - window):
        sample = values[i - window : i + window + 1]
        if values[i] == np.nanmin(sample):
            mins.append(i)
        if values[i] == np.nanmax(sample):
            maxs.append(i)
    return np.array(mins, dtype=int), np.array(maxs, dtype=int)


def _lin_slope(values: np.ndarray) -> float:
    x = np.arange(len(values))
    if len(values) < 2:
        return 0.0
    slope, _ = np.polyfit(x, values, 1)
    return float(slope)


def _detect(pattern: str, window: pd.DataFrame) -> tuple[bool, float, str]:
    close = window["close"].to_numpy(dtype=float)
    high = window["high"].to_numpy(dtype=float)
    low = window["low"].to_numpy(dtype=float)
    mins, maxs = _extrema(window["close"], window=2)

    if len(close) < 20:
        return False, 0.0, "up"

    # Reversal patterns
    if pattern in {"Double Bottom", "Triple Bottom", "Inverted Head and Shoulders", "Cup and Handle"}:
        if len(mins) < 2:
            return False, 0.0, "up"
        recent_mins = mins[-3:] if pattern == "Triple Bottom" else mins[-2:]
        lows = close[recent_mins]
        tolerance = np.std(lows) / max(np.mean(lows), 1e-6)
        clean = max(0.0, 1.0 - tolerance * 6)
        return clean > 0.35 and close[-1] > np.mean(lows), clean, "up"

    if pattern in {"Double Top", "Triple Top", "Head and Shoulders"}:
        if len(maxs) < 2:
            return False, 0.0, "down"
        recent_max = maxs[-3:] if pattern == "Triple Top" else maxs[-2:]
        highs = close[recent_max]
        tolerance = np.std(highs) / max(np.mean(highs), 1e-6)
        clean = max(0.0, 1.0 - tolerance * 6)
        return clean > 0.35 and close[-1] < np.mean(highs), clean, "down"

    # Triangle / continuation
    if pattern in {"Ascending Triangle", "Descending Triangle", "Pennant"}:
        top_slope = _lin_slope(high[-14:])
        low_slope = _lin_slope(low[-14:])
        if pattern == "Ascending Triangle":
            clean = max(0.0, min(1.0, (low_slope * 20) + max(0.0, 0.08 - abs(top_slope))))
            return clean > 0.3, clean, "up"
        if pattern == "Descending Triangle":
            clean = max(0.0, min(1.0, ((-top_slope) * 20) + max(0.0, 0.08 - abs(low_slope))))
            return clean > 0.3, clean, "down"
        clean = max(0.0, 1.0 - abs(top_slope - (-low_slope)) * 30)
        direction = "up" if close[-1] >= close[-4] else "down"
        return clean > 0.35, clean, direction

    if pattern in {"Flag", "Bearish Flag"}:
        pole_move = (close[8] - close[0]) / max(close[0], 1e-6)
        consolidation = np.std(close[8:]) / max(np.mean(close[8:]), 1e-6)
        clean = max(0.0, min(1.0, abs(pole_move) * 6 + (0.15 - consolidation)))
        bullish = pattern == "Flag"
        valid_dir = pole_move > 0 if bullish else pole_move < 0
        return valid_dir and clean > 0.25, clean, "up" if bullish else "down"

    if pattern in {"Channel", "Channel Up", "Channel Down"}:
        slope = _lin_slope(close[-20:])
        channel_width = (np.max(high[-20:]) - np.min(low[-20:])) / max(np.mean(close[-20:]), 1e-6)
        clean = max(0.0, 1.0 - abs(channel_width - 0.08) * 4)
        direction = "up" if slope > 0 else "down"
        if pattern == "Channel Up":
            return slope > 0 and clean > 0.2, clean, "up"
        if pattern == "Channel Down":
            return slope < 0 and clean > 0.2, clean, "down"
        return clean > 0.2, clean, direction

    if pattern == "Diamond":
        first = close[: len(close) // 2]
        second = close[len(close) // 2 :]
        first_vol = np.std(first) / max(np.mean(first), 1e-6)
        second_vol = np.std(second) / max(np.mean(second), 1e-6)
        clean = max(0.0, min(1.0, (first_vol - second_vol) * 15 + 0.4))
        direction = "down" if close[-1] < close[-3] else "up"
        return clean > 0.25, clean, direction

    return False, 0.0, "up"


def _scan_signals(frame: pd.DataFrame, pattern: str, lookback: int = 40) -> list[PatternSignal]:
    signals: list[PatternSignal] = []
    if len(frame) < lookback + 5:
        return signals
    for idx in range(lookback, len(frame)):
        window = frame.iloc[idx - lookback : idx + 1]
        detected, clean, direction = _detect(pattern, window)
        if detected:
            signals.append(PatternSignal(idx=idx, clean_score=float(clean), direction=direction))
    return signals


def _backtest_pattern(frame: pd.DataFrame, signals: list[PatternSignal], lookahead: int = 24) -> Dict[str, float]:
    if not signals:
        return {
            "trades": 0,
            "successes": 0,
            "success_rate": 0.0,
            "avg_reached_move": 0.0,
            "midpoint_move": 0.0,
            "midpoint_hit_rate": 0.0,
        }

    achieved_moves: list[float] = []
    successes = 0
    for signal in signals:
        if signal.idx + 1 >= len(frame):
            continue
        slice_end = min(len(frame), signal.idx + 1 + lookahead)
        look = frame.iloc[signal.idx + 1 : slice_end]
        if look.empty:
            continue
        entry = float(frame.iloc[signal.idx]["close"])
        if signal.direction == "up":
            best = (float(look["high"].max()) - entry) / max(entry, 1e-6)
        else:
            best = (entry - float(look["low"].min())) / max(entry, 1e-6)
        achieved = max(0.0, best)
        achieved_moves.append(achieved)
        if achieved > 0:
            successes += 1

    if not achieved_moves:
        return {
            "trades": 0,
            "successes": 0,
            "success_rate": 0.0,
            "avg_reached_move": 0.0,
            "midpoint_move": 0.0,
            "midpoint_hit_rate": 0.0,
        }

    midpoint = float(np.mean(achieved_moves))
    midpoint_hits = sum(1 for move in achieved_moves if move >= midpoint)
    trades = len(achieved_moves)
    return {
        "trades": trades,
        "successes": successes,
        "success_rate": successes / trades,
        "avg_reached_move": float(np.mean(achieved_moves)),
        "midpoint_move": midpoint,
        "midpoint_hit_rate": midpoint_hits / trades,
    }


def _fetch_product_frame(
    request_fn: Callable[..., Dict[str, Any] | None],
    product: str,
    start_date: date,
    end_date: date,
) -> pd.DataFrame | None:
    payload = request_fn(
        f"/data/{product}",
        params={
            "refresh": False,
            "interval": "1h",
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
        },
    )
    if payload is None:
        return None
    raw = payload.get("data") if isinstance(payload, Mapping) else payload
    frame = _coerce_dataframe(raw)
    if frame is None or frame.empty:
        return None
    return _normalise_ohlc(frame)


def render_screener(
    request_fn: Callable[..., Dict[str, Any] | None],
    *,
    default_end_date: date,
) -> None:
    st.subheader("Live Pattern Screener (1h)")
    st.caption(
        "Scannt NYSE-Aktien und Futures im 1h-Chart auf aktuelle Pattern und bewertet die Qualität "
        "über historische Treffer inklusive Backtest-Metriken."
    )

    selected_patterns = st.multiselect(
        "Pattern-Filter",
        options=list(PATTERN_CHOICES),
        default=["Double Bottom", "Double Top", "Head and Shoulders"],
    )
    custom_products = st.text_area(
        "Produkte (optional, komma-separiert)",
        value=", ".join(DEFAULT_PRODUCTS[:12]),
        help="Lasse leer, um die Standardliste aus NYSE/Futures zu verwenden.",
    )
    lookback_days = st.slider("Historie für Pattern-Backtest (Tage)", 30, 365, 180, step=10)

    if not selected_patterns:
        st.info("Bitte mindestens ein Pattern auswählen.")
        return

    if not st.button("Screener starten", type="primary"):
        return

    products = [item.strip().upper() for item in custom_products.split(",") if item.strip()] or list(DEFAULT_PRODUCTS)
    end_date = default_end_date
    start_date = end_date - timedelta(days=int(lookback_days))

    rows: list[dict[str, Any]] = []
    progress = st.progress(0.0)
    for i, product in enumerate(products, start=1):
        frame = _fetch_product_frame(request_fn, product, start_date, end_date)
        if frame is None or frame.empty:
            progress.progress(i / len(products))
            continue

        for pattern in selected_patterns:
            all_signals = _scan_signals(frame, pattern)
            if not all_signals:
                continue
            latest_signal = all_signals[-1]
            is_current = latest_signal.idx >= len(frame) - 3
            if not is_current:
                continue
            backtest = _backtest_pattern(frame, all_signals[:-1])
            quality = (
                latest_signal.clean_score * 0.4
                + backtest["success_rate"] * 0.3
                + backtest["midpoint_hit_rate"] * 0.3
            )
            rows.append(
                {
                    "Produkt": product,
                    "Pattern": pattern,
                    "Richtung": "Bullish" if latest_signal.direction == "up" else "Bearish",
                    "Pattern-Sauberkeit": round(latest_signal.clean_score * 100, 1),
                    "Gefundene Trades": int(backtest["trades"]),
                    "Erfolgreiche Trades": int(backtest["successes"]),
                    "Erfolgsquote %": round(backtest["success_rate"] * 100, 1),
                    "Durchschn. erreichter Move %": round(backtest["avg_reached_move"] * 100, 2),
                    "Mitte (Ø Move) %": round(backtest["midpoint_move"] * 100, 2),
                    "Trefferquote Mitte %": round(backtest["midpoint_hit_rate"] * 100, 1),
                    "Gesamtqualität %": round(quality * 100, 1),
                }
            )
        progress.progress(i / len(products))

    progress.empty()
    if not rows:
        st.info("Keine aktuell aktiven Pattern in den ausgewählten Produkten gefunden.")
        return

    table = pd.DataFrame(rows).sort_values(
        by=["Gesamtqualität %", "Erfolgsquote %", "Pattern-Sauberkeit"],
        ascending=False,
    )
    st.dataframe(table, use_container_width=True)
    st.caption(
        "Qualität = 40% Pattern-Sauberkeit + 30% historische Erfolgsquote + 30% Trefferquote der Mittelwert-Zielerreichung."
    )
