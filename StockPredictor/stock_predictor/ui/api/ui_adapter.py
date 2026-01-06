"""UI-facing adapter that orchestrates the modular predictor components."""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable

import pandas as pd

from stock_predictor.app import StockPredictorApplication
from stock_predictor.evaluation.backtester import BacktestConfig


LOGGER = logging.getLogger(__name__)
STRATEGY_PREFIXES = ("lorentzian_", "smc_")


def _merge_overrides(overrides: Dict[str, Any] | None) -> Dict[str, Any]:
    merged = dict(overrides or {})
    merged.setdefault("use_max_historical_data", True)
    return merged


def _classify_indicator(name: str) -> str:
    normalized = str(name).strip().lower()
    if any(normalized.startswith(prefix) for prefix in STRATEGY_PREFIXES):
        return "strategy"
    return "technical"


def _indicator_records(frame: "pd.DataFrame") -> list[dict[str, Any]]:
    if frame.empty:
        return []
    column_map = {str(col).lower(): col for col in frame.columns}
    indicator_col = column_map.get("indicator")
    value_col = column_map.get("value")
    category_col = column_map.get("category")
    if indicator_col is None or value_col is None:
        return []
    records: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        indicator = row.get(indicator_col)
        value = row.get(value_col)
        category = row.get(category_col) if category_col else None
        signal_type = _classify_indicator(indicator)
        records.append(
            {
                "indicator": indicator,
                "value": float(value) if value is not None and pd.notna(value) else None,
                "category": category,
                "signal_type": signal_type,
            }
        )
    return records


async def get_prediction(
    ticker: str,
    horizon: int | None,
    *,
    refresh: bool = False,
    targets: Iterable[str] | None = None,
    overrides: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Return a simplified prediction payload for UI consumption."""

    application = StockPredictorApplication.from_environment(
        ticker=ticker, **_merge_overrides(overrides)
    )
    prediction = application.predict(targets=targets, refresh=refresh, horizon=horizon)
    raw_payload: Dict[str, Any] = {}
    try:
        raw_payload = prediction.to_dict() if hasattr(prediction, "to_dict") else {}
    except Exception as exc:  # pragma: no cover - defensive serialization guard
        LOGGER.debug("Failed to serialise prediction payload: %s", exc)

    status = raw_payload.get("status")
    reason = raw_payload.get("reason")
    if status == "no_data":
        message = raw_payload.get("message")
        log_fn = LOGGER.info if reason == "no_data" else LOGGER.warning
        log_fn(
            "Prediction unavailable for %s (status=%s, reason=%s)",
            ticker,
            status,
            reason,
        )
        summary = {
            "ticker": ticker,
            "horizon": horizon,
            "last_price": None,
            "predicted_close": None,
            "expected_low": None,
            "expected_change_abs": None,
            "expected_change_pct": None,
            "stop_loss": None,
            "direction": None,
            "accuracy": {},
            "message": message,
            "status": status,
            "reason": reason,
        }
        return {"summary": summary, "distribution": {}, "raw": raw_payload}

    payload = raw_payload or prediction.to_dict()

    last_price = (
        payload.get("last_price")
        or payload.get("last_close")
        or payload.get("latest_price")
        or payload.get("latest_close")
    )
    anchor_price = payload.get("anchor_price") or last_price
    predicted_close = payload.get("predicted_close")
    expected_low = payload.get("expected_low")
    stop_loss = payload.get("stop_loss") or expected_low
    probability_within_tolerance = payload.get("probability_within_tolerance")
    tolerance_band = payload.get("tolerance_band")
    training_accuracy = payload.get("training_accuracy")

    change_abs = None
    change_pct = None
    direction = None
    if anchor_price is not None and predicted_close is not None:
        change_abs = float(predicted_close) - float(anchor_price)
        if anchor_price:
            change_pct = (change_abs / float(anchor_price)) * 100
        direction = "up" if change_abs >= 0 else "down"

    accuracy_summary = application.accuracy(horizon=horizon)

    summary = {
        "ticker": ticker,
        "horizon": horizon,
        "last_price": last_price,
        "predicted_close": predicted_close,
        "expected_low": expected_low,
        "expected_change_abs": change_abs,
        "expected_change_pct": change_pct,
        "stop_loss": stop_loss,
        "direction": direction,
        "accuracy": accuracy_summary,
        "probability_within_tolerance": probability_within_tolerance,
        "tolerance_band": tolerance_band,
        "training_accuracy": training_accuracy,
    }
    close_quantiles = payload.get("close_quantiles")
    if not isinstance(close_quantiles, dict):
        close_quantiles = {
            "q10": payload.get("pred_close_q10"),
            "q50": payload.get("pred_close_q50"),
            "q90": payload.get("pred_close_q90"),
        }
    hit_probabilities = payload.get("hit_probabilities")
    if not isinstance(hit_probabilities, dict):
        hit_probabilities = {
            "up": payload.get("p_hit_up"),
            "down": payload.get("p_hit_down"),
        }

    distribution = {
        "close_quantiles": close_quantiles,
        "pred_low_q10": payload.get("pred_low_q10"),
        "pred_high_q90": payload.get("pred_high_q90"),
        "range_1sigma": payload.get("range_1sigma"),
        "hit_probabilities": hit_probabilities,
        "uncertainty_calibrated": payload.get("uncertainty_calibrated"),
        "calibration_version": payload.get("calibration_version"),
        "regime_shift_risk": payload.get("regime_shift_risk"),
    }

    return {"summary": summary, "distribution": distribution, "raw": payload}


async def live_price_snapshot(
    ticker: str,
    *,
    horizon: int | None = None,
    expected_low_multiplier: float | None = None,
    stop_loss_multiplier: float | None = None,
    overrides: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Return a live price snapshot with optional expected-low scaling."""

    merged_overrides = dict(overrides or {})
    if expected_low_multiplier is not None:
        merged_overrides["expected_low_sigma"] = expected_low_multiplier
    if stop_loss_multiplier is not None:
        merged_overrides["k_stop"] = stop_loss_multiplier
    merged_overrides = _merge_overrides(merged_overrides)
    application = StockPredictorApplication.from_environment(
        ticker=ticker, **merged_overrides
    )
    return application.pipeline.live_price_snapshot(horizon=horizon)


async def run_backtest(
    ticker: str,
    *,
    targets: Iterable[str] | None = None,
    overrides: Dict[str, Any] | None = None,
    backtest_config: BacktestConfig | None = None,
) -> Dict[str, Any]:
    application = StockPredictorApplication.from_environment(
        ticker=ticker, **_merge_overrides(overrides)
    )
    return application.backtest(targets=targets, backtest_config=backtest_config)


async def run_reliability_backtest(
    ticker: str,
    *,
    targets: Iterable[str] | None = None,
    overrides: Dict[str, Any] | None = None,
    n_runs: int | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    horizon: int | None = None,
    step_size: int | None = None,
) -> Dict[str, Any]:
    application = StockPredictorApplication.from_environment(
        ticker=ticker, **_merge_overrides(overrides)
    )
    return application.reliability_backtest(
        targets=targets,
        horizon=horizon,
        n_runs=n_runs,
        start_date=start_date,
        end_date=end_date,
        step_size=step_size,
    )


async def train_models(
    ticker: str,
    *,
    targets: Iterable[str] | None = None,
    horizon: int | None = None,
    overrides: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    application = StockPredictorApplication.from_environment(
        ticker=ticker, **_merge_overrides(overrides)
    )
    return application.train(targets=targets, horizon=horizon)


async def refresh_data(
    ticker: str, *, refresh: bool = True, overrides: Dict[str, Any] | None = None
) -> Dict[str, Any]:
    application = StockPredictorApplication.from_environment(
        ticker=ticker, **_merge_overrides(overrides)
    )
    return application.refresh_data(force=refresh)


async def get_accuracy(
    ticker: str, *, horizon: int | None = None, overrides: Dict[str, Any] | None = None
) -> Dict[str, Any]:
    application = StockPredictorApplication.from_environment(
        ticker=ticker, **_merge_overrides(overrides)
    )
    return application.accuracy(horizon=horizon)


async def get_indicators(
    ticker: str,
    *,
    refresh: bool = False,
    category: str | None = None,
    limit: int | None = None,
    include_history: bool = False,
    overrides: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Return indicator values and strategy signals for the requested ticker."""

    application = StockPredictorApplication.from_environment(
        ticker=ticker, **_merge_overrides(overrides)
    )
    if refresh:
        application.refresh_data(force=True)

    fetcher = getattr(application.pipeline, "fetcher", None)
    if fetcher is None or not hasattr(fetcher, "fetch_indicator_data"):
        return {
            "latest_date": None,
            "latest": [],
            "indicator_values": [],
            "strategy_signals": [],
            "history": [],
        }

    indicator_frame = fetcher.fetch_indicator_data(category=category)
    if indicator_frame is None or indicator_frame.empty:
        return {
            "latest_date": None,
            "latest": [],
            "indicator_values": [],
            "strategy_signals": [],
            "history": [],
        }

    frame = indicator_frame.copy()
    column_map = {str(col).lower(): col for col in frame.columns}
    date_col = column_map.get("date") or column_map.get("as_of") or column_map.get("timestamp")
    if date_col is None:
        frame["__as_of"] = pd.to_datetime(frame.index, errors="coerce")
        date_col = "__as_of"
    else:
        frame[date_col] = pd.to_datetime(frame[date_col], errors="coerce")
    frame = frame.dropna(subset=[date_col]).sort_values(date_col)
    if frame.empty:
        return {
            "latest_date": None,
            "latest": [],
            "indicator_values": [],
            "strategy_signals": [],
            "history": [],
        }

    latest_date = frame[date_col].max()
    latest_frame = frame[frame[date_col] == latest_date]
    latest_records = _indicator_records(latest_frame)
    indicator_values = [row for row in latest_records if row["signal_type"] == "technical"]
    strategy_signals = [row for row in latest_records if row["signal_type"] == "strategy"]

    history_records: list[dict[str, Any]] = []
    if include_history:
        history_frame = frame
        if limit is not None and limit > 0:
            history_frame = history_frame.tail(int(limit))
        history_records = _indicator_records(history_frame)
        for entry, (_, row) in zip(history_records, history_frame.iterrows()):
            entry["as_of"] = row.get(date_col)

    return {
        "latest_date": latest_date.isoformat() if hasattr(latest_date, "isoformat") else str(latest_date),
        "latest": latest_records,
        "indicator_values": indicator_values,
        "strategy_signals": strategy_signals,
        "history": history_records,
    }


__all__ = [
    "get_prediction",
    "run_backtest",
    "run_reliability_backtest",
    "train_models",
    "refresh_data",
    "get_accuracy",
    "get_indicators",
]
