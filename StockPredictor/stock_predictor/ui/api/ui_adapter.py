"""UI-facing adapter that orchestrates the modular predictor components."""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable

from stock_predictor.app import StockPredictorApplication


LOGGER = logging.getLogger(__name__)


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
        ticker=ticker, **(overrides or {})
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
        message = raw_payload.get("message") or "Not enough historical data to generate predictions yet"
        log_fn = LOGGER.info if reason == "insufficient_samples" else LOGGER.warning
        log_fn(
            "Prediction unavailable for %s (status=%s, reason=%s)",
            ticker,
            status,
            reason,
        )
        return {
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
            "raw": raw_payload,
            "status": status,
            "reason": reason,
        }

    payload = raw_payload or prediction.to_dict()

    last_price = payload.get("latest_price") or payload.get("latest_close")
    predicted_close = payload.get("predicted_close")
    expected_low = payload.get("expected_low")
    stop_loss = payload.get("stop_loss") or expected_low

    change_abs = None
    change_pct = None
    direction = None
    if last_price is not None and predicted_close is not None:
        change_abs = float(predicted_close) - float(last_price)
        if last_price:
            change_pct = (change_abs / float(last_price)) * 100
        direction = "up" if change_abs >= 0 else "down"

    accuracy_summary = application.accuracy(horizon=horizon)

    return {
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
        "raw": payload,
    }


async def run_backtest(
    ticker: str,
    *,
    targets: Iterable[str] | None = None,
    overrides: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    application = StockPredictorApplication.from_environment(
        ticker=ticker, **(overrides or {})
    )
    return application.backtest(targets=targets)


async def train_models(
    ticker: str,
    *,
    targets: Iterable[str] | None = None,
    horizon: int | None = None,
    overrides: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    application = StockPredictorApplication.from_environment(
        ticker=ticker, **(overrides or {})
    )
    return application.train(targets=targets, horizon=horizon)


async def refresh_data(
    ticker: str, *, refresh: bool = True, overrides: Dict[str, Any] | None = None
) -> Dict[str, Any]:
    application = StockPredictorApplication.from_environment(
        ticker=ticker, **(overrides or {})
    )
    return application.refresh_data(force=refresh)


async def get_accuracy(
    ticker: str, *, horizon: int | None = None, overrides: Dict[str, Any] | None = None
) -> Dict[str, Any]:
    application = StockPredictorApplication.from_environment(
        ticker=ticker, **(overrides or {})
    )
    return application.accuracy(horizon=horizon)


__all__ = [
    "get_prediction",
    "run_backtest",
    "train_models",
    "refresh_data",
    "get_accuracy",
]
