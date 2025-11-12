"""FastAPI integration layer for the stock predictor application."""

from __future__ import annotations

from typing import Any, Dict, Optional

from stock_predictor.app import StockPredictorApplication


def create_api_app(default_overrides: Optional[Dict[str, Any]] = None) -> Any:
    """Create a FastAPI application that exposes prediction endpoints.

    Parameters
    ----------
    default_overrides:
        Optional dictionary of keyword arguments forwarded to
        :meth:`StockPredictorApplication.from_environment` for each request.

    Returns
    -------
    fastapi.FastAPI
        Configured API application.

    Notes
    -----
    FastAPI is an optional dependency. Attempting to call this function without
    FastAPI installed will raise a :class:`RuntimeError` with installation
    instructions.
    """

    try:
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel
    except ImportError as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "FastAPI and pydantic are required to use the stock_predictor.ui package."
        ) from exc

    overrides = dict(default_overrides or {})

    class PredictionRequest(BaseModel):
        ticker: str
        targets: list[str] | None = None
        refresh: bool = False
        horizon: int | None = None

    class TrainingRequest(BaseModel):
        ticker: str
        targets: list[str] | None = None
        horizon: int | None = None

    app = FastAPI(title="Stock Predictor API", version="2.0.0")

    @app.get("/health")
    async def healthcheck() -> Dict[str, str]:
        return {"status": "ok"}

    def _build_application(ticker: str) -> StockPredictorApplication:
        if not ticker:
            raise HTTPException(status_code=400, detail="Ticker symbol is required")
        payload = {**overrides, "ticker": ticker}
        return StockPredictorApplication.from_environment(**payload)

    @app.post("/predict")
    async def predict(request: PredictionRequest) -> Dict[str, Any]:
        application = _build_application(request.ticker)
        result = application.run(
            "predict",
            targets=request.targets,
            refresh=request.refresh,
            horizon=request.horizon,
        )
        return {"status": result.status, **result.payload}

    @app.post("/train")
    async def train(request: TrainingRequest) -> Dict[str, Any]:
        application = _build_application(request.ticker)
        result = application.run(
            "train",
            targets=request.targets,
            horizon=request.horizon,
        )
        return {"status": result.status, **result.payload}

    return app


__all__ = ["create_api_app"]
