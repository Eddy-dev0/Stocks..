"""FastAPI application exposing Stock Predictor services for the web UI."""

from __future__ import annotations

import inspect
import os
from functools import lru_cache
from typing import Any, Dict

from fastapi import Depends, FastAPI, HTTPException, Query, Security
from fastapi.concurrency import run_in_threadpool
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from stock_predictor.app import StockPredictorApplication
from stock_predictor.research import ResearchService


api_key_scheme = APIKeyHeader(name="X-API-Key", auto_error=False)


@lru_cache()
def _configured_api_keys() -> set[str]:
    """Return the set of API keys allowed to access the service."""

    raw_keys = os.getenv("STOCK_PREDICTOR_UI_API_KEYS", "")
    keys = {value.strip() for value in raw_keys.split(",") if value.strip()}
    return keys


async def require_api_key(api_key: str | None = Security(api_key_scheme)) -> str:
    """Validate the provided API key against the configured allow list."""

    keys = _configured_api_keys()
    if not keys:
        return api_key or ""
    if not api_key or api_key not in keys:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return api_key


class ForecastRequest(BaseModel):
    """Payload used to request new forecasts for a ticker."""

    targets: list[str] | None = Field(
        default=None,
        description="Optional list of targets to limit forecasting to.",
        example=["close", "direction"],
    )
    refresh: bool = Field(
        default=False,
        description="If true, the underlying datasets will be refreshed before predicting.",
    )
    horizon: int | None = Field(
        default=None,
        ge=1,
        description="Optional number of periods ahead to forecast.",
    )


class BacktestRequest(BaseModel):
    """Payload used to trigger a backtest run for a ticker."""

    targets: list[str] | None = Field(
        default=None,
        description="Optional list of targets to include in the backtest.",
    )


def create_app(default_overrides: Dict[str, Any] | None = None) -> FastAPI:
    """Create a configured FastAPI application for the Stock Predictor UI."""

    overrides = dict(default_overrides or {})
    app = FastAPI(title="Stock Predictor UI API", version="1.0.0")
    app.state.overrides = overrides
    app.state.research_service: ResearchService | None = None

    def _merge_overrides(ticker: str, request_overrides: Dict[str, Any]) -> Dict[str, Any]:
        payload: Dict[str, Any] = {**app.state.overrides}
        payload.update({key: value for key, value in request_overrides.items() if value is not None})
        payload["ticker"] = ticker
        return payload

    async def _build_application(ticker: str, request_overrides: Dict[str, Any]) -> StockPredictorApplication:
        payload = _merge_overrides(ticker, request_overrides)
        return await run_in_threadpool(StockPredictorApplication.from_environment, **payload)

    async def _call_with_error_handling(func, *args: Any, **kwargs: Any) -> Any:
        try:
            if inspect.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            return await run_in_threadpool(func, *args, **kwargs)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    async def _get_research_service() -> ResearchService:
        service: ResearchService | None = getattr(app.state, "research_service", None)
        if service is None:
            raise HTTPException(status_code=503, detail="Research service not initialised")
        return service

    @app.on_event("startup")
    async def _startup() -> None:
        ticker = overrides.get("ticker") or os.getenv("STOCK_PREDICTOR_DEFAULT_TICKER", "AAPL")
        application = await _build_application(ticker, {})
        app.state.research_service = ResearchService(application.config)

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        service: ResearchService | None = getattr(app.state, "research_service", None)
        if service is not None:
            await service.aclose()

    @app.get("/health")
    async def healthcheck() -> Dict[str, str]:
        return {"status": "ok"}

    @app.get("/data/{ticker}", dependencies=[Depends(require_api_key)])
    async def get_data(
        ticker: str,
        refresh: bool = Query(False, description="Refresh underlying data sources."),
        start_date: str | None = Query(None, description="Optional ISO start date override."),
        end_date: str | None = Query(None, description="Optional ISO end date override."),
        interval: str | None = Query(None, description="Historical data interval override."),
    ) -> Dict[str, Any]:
        application = await _build_application(
            ticker,
            {"start_date": start_date, "end_date": end_date, "interval": interval},
        )
        data = await _call_with_error_handling(application.refresh_data, force=refresh)
        return {"status": "ok", "data": data}

    @app.post("/forecasts/{ticker}", dependencies=[Depends(require_api_key)])
    async def forecast(ticker: str, request: ForecastRequest) -> Dict[str, Any]:
        application = await _build_application(ticker, {})
        result = await _call_with_error_handling(
            application.predict,
            targets=request.targets,
            refresh=request.refresh,
            horizon=request.horizon,
        )
        return {"status": "ok", "forecasts": result}

    @app.post("/backtests/{ticker}", dependencies=[Depends(require_api_key)])
    async def backtest(ticker: str, request: BacktestRequest) -> Dict[str, Any]:
        application = await _build_application(ticker, {})
        result = await _call_with_error_handling(application.backtest, targets=request.targets)
        return {"status": "ok", "backtest": result}

    @app.get("/research", dependencies=[Depends(require_api_key)])
    async def research_feed(
        limit: int = Query(25, ge=1, le=200, description="Maximum number of research artifacts."),
    ) -> Dict[str, Any]:
        service = await _get_research_service()
        records = await _call_with_error_handling(service.get_feed, limit=limit)
        return {"status": "ok", "items": records}

    return app


__all__ = ["create_app", "require_api_key"]
