"""FastAPI application exposing Stock Predictor services for the web UI."""

from __future__ import annotations

import inspect
import os
from functools import lru_cache
from datetime import datetime
from typing import Any, Dict

import pandas as pd

from fastapi import Depends, FastAPI, HTTPException, Query, Security
from fastapi.concurrency import run_in_threadpool
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from stock_predictor.app import StockPredictorApplication
from stock_predictor.research import ResearchService
from stock_predictor.ui.api import ui_adapter


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
        default=True,
        description="If true, the underlying datasets will be refreshed before predicting.",
    )
    horizon: int | None = Field(
        default=None,
        ge=1,
        description="Optional number of periods ahead to forecast.",
    )
    feature_toggles: dict[str, bool] | None = Field(
        default=None,
        description=(
            "Optional map of feature-group toggles keyed by registry name. "
            "Supported keys: elliott, macro, sentiment, technical, volume_liquidity."
        ),
        example={"technical": True, "macro": False},
    )


class BacktestRequest(BaseModel):
    """Payload used to trigger a backtest run for a ticker."""

    targets: list[str] | None = Field(
        default=None,
        description="Optional list of targets to include in the backtest.",
    )
    feature_toggles: dict[str, bool] | None = Field(
        default=None,
        description=(
            "Optional map of feature-group toggles keyed by registry name. "
            "Supported keys: elliott, macro, sentiment, technical, volume_liquidity."
        ),
        example={"technical": True, "macro": False},
    )


class TrainRequest(BaseModel):
    """Payload used to retrain models for a ticker."""

    targets: list[str] | None = Field(
        default=None,
        description="Optional list of targets to include during training.",
    )
    horizon: int | None = Field(
        default=None,
        ge=1,
        description="Optional forecast horizon to train against.",
    )
    feature_toggles: dict[str, bool] | None = Field(
        default=None,
        description=(
            "Optional map of feature-group toggles keyed by registry name. "
            "Supported keys: elliott, macro, sentiment, technical, volume_liquidity."
        ),
        example={"technical": True, "macro": False},
    )
    evaluation_strategy: str | None = Field(
        default=None,
        description="Evaluation strategy: holdout, time_series, or rolling.",
    )
    evaluation_folds: int | None = Field(
        default=None,
        ge=2,
        description="Number of folds to use for time-series cross validation.",
    )
    tuning_enabled: bool | None = Field(
        default=None,
        description="Whether to run hyperparameter tuning before fitting final models.",
    )
    tuning_iterations: int | None = Field(
        default=None,
        ge=1,
        description="Number of parameter samples to explore during tuning.",
    )


class BuyZoneRequest(BaseModel):
    """Payload used to request buy-zone analysis for a ticker."""

    refresh: bool = Field(
        default=True,
        description="Refresh underlying data sources before computing the buy zone.",
    )
    feature_toggles: dict[str, bool] | None = Field(
        default=None,
        description=(
            "Optional map of feature-group toggles keyed by registry name. "
            "Supported keys: elliott, macro, sentiment, technical, volume_liquidity."
        ),
        example={"technical": True, "macro": False},
    )


class TimeWindow(BaseModel):
    """Represents the time window used during buy-zone analysis."""

    start: str | None = Field(None, description="ISO timestamp for the start of the window.")
    end: str | None = Field(None, description="ISO timestamp for the end of the window.")


class PriceBounds(BaseModel):
    """Summary of the derived buy-zone price bounds."""

    lower: float | None = Field(None, description="Lower edge of the computed buy zone.")
    upper: float | None = Field(None, description="Upper edge of the computed buy zone.")
    support: float | None = Field(None, description="Nearest detected support level.")
    last_close: float | None = Field(None, description="Most recent close used in the calculation.")


class IndicatorConfirmationResponse(BaseModel):
    """Indicator-driven confirmation details returned by the buy-zone analysis."""

    confirmed: bool = Field(..., description="Whether the indicator confirms the buy setup.")
    value: float | None = Field(None, description="Latest indicator value used for the check.")
    threshold: float | None = Field(None, description="Threshold applied to the indicator value.")
    detail: str | None = Field(None, description="Human-readable explanation of the signal.")


class BuyZoneResponse(BaseModel):
    """Structured response for the buy-zone endpoint."""

    ticker: str = Field(..., description="Ticker symbol analysed.")
    window: TimeWindow
    price_bounds: PriceBounds
    confirmations: dict[str, IndicatorConfirmationResponse] = Field(
        default_factory=dict, description="Indicator confirmations supporting the buy zone."
    )
    support_components: dict[str, float] = Field(
        default_factory=dict,
        description="Raw support level components extracted from indicators.",
    )


class BuyZoneEnvelope(BaseModel):
    """Top-level envelope for buy-zone responses."""

    status: str = Field("ok", description="Outcome of the request.")
    buy_zone: BuyZoneResponse


class LivePriceRequest(BaseModel):
    """Payload describing live price context options."""

    horizon: int | None = Field(
        default=None,
        description="Optional override for the forecast horizon to evaluate.",
        ge=1,
    )


class DirectionProbabilities(BaseModel):
    """Directional probability summary for the live price snapshot."""

    up: float | None = Field(None, description="Probability of an upward move.")
    down: float | None = Field(None, description="Probability of a downward move.")


class LivePriceResponse(BaseModel):
    """Structured live price snapshot with model-derived context."""

    ticker: str = Field(..., description="Ticker symbol for the snapshot.")
    market_time: datetime | None = Field(
        None, description="Timestamp of the latest price in the market timezone."
    )
    last_price: float | None = Field(None, description="Most recent traded price.")
    predicted_close: float | None = Field(
        None, description="Price adjusted by the model's expected percentage change."
    )
    expected_change_pct: float | None = Field(
        None, description="Model-expected percentage change (decimal form)."
    )
    expected_low: float | None = Field(
        None, description="Expected downside level derived from the model percentage."
    )
    stop_loss: float | None = Field(
        None, description="Stop-loss level based on configured risk percentage or expected low."
    )
    probabilities: DirectionProbabilities = Field(
        default_factory=DirectionProbabilities,
        description="Directional probabilities from the model output.",
    )


class LivePriceEnvelope(BaseModel):
    """Envelope for live price snapshots."""

    status: str = Field("ok", description="Outcome of the request.")
    price: LivePriceResponse


class AccuracyResponse(BaseModel):
    """Aggregated accuracy details derived from experiment logs."""

    horizon: int | None = Field(None, description="Prediction horizon the metrics relate to.")
    runs_considered: int = Field(..., description="Number of backtest runs included in the summary.")
    total_predictions: int = Field(..., description="Total number of predictions evaluated.")
    correct: int = Field(..., description="Number of correct predictions.")
    incorrect: int = Field(..., description="Number of incorrect predictions.")
    correct_pct: float = Field(..., description="Fraction of predictions that were correct.")
    incorrect_pct: float = Field(..., description="Fraction of predictions that were incorrect.")


class AccuracyEnvelope(BaseModel):
    """Envelope for accuracy summaries."""

    status: str = Field("ok", description="Outcome of the request.")
    accuracy: AccuracyResponse


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
        refresh: bool = Query(True, description="Refresh underlying data sources."),
        start_date: str | None = Query(None, description="Optional ISO start date override."),
        end_date: str | None = Query(None, description="Optional ISO end date override."),
        interval: str | None = Query(None, description="Historical data interval override."),
    ) -> Dict[str, Any]:
        data = await _call_with_error_handling(
            ui_adapter.refresh_data,
            ticker,
            refresh=refresh,
            overrides={
                "start_date": start_date,
                "end_date": end_date,
                "interval": interval,
            },
        )
        return {"status": "ok", "data": data}

    @app.get("/insights/{ticker}", dependencies=[Depends(require_api_key)])
    async def get_insights(
        ticker: str,
        refresh: bool = Query(False, description="Refresh sentiment signals."),
    ) -> Dict[str, Any]:
        application = await _build_application(ticker, {})
        sentiment_df: pd.DataFrame | None = None
        try:
            sentiment_df = await _call_with_error_handling(
                application.pipeline.fetcher.fetch_sentiment_signals, force=refresh
            )
        except Exception:
            sentiment_df = None

        sentiment_payload: Dict[str, Any] = {}
        if isinstance(sentiment_df, pd.DataFrame) and not sentiment_df.empty:
            sentiment_payload["records"] = sentiment_df.tail(10).to_dict(orient="records")
            latest = sentiment_df.iloc[-1]
            for candidate in ("sentiment", "Sentiment_Avg", "sentiment_score", "Score", "score"):
                if candidate in latest and pd.notna(latest[candidate]):
                    sentiment_payload["latest_score"] = float(latest[candidate])
                    break
        return {"status": "ok", "sentiment": sentiment_payload}

    @app.post("/forecasts/{ticker}", dependencies=[Depends(require_api_key)])
    async def forecast(ticker: str, request: ForecastRequest) -> Dict[str, Any]:
        result = await _call_with_error_handling(
            ui_adapter.get_prediction,
            ticker,
            request.horizon,
            refresh=request.refresh,
            targets=request.targets,
            overrides={
                "feature_toggles": request.feature_toggles,
                "price_feature_toggles": request.feature_toggles,
            },
        )
        return {"status": "ok", "forecasts": result}

    @app.post("/backtests/{ticker}", dependencies=[Depends(require_api_key)])
    async def backtest(ticker: str, request: BacktestRequest) -> Dict[str, Any]:
        result = await _call_with_error_handling(
            ui_adapter.run_backtest,
            ticker,
            targets=request.targets,
            overrides={
                "feature_toggles": request.feature_toggles,
                "price_feature_toggles": request.feature_toggles,
            },
        )
        return {"status": "ok", "backtest": result}

    @app.post("/train/{ticker}", dependencies=[Depends(require_api_key)])
    async def retrain(ticker: str, request: TrainRequest) -> Dict[str, Any]:
        overrides = {
            "feature_toggles": request.feature_toggles,
            "price_feature_toggles": request.feature_toggles,
            "evaluation_strategy": request.evaluation_strategy,
            "evaluation_folds": request.evaluation_folds,
            "tuning_enabled": request.tuning_enabled,
            "tuning_iterations": request.tuning_iterations,
        }
        refresh_result = await _call_with_error_handling(
            ui_adapter.refresh_data, ticker, refresh=False, overrides=overrides
        )
        metrics = await _call_with_error_handling(
            ui_adapter.train_models,
            ticker,
            targets=request.targets,
            horizon=request.horizon,
            overrides=overrides,
        )
        return {"status": "ok", "refresh": refresh_result, "metrics": metrics}

    @app.post(
        "/buy-zone/{ticker}",
        dependencies=[Depends(require_api_key)],
        response_model=BuyZoneEnvelope,
    )
    async def buy_zone(ticker: str, request: BuyZoneRequest) -> BuyZoneEnvelope:
        application = await _build_application(
            ticker,
            {
                "feature_toggles": request.feature_toggles,
                "price_feature_toggles": request.feature_toggles,
            },
        )
        result = await _call_with_error_handling(
            application.buy_zone,
            refresh=request.refresh,
        )
        return BuyZoneEnvelope(status="ok", buy_zone=BuyZoneResponse(**result))

    @app.post(
        "/live-price/{ticker}",
        dependencies=[Depends(require_api_key)],
        response_model=LivePriceEnvelope,
    )
    async def live_price(ticker: str, request: LivePriceRequest) -> LivePriceEnvelope:
        application = await _build_application(ticker, {})
        snapshot_payload = await _call_with_error_handling(
            application.pipeline.live_price_snapshot,
            horizon=request.horizon,
        )

        snapshot = LivePriceResponse(**snapshot_payload)
        return LivePriceEnvelope(status="ok", price=snapshot)

    @app.get(
        "/accuracy/{ticker}",
        dependencies=[Depends(require_api_key)],
        response_model=AccuracyEnvelope,
    )
    async def accuracy(
        ticker: str, horizon: int | None = Query(None, ge=1, description="Prediction horizon to filter runs.")
    ) -> AccuracyEnvelope:
        summary = await _call_with_error_handling(
            ui_adapter.get_accuracy, ticker, horizon=horizon
        )
        return AccuracyEnvelope(status="ok", accuracy=AccuracyResponse(**summary))

    @app.get("/research", dependencies=[Depends(require_api_key)])
    async def research_feed(
        limit: int = Query(25, ge=1, le=200, description="Maximum number of research artifacts."),
    ) -> Dict[str, Any]:
        service = await _get_research_service()
        records = await _call_with_error_handling(service.get_feed, limit=limit)
        return {"status": "ok", "items": records}

    return app


__all__ = ["create_app", "require_api_key"]
