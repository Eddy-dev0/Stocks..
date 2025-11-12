"""FastAPI integration layer for the stock predictor application."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from stock_predictor.app import StockPredictorApplication
from stock_predictor.research import ResearchService


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
        from fastapi import FastAPI, HTTPException, Request
        from pydantic import AnyHttpUrl, BaseModel, Field, root_validator
    except ImportError as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "FastAPI and pydantic are required to use the stock_predictor.ui package."
        ) from exc

    overrides = dict(default_overrides or {})
    research_state: Dict[str, Any] = {"service": None}

    class PredictionRequest(BaseModel):
        ticker: str
        targets: list[str] | None = None
        refresh: bool = False
        horizon: int | None = None

    class TrainingRequest(BaseModel):
        ticker: str
        targets: list[str] | None = None
        horizon: int | None = None

    class CrawlRequest(BaseModel):
        url: AnyHttpUrl
        metadata: Dict[str, Any] | None = None

    class SnippetRequest(BaseModel):
        url: AnyHttpUrl
        html: str | None = Field(default=None, description="Optional HTML payload")
        text: str | None = Field(default=None, description="Optional plaintext snippet")
        metadata: Dict[str, Any] | None = None

        @root_validator
        def _validate_payload(cls, values: Dict[str, Any]) -> Dict[str, Any]:  # noqa: D401,N805
            text = (values.get("text") or "").strip()
            html = (values.get("html") or "").strip()
            if not text and not html:
                raise ValueError("Either 'text' or 'html' content must be provided")
            return values

    app = FastAPI(title="Stock Predictor API", version="2.0.0")

    @app.get("/health")
    async def healthcheck() -> Dict[str, str]:
        return {"status": "ok"}

    async def _get_research_service() -> ResearchService:
        service = research_state.get("service")
        if service is None:
            raise HTTPException(status_code=503, detail="Research service not initialised")
        return service

    def _build_application(ticker: str) -> StockPredictorApplication:
        if not ticker:
            raise HTTPException(status_code=400, detail="Ticker symbol is required")
        payload = {**overrides, "ticker": ticker}
        return StockPredictorApplication.from_environment(**payload)

    @app.on_event("startup")
    async def _startup() -> None:
        ticker = overrides.get("ticker") or os.getenv("STOCK_PREDICTOR_DEFAULT_TICKER", "AAPL")
        app_instance = StockPredictorApplication.from_environment(**{**overrides, "ticker": ticker})
        research_state["service"] = ResearchService(app_instance.config)

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        service = research_state.get("service")
        if service is not None:
            await service.aclose()

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

    def _require_api_key(request: Request, service: ResearchService) -> None:
        api_key = request.headers.get("X-API-Key")
        if not service.is_authorised(api_key):
            raise HTTPException(status_code=401, detail="Invalid or missing API key")

    @app.get("/research")
    async def research_feed(limit: int = 50) -> Dict[str, Any]:
        service = await _get_research_service()
        try:
            records = await service.get_feed(limit=limit)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"items": records}

    @app.post("/research/crawl")
    async def research_crawl(request: CrawlRequest, http_request: Request) -> Dict[str, Any]:
        service = await _get_research_service()
        _require_api_key(http_request, service)
        try:
            record = await service.crawl(str(request.url), metadata=request.metadata)
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover - defensive
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        return {"item": record}

    @app.post("/research/snippets")
    async def research_snippet(request: SnippetRequest, http_request: Request) -> Dict[str, Any]:
        service = await _get_research_service()
        _require_api_key(http_request, service)
        try:
            record = await service.ingest_snippet(
                str(request.url),
                text=request.text,
                html=request.html,
                metadata=request.metadata,
            )
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover - defensive
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        return {"item": record}

    return app


__all__ = ["create_api_app"]
