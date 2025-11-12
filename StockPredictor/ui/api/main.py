"""ASGI entry point for running the Stock Predictor UI API via uvicorn."""

from __future__ import annotations

from .app import create_app

app = create_app()

__all__ = ["app"]
