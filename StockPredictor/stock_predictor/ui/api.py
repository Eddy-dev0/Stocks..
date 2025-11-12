"""Compatibility wrapper that exposes the deployment FastAPI application."""

from __future__ import annotations

from ui.api import create_app as create_api_app

__all__ = ["create_api_app"]
