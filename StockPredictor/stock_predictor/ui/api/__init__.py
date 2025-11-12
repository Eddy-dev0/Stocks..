"""Compatibility wrapper that exposes the deployment FastAPI application."""

from __future__ import annotations

from ui.api import create_app as create_api_app
from ui.api.main import app as deployment_app

app = deployment_app

__all__ = ["app", "create_api_app"]
