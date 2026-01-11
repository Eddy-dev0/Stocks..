"""Compatibility wrapper that exposes the deployment FastAPI application."""

from __future__ import annotations


def create_api_app():
    """Create the FastAPI application on demand to avoid circular imports."""

    from ui.api.app import create_app

    return create_app()


__all__ = ["create_api_app"]
