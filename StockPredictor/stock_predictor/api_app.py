"""Convenience wrapper for launching the Stock Predictor FastAPI service."""

from __future__ import annotations

import argparse
import logging
from typing import Any

import uvicorn

LOGGER = logging.getLogger(__name__)


def run_api(host: str = "127.0.0.1", port: int = 8000, **uvicorn_options: Any) -> None:
    """Start the FastAPI application using uvicorn."""

    LOGGER.info("Starting API server on http://%s:%s", host, port)
    uvicorn.run(
        "stock_predictor.ui.api.main:app",
        host=host,
        port=port,
        **uvicorn_options,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Stock Predictor API server.")
    parser.add_argument("--host", default="127.0.0.1", help="Hostname or IP address to bind.")
    parser.add_argument("--port", type=int, default=8000, help="Port number to bind.")
    return parser


def _main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    run_api(host=args.host, port=args.port)


if __name__ == "__main__":  # pragma: no cover - manual invocation helper
    _main()


__all__ = ["run_api"]
