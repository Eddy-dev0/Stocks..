"""Wrapper utilities for launching the Streamlit dashboard."""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Mapping

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_PATH = PROJECT_ROOT / "ui" / "frontend" / "app.py"

LOGGER = logging.getLogger(__name__)


def run_streamlit_app(
    port: int = 8501,
    *,
    headless: bool = True,
    gather_usage_stats: bool = False,
    env: Mapping[str, str] | None = None,
) -> int:
    """Launch the Streamlit dashboard script and return the exit code."""

    if not FRONTEND_PATH.exists():
        LOGGER.error("Streamlit dashboard entry point not found at %s", FRONTEND_PATH)
        return 1

    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(FRONTEND_PATH),
        "--server.port",
        str(port),
        "--server.headless",
        "true" if headless else "false",
        "--browser.gatherUsageStats",
        "true" if gather_usage_stats else "false",
    ]

    environment = os.environ.copy()
    if env:
        environment.update(env)

    root_str = str(PROJECT_ROOT)
    pythonpath = environment.get("PYTHONPATH")
    if pythonpath:
        paths = pythonpath.split(os.pathsep)
        if root_str not in paths:
            environment["PYTHONPATH"] = os.pathsep.join([root_str, pythonpath])
    else:
        environment["PYTHONPATH"] = root_str

    LOGGER.info("Starting Streamlit dashboard on http://localhost:%s", port)
    result = subprocess.run(command, env=environment, cwd=PROJECT_ROOT, check=False)
    return result.returncode


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Stock Predictor Streamlit dashboard.")
    parser.add_argument("--port", type=int, default=8501, help="Port number for the dashboard server.")
    parser.add_argument(
        "--headless",
        choices={"true", "false"},
        default="true",
        help="Control Streamlit's headless mode.",
    )
    parser.add_argument(
        "--gather-usage-stats",
        choices={"true", "false"},
        default="false",
        help="Control whether Streamlit gathers usage statistics.",
    )
    return parser


def _main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    headless = args.headless.lower() == "true"
    gather_stats = args.gather_usage_stats.lower() == "true"
    return run_streamlit_app(port=args.port, headless=headless, gather_usage_stats=gather_stats)


if __name__ == "__main__":  # pragma: no cover - manual invocation helper
    raise SystemExit(_main())


__all__ = ["run_streamlit_app"]
