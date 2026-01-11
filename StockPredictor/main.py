"""Entry point for launching the Stock Predictor user interfaces."""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import threading
import time
from datetime import timezone
from pathlib import Path
from typing import Any

from stock_predictor.api_app import run_api
from stock_predictor.dashboard_app import run_streamlit_app
from stock_predictor import dashboard_app as dashboard_module
from stock_predictor.app import StockPredictorApplication
from stock_predictor.core.clock import app_clock
from stock_predictor.ui_app import run_tkinter_app


REPOSITORY_ROOT = Path(__file__).resolve().parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

_existing_pythonpath = os.environ.get("PYTHONPATH")
if _existing_pythonpath:
    pythonpath_parts = _existing_pythonpath.split(os.pathsep)
    if str(REPOSITORY_ROOT) not in pythonpath_parts:
        os.environ["PYTHONPATH"] = os.pathsep.join([str(REPOSITORY_ROOT), _existing_pythonpath])
else:
    os.environ["PYTHONPATH"] = str(REPOSITORY_ROOT)


LOGGER = logging.getLogger(__name__)


def _ensure_dependencies(*, require_api: bool, require_dashboard: bool) -> bool:
    """Verify optional dependencies needed for the requested mode."""

    missing: list[str] = []

    if require_api:
        try:  # pragma: no cover - runtime guard for optional dependency
            import uvicorn  # noqa: F401  # pylint: disable=unused-import
        except ModuleNotFoundError:  # pragma: no cover - runtime guard
            missing.append("uvicorn")

    if require_dashboard:
        try:  # pragma: no cover - runtime guard for optional dependency
            import streamlit  # noqa: F401  # pylint: disable=unused-import
        except ModuleNotFoundError:  # pragma: no cover - runtime guard
            missing.append("streamlit")

    if missing:
        formatted = ", ".join(sorted(missing))
        print(
            "Missing dependency: the requested mode requires the following packages: "
            f"{formatted}. Install them with `pip install {formatted}`.",
            file=sys.stderr,
        )
        return False

    return True


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def _build_streamlit_environment() -> tuple[dict[str, str], bool, bool]:
    """Prepare environment variables for launching the Streamlit dashboard."""

    env = os.environ.copy()
    root_str = str(dashboard_module.PROJECT_ROOT)
    pythonpath = env.get("PYTHONPATH")
    if pythonpath:
        paths = pythonpath.split(os.pathsep)
        if root_str not in paths:
            env["PYTHONPATH"] = os.pathsep.join([root_str, pythonpath])
    else:
        env["PYTHONPATH"] = root_str

    gather_stats_env = env.get("STREAMLIT_BROWSER_GATHER_USAGE_STATS")
    gather_stats_value = gather_stats_env.lower() if gather_stats_env else "false"
    env["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = gather_stats_value
    gather_stats_flag = gather_stats_value == "true"

    headless_env = env.get("STREAMLIT_HEADLESS")
    headless_value = headless_env.lower() if headless_env else "true"
    env["STREAMLIT_HEADLESS"] = headless_value
    headless_flag = headless_value == "true"

    return env, headless_flag, gather_stats_flag


def _spawn_streamlit_process(port: int, env: dict[str, str], *, headless: bool, gather_usage_stats: bool) -> subprocess.Popen[Any]:
    """Launch Streamlit as a subprocess and return the handle."""

    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(dashboard_module.FRONTEND_PATH),
        "--server.port",
        str(port),
        "--server.headless",
        "true" if headless else "false",
        "--browser.gatherUsageStats",
        "true" if gather_usage_stats else "false",
    ]

    LOGGER.info("Starting Streamlit dashboard on http://localhost:%s", port)
    return subprocess.Popen(command, env=env, cwd=dashboard_module.PROJECT_ROOT)


def _start_api_thread(host: str, port: int) -> threading.Thread:
    """Launch the FastAPI service in a daemon thread."""

    def _run() -> None:
        run_api(host=host, port=port, log_level="info")

    thread = threading.Thread(target=_run, name="uvicorn-server", daemon=True)
    thread.start()
    return thread


def _apply_ui_flags(*, no_train: bool, no_refresh: bool) -> None:
    """Expose UI-specific toggles via environment variables."""

    if no_train:
        os.environ["STOCK_PREDICTOR_UI_DISABLE_TRAINING"] = "1"
    if no_refresh:
        os.environ["STOCK_PREDICTOR_UI_DISABLE_REFRESH"] = "1"


def _parse_interval(raw_value: str) -> float:
    """Parse an interval string supporting seconds or minutes suffixes."""

    normalized = raw_value.strip().lower()
    multiplier = 1.0
    if normalized.endswith("m"):
        multiplier = 60.0
        normalized = normalized[:-1]
    elif normalized.endswith("s"):
        normalized = normalized[:-1]

    try:
        value = float(normalized)
    except ValueError as exc:  # pragma: no cover - defensive user input guard
        raise argparse.ArgumentTypeError(
            "Interval must be a number optionally suffixed with 's' or 'm'."
        ) from exc

    if value <= 0:
        raise argparse.ArgumentTypeError("Interval must be greater than zero.")

    return value * multiplier


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch the Stock Predictor desktop UI, API server, or dashboard.",
    )
    parser.add_argument(
        "--mode",
        choices=["tk", "dash", "api", "both", "full", "live-loop"],
        default="tk",
        help=(
            "Execution mode: 'tk' launches the desktop UI, 'dash' launches only the "
            "Streamlit dashboard, 'api' starts the FastAPI service, 'both' launches "
            "the desktop UI together with the API, 'full' launches all three, and "
            "'live-loop' runs repeated live analyses without starting any UIs."
        ),
    )
    parser.add_argument(
        "--host",
        default=os.getenv("STOCK_PREDICTOR_API_HOST", "127.0.0.1"),
        help="Host interface for the FastAPI service (api/both/full modes).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("STOCK_PREDICTOR_API_PORT", "8000")),
        help="Port for the FastAPI service (api/both/full modes).",
    )
    parser.add_argument(
        "--dash-port",
        type=int,
        default=int(os.getenv("STOCK_PREDICTOR_DASHBOARD_PORT", "8501")),
        help="Port for the Streamlit dashboard (dash/full modes).",
    )
    parser.add_argument(
        "--log-level",
        default=os.getenv("STOCK_PREDICTOR_LOG_LEVEL", "INFO"),
        help="Logging level (DEBUG, INFO, WARNING, ...).",
    )
    parser.add_argument(
        "--no-train",
        action="store_true",
        help="Disable training-related controls in the Tkinter UI (tk/both/full modes).",
    )
    parser.add_argument(
        "--no-refresh",
        action="store_true",
        help="Disable refresh controls in the Tkinter UI (tk/both/full modes).",
    )
    parser.add_argument(
        "--interval",
        type=_parse_interval,
        default=float(os.getenv("STOCK_PREDICTOR_LIVE_INTERVAL_SECONDS", "300")),
        help=(
            "Interval between live analyses when running in 'live-loop' mode. "
            "Accepts seconds by default or a value suffixed with 's' or 'm' (e.g. '90s', '5m')."
        ),
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""

    parser = _build_parser()
    return parser.parse_args(argv)


def _run_tk_mode(args: argparse.Namespace) -> int:
    _apply_ui_flags(no_train=args.no_train, no_refresh=args.no_refresh)
    LOGGER.info("Launching the Tkinter desktop application. Close the window to exit.")
    run_tkinter_app()
    return 0


def _run_api_mode(args: argparse.Namespace) -> int:
    LOGGER.info(
        "Starting the FastAPI service on http://%s:%s (press Ctrl+C to stop)",
        args.host,
        args.port,
    )
    run_api(host=args.host, port=args.port)
    return 0


def _run_dash_mode(args: argparse.Namespace) -> int:
    env, headless_flag, gather_flag = _build_streamlit_environment()
    return run_streamlit_app(
        port=args.dash_port,
        headless=headless_flag,
        gather_usage_stats=gather_flag,
        env=env,
    )


def _run_both_mode(args: argparse.Namespace) -> int:
    LOGGER.info(
        "Starting the FastAPI service on http://%s:%s in the background.",
        args.host,
        args.port,
    )
    _start_api_thread(args.host, args.port)
    return _run_tk_mode(args)


def _run_full_mode(args: argparse.Namespace) -> int:
    LOGGER.info(
        "Starting the FastAPI service on http://%s:%s in the background.",
        args.host,
        args.port,
    )
    _start_api_thread(args.host, args.port)

    env, headless_flag, gather_flag = _build_streamlit_environment()
    streamlit_process = _spawn_streamlit_process(
        args.dash_port,
        env,
        headless=headless_flag,
        gather_usage_stats=gather_flag,
    )

    exit_code = 0
    try:
        exit_code = _run_tk_mode(args)
    finally:
        if streamlit_process.poll() is None:
            LOGGER.info("Closing Streamlit dashboard...")
            streamlit_process.terminate()
            try:
                streamlit_process.wait(timeout=5)
            except subprocess.TimeoutExpired:  # pragma: no cover - defensive shutdown
                streamlit_process.kill()
        else:
            exit_code = streamlit_process.returncode

    return exit_code


def _run_live_analysis(application: StockPredictorApplication) -> dict[str, Any]:
    """Execute a single live prediction pass and return the payload."""

    LOGGER.info("Running live analysis for ticker %s", application.config.ticker)
    return application.predict(refresh=True)


def _run_live_loop_mode(args: argparse.Namespace) -> int:
    """Continuously run live analyses at a fixed interval without launching UIs."""

    application = StockPredictorApplication.from_environment()
    interval_seconds = args.interval

    LOGGER.info(
        "Starting live analysis loop for %s every %.0f seconds. Press Ctrl+C to stop.",
        application.config.ticker,
        interval_seconds,
    )

    try:
        while True:
            timestamp = app_clock.system_now(timezone.utc)
            try:
                payload = _run_live_analysis(application)
                LOGGER.info("Live analysis at %s: %s", timestamp.isoformat(), payload)
            except Exception:  # pragma: no cover - runtime guard around long-running loop
                LOGGER.exception("Live analysis failed at %s", timestamp.isoformat())

            time.sleep(interval_seconds)
    except KeyboardInterrupt:  # pragma: no cover - interactive exit
        LOGGER.info("Stopping live analysis loop.")
        return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    _configure_logging(args.log_level)

    mode = args.mode
    require_api = mode in {"api", "both", "full"}
    require_dashboard = mode in {"dash", "full"}

    if not _ensure_dependencies(require_api=require_api, require_dashboard=require_dashboard):
        return 1

    try:
        if mode == "tk":
            return _run_tk_mode(args)
        if mode == "api":
            return _run_api_mode(args)
        if mode == "dash":
            return _run_dash_mode(args)
        if mode == "both":
            return _run_both_mode(args)
        if mode == "full":
            return _run_full_mode(args)
        if mode == "live-loop":
            return _run_live_loop_mode(args)
    except KeyboardInterrupt:  # pragma: no cover - interactive exit
        LOGGER.info("Interrupted by user. Shutting down...")
        return 0

    raise ValueError(f"Unsupported mode: {mode}")  # pragma: no cover - defensive guard


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
