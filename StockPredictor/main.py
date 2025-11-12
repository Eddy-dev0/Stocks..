"""Deprecated command line entry point for the stock predictor platform."""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

from stock_predictor.app import StockPredictorApplication


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    default_mode = os.getenv("STOCK_PREDICTOR_DEFAULT_MODE", "predict")
    default_ticker = os.getenv("STOCK_PREDICTOR_DEFAULT_TICKER", "AAPL")

    parser = argparse.ArgumentParser(
        description="Legacy CLI interface for the Stock Predictor application.",
    )
    parser.add_argument(
        "--mode",
        choices=[
            "download-data",
            "train",
            "predict",
            "backtest",
            "importance",
            "list-models",
            "dashboard",
        ],
        default=default_mode,
        help="Pipeline mode to run (default: %(default)s).",
    )
    parser.add_argument("--ticker", default=default_ticker, help="Ticker symbol to process.")
    parser.add_argument("--start-date", help="Historical start date (YYYY-MM-DD).")
    parser.add_argument("--end-date", help="Historical end date (YYYY-MM-DD).")
    parser.add_argument("--interval", default="1d", help="Historical data interval (default: 1d).")
    parser.add_argument(
        "--model-type",
        default="random_forest",
        help="Model type to use (random_forest, lightgbm, xgboost, hist_gb, mlp, logistic).",
    )
    parser.add_argument("--data-dir", help="Directory to store downloaded data.")
    parser.add_argument("--models-dir", help="Directory to store trained models.")
    parser.add_argument("--news-api-key", help="API key for the news provider integration.")
    parser.add_argument("--news-limit", type=int, default=50, help="Maximum number of news items.")
    parser.add_argument("--database-url", help="Database connection URL.")
    parser.add_argument(
        "--no-sentiment",
        action="store_true",
        help="Disable sentiment analysis even if news data is available.",
    )
    parser.add_argument(
        "--refresh-data",
        action="store_true",
        help="Force redownload of remote data before processing.",
    )
    parser.add_argument(
        "--targets",
        help=(
            "Comma separated list of prediction targets to train/predict/backtest "
            "(default: close,direction,return,volatility)."
        ),
    )
    parser.add_argument(
        "--feature-sets",
        help=(
            "Comma separated list of feature groups to enable (technical, elliott, "
            "fundamental, macro, sentiment, identification, volume_liquidity, options, esg)."
        ),
    )
    parser.add_argument("--model-params", help="JSON string of model hyper-parameters.")
    parser.add_argument(
        "--volatility-window",
        type=int,
        default=20,
        help="Rolling window (in days) for volatility label generation.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        help="Override the default prediction horizon for training/prediction.",
    )
    parser.add_argument("--target", help="Focus on a single target when reporting importance.")
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ...).",
    )
    parser.add_argument(
        "--api-host",
        default="127.0.0.1",
        help="Host for the embedded API server when launching the dashboard.",
    )
    parser.add_argument(
        "--api-port",
        type=int,
        default=8000,
        help="Port for the embedded API server when launching the dashboard.",
    )
    parser.add_argument(
        "--ui-port",
        type=int,
        default=8501,
        help="Port for the Streamlit dashboard when launching the UI.",
    )
    parser.add_argument(
        "--ui-headless",
        action="store_true",
        help="Launch the dashboard without opening a browser window.",
    )
    parser.add_argument(
        "--ui-api-key",
        help="API key injected into the dashboard session for authenticated API calls.",
    )

    args = parser.parse_args(argv)

    supplied_argv = sys.argv[1:] if argv is None else list(argv)
    provided_mode = any(arg.startswith("--mode") for arg in supplied_argv)
    if not provided_mode and args.mode == default_mode == "predict":
        setattr(args, "_auto_mode", "dashboard")
        args.mode = "dashboard"
    else:
        setattr(args, "_auto_mode", None)

    return args


def _parse_csv(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.log_level)

    logging.warning(
        "The command line interface is deprecated. Instantiate StockPredictorApplication "
        "and use the API or UI packages instead."
    )

    if getattr(args, "_auto_mode", None):
        logging.info(
            "No mode supplied. Launching the interactive dashboard instead (override with --mode)."
        )

    try:
        model_params = json.loads(args.model_params) if args.model_params else None
    except json.JSONDecodeError as exc:  # pragma: no cover - user error path
        print(json.dumps({"status": "error", "message": str(exc)}), file=sys.stderr)
        return 1

    targets = _parse_csv(args.targets)

    overrides: dict[str, Any] = {
        "ticker": args.ticker,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "interval": args.interval,
        "model_type": args.model_type,
        "data_dir": args.data_dir,
        "models_dir": args.models_dir,
        "news_api_key": args.news_api_key,
        "news_limit": args.news_limit,
        "sentiment": not args.no_sentiment,
        "database_url": args.database_url,
        "feature_sets": args.feature_sets,
        "prediction_targets": targets,
        "model_params": model_params,
        "volatility_window": args.volatility_window,
    }

    if args.mode == "dashboard":
        return launch_dashboard(args, overrides)

    app = StockPredictorApplication.from_environment(**overrides)

    try:
        result = app.run(
            args.mode,
            targets=targets,
            refresh=args.refresh_data,
            force=args.refresh_data,
            horizon=args.horizon,
            target=args.target,
        )
    except Exception as exc:  # pylint: disable=broad-except
        logging.exception("Pipeline execution failed")
        print(json.dumps({"status": "error", "message": str(exc)}), file=sys.stderr)
        return 1

    output = {"status": result.status, **result.payload}
    print(json.dumps(output, indent=2))
    return 0


def launch_dashboard(args: argparse.Namespace, overrides: dict[str, Any]) -> int:
    """Launch the embedded API service and Streamlit dashboard."""

    frontend_path = Path(__file__).resolve().parent / "ui" / "frontend" / "app.py"
    if not frontend_path.exists():
        logging.error("Streamlit dashboard entry point not found at %s", frontend_path)
        return 1

    env = os.environ.copy()
    default_ticker = overrides.get("ticker") or env.get("STOCK_PREDICTOR_DEFAULT_TICKER") or "AAPL"
    env.setdefault("STOCK_PREDICTOR_DEFAULT_TICKER", default_ticker)
    env.setdefault("STOCK_PREDICTOR_API_URL", f"http://{args.api_host}:{args.api_port}")
    if args.ui_api_key:
        env["STOCK_PREDICTOR_UI_API_KEY"] = args.ui_api_key
        env["STOCK_PREDICTOR_UI_API_KEYS"] = args.ui_api_key

    logging.info(
        "Starting API server on http://%s:%s and dashboard on http://localhost:%s",
        args.api_host,
        args.api_port,
        args.ui_port,
    )

    api_cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "stock_predictor.ui.api.main:app",
        "--host",
        str(args.api_host),
        "--port",
        str(args.api_port),
    ]

    ui_cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(frontend_path),
        "--server.port",
        str(args.ui_port),
    ]
    if args.ui_headless:
        ui_cmd.extend(["--server.headless", "true"])

    api_process = subprocess.Popen(api_cmd, env=env)
    try:
        result = subprocess.run(ui_cmd, env=env, check=False)
        return result.returncode
    except KeyboardInterrupt:
        logging.info("Dashboard interrupted by user.")
        return 0
    finally:
        api_process.send_signal(signal.SIGINT)
        try:
            api_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            logging.debug("Force terminating API server")
            api_process.kill()


if __name__ == "__main__":
    raise SystemExit(main())
