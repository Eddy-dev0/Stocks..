"""Command line interface for the stock predictor pipeline."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any, Dict

from stock_predictor.config import build_config, load_environment
from stock_predictor.model import StockPredictorAI


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    load_environment()

    default_mode = os.getenv("STOCK_PREDICTOR_DEFAULT_MODE", "predict")
    default_ticker = os.getenv("STOCK_PREDICTOR_DEFAULT_TICKER", "AAPL")

    parser = argparse.ArgumentParser(
        description="Run the Stock Predictor AI pipeline.",
    )
    parser.add_argument(
        "--mode",
        choices=["download-data", "train", "predict"],
        default=default_mode,
        help=(
            "Pipeline mode to run. Defaults to the value of the "
            "STOCK_PREDICTOR_DEFAULT_MODE environment variable or 'predict'."
        ),
    )
    parser.add_argument(
        "--ticker",
        default=default_ticker,
        help=(
            "Ticker symbol to process. Defaults to the value of the "
            "STOCK_PREDICTOR_DEFAULT_TICKER environment variable or 'AAPL'."
        ),
    )
    parser.add_argument(
        "--start-date",
        help="Historical start date (YYYY-MM-DD). Defaults to one year ago.",
    )
    parser.add_argument("--end-date", help="Historical end date (YYYY-MM-DD).")
    parser.add_argument(
        "--interval",
        default="1d",
        help="Historical data interval supported by yfinance (default: 1d).",
    )
    parser.add_argument(
        "--model-type",
        default="random_forest",
        help="Model type to use (currently only random_forest is implemented).",
    )
    parser.add_argument(
        "--data-dir",
        help="Directory to store downloaded data (defaults to ./data).",
    )
    parser.add_argument(
        "--models-dir",
        help="Directory to store trained models (defaults to ./models).",
    )
    parser.add_argument(
        "--news-api-key",
        help="API key for the Financial Modeling Prep news endpoint.",
    )
    parser.add_argument(
        "--news-limit",
        type=int,
        default=50,
        help="Maximum number of news articles to download (default: 50).",
    )
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
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ...).",
    )
    parser.add_argument(
        "--ui",
        action="store_true",
        help=(
            "Display prediction results in a graphical window. Only applies in predict mode."
        ),
    )
    return parser.parse_args(argv)


def display_prediction_ui(prediction: Dict[str, Any]) -> None:
    """Render a simple Tkinter window summarizing the prediction."""

    import logging
    import tkinter as tk
    from tkinter import ttk

    ticker = prediction.get("ticker", "-")
    predicted_close = prediction.get("predicted_close")
    last_close = prediction.get("last_close")
    change = prediction.get("expected_change")
    change_pct = prediction.get("expected_change_pct")
    as_of = prediction.get("as_of", "-")
    training_metrics = prediction.get("training_metrics")

    logger = logging.getLogger(__name__)

    try:
        root = tk.Tk()
    except tk.TclError as exc:  # pragma: no cover - depends on system display
        logger.warning("Unable to start UI: %s", exc)
        return
    root.title(f"Stock Prediction - {ticker}")

    mainframe = ttk.Frame(root, padding=20)
    mainframe.grid(column=0, row=0, sticky="nsew")
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)

    header = ttk.Label(
        mainframe,
        text=f"Prediction Summary for {ticker}",
        font=("Helvetica", 18, "bold"),
    )
    header.grid(column=0, row=0, columnspan=2, pady=(0, 15))

    def format_currency(value: Any) -> str:
        return "-" if value is None else f"${value:,.2f}"

    def format_change(value: Any, pct: Any) -> str:
        if value is None or pct is None:
            return "-"
        arrow = "▲" if value >= 0 else "▼"
        return f"{arrow} {value:,.2f} ({pct * 100:,.2f}%)"

    rows = [
        ("As of", as_of),
        ("Last Close", format_currency(last_close)),
        ("Predicted Close", format_currency(predicted_close)),
        ("Expected Change", format_change(change, change_pct)),
    ]

    for idx, (label, value) in enumerate(rows, start=1):
        ttk.Label(mainframe, text=label + ":", font=("Helvetica", 12, "bold")).grid(
            column=0,
            row=idx,
            sticky="w",
            pady=5,
            padx=(0, 10),
        )
        ttk.Label(mainframe, text=value, font=("Helvetica", 12)).grid(
            column=1,
            row=idx,
            sticky="w",
            pady=5,
        )

    if isinstance(training_metrics, dict):
        ttk.Separator(mainframe, orient="horizontal").grid(
            column=0, row=len(rows) + 1, columnspan=2, sticky="ew", pady=(15, 10)
        )
        ttk.Label(
            mainframe,
            text="Training Metrics",
            font=("Helvetica", 14, "bold"),
        ).grid(column=0, row=len(rows) + 2, columnspan=2, sticky="w")

        metric_rows = [
            ("MAE", training_metrics.get("mae")),
            ("RMSE", training_metrics.get("rmse")),
            ("R²", training_metrics.get("r2")),
        ]

        for offset, (label, value) in enumerate(metric_rows, start=len(rows) + 3):
            display_value = "-" if value is None else f"{value:,.4f}"
            ttk.Label(mainframe, text=label + ":", font=("Helvetica", 12, "bold")).grid(
                column=0,
                row=offset,
                sticky="w",
                pady=3,
                padx=(0, 10),
            )
            ttk.Label(mainframe, text=display_value, font=("Helvetica", 12)).grid(
                column=1,
                row=offset,
                sticky="w",
                pady=3,
            )

    ttk.Button(mainframe, text="Close", command=root.destroy).grid(
        column=0,
        row=99,
        columnspan=2,
        pady=(20, 0),
    )

    root.mainloop()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.log_level)

    config = build_config(
        ticker=args.ticker,
        start_date=args.start_date,
        end_date=args.end_date,
        interval=args.interval,
        model_type=args.model_type,
        data_dir=args.data_dir,
        models_dir=args.models_dir,
        news_api_key=args.news_api_key,
        news_limit=args.news_limit,
        sentiment=not args.no_sentiment,
    )

    ai = StockPredictorAI(config)

    try:
        if args.mode == "download-data":
            result = ai.download_data(force=args.refresh_data)
            print(json.dumps({"status": "ok", "downloaded": {k: str(v) for k, v in result.items()}}, indent=2))
        elif args.mode == "train":
            if args.refresh_data:
                ai.download_data(force=True)
            metrics = ai.train_model()
            print(json.dumps({"status": "ok", "metrics": metrics}, indent=2))
        elif args.mode == "predict":
            prediction = ai.predict(refresh_data=args.refresh_data)
            print(json.dumps({"status": "ok", "prediction": prediction}, indent=2))
            if args.ui:
                display_prediction_ui(prediction)
        else:
            raise ValueError(f"Unsupported mode: {args.mode}")
    except Exception as exc:  # pylint: disable=broad-except
        logging.exception("Pipeline failed")
        print(json.dumps({"status": "error", "message": str(exc)}), file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
