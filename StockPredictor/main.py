"""Command line interface for the stock predictor pipeline."""

from __future__ import annotations

import argparse
import json
import logging
import sys

from stock_predictor.config import build_config
from stock_predictor.model import StockPredictorAI


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Stock Predictor AI pipeline.",
    )
    parser.add_argument(
        "--mode",
        choices=["download-data", "train", "predict"],
        required=True,
        help="Pipeline mode to run.",
    )
    parser.add_argument("--ticker", required=True, help="Ticker symbol to process.")
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
    return parser.parse_args(argv)


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
        else:
            raise ValueError(f"Unsupported mode: {args.mode}")
    except Exception as exc:  # pylint: disable=broad-except
        logging.exception("Pipeline failed")
        print(json.dumps({"status": "error", "message": str(exc)}), file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
