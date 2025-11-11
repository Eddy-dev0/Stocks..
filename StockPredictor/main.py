"""Command line interface and GUI entry point for the stock predictor pipeline."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any

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
        description="Run the Stock Predictor AI pipeline or launch the desktop UI.",
    )
    parser.add_argument(
        "--mode",
        choices=["download-data", "train", "predict", "backtest", "importance", "list-models"],
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
        help="Model type to use (random_forest, lightgbm, xgboost, hist_gb, mlp, logistic).",
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
        "--targets",
        help="Comma separated list of prediction targets to train/predict/backtest.",
    )
    parser.add_argument(
        "--feature-sets",
        help="Comma separated list of feature blocks to enable (technical, elliott, fundamental, sentiment, macro).",
    )
    parser.add_argument(
        "--model-params",
        help="JSON string describing model hyper-parameters (global/target overrides).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test set proportion for holdout validation (default: 0.2).",
    )
    parser.add_argument(
        "--shuffle-training",
        action="store_true",
        help="Shuffle rows before the train/test split (useful for non-time-series experimentation).",
    )
    parser.add_argument(
        "--backtest-window",
        type=int,
        default=252,
        help="Training window length for rolling/expanding backtests (default: 252).",
    )
    parser.add_argument(
        "--backtest-step",
        type=int,
        default=20,
        help="Number of rows to advance between backtest splits (default: 20).",
    )
    parser.add_argument(
        "--backtest-strategy",
        choices=["rolling", "expanding"],
        default="rolling",
        help="Backtest strategy to use when evaluating models.",
    )
    parser.add_argument(
        "--target",
        help="Focus on a single target when reporting importance or predictions.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ...).",
    )
    parser.add_argument(
        "--ui",
        dest="ui",
        action="store_true",
        help="Display prediction results in a graphical window.",
    )
    parser.add_argument(
        "--no-ui",
        dest="ui",
        action="store_false",
        help="Disable the graphical interface and output JSON to the console.",
    )
    parser.set_defaults(ui=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.log_level)

    use_ui = args.ui if args.ui is not None else args.mode == "predict"

    if args.mode == "predict" and use_ui:
        from stock_predictor.ui import UIOptions, run_app

        options = UIOptions(
            interval=args.interval,
            model_type=args.model_type,
            data_dir=args.data_dir,
            models_dir=args.models_dir,
            news_api_key=args.news_api_key,
            news_limit=args.news_limit,
            sentiment=not args.no_sentiment,
            start_date=args.start_date,
            end_date=args.end_date,
            refresh_data=args.refresh_data,
        )
        run_app(default_ticker=args.ticker, options=options)
        return 0

    def parse_csv(value: str | None) -> list[str] | None:
        if not value:
            return None
        return [item.strip() for item in value.split(",") if item.strip()]

    model_params = None
    if args.model_params:
        try:
            model_params = json.loads(args.model_params)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Failed to parse --model-params: {exc}") from exc

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
        feature_sets=parse_csv(args.feature_sets) or None,
        prediction_targets=parse_csv(args.targets) or None,
        model_params=model_params,
        test_size=args.test_size,
        shuffle_training=args.shuffle_training,
        backtest_strategy=args.backtest_strategy,
        backtest_window=args.backtest_window,
        backtest_step=args.backtest_step,
    )

    ai = StockPredictorAI(config)

    try:
        if args.mode == "download-data":
            result = ai.download_data(force=args.refresh_data)
            payload: dict[str, Any] = {"status": "ok", "downloaded": {k: str(v) for k, v in result.items()}}
            print(json.dumps(payload, indent=2))
        elif args.mode == "train":
            if args.refresh_data:
                ai.download_data(force=True)
            metrics = ai.train_model(targets=parse_csv(args.targets))
            print(json.dumps({"status": "ok", "metrics": metrics}, indent=2))
        elif args.mode == "predict":
            targets = parse_csv(args.targets) if args.targets else ([args.target] if args.target else None)
            prediction = ai.predict(refresh_data=args.refresh_data, targets=targets)
            print(json.dumps({"status": "ok", "prediction": prediction}, indent=2))
        elif args.mode == "backtest":
            results = ai.run_backtest(targets=parse_csv(args.targets))
            print(json.dumps({"status": "ok", "backtest": results}, indent=2))
        elif args.mode == "importance":
            target = args.target or "close"
            importance = ai.feature_importance(target=target)
            print(json.dumps({"status": "ok", "target": target, "importance": importance}, indent=2))
        elif args.mode == "list-models":
            models = ai.list_available_models()
            print(json.dumps({"status": "ok", "models": models}, indent=2))
        else:
            raise ValueError(f"Unsupported mode: {args.mode}")
    except Exception as exc:  # pylint: disable=broad-except
        logging.exception("Pipeline failed")
        print(json.dumps({"status": "error", "message": str(exc)}), file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
