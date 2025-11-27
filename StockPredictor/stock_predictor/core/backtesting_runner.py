"""Scheduled backtesting runner for automated calibration checks.

This module refreshes the latest historical dataset, executes backtests
across configured targets, and writes a dated report that can be consumed
by schedulers such as cron or GitHub Actions.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from stock_predictor.core.config import (
    PredictorConfig,
    build_config,
    load_config_from_file,
    load_environment,
)
from stock_predictor.core.modeling import StockPredictorAI

LOGGER = logging.getLogger(__name__)


DEFAULT_TICKER = os.getenv("STOCK_PREDICTOR_TICKER", "SPY")


class BacktestingJob:
    """Orchestrates a full backtesting pass for scheduled execution."""

    def __init__(
        self,
        config: PredictorConfig,
        *,
        targets: Iterable[str] | None = None,
        horizon: int | None = None,
        refresh_data: bool = True,
        output_dir: Path | None = None,
    ) -> None:
        self.config = config
        self.targets = list(targets) if targets is not None else None
        self.horizon = horizon
        self.refresh_data = refresh_data
        self.output_dir = output_dir or (self.config.data_dir / "reports" / "backtests")

    def run(self) -> Path:
        """Execute the backtest and persist a dated JSON report."""

        LOGGER.info(
            "Starting backtest for %s (targets=%s, horizon=%s)",
            self.config.ticker,
            ",".join(self.targets) if self.targets else "default",
            self.horizon if self.horizon is not None else "auto",
        )
        pipeline = StockPredictorAI(self.config, horizon=self.horizon)
        if self.refresh_data:
            LOGGER.info("Refreshing latest historical datasets before backtesting.")
            pipeline.download_data(force=True)

        backtest_results = pipeline.run_backtest(targets=self.targets, horizon=self.horizon)
        resolved_horizon = pipeline.horizon

        timestamp = datetime.now(timezone.utc)
        report = {
            "ticker": self.config.ticker,
            "horizon": resolved_horizon,
            "generated_at": timestamp.isoformat(),
            "targets": backtest_results,
        }
        report_path = self._report_path(timestamp, resolved_horizon)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        LOGGER.info("Backtest report written to %s", report_path)
        return report_path

    def _report_path(self, timestamp: datetime, horizon: int) -> Path:
        date_str = timestamp.strftime("%Y%m%dT%H%M%SZ")
        return self.output_dir / f"{self.config.ticker}_h{horizon}_{date_str}.json"


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------


def _parse_targets(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    tokens = [part.strip().lower() for part in raw.split(",") if part.strip()]
    return tokens or None


def _build_config(args: argparse.Namespace) -> PredictorConfig:
    load_environment()
    if args.config_file:
        return load_config_from_file(args.config_file)
    config = build_config(
        ticker=args.ticker,
        start_date=args.start_date,
        end_date=args.end_date,
        interval=args.interval,
        model_type=args.model_type,
        data_dir=args.data_dir,
        models_dir=args.models_dir,
        sentiment=not args.disable_sentiment,
        backtest_strategy=args.backtest_strategy,
        backtest_window=args.backtest_window,
        backtest_step=args.backtest_step,
    )
    return config


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", default=DEFAULT_TICKER, help="Ticker symbol to backtest (default: %(default)s)")
    parser.add_argument("--targets", help="Comma separated list of targets to backtest (default: config targets)")
    parser.add_argument("--horizon", type=int, help="Prediction horizon to backtest (default: config value)")
    parser.add_argument("--interval", default="1d", help="Price interval for data refresh (default: %(default)s)")
    parser.add_argument("--start-date", dest="start_date", help="Override start date for historical data (YYYY-MM-DD)")
    parser.add_argument("--end-date", dest="end_date", help="Override end date for historical data (YYYY-MM-DD)")
    parser.add_argument("--model-type", default="random_forest", help="Model type to evaluate (default: %(default)s)")
    parser.add_argument("--backtest-strategy", default=None, help="Backtest strategy (e.g. rolling)")
    parser.add_argument("--backtest-window", type=int, default=None, help="Backtest lookback window")
    parser.add_argument("--backtest-step", type=int, default=None, help="Step size between backtest windows")
    parser.add_argument("--data-dir", help="Directory containing cached datasets")
    parser.add_argument("--models-dir", help="Directory containing persisted models")
    parser.add_argument("--config-file", help="Path to JSON/YAML configuration file")
    parser.add_argument("--no-refresh", action="store_true", help="Skip dataset refresh before running the backtest")
    parser.add_argument("--disable-sentiment", action="store_true", help="Disable sentiment features")
    parser.add_argument("--output-dir", help="Directory for saving backtest reports")
    parser.add_argument("--log-level", default=os.getenv("LOG_LEVEL", "INFO"), help="Logging level (default: %(default)s)")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_argument_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    LOGGER.debug("Parsed arguments: %s", args)

    try:
        config = _build_config(args)
    except Exception as exc:  # pragma: no cover - defensive CLI guard
        LOGGER.error("Failed to load configuration: %s", exc)
        return 1

    targets = _parse_targets(args.targets)
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else None
    job = BacktestingJob(
        config,
        targets=targets,
        horizon=args.horizon,
        refresh_data=not args.no_refresh,
        output_dir=output_dir,
    )

    try:
        job.run()
    except Exception as exc:  # pragma: no cover - defensive CLI guard
        LOGGER.exception("Backtesting job failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
