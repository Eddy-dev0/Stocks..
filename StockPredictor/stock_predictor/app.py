"""Top-level application orchestration for the stock predictor platform."""

from __future__ import annotations

import logging
import os
import platform
import signal
from multiprocessing import Process
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable

from stock_predictor.api_app import run_api
from stock_predictor.core import (
    PredictorConfig,
    PredictionResult,
    BuyZoneAnalyzer,
    build_config,
    load_environment,
)
from stock_predictor.data.indicator_store import IndicatorDataStore
from stock_predictor.features.engineer import IndicatorFeatureEngineer
from stock_predictor.models.trainer import HorizonModelTrainer
from stock_predictor.evaluation.backtester import Backtester
from stock_predictor.dashboard_app import run_streamlit_app

PROJECT_ROOT = Path(__file__).resolve().parent.parent

LOGGER = logging.getLogger(__name__)


def _run_api_process(host: str, port: int, env: dict[str, str]) -> None:
    """Entry point used when launching the API server in a separate process."""

    os.environ.update(env)
    run_api(host=host, port=port)


@dataclass(slots=True)
class RunResult:
    """Wrapper used by the application to provide consistent responses."""

    status: str
    payload: dict[str, Any]


class StockPredictorApplication:
    """Coordinate data providers, modelling pipelines, and presentation layers."""

    def __init__(self, config: PredictorConfig) -> None:
        self.config = config
        self.indicator_store = IndicatorDataStore(config)
        self.feature_engineer = IndicatorFeatureEngineer(self.indicator_store)
        self.model_trainer = HorizonModelTrainer(config, self.feature_engineer)
        self.evaluator = Backtester(config, self.model_trainer)
        self.pipeline = self.model_trainer.pipeline

    @classmethod
    def from_environment(cls, **overrides: Any) -> "StockPredictorApplication":
        """Create an application instance using environment variables and overrides."""

        load_environment()
        if "feature_toggles" in overrides and "price_feature_toggles" not in overrides:
            overrides["price_feature_toggles"] = overrides["feature_toggles"]
        overrides.setdefault(
            "ticker", os.getenv("STOCK_PREDICTOR_DEFAULT_TICKER", "AAPL")
        )
        config = build_config(**overrides)
        LOGGER.debug("Initialised configuration for ticker %s", config.ticker)
        return cls(config)

    # ------------------------------------------------------------------
    # High level orchestration helpers
    # ------------------------------------------------------------------
    def refresh_data(self, force: bool = False) -> dict[str, Any]:
        """Refresh cached datasets via the provider layer."""

        LOGGER.info("Refreshing market data (force=%s)", force)
        indicator_df = self.indicator_store.fetch(force=force)
        summary = {"indicator_rows": len(indicator_df.index)}
        try:
            pipeline_summary = self.pipeline.download_data(force=force)
            summary.update(pipeline_summary)
        except Exception as exc:
            LOGGER.debug("Pipeline refresh failed: %s", exc)
        return summary

    def train(self, *, targets: Iterable[str] | None = None, horizon: int | None = None) -> dict[str, Any]:
        """Train configured models and return evaluation metrics."""

        LOGGER.info("Training models for ticker %s", self.config.ticker)
        return self.model_trainer.train(targets=targets, horizon=horizon)

    def predict(
        self,
        *,
        targets: Iterable[str] | None = None,
        refresh: bool = False,
        horizon: int | None = None,
    ) -> PredictionResult:
        """Generate predictions for the configured ticker."""

        LOGGER.info("Generating predictions (refresh=%s)", refresh)
        return self.model_trainer.predict(
            targets=targets, refresh=refresh, horizon=horizon
        )

    def update_ticker(self, ticker: str) -> None:
        """Update the active ticker and rebuild dependent components."""

        new_ticker = ticker.strip().upper()
        if not new_ticker:
            raise ValueError("Ticker symbol cannot be empty.")
        if new_ticker == self.config.ticker:
            return

        updated_config = replace(self.config, ticker=new_ticker)
        updated_config.ensure_directories()
        self.config = updated_config
        self.indicator_store = IndicatorDataStore(updated_config)
        self.feature_engineer = IndicatorFeatureEngineer(self.indicator_store)
        self.model_trainer = HorizonModelTrainer(updated_config, self.feature_engineer)
        self.evaluator = Backtester(updated_config, self.model_trainer)
        self.pipeline = self.model_trainer.pipeline
        LOGGER.info("Switched application context to ticker %s", new_ticker)

    def backtest(self, *, targets: Iterable[str] | None = None) -> dict[str, Any]:
        """Run historical simulations to evaluate the active models."""

        LOGGER.info("Running backtest for ticker %s", self.config.ticker)
        return self.evaluator.run(targets=targets)

    def accuracy(self, *, horizon: int | None = None) -> dict[str, Any]:
        """Summarise stored prediction accuracy for the configured ticker."""

        LOGGER.info("Summarising accuracy for ticker %s", self.config.ticker)
        return self.pipeline.accuracy_summary(horizon=horizon)

    def buy_zone(self, *, refresh: bool = False) -> dict[str, Any]:
        """Compute a tactical buy zone for the configured ticker."""

        LOGGER.info("Computing buy zone for %s (refresh=%s)", self.config.ticker, refresh)
        analyzer = BuyZoneAnalyzer(self.config, fetcher=self.pipeline.fetcher)
        result = analyzer.compute(refresh=refresh)
        return result.as_dict()

    def feature_importance(self, target: str, *, horizon: int | None = None) -> dict[str, Any]:
        """Return feature importance scores for the specified target."""

        LOGGER.info("Computing feature importance for target %s", target)
        return self.pipeline.feature_importance(target=target, horizon=horizon)

    def list_models(self) -> dict[str, Any]:
        """List persisted models available to the application."""

        LOGGER.debug("Listing available models from %s", self.config.models_dir)
        return self.model_trainer.list_models()

    def run(self, mode: str, **kwargs: Any) -> RunResult:
        """Dispatch execution based on the requested mode."""

        handlers = {
            "download-data": lambda: self.refresh_data(force=kwargs.get("force", False)),
            "train": lambda: self.train(targets=kwargs.get("targets"), horizon=kwargs.get("horizon")),
            "predict": lambda: self.predict(
                targets=kwargs.get("targets"),
                refresh=kwargs.get("refresh", False),
                horizon=kwargs.get("horizon"),
            ),
            "backtest": lambda: self.backtest(targets=kwargs.get("targets")),
            "importance": lambda: self.feature_importance(
                kwargs.get("target", "close"),
                horizon=kwargs.get("horizon"),
            ),
            "list-models": self.list_models,
        }

        if mode not in handlers:
            raise ValueError(f"Unsupported application mode: {mode}")

        payload = handlers[mode]()
        return RunResult(status="ok", payload={mode.replace("-", "_"): payload})

    def launch_dashboard(
        self,
        *,
        api_host: str = "127.0.0.1",
        api_port: int = 8000,
        ui_port: int = 8501,
        ui_headless: bool | None = None,
        ui_api_key: str | None = None,
    ) -> int:
        """Launch the dashboard alongside the embedded API service."""

        env = os.environ.copy()
        root_str = str(PROJECT_ROOT)
        pythonpath = env.get("PYTHONPATH")
        if pythonpath:
            paths = pythonpath.split(os.pathsep)
            if root_str not in paths:
                env["PYTHONPATH"] = os.pathsep.join([root_str, pythonpath])
        else:
            env["PYTHONPATH"] = root_str

        env.setdefault("STOCK_PREDICTOR_DEFAULT_TICKER", self.config.ticker)
        env.setdefault("STOCK_PREDICTOR_API_URL", f"http://{api_host}:{api_port}")

        gather_stats_env = env.get("STREAMLIT_BROWSER_GATHER_USAGE_STATS")
        if gather_stats_env is None:
            gather_stats_value = "false"
        else:
            gather_stats_value = gather_stats_env.lower()
        env["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = gather_stats_value
        gather_stats_flag = gather_stats_value == "true"

        if ui_headless is not None:
            headless_value = "true" if ui_headless else "false"
        else:
            headless_env = env.get("STREAMLIT_HEADLESS")
            if headless_env is None:
                headless_value = "true"
            else:
                headless_value = headless_env.lower()
        env["STREAMLIT_HEADLESS"] = headless_value
        headless_flag = headless_value == "true"

        if ui_api_key:
            env["STOCK_PREDICTOR_UI_API_KEY"] = ui_api_key
            env["STOCK_PREDICTOR_UI_API_KEYS"] = ui_api_key

        LOGGER.info(
            "Starting API server on http://%s:%s and dashboard on http://localhost:%s",
            api_host,
            api_port,
            ui_port,
        )

        api_process = Process(
            target=_run_api_process,
            kwargs={
                "host": api_host,
                "port": api_port,
                "env": {key: value for key, value in env.items()},
            },
            daemon=True,
        )
        api_process.start()

        try:
            return run_streamlit_app(
                port=ui_port,
                headless=headless_flag,
                gather_usage_stats=gather_stats_flag,
                env=env,
            )
        except KeyboardInterrupt:
            LOGGER.info("Dashboard interrupted by user.")
            return 0
        finally:
            if api_process.is_alive():
                try:
                    if platform.system() == "Windows":
                        api_process.terminate()
                    else:
                        os.kill(api_process.pid, signal.SIGINT)
                except (OSError, ValueError):  # pragma: no cover - best effort cleanup
                    LOGGER.debug("Failed to gracefully stop API server", exc_info=True)

                api_process.join(timeout=10)
                if api_process.is_alive():
                    LOGGER.debug("Force terminating API server")
                    api_process.kill()


__all__ = ["StockPredictorApplication", "RunResult"]
