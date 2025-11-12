"""Tkinter-based graphical interface for the stock predictor pipeline."""

from __future__ import annotations

import json
import logging
import math
import queue
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import pandas as pd
import tkinter as tk
from tkinter import messagebox, ttk
import yfinance as yf

from matplotlib import dates as mdates
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from .config import PredictorConfig, build_config
from .elliott import WaveSegment, apply_wave_features
from .etl import NoPriceDataError
from .model import ModelNotFoundError, StockPredictorAI
from .preprocessing import compute_price_features

LOGGER = logging.getLogger(__name__)

APP_VERSION = "StockPredictor v1.0.0"

TIMEFRAME_WINDOWS: dict[str, int] = {
    "1D": 1,
    "1W": 7,
    "1M": 30,
    "3M": 90,
    "1Y": 365,
}

HORIZON_OPTIONS: dict[str, int] = {
    "1D": 1,
    "1W": 5,
    "1M": 21,
    "3M": 63,
}

COLOR_GAIN = "#15803d"
COLOR_LOSS = "#dc2626"
COLOR_NEUTRAL = "#1f2937"

SMA_COLORS = {
    "SMA_20": "#2563eb",
    "SMA_50": "#9333ea",
    "SMA_200": "#ef4444",
}
BOLLINGER_COLOR = "#93c5fd"
MACD_LINE_COLOR = "#0f172a"
MACD_SIGNAL_COLOR = "#22d3ee"
PREDICTION_COLOR = "#f97316"


@dataclass(slots=True)
class UIOptions:
    """Runtime options forwarded from the CLI to the UI layer."""

    interval: str = "1d"
    model_type: str = "random_forest"
    data_dir: Optional[str] = None
    models_dir: Optional[str] = None
    news_api_key: Optional[str] = None
    news_limit: int = 50
    sentiment: bool = True
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    refresh_data: bool = False
    database_url: Optional[str] = None

    def to_kwargs(self) -> Dict[str, Any]:
        return {
            "interval": self.interval,
            "model_type": self.model_type,
            "data_dir": self.data_dir,
            "models_dir": self.models_dir,
            "news_api_key": self.news_api_key,
            "news_limit": self.news_limit,
            "sentiment": self.sentiment,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "database_url": self.database_url,
        }


class UILogHandler(logging.Handler):
    """Forward log messages to the Tkinter UI in a thread-safe manner."""

    def __init__(self, message_queue: "queue.Queue[str]") -> None:
        super().__init__()
        self.queue = message_queue
        self.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))

    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - GUI glue
        try:
            msg = self.format(record)
        except Exception:  # pylint: disable=broad-except
            self.handleError(record)
        else:
            self.queue.put(msg)


class StockPredictorApp(tk.Tk):  # pragma: no cover - UI side effects dominate
    """Desktop application shell presenting prediction results."""

    def __init__(self, default_ticker: str = "AAPL", options: Optional[UIOptions] = None) -> None:
        super().__init__()
        self.title("Stock Predictor")
        self.geometry("1100x750")
        self.minsize(960, 640)

        self.options = options or UIOptions()
        self.default_ticker = default_ticker.upper()

        self.log_queue: "queue.Queue[str]" = queue.Queue()
        self.log_messages: list[str] = []
        self.log_handler = UILogHandler(self.log_queue)
        logging.getLogger().addHandler(self.log_handler)

        self.price_data = pd.DataFrame()
        self.price_data_base = pd.DataFrame()
        self.current_ticker: str | None = None
        self.data_source_var = tk.StringVar(value="Data source: -")
        self.last_updated_var = tk.StringVar(value="Last updated: -")
        self.version_var = tk.StringVar(value=APP_VERSION)
        self.status_var = tk.StringVar(value="Enter a ticker symbol and click Predict")
        self.timeframe_var = tk.StringVar(value="3M")
        self.horizon_var = tk.StringVar(value=self._default_horizon_key())

        self.predicted_close_var = tk.StringVar(value="-")
        self.last_close_var = tk.StringVar(value="-")
        self.absolute_change_var = tk.StringVar(value="-")
        self.percent_change_var = tk.StringVar(value="-")
        self.prediction_header_var = tk.StringVar(value="Prediction details will appear here.")
        self.market_data_time_var = tk.StringVar(value="Market data as of: -")
        self.prediction_time_var = tk.StringVar(value="Prediction generated at: -")
        self.horizon_info_var = tk.StringVar(value="Horizon: -")
        self.target_date_var = tk.StringVar(value="Target date: -")
        self.predicted_return_var = tk.StringVar(value="-")
        self.predicted_volatility_var = tk.StringVar(value="-")
        self.direction_prob_up_var = tk.StringVar(value="-")
        self.direction_prob_down_var = tk.StringVar(value="-")
        self.currency_var = tk.StringVar(value="USD")
        self.currency_button_text = tk.StringVar(value="Display in EUR (€)")
        self.metric_vars: dict[str, tk.StringVar] = {
            "mae": tk.StringVar(value="-"),
            "rmse": tk.StringVar(value="-"),
            "directional_accuracy": tk.StringVar(value="-"),
            "r2": tk.StringVar(value="-"),
            "training_rows": tk.StringVar(value="-"),
            "test_rows": tk.StringVar(value="-"),
        }

        self.explanation_horizon_var = tk.StringVar(value="Horizon: -")
        self.explanation_target_var = tk.StringVar(value="Target date: -")
        self.explanation_return_var = tk.StringVar(value="Projected return: -")
        self.explanation_volatility_var = tk.StringVar(value="Projected volatility: -")
        self.explanation_prob_up_var = tk.StringVar(value="Upside probability: -")
        self.explanation_prob_down_var = tk.StringVar(value="Downside probability: -")

        self.refresh_var = tk.BooleanVar(value=self.options.refresh_data)
        self.ticker_var = tk.StringVar(value=self.default_ticker)
        self.usd_to_eur_rate: float | None = None
        self.latest_prediction: Optional[Dict[str, Any]] = None
        self.latest_metrics: Optional[Dict[str, Any]] = None
        self.wave_annotations: list[WaveSegment] = []
        self.wave_annotations_base: list[WaveSegment] = []

        self.explanation_summary_var = tk.StringVar(value="Run a prediction to see explanation details.")
        self.explanation_reason_vars: dict[str, tk.StringVar] = {}
        self.feature_importance_tree: ttk.Treeview | None = None
        self.sources_text: tk.Text | None = None

        self._interactive_widgets: list[tk.Widget] = []
        self._horizon_controls: dict[str, ttk.Radiobutton] = {}
        self._available_horizons: tuple[int, ...] = tuple(HORIZON_OPTIONS.values())
        self._inputs_disabled = False

        self._build_ui()
        self._after_id = self.after(150, self._poll_log_queue)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------------
    # UI assembly helpers
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        top_bar = ttk.Frame(self, padding=(20, 15))
        top_bar.pack(fill="x")

        ttk.Label(top_bar, text="Ticker", font=("Helvetica", 12, "bold")).pack(side="left")
        ticker_entry = ttk.Entry(top_bar, textvariable=self.ticker_var, width=12, font=("Helvetica", 12))
        ticker_entry.pack(side="left", padx=(10, 15))
        ticker_entry.bind("<Return>", lambda _event: self.on_predict())
        self._interactive_widgets.append(ticker_entry)

        predict_button = ttk.Button(top_bar, text="Predict", command=self.on_predict)
        predict_button.pack(side="left")
        self._interactive_widgets.append(predict_button)

        refresh_check = ttk.Checkbutton(top_bar, text="Refresh data", variable=self.refresh_var)
        refresh_check.pack(side="left", padx=(15, 0))
        self._interactive_widgets.append(refresh_check)

        currency_button = ttk.Button(
            top_bar,
            textvariable=self.currency_button_text,
            command=self._on_currency_toggle,
        )
        currency_button.pack(side="left", padx=(15, 0))
        self._interactive_widgets.append(currency_button)
        self.currency_button = currency_button

        timeframe_frame = ttk.Frame(top_bar)
        timeframe_frame.pack(side="left", padx=(25, 0))
        ttk.Label(timeframe_frame, text="Timeframe:", font=("Helvetica", 11, "bold")).pack(side="left")
        for key in TIMEFRAME_WINDOWS:
            button = ttk.Radiobutton(
                timeframe_frame,
                text=key,
                value=key,
                variable=self.timeframe_var,
                command=self._on_timeframe_change,
            )
            button.pack(side="left", padx=4)
            self._interactive_widgets.append(button)

        horizon_frame = ttk.Frame(top_bar)
        horizon_frame.pack(side="left", padx=(25, 0))
        ttk.Label(
            horizon_frame,
            text="Forecast horizon:",
            font=("Helvetica", 11, "bold"),
        ).pack(side="left")
        for label in HORIZON_OPTIONS:
            button = ttk.Radiobutton(
                horizon_frame,
                text=label,
                value=label,
                variable=self.horizon_var,
            )
            button.pack(side="left", padx=4)
            self._interactive_widgets.append(button)
            self._horizon_controls[label] = button

        status_label = ttk.Label(
            top_bar,
            textvariable=self.status_var,
            font=("Helvetica", 11),
            foreground=COLOR_NEUTRAL,
        )
        status_label.pack(side="left", padx=(25, 0))
        self.status_label = status_label

        content = ttk.Frame(self, padding=(15, 5))
        content.pack(fill="both", expand=True)

        notebook = ttk.Notebook(content)
        notebook.pack(fill="both", expand=True)

        # Chart tab
        chart_tab = ttk.Frame(notebook, padding=10)
        notebook.add(chart_tab, text="Chart")

        self.figure = Figure(figsize=(6, 4), dpi=100)
        grid = self.figure.add_gridspec(4, 1, height_ratios=[3, 1, 1, 1], hspace=0.06)
        self.price_ax = self.figure.add_subplot(grid[0])
        self.rsi_ax = self.figure.add_subplot(grid[1], sharex=self.price_ax)
        self.macd_ax = self.figure.add_subplot(grid[2], sharex=self.price_ax)
        self.volume_ax = self.figure.add_subplot(grid[3], sharex=self.price_ax)
        self.volume_obv_ax = self.volume_ax.twinx()
        self.chart_ax = self.price_ax

        for axis in (self.price_ax, self.rsi_ax, self.macd_ax, self.volume_ax):
            axis.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

        self.price_ax.set_title("Historical Prices")
        self.price_ax.set_ylabel(f"Price ({self._get_currency_symbol()})")
        self.rsi_ax.set_ylabel("RSI")
        self.macd_ax.set_ylabel("MACD")
        self.volume_ax.set_ylabel("Volume")
        self.volume_obv_ax.set_ylabel("OBV")
        self.volume_obv_ax.grid(False)

        self.canvas = FigureCanvasTkAgg(self.figure, master=chart_tab)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # Prediction tab
        prediction_tab = ttk.Frame(notebook, padding=20)
        notebook.add(prediction_tab, text="Prediction")

        header = ttk.Label(
            prediction_tab,
            textvariable=self.prediction_header_var,
            font=("Helvetica", 16, "bold"),
        )
        header.grid(column=0, row=0, columnspan=2, sticky="w")

        ttk.Label(
            prediction_tab,
            textvariable=self.market_data_time_var,
            font=("Helvetica", 11),
        ).grid(column=0, row=1, columnspan=2, sticky="w", pady=(6, 0))

        ttk.Label(
            prediction_tab,
            textvariable=self.prediction_time_var,
            font=("Helvetica", 11),
        ).grid(column=0, row=2, columnspan=2, sticky="w", pady=(0, 12))

        ttk.Label(
            prediction_tab,
            text="Predicted close (forecast for next close):",
            font=("Helvetica", 13, "bold"),
        ).grid(column=0, row=3, sticky="w")
        self.predicted_close_label = ttk.Label(
            prediction_tab,
            textvariable=self.predicted_close_var,
            font=("Helvetica", 24, "bold"),
        )
        self.predicted_close_label.grid(column=0, row=4, sticky="w")

        ttk.Label(
            prediction_tab,
            text="Last close (actual, previous trading day):",
            font=("Helvetica", 12, "bold"),
        ).grid(column=0, row=5, sticky="w", pady=(18, 4))
        ttk.Label(
            prediction_tab,
            textvariable=self.last_close_var,
            font=("Helvetica", 14),
        ).grid(column=0, row=6, sticky="w")

        ttk.Label(
            prediction_tab,
            text="Expected change (predicted − last close):",
            font=("Helvetica", 12, "bold"),
        ).grid(column=1, row=3, sticky="w", padx=(40, 0))
        self.absolute_change_label = ttk.Label(
            prediction_tab,
            textvariable=self.absolute_change_var,
            font=("Helvetica", 16, "bold"),
        )
        self.absolute_change_label.grid(column=1, row=4, sticky="w", padx=(40, 0))

        ttk.Label(
            prediction_tab,
            text="% change (relative to last close):",
            font=("Helvetica", 12, "bold"),
        ).grid(column=1, row=5, sticky="w", padx=(40, 0), pady=(18, 4))
        self.percent_change_label = ttk.Label(
            prediction_tab,
            textvariable=self.percent_change_var,
            font=("Helvetica", 16, "bold"),
        )
        self.percent_change_label.grid(column=1, row=6, sticky="w", padx=(40, 0))

        ttk.Label(
            prediction_tab,
            textvariable=self.horizon_info_var,
            font=("Helvetica", 11, "bold"),
        ).grid(column=0, row=7, sticky="w", pady=(12, 2))
        ttk.Label(
            prediction_tab,
            textvariable=self.target_date_var,
            font=("Helvetica", 11),
        ).grid(column=0, row=8, sticky="w", pady=(0, 10))

        ttk.Label(
            prediction_tab,
            text="Predicted return over horizon:",
            font=("Helvetica", 12, "bold"),
        ).grid(column=0, row=9, sticky="w", pady=(0, 4))
        ttk.Label(
            prediction_tab,
            textvariable=self.predicted_return_var,
            font=("Helvetica", 14),
        ).grid(column=1, row=9, sticky="w", padx=(8, 0))

        ttk.Label(
            prediction_tab,
            text="Predicted volatility over horizon:",
            font=("Helvetica", 12, "bold"),
        ).grid(column=0, row=10, sticky="w", pady=(0, 4))
        ttk.Label(
            prediction_tab,
            textvariable=self.predicted_volatility_var,
            font=("Helvetica", 14),
        ).grid(column=1, row=10, sticky="w", padx=(8, 0))

        ttk.Label(
            prediction_tab,
            text="Upside probability:",
            font=("Helvetica", 12, "bold"),
        ).grid(column=0, row=11, sticky="w", pady=(0, 4))
        ttk.Label(
            prediction_tab,
            textvariable=self.direction_prob_up_var,
            font=("Helvetica", 14),
        ).grid(column=1, row=11, sticky="w", padx=(8, 0))

        ttk.Label(
            prediction_tab,
            text="Downside probability:",
            font=("Helvetica", 12, "bold"),
        ).grid(column=0, row=12, sticky="w", pady=(0, 8))
        ttk.Label(
            prediction_tab,
            textvariable=self.direction_prob_down_var,
            font=("Helvetica", 14),
        ).grid(column=1, row=12, sticky="w", padx=(8, 0))

        ttk.Separator(prediction_tab, orient="horizontal").grid(
            column=0, row=13, columnspan=2, sticky="ew", pady=(20, 15)
        )

        ttk.Label(
            prediction_tab,
            text="Training Metrics",
            font=("Helvetica", 13, "bold"),
        ).grid(column=0, row=14, columnspan=2, sticky="w", pady=(0, 10))

        metrics_grid = ttk.Frame(prediction_tab)
        metrics_grid.grid(column=0, row=15, columnspan=2, sticky="w")

        metric_labels = [
            ("MAE", "mae"),
            ("RMSE", "rmse"),
            ("Directional Accuracy", "directional_accuracy"),
            ("R²", "r2"),
            ("Training Rows", "training_rows"),
            ("Test Rows", "test_rows"),
        ]
        for row_index, (label, key) in enumerate(metric_labels):
            ttk.Label(metrics_grid, text=label + ":", font=("Helvetica", 11, "bold")).grid(
                column=0, row=row_index, sticky="w", padx=(0, 12), pady=4
            )
            ttk.Label(metrics_grid, textvariable=self.metric_vars[key], font=("Helvetica", 11)).grid(
                column=1, row=row_index, sticky="w", pady=4
            )

        prediction_tab.columnconfigure(0, weight=1)
        prediction_tab.columnconfigure(1, weight=1)

        # Explanation tab
        explanation_tab = ttk.Frame(notebook, padding=20)
        notebook.add(explanation_tab, text="Explanation")

        meta_frame = ttk.Frame(explanation_tab)
        meta_frame.grid(column=0, row=0, columnspan=2, sticky="ew", pady=(0, 10))
        meta_frame.columnconfigure(0, weight=1)
        meta_frame.columnconfigure(1, weight=1)

        meta_labels = [
            (self.explanation_horizon_var, 0, 0),
            (self.explanation_target_var, 0, 1),
            (self.explanation_return_var, 1, 0),
            (self.explanation_volatility_var, 1, 1),
            (self.explanation_prob_up_var, 2, 0),
            (self.explanation_prob_down_var, 2, 1),
        ]
        for var, row_index, column_index in meta_labels:
            ttk.Label(
                meta_frame,
                textvariable=var,
                font=("Helvetica", 11),
                justify="left",
            ).grid(column=column_index, row=row_index, sticky="w", padx=(0, 12), pady=2)

        summary_label = ttk.Label(
            explanation_tab,
            textvariable=self.explanation_summary_var,
            font=("Helvetica", 13, "bold"),
            justify="left",
            wraplength=780,
        )
        summary_label.grid(column=0, row=1, columnspan=2, sticky="w", pady=(0, 10))

        sections = [
            ("Technical signals", "technical_reasons"),
            ("Fundamental context", "fundamental_reasons"),
            ("Sentiment", "sentiment_reasons"),
            ("Macro backdrop", "macro_reasons"),
        ]
        for index, (title, key) in enumerate(sections, start=1):
            frame = ttk.LabelFrame(explanation_tab, text=title)
            frame.grid(column=0, row=index + 1, columnspan=2, sticky="ew", pady=6)
            frame.columnconfigure(0, weight=1)
            var = tk.StringVar(value="No signals available yet.")
            self.explanation_reason_vars[key] = var
            ttk.Label(
                frame,
                textvariable=var,
                justify="left",
                wraplength=760,
            ).grid(column=0, row=0, sticky="w", padx=6, pady=4)

        importance_frame = ttk.LabelFrame(explanation_tab, text="Top features")
        importance_frame.grid(column=0, row=len(sections) + 2, sticky="nsew", pady=(10, 6))
        columns = ("feature", "importance", "category")
        tree = ttk.Treeview(
            importance_frame,
            columns=columns,
            show="headings",
            height=8,
        )
        tree.heading("feature", text="Feature")
        tree.heading("importance", text="Importance")
        tree.heading("category", text="Category")
        tree.column("feature", width=260, anchor="w")
        tree.column("importance", width=120, anchor="center")
        tree.column("category", width=120, anchor="center")
        tree.grid(column=0, row=0, sticky="nsew")
        importance_frame.columnconfigure(0, weight=1)
        importance_frame.rowconfigure(0, weight=1)
        self.feature_importance_tree = tree

        sources_frame = ttk.LabelFrame(explanation_tab, text="Data sources")
        sources_frame.grid(column=0, row=len(sections) + 3, sticky="nsew", pady=(10, 0))
        sources_frame.columnconfigure(0, weight=1)
        text_widget = tk.Text(
            sources_frame,
            height=6,
            wrap="word",
            state="disabled",
            font=("Helvetica", 11),
        )
        text_widget.grid(column=0, row=0, sticky="nsew")
        self.sources_text = text_widget

        explanation_tab.columnconfigure(0, weight=1)

        # Logs tab
        logs_tab = ttk.Frame(notebook, padding=10)
        notebook.add(logs_tab, text="Details & Logs")

        self.log_text = tk.Text(logs_tab, height=20, wrap="word", state="disabled", font=("Courier New", 11))
        self.log_text.pack(fill="both", expand=True)

        footer = ttk.Frame(self, padding=(20, 10))
        footer.pack(fill="x")

        ttk.Label(footer, textvariable=self.version_var, font=("Helvetica", 10)).pack(side="left")
        ttk.Label(footer, textvariable=self.data_source_var, font=("Helvetica", 10)).pack(
            side="left", padx=(20, 0)
        )
        ttk.Label(footer, textvariable=self.last_updated_var, font=("Helvetica", 10)).pack(
            side="left", padx=(20, 0)
        )

        self._clear_explanation_tab()
        self._refresh_horizon_states(self._available_horizons)

    # ------------------------------------------------------------------
    # UI event handlers
    # ------------------------------------------------------------------
    def on_predict(self) -> None:
        ticker = self.ticker_var.get().strip().upper()
        if not ticker:
            self._set_status("Please provide a ticker symbol.", error=True)
            return

        self._set_status(f"Running prediction for {ticker}…")
        self._toggle_inputs(state=tk.DISABLED)
        thread = threading.Thread(target=self._run_prediction, args=(ticker,), daemon=True)
        thread.start()

    def _on_timeframe_change(self) -> None:
        if not self.price_data.empty:
            self._plot_price_data()

    def _toggle_inputs(self, state: str) -> None:
        for widget in self._interactive_widgets:
            try:
                if hasattr(widget, "state"):
                    widget.state(["disabled" if state == tk.DISABLED else "!disabled"])
                else:
                    widget.configure(state=state)
            except tk.TclError:
                continue
        self._inputs_disabled = state == tk.DISABLED

    # ------------------------------------------------------------------
    # Prediction pipeline glue
    # ------------------------------------------------------------------
    def _run_prediction(self, ticker: str) -> None:
        refresh = self.refresh_var.get()
        try:
            selected_horizon_label = self.horizon_var.get()
            selected_horizon_value = self._horizon_value_from_label(selected_horizon_label)
            kwargs = {k: v for k, v in self.options.to_kwargs().items() if v is not None}
            if "start_date" not in kwargs:
                days = max(TIMEFRAME_WINDOWS.values())
                kwargs["start_date"] = (datetime.today() - timedelta(days=days)).date().isoformat()
            kwargs["ticker"] = ticker
            config = build_config(**kwargs)
            self._available_horizons = tuple(int(h) for h in config.prediction_horizons)
            self.after(
                0,
                lambda horizons=self._available_horizons: self._refresh_horizon_states(horizons),
            )

            try:
                resolved_horizon = config.resolve_horizon(selected_horizon_value)
            except ValueError:
                resolved_horizon = config.default_horizon

            resolved_label = self._label_for_horizon(resolved_horizon)
            if resolved_label:
                if resolved_label != self.horizon_var.get():
                    self.after(0, lambda label=resolved_label: self.horizon_var.set(label))

            LOGGER.info("Starting prediction workflow for %s", ticker)

            ai = StockPredictorAI(config, horizon=resolved_horizon)
            prediction = ai.predict(refresh_data=refresh)
            source_before_fetch = ai.fetcher.last_price_source
            prices = ai.fetcher.fetch_price_data(force=False)

            data_source = (
                "Remote download" if source_before_fetch == "remote" else "Database cache"
            )

            metrics = prediction.get("training_metrics")
            if isinstance(metrics, dict) and "mae" not in metrics:
                metrics = metrics.get("close") or next(iter(metrics.values()), {})
            if not metrics:
                metrics = self._load_metrics(config)

            prices = self._prepare_prices(prices)
            LOGGER.info(
                "Prediction complete for %s - close %.2f (change %.2f / %.2f%%)",
                ticker,
                prediction.get("predicted_close", float("nan")),
                prediction.get("expected_change", float("nan")),
                (prediction.get("expected_change_pct", 0.0) or 0.0) * 100,
            )
            self.after(
                0,
                lambda: self._on_prediction_success(
                    ticker=ticker,
                    prices=prices,
                    prediction=prediction,
                    metrics=metrics,
                    data_source=data_source,
                    last_updated=ai.fetcher.get_last_updated("prices"),
                ),
            )
        except NoPriceDataError as exc:
            LOGGER.error("Prediction failed for %s - no price data", ticker, exc_info=True)
            self.after(0, lambda msg=str(exc): self._show_error_message(msg))
        except Exception as exc:  # pylint: disable=broad-except
            LOGGER.exception("Prediction failed for %s", ticker)
            if isinstance(exc, ModelNotFoundError):
                message = (
                    "Prediction failed: could not load saved model for target "
                    f"'{exc.target}' (horizon {exc.horizon}). Please retrain the model or verify the file at {exc.path}."
                )
            else:
                message = f"Prediction failed: {exc}"
            self.after(0, lambda msg=message: self._on_prediction_error(msg))

    def _prepare_prices(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "Date" not in df.columns:
            date_col = next((col for col in df.columns if col.lower() == "date"), None)
            if date_col:
                df = df.rename(columns={date_col: "Date"})
            else:
                fuzzy_date_col = next((col for col in df.columns if "date" in col.lower()), None)
                if fuzzy_date_col:
                    df = df.rename(columns={fuzzy_date_col: "Date"})
        if "Date" not in df.columns and isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index().rename(columns={"index": "Date"})
        if "Date" not in df.columns:
            raise RuntimeError("Price data is missing a date column.")
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"])
        rename_map = {col: col.title() for col in df.columns}
        df = df.rename(columns=rename_map)
        expected_cols = {"Open", "High", "Low", "Close"}
        if not expected_cols.issubset(df.columns):
            raise RuntimeError("Price data is missing OHLC columns.")
        df = df.sort_values("Date")
        return compute_price_features(df)

    def _load_metrics(self, config: PredictorConfig) -> Optional[Dict[str, Any]]:
        path: Path = config.metrics_path
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, ValueError) as exc:
            LOGGER.warning("Unable to load metrics from %s: %s", path, exc)
            return None
        return data

    def _on_prediction_success(
        self,
        *,
        ticker: str,
        prices: pd.DataFrame,
        prediction: Dict[str, Any],
        metrics: Optional[Dict[str, Any]],
        data_source: str,
        last_updated: Optional[datetime],
    ) -> None:
        self.current_ticker = ticker
        self.price_data_base = prices
        self.latest_prediction = prediction
        self.latest_metrics = metrics
        self._refresh_currency_views()
        self.data_source_var.set(f"Data source: {data_source}")
        timestamp = last_updated or datetime.now()
        self.last_updated_var.set(
            f"Last updated: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
        )
        self._set_status(f"Prediction ready for {ticker}", error=False)
        self._toggle_inputs(state=tk.NORMAL)
        self._update_explanation_tab(prediction.get("explanation"), prediction)
        self._refresh_horizon_states(self._available_horizons)

    def _on_prediction_error(self, message: str) -> None:
        self._set_status(message, error=True)
        messagebox.showerror("Prediction failed", message, parent=self)
        self._toggle_inputs(state=tk.NORMAL)
        self._refresh_horizon_states(self._available_horizons)

    # ------------------------------------------------------------------
    # Rendering helpers
    # ------------------------------------------------------------------
    def _plot_price_data(self) -> None:
        df = self.price_data
        axes = [
            getattr(self, "price_ax", self.chart_ax),
            getattr(self, "rsi_ax", None),
            getattr(self, "macd_ax", None),
            getattr(self, "volume_ax", None),
            getattr(self, "volume_obv_ax", None),
        ]
        for axis in filter(None, axes):
            axis.clear()

        if df.empty:
            self.price_ax.set_ylabel(f"Price ({self._get_currency_symbol()})")
            self.price_ax.text(0.5, 0.5, "No price data", ha="center", va="center")
            if hasattr(self, "rsi_ax"):
                self.rsi_ax.set_ylabel("RSI")
                self.rsi_ax.set_ylim(0, 100)
            if hasattr(self, "macd_ax"):
                self.macd_ax.set_ylabel("MACD")
            if hasattr(self, "volume_ax"):
                self.volume_ax.set_ylabel("Volume")
            if hasattr(self, "volume_obv_ax"):
                self.volume_obv_ax.set_ylabel("OBV")
            self.canvas.draw_idle()
            return

        window_days = TIMEFRAME_WINDOWS.get(self.timeframe_var.get(), 90)
        end_date = df["Date"].max()
        start_date = end_date - timedelta(days=window_days)
        window_df = df[df["Date"] >= start_date]
        if window_df.empty:
            window_df = df

        for axis in (self.price_ax, self.rsi_ax, self.macd_ax, self.volume_ax):
            axis.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

        self.price_ax.set_title(f"{self.current_ticker or ''} Price History")
        self.price_ax.set_ylabel(f"Price ({self._get_currency_symbol()})")
        self.price_ax.tick_params(labelbottom=False)
        self.rsi_ax.set_ylabel("RSI")
        self.rsi_ax.set_ylim(0, 100)
        self.rsi_ax.axhline(70, color=COLOR_LOSS, linestyle="--", linewidth=0.8, alpha=0.7)
        self.rsi_ax.axhline(30, color=COLOR_GAIN, linestyle="--", linewidth=0.8, alpha=0.7)
        self.rsi_ax.tick_params(labelbottom=False)
        self.macd_ax.set_ylabel("MACD")
        self.macd_ax.axhline(0, color=COLOR_NEUTRAL, linestyle="--", linewidth=0.8)
        self.macd_ax.tick_params(labelbottom=False)
        self.volume_ax.set_ylabel("Volume")
        self.volume_obv_ax.set_ylabel("OBV")
        self.volume_obv_ax.grid(False)

        dates = mdates.date2num(window_df["Date"].to_list())
        width = 0.6
        for x, (_, row) in zip(dates, window_df.iterrows()):
            open_price = float(row["Open"])
            close_price = float(row["Close"])
            high_price = float(row["High"])
            low_price = float(row["Low"])
            color = COLOR_GAIN if close_price >= open_price else COLOR_LOSS
            self.price_ax.plot([x, x], [low_price, high_price], color=color, linewidth=1.2)
            body_bottom = min(open_price, close_price)
            body_height = max(abs(close_price - open_price), 0.01)
            rect = Rectangle(
                (x - width / 2, body_bottom),
                width,
                body_height,
                facecolor=color,
                edgecolor=color,
                alpha=0.8,
            )
            self.price_ax.add_patch(rect)

        for name, color in SMA_COLORS.items():
            if name in window_df.columns:
                self.price_ax.plot(dates, window_df[name], color=color, linewidth=1.2, label=name.replace("_", " "))

        if {"BB_Upper_20", "BB_Lower_20"}.issubset(window_df.columns):
            upper = window_df["BB_Upper_20"].to_numpy()
            lower = window_df["BB_Lower_20"].to_numpy()
            self.price_ax.fill_between(
                dates,
                lower,
                upper,
                color=BOLLINGER_COLOR,
                alpha=0.15,
                label="Bollinger Bands",
            )

        last_close = float(window_df["Close"].iloc[-1])
        self.price_ax.scatter(dates[-1], last_close, color=COLOR_NEUTRAL, s=35, zorder=5, label="Last Close")

        predicted_value = None
        target_date_raw: Any = None
        horizon_days: Optional[int] = None
        if self.latest_prediction:
            predicted_value = self.latest_prediction.get("predicted_close")
            target_date_raw = self.latest_prediction.get("target_date")
            try:
                horizon_days = (
                    int(self.latest_prediction.get("horizon"))
                    if self.latest_prediction.get("horizon") is not None
                    else None
                )
            except (TypeError, ValueError):
                horizon_days = None
        if predicted_value is not None:
            try:
                predicted_value = float(predicted_value)
            except (TypeError, ValueError):
                predicted_value = None
        predicted_timestamp: Optional[datetime] = None
        if target_date_raw:
            try:
                parsed = pd.to_datetime(target_date_raw)
            except Exception:  # pragma: no cover - parsing guard
                parsed = None
            if isinstance(parsed, pd.Timestamp):
                predicted_timestamp = parsed.to_pydatetime()
            elif isinstance(parsed, datetime):
                predicted_timestamp = parsed
        if predicted_value is not None and not math.isnan(predicted_value):
            if predicted_timestamp is None:
                offset_days = horizon_days if horizon_days and horizon_days > 0 else 1
                predicted_timestamp = window_df["Date"].max() + timedelta(days=offset_days)
            predicted_date = predicted_timestamp
            predicted_x = mdates.date2num(predicted_date)
            converted_prediction = self._convert_currency_value(predicted_value)
            y_value = converted_prediction if converted_prediction is not None else predicted_value
            self.price_ax.scatter(
                predicted_x,
                y_value,
                color=PREDICTION_COLOR,
                marker="*",
                s=140,
                zorder=6,
                label="Predicted Close",
            )
            self.price_ax.annotate(
                self._format_currency(predicted_value),
                (predicted_x, y_value),
                xytext=(8, 8),
                textcoords="offset points",
                color=PREDICTION_COLOR,
                fontsize=8,
            )
            self.price_ax.set_xlim(dates[0], predicted_x + 1)

        if "RSI_14" in window_df.columns:
            rsi_values = window_df["RSI_14"].clip(0, 100)
            self.rsi_ax.plot(dates, rsi_values, color="#6366f1", linewidth=1.2)

        if "MACD" in window_df.columns:
            self.macd_ax.plot(dates, window_df["MACD"], color=MACD_LINE_COLOR, linewidth=1.2, label="MACD")
        if "MACD_Signal" in window_df.columns:
            self.macd_ax.plot(
                dates,
                window_df["MACD_Signal"],
                color=MACD_SIGNAL_COLOR,
                linewidth=1.1,
                label="Signal",
            )
        if "MACD_Hist" in window_df.columns:
            hist_series = window_df["MACD_Hist"].fillna(0.0)
            colors = [COLOR_GAIN if value >= 0 else COLOR_LOSS for value in hist_series]
            self.macd_ax.bar(dates, hist_series, color=colors, width=0.6, alpha=0.55, label="Histogram")

        if "Volume" in window_df.columns:
            volume_colors = [
                COLOR_GAIN if close >= open_ else COLOR_LOSS
                for close, open_ in zip(window_df["Close"], window_df["Open"])
            ]
            self.volume_ax.bar(dates, window_df["Volume"], color=volume_colors, alpha=0.35, label="Volume")
        if "Volume_SMA_20" in window_df.columns:
            self.volume_ax.plot(dates, window_df["Volume_SMA_20"], color="#1d4ed8", linewidth=1.1, label="Vol SMA20")
        if "Volume_EMA_20" in window_df.columns:
            self.volume_ax.plot(dates, window_df["Volume_EMA_20"], color="#0ea5e9", linewidth=1.1, linestyle="--", label="Vol EMA20")
        if "OBV" in window_df.columns:
            self.volume_obv_ax.plot(dates, window_df["OBV"], color="#2563eb", linewidth=1.0, label="OBV")

        price_handles, price_labels = self.price_ax.get_legend_handles_labels()
        if price_handles:
            self.price_ax.legend(price_handles, price_labels, loc="upper left", fontsize=8)

        macd_handles, macd_labels = self.macd_ax.get_legend_handles_labels()
        if macd_handles:
            self.macd_ax.legend(macd_handles, macd_labels, loc="upper left", fontsize=8)

        volume_handles, volume_labels = self.volume_ax.get_legend_handles_labels()
        if volume_handles:
            self.volume_ax.legend(volume_handles, volume_labels, loc="upper left", fontsize=8)

        if hasattr(self.volume_obv_ax, "get_legend_handles_labels"):
            obv_handles, obv_labels = self.volume_obv_ax.get_legend_handles_labels()
            if obv_handles:
                self.volume_obv_ax.legend(obv_handles, obv_labels, loc="upper right", fontsize=8)

        self.volume_ax.xaxis_date()
        self.volume_ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        self.figure.autofmt_xdate()

        if self.wave_annotations:
            start_timestamp = window_df["Date"].min()
            end_timestamp = window_df["Date"].max()
            for wave in self.wave_annotations:
                if pd.isna(wave.start_date) or pd.isna(wave.end_date):
                    continue
                if wave.end_date < start_timestamp or wave.start_date > end_timestamp:
                    continue
                x1 = mdates.date2num(wave.start_date.to_pydatetime())
                x2 = mdates.date2num(wave.end_date.to_pydatetime())
                color = "#2563eb" if wave.direction >= 0 else "#f97316"
                self.chart_ax.plot(
                    [x1, x2],
                    [wave.start_price, wave.end_price],
                    color=color,
                    linewidth=1.6,
                    alpha=0.85,
                )
                label_y = wave.end_price * (1.002 if wave.direction >= 0 else 0.998)
                self.chart_ax.text(
                    x2,
                    label_y,
                    wave.label,
                    color=color,
                    fontsize=9,
                    fontweight="bold",
                    ha="left",
                    va="bottom" if wave.direction >= 0 else "top",
                )

        self.canvas.draw_idle()

    def _update_prediction_panel(self, prediction: Dict[str, Any], metrics: Optional[Dict[str, Any]]) -> None:
        self.predicted_close_var.set(self._format_currency(prediction.get("predicted_close")))
        self.last_close_var.set(self._format_currency(prediction.get("last_close")))

        change = prediction.get("expected_change")
        change_pct = prediction.get("expected_change_pct")
        color = COLOR_NEUTRAL
        if change is not None:
            if change > 0:
                color = COLOR_GAIN
            elif change < 0:
                color = COLOR_LOSS
        self.absolute_change_var.set(self._format_currency(change))
        self.percent_change_var.set(self._format_percent(change_pct))
        self.absolute_change_label.configure(foreground=color)
        self.percent_change_label.configure(foreground=color)

        ticker = prediction.get("ticker") or (self.current_ticker or "-")
        horizon_raw = prediction.get("horizon")
        try:
            horizon_value = int(horizon_raw) if horizon_raw is not None else None
        except (TypeError, ValueError):
            horizon_value = None
        if horizon_value and horizon_value > 0:
            horizon_display = self._format_horizon_display(horizon_value)
            self.horizon_info_var.set(f"Horizon: {horizon_display}")
            horizon_label = self._label_for_horizon(horizon_value)
            suffix = f" ({horizon_label})" if horizon_label else f" (H{horizon_value})"
        else:
            suffix = ""
            self.horizon_info_var.set("Horizon: -")
        self.prediction_header_var.set(f"Prediction for {ticker}{suffix}")

        target_date = prediction.get("target_date")
        target_display = self._format_timestamp(target_date)
        if target_display != "-" and " " in target_display:
            target_display = target_display.split(" ")[0]
        self.target_date_var.set(
            f"Target date: {target_display if target_display != '-' else '-'}"
        )

        self.predicted_return_var.set(self._format_percent(prediction.get("predicted_return")))
        self.predicted_volatility_var.set(
            self._format_percent(prediction.get("predicted_volatility"))
        )
        self.direction_prob_up_var.set(
            self._format_percent(prediction.get("direction_probability_up"))
        )
        self.direction_prob_down_var.set(
            self._format_percent(prediction.get("direction_probability_down"))
        )

        market_timestamp = prediction.get("market_data_as_of") or prediction.get("as_of")
        generated_timestamp = prediction.get("generated_at")
        self.market_data_time_var.set(
            f"Market data as of: {self._format_timestamp(market_timestamp)}"
        )
        self.prediction_time_var.set(
            f"Prediction generated at: {self._format_timestamp(generated_timestamp)}"
        )

        metrics = metrics or {}
        self.metric_vars["mae"].set(self._format_number(metrics.get("mae")))
        self.metric_vars["rmse"].set(self._format_number(metrics.get("rmse")))
        self.metric_vars["directional_accuracy"].set(
            self._format_percent(metrics.get("directional_accuracy"))
        )
        self.metric_vars["r2"].set(self._format_number(metrics.get("r2")))
        self.metric_vars["training_rows"].set(self._format_int(metrics.get("training_rows")))
        self.metric_vars["test_rows"].set(self._format_int(metrics.get("test_rows")))

    def _update_explanation_tab(
        self,
        explanation: Optional[Dict[str, Any]],
        prediction: Optional[Dict[str, Any]] = None,
    ) -> None:
        if explanation is None:
            self._clear_explanation_tab("No detailed explanation available for this prediction.")
            return

        summary = explanation.get("summary") or "No summary available."
        self.explanation_summary_var.set(summary)

        context_prediction = prediction or self.latest_prediction or {}
        horizon_value = explanation.get("horizon")
        if horizon_value is None:
            horizon_value = context_prediction.get("horizon")
        horizon_display = self._format_horizon_display(horizon_value)
        self.explanation_horizon_var.set(f"Horizon: {horizon_display}")

        target_date = context_prediction.get("target_date") or explanation.get("target_date")
        target_display = self._format_timestamp(target_date)
        if target_display != "-" and " " in target_display:
            target_display = target_display.split(" ")[0]
        self.explanation_target_var.set(f"Target date: {target_display}")

        self.explanation_return_var.set(
            f"Projected return: {self._format_percent(context_prediction.get('predicted_return'))}"
        )
        self.explanation_volatility_var.set(
            f"Projected volatility: {self._format_percent(context_prediction.get('predicted_volatility'))}"
        )
        self.explanation_prob_up_var.set(
            f"Upside probability: {self._format_percent(context_prediction.get('direction_probability_up'))}"
        )
        self.explanation_prob_down_var.set(
            f"Downside probability: {self._format_percent(context_prediction.get('direction_probability_down'))}"
        )

        for key, var in self.explanation_reason_vars.items():
            entries = explanation.get(key) or []
            if entries:
                formatted = "\n".join(f"• {item}" for item in entries)
            else:
                formatted = "No signals identified."
            var.set(formatted)

        if self.feature_importance_tree is not None:
            tree = self.feature_importance_tree
            for item in tree.get_children():
                tree.delete(item)
            items = explanation.get("feature_importance") or []
            if items:
                for entry in items:
                    name = entry.get("name", "-")
                    importance = entry.get("importance")
                    try:
                        importance_str = f"{float(importance):.4f}"
                    except (TypeError, ValueError):
                        importance_str = "-"
                    category = entry.get("category", "-")
                    tree.insert("", "end", values=(name, importance_str, category))
            else:
                tree.insert("", "end", values=("No feature importance available", "-", "-"))

        if self.sources_text is not None:
            sources = explanation.get("sources") or []
            self.sources_text.configure(state="normal")
            self.sources_text.delete("1.0", tk.END)
            if sources:
                self.sources_text.insert("1.0", "\n".join(f"• {source}" for source in sources))
            else:
                self.sources_text.insert("1.0", "No data sources were recorded for this run.")
            self.sources_text.configure(state="disabled")

    def _clear_explanation_tab(self, message: str | None = None) -> None:
        summary = message or "Run a prediction to see explanation details."
        self.explanation_summary_var.set(summary)
        for var in self.explanation_reason_vars.values():
            var.set("No signals available yet.")
        if self.feature_importance_tree is not None:
            tree = self.feature_importance_tree
            for item in tree.get_children():
                tree.delete(item)
            tree.insert("", "end", values=("No data", "-", "-"))
        if self.sources_text is not None:
            self.sources_text.configure(state="normal")
            self.sources_text.delete("1.0", tk.END)
            self.sources_text.insert("1.0", "No sources recorded yet.")
            self.sources_text.configure(state="disabled")
        self.explanation_horizon_var.set("Horizon: -")
        self.explanation_target_var.set("Target date: -")
        self.explanation_return_var.set("Projected return: -")
        self.explanation_volatility_var.set("Projected volatility: -")
        self.explanation_prob_up_var.set("Upside probability: -")
        self.explanation_prob_down_var.set("Downside probability: -")

    def _show_error_message(self, message: str) -> None:
        self._set_status(message, error=True)
        try:
            messagebox.showerror("Prediction error", message, parent=self)
        except tk.TclError:
            LOGGER.warning("Unable to display error dialog: %s", message)
        self._toggle_inputs(state=tk.NORMAL)

    # ------------------------------------------------------------------
    # Logging integration
    # ------------------------------------------------------------------
    def _poll_log_queue(self) -> None:
        while True:
            try:
                message = self.log_queue.get_nowait()
            except queue.Empty:
                break
            else:
                self._append_log(message)
        self._after_id = self.after(250, self._poll_log_queue)

    def _append_log(self, message: str) -> None:
        self.log_messages.append(message)
        if len(self.log_messages) > 400:
            self.log_messages = self.log_messages[-400:]
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", tk.END)
        self.log_text.insert("1.0", "\n".join(self.log_messages))
        self.log_text.see(tk.END)
        self.log_text.configure(state="disabled")

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def _default_horizon_key(self) -> str:
        return next(iter(HORIZON_OPTIONS.keys()), "1D")

    def _horizon_value_from_label(self, label: Optional[str]) -> Optional[int]:
        if not label:
            return None
        lookup = label.strip().upper()
        if not lookup:
            return None
        return HORIZON_OPTIONS.get(lookup)

    def _label_for_horizon(self, horizon: Optional[int]) -> str:
        if horizon is None:
            return ""
        try:
            numeric = int(horizon)
        except (TypeError, ValueError):
            return ""
        for label, value in HORIZON_OPTIONS.items():
            if value == numeric:
                return label
        return f"H{numeric}"

    def _format_horizon_display(self, horizon: Optional[int]) -> str:
        if horizon is None:
            return "-"
        try:
            numeric = int(horizon)
        except (TypeError, ValueError):
            return "-"
        if numeric <= 0:
            return "-"
        label = self._label_for_horizon(numeric)
        day_label = "trading day" if numeric == 1 else "trading days"
        if label and label.startswith("H") and label[1:].isdigit():
            return f"{numeric} {day_label}"
        if label:
            return f"{label} ({numeric} {day_label})"
        return f"{numeric} {day_label}"

    def _refresh_horizon_states(self, available_horizons: Iterable[int]) -> None:
        available_set: set[int] = set()
        for value in available_horizons:
            try:
                numeric = int(value)
            except (TypeError, ValueError):
                continue
            if numeric > 0:
                available_set.add(numeric)
        if not available_set:
            available_set = set(HORIZON_OPTIONS.values())
        for label, widget in self._horizon_controls.items():
            value = HORIZON_OPTIONS.get(label)
            if widget is None or value is None:
                continue
            if value in available_set:
                if not self._inputs_disabled:
                    widget.state(["!disabled"])
            else:
                widget.state(["disabled"])
        current_value = self._horizon_value_from_label(self.horizon_var.get())
        if current_value not in available_set:
            fallback_label = next(
                (label for label, value in HORIZON_OPTIONS.items() if value in available_set),
                self._default_horizon_key(),
            )
            self.horizon_var.set(fallback_label)

    def _set_status(self, message: str, error: bool | None = None) -> None:
        if error is None:
            color = COLOR_NEUTRAL
        elif error:
            color = COLOR_LOSS
        else:
            color = COLOR_GAIN
        self.status_var.set(message)
        self.status_label.configure(foreground=color)

    def _format_currency(self, value: Any) -> str:
        converted = self._convert_currency_value(value)
        if converted is None:
            return "-"
        symbol = self._get_currency_symbol()
        return f"{symbol}{converted:,.2f}"

    def _format_percent(self, value: Any) -> str:
        if value is None:
            return "-"
        try:
            return f"{float(value) * 100:,.2f}%"
        except (TypeError, ValueError):
            return "-"

    def _format_number(self, value: Any) -> str:
        if value is None:
            return "-"
        try:
            return f"{float(value):,.4f}"
        except (TypeError, ValueError):
            return "-"

    def _format_int(self, value: Any) -> str:
        if value is None:
            return "-"
        try:
            return f"{int(value):,}"
        except (TypeError, ValueError):
            return "-"

    def _format_timestamp(self, value: Any) -> str:
        if value is None:
            return "-"
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return "-"
        try:
            timestamp = pd.to_datetime(value)
        except (TypeError, ValueError):
            return "-"
        if pd.isna(timestamp):
            return "-"
        if isinstance(timestamp, pd.Timestamp):
            dt_value = timestamp.to_pydatetime()
        else:
            dt_value = timestamp
        if not isinstance(dt_value, datetime):
            return "-"
        return dt_value.strftime("%Y-%m-%d %H:%M")

    def _convert_currency_value(self, value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        if self.currency_var.get() == "EUR":
            rate = self.usd_to_eur_rate
            if rate:
                return numeric * rate
            LOGGER.warning("Currency set to EUR without available conversion rate; falling back to USD values.")
        return numeric

    def _fetch_latest_fx_rate(self) -> Optional[float]:
        try:
            data = yf.download(
                "EURUSD=X",
                period="5d",
                interval="1d",
                progress=False,
                auto_adjust=False,
            )
        except Exception as exc:  # pylint: disable=broad-except
            LOGGER.warning("Failed to download EUR/USD rate: %s", exc)
            return None
        if data.empty or "Close" not in data.columns:
            return None
        close_data = data["Close"]
        if isinstance(close_data, pd.DataFrame):
            if close_data.empty:
                return None
            close_series = close_data.iloc[:, -1]
        else:
            close_series = close_data
        close_prices = pd.to_numeric(close_series, errors="coerce")
        if close_prices.empty:
            return None
        finite_mask = close_prices.apply(math.isfinite)
        filtered_count = int((~finite_mask).sum())
        if filtered_count:
            LOGGER.warning(
                "Filtered out %d non-finite EUR/USD close price value(s) when fetching FX rate.",
                filtered_count,
            )
        finite_prices = close_prices[finite_mask]
        if finite_prices.empty:
            return None
        latest = float(finite_prices.iloc[-1])
        if not math.isfinite(latest) or latest == 0:
            LOGGER.warning(
                "Latest EUR/USD close price %r is not usable for conversion; ignoring value.",
                latest,
            )
            return None
        return 1.0 / latest

    def _on_currency_toggle(self) -> None:
        current = self.currency_var.get()
        if current == "USD":
            if self.usd_to_eur_rate is not None:
                self.currency_button.state(["!disabled"])
                self.currency_var.set("EUR")
                self._update_currency_button()
                self._refresh_currency_views()
                self._set_status("Displaying prices in EUR", error=False)
                return
            self.currency_button.state(["disabled"])
            self._set_status("Fetching EUR/USD rate…")
            thread = threading.Thread(target=self._switch_to_eur_async, daemon=True)
            thread.start()
        else:
            self.currency_button.state(["!disabled"])
            self.currency_var.set("USD")
            self._update_currency_button()
            self._refresh_currency_views()
            self._set_status("Displaying prices in USD", error=False)

    def _switch_to_eur_async(self) -> None:
        rate = self._fetch_latest_fx_rate()
        self.after(0, lambda: self._finalise_currency_switch(rate))

    def _finalise_currency_switch(self, rate: Optional[float]) -> None:
        valid_rate: Optional[float]
        if rate is None:
            valid_rate = None
        else:
            try:
                valid_rate = float(rate)
            except (TypeError, ValueError):
                LOGGER.warning(
                    "Discarding non-numeric EUR/USD rate %r; keeping USD currency display.",
                    rate,
                )
                valid_rate = None
        if valid_rate is None or not math.isfinite(valid_rate):
            if valid_rate is not None:
                LOGGER.warning(
                    "Discarding non-finite EUR/USD rate %r; keeping USD currency display.",
                    valid_rate,
                )
            self.currency_button.state(["!disabled"])
            self._set_status("Unable to fetch EUR/USD rate", error=True)
            messagebox.showerror(
                "Currency conversion",
                "Der Wechselkurs EUR/USD konnte nicht geladen werden.",
                parent=self,
            )
            return
        self.usd_to_eur_rate = valid_rate
        self.currency_var.set("EUR")
        self._update_currency_button()
        self._refresh_currency_views()
        self.currency_button.state(["!disabled"])
        self._set_status("Displaying prices in EUR", error=False)

    def _update_currency_button(self) -> None:
        if self.currency_var.get() == "EUR":
            self.currency_button_text.set("Display in USD ($)")
        else:
            self.currency_button_text.set("Display in EUR (€)")

    def _refresh_currency_views(self) -> None:
        if self.price_data_base.empty:
            self.price_data = pd.DataFrame()
            self.wave_annotations = []
        elif self.currency_var.get() == "USD":
            self.price_data = self.price_data_base.copy()
            self.wave_annotations = list(self.wave_annotations_base)
        else:
            converted = self._convert_price_dataframe(self.price_data_base)
            converted_with_waves, waves = apply_wave_features(converted)
            self.price_data = converted_with_waves
            self.wave_annotations = waves
        self._plot_price_data()
        if self.latest_prediction is not None:
            self._update_prediction_panel(self.latest_prediction, self.latest_metrics)

    def _convert_price_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.currency_var.get() != "EUR":
            return df.copy()
        if self.usd_to_eur_rate is None:
            return df.copy()
        try:
            rate = float(self.usd_to_eur_rate)
        except (TypeError, ValueError):
            LOGGER.warning(
                "Currency conversion requested but usd_to_eur_rate=%r is not numeric; skipping conversion",
                self.usd_to_eur_rate,
            )
            return df.copy()
        converted = df.copy()
        converted.attrs.update(df.attrs)
        price_columns = df.attrs.get("price_columns")
        if not price_columns:
            price_columns = [
                column
                for column in converted.columns
                if column in {"Open", "High", "Low", "Close", "Adj Close"}
                or column.startswith(("SMA_", "EMA_"))
                or column in {"MACD", "MACD_Signal", "MACD_Hist", "BB_Middle_20", "BB_Upper_20", "BB_Lower_20"}
            ]
        if not price_columns:
            LOGGER.warning(
                "Currency conversion requested but dataframe has no price-like columns; leaving values unchanged"
            )
            return converted
        for column in price_columns:
            if column in converted.columns:
                converted[column] = pd.to_numeric(converted[column], errors="coerce") * rate
        return converted

    def _get_currency_symbol(self) -> str:
        return "€" if self.currency_var.get() == "EUR" else "$"

    def _on_close(self) -> None:
        if self._after_id is not None:
            self.after_cancel(self._after_id)
        logging.getLogger().removeHandler(self.log_handler)
        self.destroy()

    def run(self) -> None:
        """Start the Tkinter main event loop."""
        self.mainloop()


def run_app(default_ticker: str = "AAPL", options: Optional[UIOptions] = None) -> None:
    """Convenience helper to start the GUI from procedural code."""

    app = StockPredictorApp(default_ticker=default_ticker, options=options)
    app.run()
