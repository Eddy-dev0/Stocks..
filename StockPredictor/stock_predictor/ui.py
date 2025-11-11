"""Tkinter-based graphical interface for the stock predictor pipeline."""

from __future__ import annotations

import json
import logging
import queue
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import tkinter as tk
from tkinter import messagebox, ttk
import yfinance as yf

from matplotlib import dates as mdates
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from .config import PredictorConfig, build_config
from .model import StockPredictorAI

LOGGER = logging.getLogger(__name__)

APP_VERSION = "StockPredictor v1.0.0"

TIMEFRAME_WINDOWS: dict[str, int] = {
    "1D": 1,
    "1W": 7,
    "1M": 30,
    "3M": 90,
    "1Y": 365,
}

COLOR_GAIN = "#15803d"
COLOR_LOSS = "#dc2626"
COLOR_NEUTRAL = "#1f2937"


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

        self.predicted_close_var = tk.StringVar(value="-")
        self.last_close_var = tk.StringVar(value="-")
        self.absolute_change_var = tk.StringVar(value="-")
        self.percent_change_var = tk.StringVar(value="-")
        self.as_of_var = tk.StringVar(value="-")
        self.currency_var = tk.StringVar(value="USD")
        self.currency_button_text = tk.StringVar(value="Display in EUR (€)")
        self.metric_vars: dict[str, tk.StringVar] = {
            "mae": tk.StringVar(value="-"),
            "rmse": tk.StringVar(value="-"),
            "r2": tk.StringVar(value="-"),
            "training_rows": tk.StringVar(value="-"),
            "test_rows": tk.StringVar(value="-"),
        }

        self.refresh_var = tk.BooleanVar(value=self.options.refresh_data)
        self.ticker_var = tk.StringVar(value=self.default_ticker)
        self.usd_to_eur_rate: float | None = None
        self.latest_prediction: Optional[Dict[str, Any]] = None
        self.latest_metrics: Optional[Dict[str, Any]] = None

        self._interactive_widgets: list[tk.Widget] = []

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
        self.chart_ax = self.figure.add_subplot(111)
        self.chart_ax.set_title("Historical Prices")
        self.chart_ax.set_ylabel(f"Price ({self._get_currency_symbol()})")
        self.chart_ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

        self.canvas = FigureCanvasTkAgg(self.figure, master=chart_tab)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # Prediction tab
        prediction_tab = ttk.Frame(notebook, padding=20)
        notebook.add(prediction_tab, text="Prediction")

        header = ttk.Label(
            prediction_tab,
            textvariable=self.as_of_var,
            font=("Helvetica", 12),
        )
        header.grid(column=0, row=0, columnspan=2, sticky="w")

        ttk.Label(
            prediction_tab,
            text="Predicted Close",
            font=("Helvetica", 16, "bold"),
        ).grid(column=0, row=1, sticky="w", pady=(10, 5))
        self.predicted_close_label = ttk.Label(
            prediction_tab,
            textvariable=self.predicted_close_var,
            font=("Helvetica", 24, "bold"),
        )
        self.predicted_close_label.grid(column=0, row=2, sticky="w")

        ttk.Label(
            prediction_tab,
            text="Last Close",
            font=("Helvetica", 12, "bold"),
        ).grid(column=0, row=3, sticky="w", pady=(20, 5))
        ttk.Label(
            prediction_tab,
            textvariable=self.last_close_var,
            font=("Helvetica", 14),
        ).grid(column=0, row=4, sticky="w")

        ttk.Label(
            prediction_tab,
            text="Expected Change",
            font=("Helvetica", 12, "bold"),
        ).grid(column=1, row=1, sticky="w", padx=(40, 0), pady=(10, 5))
        self.absolute_change_label = ttk.Label(
            prediction_tab,
            textvariable=self.absolute_change_var,
            font=("Helvetica", 16, "bold"),
        )
        self.absolute_change_label.grid(column=1, row=2, sticky="w", padx=(40, 0))

        ttk.Label(
            prediction_tab,
            text="% Change",
            font=("Helvetica", 12, "bold"),
        ).grid(column=1, row=3, sticky="w", padx=(40, 0), pady=(20, 5))
        self.percent_change_label = ttk.Label(
            prediction_tab,
            textvariable=self.percent_change_var,
            font=("Helvetica", 16, "bold"),
        )
        self.percent_change_label.grid(column=1, row=4, sticky="w", padx=(40, 0))

        ttk.Separator(prediction_tab, orient="horizontal").grid(
            column=0, row=5, columnspan=2, sticky="ew", pady=(25, 15)
        )

        ttk.Label(
            prediction_tab,
            text="Training Metrics",
            font=("Helvetica", 13, "bold"),
        ).grid(column=0, row=6, columnspan=2, sticky="w", pady=(0, 10))

        metrics_grid = ttk.Frame(prediction_tab)
        metrics_grid.grid(column=0, row=7, columnspan=2, sticky="w")

        metric_labels = [
            ("MAE", "mae"),
            ("RMSE", "rmse"),
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

    # ------------------------------------------------------------------
    # Prediction pipeline glue
    # ------------------------------------------------------------------
    def _run_prediction(self, ticker: str) -> None:
        refresh = self.refresh_var.get()
        try:
            kwargs = {k: v for k, v in self.options.to_kwargs().items() if v is not None}
            if "start_date" not in kwargs:
                days = max(TIMEFRAME_WINDOWS.values())
                kwargs["start_date"] = (datetime.today() - timedelta(days=days)).date().isoformat()
            kwargs["ticker"] = ticker
            config = build_config(**kwargs)
            LOGGER.info("Starting prediction workflow for %s", ticker)

            ai = StockPredictorAI(config)
            prediction = ai.predict(refresh_data=refresh)
            source_before_fetch = ai.fetcher.last_price_source
            prices = ai.fetcher.fetch_price_data(force=False)

            data_source = (
                "Remote download" if source_before_fetch == "remote" else "Database cache"
            )

            metrics = prediction.get("training_metrics")
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
        except Exception as exc:  # pylint: disable=broad-except
            LOGGER.exception("Prediction failed for %s", ticker)
            message = str(exc)
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
        return df.sort_values("Date")

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

    def _on_prediction_error(self, message: str) -> None:
        self._set_status(message, error=True)
        messagebox.showerror("Prediction failed", message, parent=self)
        self._toggle_inputs(state=tk.NORMAL)

    # ------------------------------------------------------------------
    # Rendering helpers
    # ------------------------------------------------------------------
    def _plot_price_data(self) -> None:
        df = self.price_data
        if df.empty:
            self.chart_ax.clear()
            self.chart_ax.set_ylabel(f"Price ({self._get_currency_symbol()})")
            self.chart_ax.text(0.5, 0.5, "No price data", ha="center", va="center")
            self.canvas.draw_idle()
            return

        window_days = TIMEFRAME_WINDOWS.get(self.timeframe_var.get(), 90)
        end_date = df["Date"].max()
        start_date = end_date - timedelta(days=window_days)
        window_df = df[df["Date"] >= start_date]
        if window_df.empty:
            window_df = df

        self.chart_ax.clear()
        self.chart_ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        self.chart_ax.set_title(f"{self.current_ticker or ''} Price History")
        self.chart_ax.set_ylabel(f"Price ({self._get_currency_symbol()})")

        dates = mdates.date2num(window_df["Date"].to_list())
        width = 0.6
        for x, (_, row) in zip(dates, window_df.iterrows()):
            open_price = float(row["Open"])
            close_price = float(row["Close"])
            high_price = float(row["High"])
            low_price = float(row["Low"])
            color = COLOR_GAIN if close_price >= open_price else COLOR_LOSS
            self.chart_ax.plot([x, x], [low_price, high_price], color=color, linewidth=1.2)
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
            self.chart_ax.add_patch(rect)

        self.chart_ax.xaxis_date()
        self.chart_ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        self.figure.autofmt_xdate()
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

        as_of = prediction.get("as_of") or "-"
        ticker = prediction.get("ticker") or (self.current_ticker or "-")
        self.as_of_var.set(f"{ticker} – as of {as_of}")

        metrics = metrics or {}
        self.metric_vars["mae"].set(self._format_number(metrics.get("mae")))
        self.metric_vars["rmse"].set(self._format_number(metrics.get("rmse")))
        self.metric_vars["r2"].set(self._format_number(metrics.get("r2")))
        self.metric_vars["training_rows"].set(self._format_int(metrics.get("training_rows")))
        self.metric_vars["test_rows"].set(self._format_int(metrics.get("test_rows")))

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
            data = yf.download("EURUSD=X", period="5d", interval="1d", progress=False)
        except Exception as exc:  # pylint: disable=broad-except
            LOGGER.warning("Failed to download EUR/USD rate: %s", exc)
            return None
        if data.empty or "Close" not in data.columns:
            return None
        close_prices = data["Close"].dropna()
        if close_prices.empty:
            return None
        latest = float(close_prices.iloc[-1])
        if not latest:
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
        if rate is None:
            self.currency_button.state(["!disabled"])
            self._set_status("Unable to fetch EUR/USD rate", error=True)
            messagebox.showerror(
                "Currency conversion",
                "Der Wechselkurs EUR/USD konnte nicht geladen werden.",
                parent=self,
            )
            return
        self.usd_to_eur_rate = rate
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
        if not self.price_data_base.empty:
            self.price_data = self._convert_price_dataframe(self.price_data_base)
        else:
            self.price_data = pd.DataFrame()
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
        numeric_cols = converted.select_dtypes(include=["number"]).columns
        if not len(numeric_cols):
            LOGGER.warning(
                "Currency conversion requested but dataframe has no numeric columns; leaving values unchanged"
            )
            return converted
        converted.loc[:, numeric_cols] = converted.loc[:, numeric_cols] * rate
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
