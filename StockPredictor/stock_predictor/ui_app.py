"""Tkinter desktop application for the stock predictor platform."""

from __future__ import annotations

import logging
import os
import threading
import tkinter as tk
from dataclasses import dataclass
from tkinter import messagebox, ttk
from typing import Any, Callable, Iterable, Mapping

import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from stock_predictor.app import StockPredictorApplication
from stock_predictor.core import DEFAULT_PREDICTION_HORIZONS, StockPredictorAI

LOGGER = logging.getLogger(__name__)


def _safe_float(value: Any) -> float | None:
    """Best-effort conversion used for rendering numeric values."""

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric):
        return None
    return numeric


@dataclass(slots=True)
class CurrencyFormatter:
    """Utility that handles currency selection and formatting."""

    base_label: str
    quote_label: str
    fx_rate: float = 1.0
    mode: str = "base"

    def set_mode(self, mode: str) -> None:
        """Switch between base and quote currency modes."""

        if mode not in {"base", "quote"}:
            LOGGER.debug("Unsupported currency mode '%s'; defaulting to base", mode)
            mode = "base"
        self.mode = mode

    def set_rate(self, rate: float) -> None:
        """Update the FX conversion rate."""

        if rate <= 0:
            raise ValueError("FX rate must be positive.")
        self.fx_rate = rate

    def format(self, value: Any, *, allow_sign: bool = True) -> str:
        """Format a numeric value according to the active mode."""

        numeric = _safe_float(value)
        if numeric is None:
            return "—"
        label = self.base_label
        rendered = numeric
        if self.mode == "quote":
            label = self.quote_label
            rendered = numeric * self.fx_rate
        formatted = f"{abs(rendered):,.2f}"
        if not allow_sign:
            return f"{label} {formatted}".strip()
        if rendered < 0:
            return f"{label} -{formatted}"
        return f"{label} {formatted}"


class StockPredictorDesktopApp:
    """Manage the lifecycle of the Tkinter desktop interface."""

    def __init__(self, root: tk.Tk, application: StockPredictorApplication | None = None) -> None:
        self.root = root
        self.application = application or StockPredictorApplication.from_environment()
        self.config = self.application.config
        self.root.title(f"Stock Predictor – {self.config.ticker}")
        self.root.geometry("1280x840")

        self.ticker_var = tk.StringVar(value=self.config.ticker)
        base_currency = os.environ.get("STOCK_PREDICTOR_UI_BASE_CURRENCY", "Local")
        quote_currency = os.environ.get("STOCK_PREDICTOR_UI_QUOTE_CURRENCY", "USD")
        fx_rate = self._resolve_fx_rate(os.environ.get("STOCK_PREDICTOR_UI_FX_RATE"))
        self.currency_formatter = CurrencyFormatter(
            base_label=base_currency,
            quote_label=quote_currency,
            fx_rate=fx_rate,
        )

        horizons = self.config.prediction_horizons or DEFAULT_PREDICTION_HORIZONS
        self.selected_horizon = horizons[0]
        self.current_prediction: dict[str, Any] = {}
        self.price_history: pd.DataFrame | None = None
        self.feature_snapshot: pd.DataFrame | None = None
        self.feature_history: pd.DataFrame | None = None
        self.indicator_history: pd.DataFrame | None = None
        self.feature_toggle_vars: dict[str, tk.BooleanVar] = {}
        self._busy = False

        self._build_layout(horizons)
        self.root.after(200, self._initialise_prediction)

    # ------------------------------------------------------------------
    # Layout helpers
    # ------------------------------------------------------------------
    def _build_layout(self, horizons: Iterable[int]) -> None:
        self._build_toolbar(list(horizons))
        self._build_notebook()
        self._build_statusbar()

    def _build_toolbar(self, horizons: list[int]) -> None:
        toolbar = ttk.Frame(self.root, padding=(12, 6))
        toolbar.pack(fill=tk.X)

        self.refresh_button = ttk.Button(toolbar, text="Refresh data", command=self._on_refresh)
        self.refresh_button.pack(side=tk.LEFT, padx=(0, 6))

        self.predict_button = ttk.Button(toolbar, text="Run prediction", command=self._on_predict)
        self.predict_button.pack(side=tk.LEFT, padx=(0, 12))

        ttk.Label(toolbar, text="Horizon").pack(side=tk.LEFT)
        self.horizon_var = tk.StringVar(value=str(self.selected_horizon))
        self.horizon_box = ttk.Combobox(
            toolbar,
            width=6,
            state="readonly",
            textvariable=self.horizon_var,
            values=[str(value) for value in horizons],
        )
        self.horizon_box.pack(side=tk.LEFT, padx=(4, 12))
        self.horizon_box.bind("<<ComboboxSelected>>", self._on_horizon_changed)

        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)

        ttk.Label(toolbar, text="Ticker").pack(side=tk.LEFT, padx=(0, 4))
        self.ticker_entry = ttk.Entry(toolbar, width=10, textvariable=self.ticker_var)
        self.ticker_entry.pack(side=tk.LEFT)
        self.ticker_entry.bind("<Return>", self._on_ticker_submitted)
        self.ticker_entry.bind("<FocusOut>", self._on_ticker_focus_out)
        self.ticker_apply_button = ttk.Button(
            toolbar,
            text="Apply",
            command=self._on_ticker_button,
        )
        self.ticker_apply_button.pack(side=tk.LEFT, padx=(4, 12))

        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)

        ttk.Label(toolbar, text="Currency").pack(side=tk.LEFT, padx=(0, 4))
        self.currency_var = tk.StringVar(value="base")
        self.base_currency_button = ttk.Radiobutton(
            toolbar,
            text=self.currency_formatter.base_label,
            value="base",
            variable=self.currency_var,
            command=self._on_currency_changed,
        )
        self.base_currency_button.pack(side=tk.LEFT)
        self.quote_currency_button = ttk.Radiobutton(
            toolbar,
            text=self.currency_formatter.quote_label,
            value="quote",
            variable=self.currency_var,
            command=self._on_currency_changed,
        )
        self.quote_currency_button.pack(side=tk.LEFT, padx=(4, 8))

        ttk.Label(toolbar, text="FX rate").pack(side=tk.LEFT)
        self.fx_rate_var = tk.StringVar(value=f"{self.currency_formatter.fx_rate:.4f}")
        self.fx_rate_entry = ttk.Entry(toolbar, width=10, textvariable=self.fx_rate_var)
        self.fx_rate_entry.pack(side=tk.LEFT, padx=(4, 12))
        self.fx_rate_entry.bind("<Return>", self._on_fx_rate_changed)
        self.fx_rate_entry.bind("<FocusOut>", self._on_fx_rate_changed)

        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)

        self.progress = ttk.Progressbar(toolbar, mode="indeterminate", length=180)
        self.progress.pack(side=tk.RIGHT)

    def _build_notebook(self) -> None:
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self._build_overview_tab()
        self._build_indicators_tab()
        self._build_explanation_tab()
        self._build_settings_tab()

    def _build_statusbar(self) -> None:
        status_frame = ttk.Frame(self.root, padding=(12, 4))
        status_frame.pack(fill=tk.X)
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(status_frame, textvariable=self.status_var, anchor=tk.W).pack(fill=tk.X)

    # ------------------------------------------------------------------
    # Tab builders
    # ------------------------------------------------------------------
    def _build_overview_tab(self) -> None:
        frame = ttk.Frame(self.notebook, padding=12)
        self.notebook.add(frame, text="Overview")

        summary_frame = ttk.Frame(frame)
        summary_frame.pack(fill=tk.X, padx=4, pady=(0, 12))

        self.metric_vars: dict[str, tk.StringVar] = {}
        metric_specs = [
            ("ticker", "Ticker"),
            ("as_of", "Market data as of"),
            ("last_close", "Last close"),
            ("predicted_close", "Predicted close"),
            ("expected_change", "Expected change"),
            ("expected_change_pct", "Expected change %"),
            ("direction_probability_up", "Direction ↑"),
            ("direction_probability_down", "Direction ↓"),
        ]
        for idx, (key, label) in enumerate(metric_specs):
            column = idx % 4
            row = idx // 4
            container = ttk.Frame(summary_frame, padding=4)
            container.grid(row=row, column=column, sticky=tk.W)
            ttk.Label(container, text=f"{label}:", font=("TkDefaultFont", 9, "bold")).pack(anchor=tk.W)
            var = tk.StringVar(value="—")
            ttk.Label(container, textvariable=var).pack(anchor=tk.W)
            self.metric_vars[key] = var

        chart_frame = ttk.LabelFrame(frame, text="Price history", padding=8)
        chart_frame.pack(fill=tk.BOTH, expand=True)
        self.price_figure = Figure(figsize=(8, 4), dpi=100)
        self.price_ax = self.price_figure.add_subplot(111)
        self.price_canvas = FigureCanvasTkAgg(self.price_figure, master=chart_frame)
        self.price_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _build_indicators_tab(self) -> None:
        frame = ttk.Frame(self.notebook, padding=12)
        self.notebook.add(frame, text="Indicators")

        tree_frame = ttk.Frame(frame)
        tree_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        columns = ("indicator", "value", "category")
        self.indicator_tree = ttk.Treeview(tree_frame, columns=columns, show="headings", height=20)
        self.indicator_tree.heading("indicator", text="Indicator")
        self.indicator_tree.heading("value", text="Latest value")
        self.indicator_tree.heading("category", text="Category")
        self.indicator_tree.column("indicator", width=220, anchor=tk.W)
        self.indicator_tree.column("value", width=120, anchor=tk.E)
        self.indicator_tree.column("category", width=140, anchor=tk.W)
        self.indicator_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.indicator_tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.indicator_tree.configure(yscrollcommand=scrollbar.set)
        self.indicator_tree.bind("<<TreeviewSelect>>", self._on_indicator_selected)

        chart_frame = ttk.LabelFrame(frame, text="Indicator snapshot", padding=8)
        chart_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(12, 0))
        self.indicator_figure = Figure(figsize=(5, 4), dpi=100)
        self.indicator_ax = self.indicator_figure.add_subplot(111)
        self.indicator_canvas = FigureCanvasTkAgg(self.indicator_figure, master=chart_frame)
        self.indicator_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _build_explanation_tab(self) -> None:
        frame = ttk.Frame(self.notebook, padding=12)
        self.notebook.add(frame, text="Explanation")

        summary_box = ttk.LabelFrame(frame, text="Narrative", padding=8)
        summary_box.pack(fill=tk.BOTH, expand=False)
        self.summary_text = tk.Text(summary_box, height=8, wrap=tk.WORD, state=tk.DISABLED)
        self.summary_text.pack(fill=tk.BOTH, expand=True)

        reasons_frame = ttk.Frame(frame)
        reasons_frame.pack(fill=tk.BOTH, expand=True, pady=12)

        self.reason_lists: dict[str, tk.Text] = {}
        for column, (key, title) in enumerate(
            (
                ("technical_reasons", "Technical"),
                ("fundamental_reasons", "Fundamental"),
                ("sentiment_reasons", "Sentiment"),
                ("macro_reasons", "Macro"),
            )
        ):
            box = ttk.LabelFrame(reasons_frame, text=title, padding=6)
            box.grid(row=0, column=column, sticky=tk.NSEW, padx=4)
            reasons_frame.columnconfigure(column, weight=1)
            widget = tk.Text(box, height=8, wrap=tk.WORD, state=tk.DISABLED)
            widget.pack(fill=tk.BOTH, expand=True)
            self.reason_lists[key] = widget

        self.feature_importance_frame = ttk.LabelFrame(frame, text="Feature importance", padding=8)
        self.feature_importance_frame.pack(fill=tk.BOTH, expand=True)
        self.feature_tree = ttk.Treeview(
            self.feature_importance_frame,
            columns=("feature", "importance", "category"),
            show="headings",
            height=8,
        )
        self.feature_tree.heading("feature", text="Feature")
        self.feature_tree.heading("importance", text="Weight")
        self.feature_tree.heading("category", text="Category")
        self.feature_tree.column("feature", width=260, anchor=tk.W)
        self.feature_tree.column("importance", width=120, anchor=tk.E)
        self.feature_tree.column("category", width=140, anchor=tk.W)
        self.feature_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        feature_scroll = ttk.Scrollbar(
            self.feature_importance_frame, orient=tk.VERTICAL, command=self.feature_tree.yview
        )
        feature_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.feature_tree.configure(yscrollcommand=feature_scroll.set)

        self.feature_figure = Figure(figsize=(6, 3), dpi=100)
        self.feature_ax = self.feature_figure.add_subplot(111)
        figure_container = ttk.LabelFrame(frame, text="Top drivers", padding=8)
        figure_container.pack(fill=tk.BOTH, expand=True, pady=(12, 0))
        self.feature_canvas = FigureCanvasTkAgg(self.feature_figure, master=figure_container)
        self.feature_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _build_settings_tab(self) -> None:
        frame = ttk.Frame(self.notebook, padding=12)
        self.notebook.add(frame, text="Settings")

        info_box = ttk.LabelFrame(frame, text="Configuration", padding=8)
        info_box.pack(fill=tk.X)
        ttk.Label(info_box, text=f"Ticker: {self.config.ticker}").pack(anchor=tk.W)
        ttk.Label(info_box, text=f"Interval: {self.config.interval}").pack(anchor=tk.W)
        ttk.Label(info_box, text=f"Model: {self.config.model_type}").pack(anchor=tk.W)
        ttk.Label(info_box, text=f"Prediction targets: {', '.join(self.config.prediction_targets)}").pack(anchor=tk.W)

        toggles_box = ttk.LabelFrame(frame, text="Feature toggles", padding=8)
        toggles_box.pack(fill=tk.BOTH, expand=True, pady=(12, 0))

        for idx, (name, enabled) in enumerate(sorted(self.config.feature_toggles.items())):
            var = tk.BooleanVar(value=enabled)
            self.feature_toggle_vars[name] = var
            check = ttk.Checkbutton(
                toggles_box,
                text=name.replace("_", " ").title(),
                variable=var,
                command=self._on_feature_toggle_changed,
            )
            check.grid(row=idx // 2, column=idx % 2, sticky=tk.W, padx=6, pady=4)
        toggles_box.columnconfigure(0, weight=1)
        toggles_box.columnconfigure(1, weight=1)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------
    def _on_refresh(self) -> None:
        self._run_async(self._refresh_and_predict, "Refreshing market data…")

    def _on_predict(self) -> None:
        self._run_async(self._predict_only, "Generating prediction…")

    def _on_horizon_changed(self, _event: Any) -> None:
        try:
            self.selected_horizon = int(self.horizon_var.get())
        except (TypeError, ValueError):
            messagebox.showwarning("Invalid horizon", "Please select a valid horizon value.")
            self.horizon_var.set(str(self.selected_horizon))
            return
        self._on_predict()

    def _on_currency_changed(self) -> None:
        self.currency_formatter.set_mode(self.currency_var.get())
        self._update_metrics()

    def _on_fx_rate_changed(self, _event: Any) -> None:
        raw = self.fx_rate_var.get()
        try:
            rate = float(raw)
        except (TypeError, ValueError):
            messagebox.showwarning("Invalid FX rate", "Please provide a numeric FX conversion rate.")
            self.fx_rate_var.set(f"{self.currency_formatter.fx_rate:.4f}")
            return
        if rate <= 0:
            messagebox.showwarning("Invalid FX rate", "FX conversion rate must be positive.")
            self.fx_rate_var.set(f"{self.currency_formatter.fx_rate:.4f}")
            return
        self.currency_formatter.set_rate(rate)
        self.fx_rate_var.set(f"{self.currency_formatter.fx_rate:.4f}")
        self._update_metrics()

    def _on_ticker_submitted(self, _event: Any) -> None:
        self._apply_ticker_change(self.ticker_var.get())

    def _on_ticker_focus_out(self, _event: Any) -> None:
        value = (self.ticker_var.get() or "").strip().upper()
        if not value:
            self.ticker_var.set(self.config.ticker)
        else:
            self.ticker_var.set(value)

    def _on_ticker_button(self) -> None:
        self._apply_ticker_change(self.ticker_var.get())

    def _on_indicator_selected(self, _event: Any) -> None:
        selection = self.indicator_tree.selection()
        if not selection:
            return
        indicator = selection[0]
        series = None
        if isinstance(self.feature_history, pd.DataFrame) and indicator in self.feature_history.columns:
            series = pd.to_numeric(self.feature_history[indicator], errors="coerce").dropna()
        elif isinstance(self.indicator_history, pd.DataFrame):
            columns = {str(column).lower(): column for column in self.indicator_history.columns}
            indicator_col = columns.get("indicator")
            value_col = columns.get("value")
            date_col = columns.get("date")
            if indicator_col and value_col:
                subset = self.indicator_history[self.indicator_history[indicator_col] == indicator]
                if date_col:
                    subset = subset.copy()
                    subset[date_col] = pd.to_datetime(subset[date_col], errors="coerce")
                    subset = subset.dropna(subset=[date_col])
                    subset = subset.sort_values(date_col)
                    x_values = subset[date_col]
                else:
                    subset = subset.copy()
                    subset.index = pd.RangeIndex(start=0, stop=len(subset))
                    x_values = subset.index
                values = pd.to_numeric(subset[value_col], errors="coerce")
                series = pd.Series(values.values, index=x_values)
        if series is None or series.empty:
            return

        self.indicator_ax.clear()
        self.indicator_ax.plot(series.index, series.values, label=indicator)
        self.indicator_ax.set_title(indicator)
        self.indicator_ax.set_xlabel("Date")
        self.indicator_ax.set_ylabel("Value")
        self.indicator_ax.grid(True, linestyle="--", alpha=0.3)
        self.indicator_ax.legend(loc="best")
        self.indicator_figure.tight_layout()
        self.indicator_canvas.draw_idle()

    def _on_feature_toggle_changed(self) -> None:
        if self._busy:
            return
        new_toggles = {name: bool(var.get()) for name, var in self.feature_toggle_vars.items()}
        self.config.feature_toggles.update(new_toggles)
        self.application.pipeline = StockPredictorAI(self.config)
        self._set_status("Feature toggles updated. Re-run prediction to apply changes.")

    def _apply_ticker_change(self, raw_value: str | None) -> None:
        if self._busy:
            self._set_status("Operation in progress. Please wait before changing the ticker.")
            self.ticker_var.set(self.config.ticker)
            return

        ticker = (raw_value or "").strip().upper()
        if not ticker:
            messagebox.showwarning("Invalid ticker", "Please provide a non-empty ticker symbol.")
            self.ticker_var.set(self.config.ticker)
            return
        if ticker == self.config.ticker:
            self.ticker_var.set(ticker)
            return

        try:
            self.application.update_ticker(ticker)
        except Exception as exc:  # pragma: no cover - defensive path
            LOGGER.exception("Failed to update ticker to %s", ticker)
            messagebox.showerror("Ticker update failed", str(exc))
            self.ticker_var.set(self.config.ticker)
            return

        self.config = self.application.config
        self.root.title(f"Stock Predictor – {self.config.ticker}")
        self.ticker_var.set(self.config.ticker)
        self._run_async(self._refresh_and_predict, f"Loading data for {self.config.ticker}…")

    # ------------------------------------------------------------------
    # Core actions
    # ------------------------------------------------------------------
    def _initialise_prediction(self) -> None:
        self._on_predict()

    def _refresh_and_predict(self) -> dict[str, Any]:
        self.application.refresh_data(force=True)
        return self._predict_payload()

    def _predict_only(self) -> dict[str, Any]:
        return self._predict_payload()

    def _predict_payload(self) -> dict[str, Any]:
        prediction = self.application.predict(horizon=self.selected_horizon)
        metadata = self.application.pipeline.metadata
        snapshot = metadata.get("latest_features") if isinstance(metadata, Mapping) else None
        feature_history: pd.DataFrame | None = None
        if isinstance(snapshot, pd.DataFrame) and not snapshot.empty:
            try:
                feature_history, _, _ = self.application.pipeline.prepare_features()
            except Exception as exc:  # pragma: no cover - defensive path for optional deps
                LOGGER.debug("Failed to rebuild feature history: %s", exc)
                feature_history = snapshot
        elif not isinstance(snapshot, pd.DataFrame):
            try:
                feature_history, _, _ = self.application.pipeline.prepare_features()
            except Exception as exc:  # pragma: no cover - fallback when metadata missing
                LOGGER.debug("Failed to construct feature history: %s", exc)
                feature_history = None
        price_history: pd.DataFrame | None = None
        try:
            price_history = self.application.pipeline.fetcher.fetch_price_data()
        except Exception as exc:  # pragma: no cover - provider optional dependencies
            LOGGER.debug("Failed to fetch price history: %s", exc)
        indicators: pd.DataFrame | None = None
        try:
            indicators = self.application.pipeline.fetcher.fetch_indicator_data()
        except Exception as exc:  # pragma: no cover - optional dataset
            LOGGER.debug("Indicator dataset unavailable: %s", exc)
        return {
            "prediction": prediction,
            "snapshot": snapshot if isinstance(snapshot, pd.DataFrame) else None,
            "feature_history": feature_history,
            "price_history": price_history if isinstance(price_history, pd.DataFrame) else None,
            "indicator_history": indicators if isinstance(indicators, pd.DataFrame) else None,
        }

    # ------------------------------------------------------------------
    # Async execution helpers
    # ------------------------------------------------------------------
    def _run_async(self, func: Callable[[], Any], status: str) -> None:
        if self._busy:
            return
        self._set_busy(True, status)

        def worker() -> None:
            try:
                result = func()
            except Exception as exc:  # pragma: no cover - defensive wrapper around worker
                LOGGER.exception("Desktop worker failed: %s", exc)
                self.root.after(0, lambda err=exc: self._on_async_failure(err))
            else:
                self.root.after(0, lambda payload=result: self._on_async_success(payload))

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

    def _on_async_success(self, payload: Mapping[str, Any] | Any) -> None:
        if not isinstance(payload, Mapping):
            payload = {}

        prediction = payload.get("prediction", {})
        snapshot = payload.get("snapshot")
        feature_history = payload.get("feature_history")
        price_history = payload.get("price_history")
        indicator_history = payload.get("indicator_history")

        self.current_prediction = prediction if isinstance(prediction, Mapping) else {}
        if snapshot is None and isinstance(feature_history, pd.DataFrame) and not feature_history.empty:
            snapshot = feature_history.iloc[[-1]]
        self.feature_snapshot = snapshot if isinstance(snapshot, pd.DataFrame) else None
        self.feature_history = feature_history if isinstance(feature_history, pd.DataFrame) else None
        self.price_history = price_history if isinstance(price_history, pd.DataFrame) else None
        self.indicator_history = indicator_history if isinstance(indicator_history, pd.DataFrame) else None
        self._update_metrics()
        self._update_price_chart()
        self._update_indicator_view()
        self._update_explanation()
        self._set_busy(False, "Prediction updated.")

    def _on_async_failure(self, exc: Exception) -> None:
        self._set_busy(False, "An error occurred.")
        messagebox.showerror("Prediction failed", str(exc))

    def _set_busy(self, busy: bool, status: str | None = None) -> None:
        self._busy = busy
        state = tk.DISABLED if busy else tk.NORMAL
        for widget in (
            self.refresh_button,
            self.predict_button,
            self.horizon_box,
            self.ticker_entry,
            self.ticker_apply_button,
            self.base_currency_button,
            self.quote_currency_button,
            self.fx_rate_entry,
        ):
            widget.configure(state=state)
        if busy:
            self.progress.start(12)
        else:
            self.progress.stop()
        if status:
            self._set_status(status)

    def _set_status(self, message: str) -> None:
        self.status_var.set(message)

    # ------------------------------------------------------------------
    # UI updates
    # ------------------------------------------------------------------
    def _update_metrics(self) -> None:
        prediction = self.current_prediction or {}
        explanation = prediction.get("explanation") if isinstance(prediction, Mapping) else None
        self.metric_vars["ticker"].set(str(prediction.get("ticker") or self.config.ticker))
        self.metric_vars["as_of"].set(str(prediction.get("market_data_as_of") or "—"))

        last_close = prediction.get("last_close")
        predicted_close = prediction.get("predicted_close")
        expected_change = prediction.get("expected_change")
        expected_change_pct = prediction.get("expected_change_pct")

        self.metric_vars["last_close"].set(self.currency_formatter.format(last_close))
        self.metric_vars["predicted_close"].set(self.currency_formatter.format(predicted_close))
        self.metric_vars["expected_change"].set(self.currency_formatter.format(expected_change))

        pct = _safe_float(expected_change_pct)
        if pct is None:
            self.metric_vars["expected_change_pct"].set("—")
        else:
            self.metric_vars["expected_change_pct"].set(f"{pct * 100:+.2f}%")

        prob_up = prediction.get("direction_probability_up")
        prob_down = prediction.get("direction_probability_down")
        prob_up_str = self._format_probability(prob_up)
        prob_down_str = self._format_probability(prob_down)
        self.metric_vars["direction_probability_up"].set(prob_up_str)
        self.metric_vars["direction_probability_down"].set(prob_down_str)

        if explanation and isinstance(explanation, Mapping):
            summary = explanation.get("summary")
            if summary:
                self._set_status(summary)

    def _update_price_chart(self) -> None:
        self.price_ax.clear()
        if isinstance(self.price_history, pd.DataFrame) and not self.price_history.empty:
            frame = self.price_history.copy()
            lower_map = {str(column).lower(): column for column in frame.columns}
            if "date" in lower_map:
                frame[lower_map["date"]] = pd.to_datetime(frame[lower_map["date"]], errors="coerce")
                frame = frame.dropna(subset=[lower_map["date"]])
                frame = frame.sort_values(lower_map["date"])
                x_values = frame[lower_map["date"]]
            else:
                frame.index = pd.to_datetime(frame.index, errors="coerce")
                frame = frame.dropna(subset=[frame.columns[0]])
                x_values = frame.index
            close_column = None
            for candidate in ("close", "adjclose", "adj_close"):
                if candidate in lower_map:
                    close_column = lower_map[candidate]
                    break
            if close_column is None:
                close_column = frame.select_dtypes(include=[np.number]).columns[0]
            series = pd.to_numeric(frame[close_column], errors="coerce")
            self.price_ax.plot(x_values, series, label="Close")
            predicted_close = _safe_float(self.current_prediction.get("predicted_close"))
            if predicted_close is not None:
                target_date = self.current_prediction.get("target_date") or self.current_prediction.get("horizon")
                self.price_ax.axhline(predicted_close, color="tab:orange", linestyle="--", label="Predicted close")
                if hasattr(x_values, "iloc"):
                    last_x = x_values.iloc[-1]
                else:
                    last_x = x_values[-1]
                self.price_ax.text(
                    last_x,
                    predicted_close,
                    f"Predicted: {predicted_close:.2f}",
                    color="tab:orange",
                    va="bottom",
                )
                if target_date:
                    self.price_ax.set_title(f"Forecast horizon: {target_date}")
            self.price_ax.set_ylabel("Price")
            self.price_ax.grid(True, linestyle="--", alpha=0.3)
            self.price_ax.legend(loc="best")
        else:
            self.price_ax.text(0.5, 0.5, "Price history unavailable", ha="center", va="center")
        self.price_figure.tight_layout()
        self.price_canvas.draw_idle()

    def _update_indicator_view(self) -> None:
        for item in self.indicator_tree.get_children():
            self.indicator_tree.delete(item)
        if self.feature_snapshot is not None and not self.feature_snapshot.empty:
            row = self.feature_snapshot.iloc[0]
            metadata = self.application.pipeline.metadata
            categories = {}
            if isinstance(metadata, Mapping):
                categories = metadata.get("feature_categories", {}) or {}
            for column, value in row.items():
                numeric = _safe_float(value)
                if numeric is None:
                    continue
                category = categories.get(column, "feature")
                self.indicator_tree.insert(
                    "",
                    tk.END,
                    iid=str(column),
                    values=(str(column), f"{numeric:.4f}", str(category)),
                )
        if not self.indicator_tree.get_children() and isinstance(self.indicator_history, pd.DataFrame):
            frame = self.indicator_history
            columns = {str(column).lower(): column for column in frame.columns}
            indicator_col = columns.get("indicator")
            value_col = columns.get("value")
            date_col = columns.get("date")
            if indicator_col and value_col:
                latest = frame.copy()
                if date_col:
                    latest[date_col] = pd.to_datetime(latest[date_col], errors="coerce")
                    latest = latest.dropna(subset=[date_col])
                    latest = latest.sort_values(date_col)
                    latest = latest.groupby(indicator_col).tail(1)
                else:
                    latest = latest.groupby(indicator_col).tail(1)
                for _, row in latest.iterrows():
                    name = str(row.get(indicator_col))
                    numeric = _safe_float(row.get(value_col))
                    if numeric is None:
                        continue
                    self.indicator_tree.insert(
                        "",
                        tk.END,
                        iid=name,
                        values=(name, f"{numeric:.4f}", "indicator"),
                    )
        if isinstance(self.feature_history, pd.DataFrame):
            cleaned_history = self.feature_history.copy()
            for column in cleaned_history.columns:
                try:
                    numeric = pd.to_numeric(cleaned_history[column], errors="coerce")
                except Exception:  # pragma: no cover - defensive conversion
                    continue
                if not numeric.isna().all():
                    cleaned_history[column] = numeric
            self.feature_history = cleaned_history

    def _update_explanation(self) -> None:
        prediction = self.current_prediction or {}
        explanation = prediction.get("explanation") if isinstance(prediction, Mapping) else None
        if not isinstance(explanation, Mapping):
            self._clear_explanation()
            return

        self._set_text(self.summary_text, explanation.get("summary") or "No explanation available.")
        for key, widget in self.reason_lists.items():
            reasons = explanation.get(key)
            if isinstance(reasons, Iterable) and not isinstance(reasons, (str, bytes)):
                text = "\n".join(f"• {item}" for item in reasons)
            else:
                text = "No signals."
            self._set_text(widget, text)

        for item in self.feature_tree.get_children():
            self.feature_tree.delete(item)
        importance = explanation.get("feature_importance")
        if isinstance(importance, Iterable) and not isinstance(importance, (str, bytes)):
            bars: list[tuple[str, float, str]] = []
            for entry in importance:
                if not isinstance(entry, Mapping):
                    continue
                name = str(entry.get("name") or "")
                weight = _safe_float(entry.get("importance"))
                if weight is None:
                    continue
                category = str(entry.get("category") or "feature")
                self.feature_tree.insert("", tk.END, values=(name, f"{weight:+.4f}", category))
                bars.append((name, weight, category))
            self._update_feature_chart(bars)
        else:
            self._update_feature_chart([])

    def _update_feature_chart(self, bars: list[tuple[str, float, str]]) -> None:
        self.feature_ax.clear()
        if not bars:
            self.feature_ax.text(0.5, 0.5, "Feature importance unavailable", ha="center", va="center")
        else:
            names = [name for name, _, _ in bars]
            weights = [weight for _, weight, _ in bars]
            colors = ["tab:green" if weight >= 0 else "tab:red" for weight in weights]
            y_positions = np.arange(len(names))
            self.feature_ax.barh(y_positions, weights, color=colors)
            self.feature_ax.set_yticks(y_positions)
            self.feature_ax.set_yticklabels(names)
            self.feature_ax.axvline(0, color="black", linewidth=0.8)
            self.feature_ax.set_xlabel("Importance")
            self.feature_ax.set_ylabel("Feature")
            self.feature_ax.grid(True, axis="x", linestyle="--", alpha=0.3)
        self.feature_figure.tight_layout()
        self.feature_canvas.draw_idle()

    def _clear_explanation(self) -> None:
        self._set_text(self.summary_text, "Explanation unavailable.")
        for widget in self.reason_lists.values():
            self._set_text(widget, "No data.")
        for item in self.feature_tree.get_children():
            self.feature_tree.delete(item)
        self._update_feature_chart([])

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def _set_text(self, widget: tk.Text, value: str) -> None:
        widget.configure(state=tk.NORMAL)
        widget.delete("1.0", tk.END)
        widget.insert(tk.END, value)
        widget.configure(state=tk.DISABLED)

    def _format_probability(self, value: Any) -> str:
        numeric = _safe_float(value)
        if numeric is None:
            return "—"
        return f"{numeric * 100:.1f}%"

    def _resolve_fx_rate(self, raw: str | None) -> float:
        if raw is None:
            return 1.0
        try:
            rate = float(raw)
        except (TypeError, ValueError):
            LOGGER.warning("Invalid FX rate '%s'; falling back to 1.0", raw)
            return 1.0
        if rate <= 0:
            LOGGER.warning("Non-positive FX rate '%s'; falling back to 1.0", raw)
            return 1.0
        return rate


def run_tkinter_app() -> None:
    """Launch the Tkinter desktop application."""

    root = tk.Tk()
    try:
        application = StockPredictorApplication.from_environment()
    except Exception as exc:  # pragma: no cover - initialisation guard
        LOGGER.exception("Failed to initialise application: %s", exc)
        messagebox.showerror("Initialisation failed", str(exc))
        root.destroy()
        return

    try:
        StockPredictorDesktopApp(root, application=application)
    except Exception as exc:  # pragma: no cover - defensive UI guard
        LOGGER.exception("Failed to start desktop UI: %s", exc)
        messagebox.showerror("Startup failed", str(exc))
        root.destroy()
        return

    root.mainloop()


if __name__ == "__main__":
    run_tkinter_app()
