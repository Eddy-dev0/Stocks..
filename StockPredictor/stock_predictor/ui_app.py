"""Tkinter desktop application for the stock predictor platform."""

from __future__ import annotations

import logging
import os
import threading
import tkinter as tk
from dataclasses import dataclass
from tkinter import messagebox, ttk
from typing import Any, Callable, Iterable, Mapping

import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from matplotlib import style as mpl_style
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from stock_predictor.app import StockPredictorApplication
from stock_predictor.core import DEFAULT_PREDICTION_HORIZONS, StockPredictorAI

LOGGER = logging.getLogger(__name__)


KNOWN_CURRENCY_SYMBOLS = (
    "$",
    "€",
    "£",
    "¥",
    "₹",
    "₩",
    "₽",
    "₪",
    "₫",
    "₱",
    "฿",
    "₦",
    "₴",
    "₭",
    "₲",
    "₵",
)


def _safe_float(value: Any) -> float | None:
    """Best-effort conversion used for rendering numeric values."""

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric):
        return None
    return numeric


def _detect_currency_symbol(label: str, fallback: str = "$") -> str:
    """Extract a representative currency symbol from a display label."""

    for char in label:
        if char in KNOWN_CURRENCY_SYMBOLS:
            return char
    trimmed = label.strip()
    if len(trimmed) == 1 and not trimmed.isdigit():
        return trimmed
    return fallback


def fmt_ccy(value: Any, symbol: str, *, decimals: int = 2) -> str:
    """Format a numeric value as currency with thousands separators."""

    numeric = _safe_float(value)
    if numeric is None:
        return "—"
    try:
        decimals_int = max(0, int(decimals))
    except (TypeError, ValueError):  # pragma: no cover - defensive
        decimals_int = 2
    formatted = f"{abs(numeric):,.{decimals_int}f}"
    sign = "-" if numeric < 0 else ""
    clean_symbol = symbol or ""
    return f"{sign}{clean_symbol}{formatted}" if clean_symbol else f"{sign}{formatted}"


def fmt_pct(value: Any, *, decimals: int = 2, show_sign: bool = False) -> str:
    """Format a numeric value as a percentage string."""

    numeric = _safe_float(value)
    if numeric is None:
        return "—"
    scaled = numeric * 100
    if show_sign:
        return f"{scaled:+.{decimals}f}%"
    return f"{scaled:.{decimals}f}%"


@dataclass(slots=True)
class HorizonOption:
    """Represents a forecast horizon option exposed in the UI."""

    label: str
    code: str
    business_days: int


HORIZON_OPTIONS: tuple[HorizonOption, ...] = (
    HorizonOption("Tomorrow", "1d", 1),
    HorizonOption("1 Week", "1w", 5),
    HorizonOption("1 Month", "1m", 21),
    HorizonOption("3 Months", "3m", 63),
)


class Tooltip:
    """Lightweight tooltip helper for Tkinter widgets."""

    def __init__(self, widget: tk.Widget, text: str) -> None:
        self.widget = widget
        self.text = text
        self.tip_window: tk.Toplevel | None = None
        self.widget.bind("<Enter>", self._on_enter)
        self.widget.bind("<Leave>", self._on_leave)

    def _on_enter(self, _event: Any) -> None:
        if self.tip_window is not None:
            return
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        self.tip_window = tk.Toplevel(self.widget)
        self.tip_window.wm_overrideredirect(True)
        self.tip_window.wm_geometry(f"+{x}+{y}")
        label = tk.Label(
            self.tip_window,
            text=self.text,
            bg="white",
            relief=tk.SOLID,
            borderwidth=1,
        )
        label.pack(ipadx=6, ipady=3)

    def _on_leave(self, _event: Any) -> None:
        if self.tip_window is not None:
            self.tip_window.destroy()
            self.tip_window = None


class StockPredictorDesktopApp:
    """Manage the lifecycle of the Tkinter desktop interface."""

    def __init__(self, root: tk.Tk, application: StockPredictorApplication | None = None) -> None:
        self.root = root
        self.application = application or StockPredictorApplication.from_environment()
        self.config = self.application.config
        self.root.title(f"Stock Predictor – {self.config.ticker}")
        self.root.geometry("1280x840")

        self.style = ttk.Style()
        self.default_theme = self.style.theme_use()

        self.ticker_var = tk.StringVar(value=self.config.ticker)
        base_currency = os.environ.get("STOCK_PREDICTOR_UI_BASE_CURRENCY", "Local")
        usd_rate = self._resolve_fx_rate(os.environ.get("STOCK_PREDICTOR_UI_FX_RATE"))
        base_symbol = _detect_currency_symbol(base_currency, fallback="$")
        self.currency_profiles: dict[str, dict[str, str]] = {
            "local": {"label": base_currency, "symbol": base_symbol.strip()},
            "usd": {"label": "USD $", "symbol": "$"},
            "eur": {"label": "EUR €", "symbol": "€"},
        }
        self.currency_rates: dict[str, float] = {
            "local": 1.0,
            "usd": usd_rate,
            "eur": 1.0,
        }
        self.currency_mode: str = "local"
        self.currency_symbol: str = self._currency_symbol("local")

        horizons = self.config.prediction_horizons or DEFAULT_PREDICTION_HORIZONS
        self.horizon_options = HORIZON_OPTIONS
        self.horizon_option_map = {option.label: option for option in self.horizon_options}
        self.code_to_horizon = {option.code: option for option in self.horizon_options}
        self.label_to_code = {option.label: option.code for option in self.horizon_options}
        self.selected_horizon_option = self._resolve_initial_horizon(horizons)
        self.current_prediction: dict[str, Any] = {}
        self.price_history: pd.DataFrame | None = None
        self.feature_snapshot: pd.DataFrame | None = None
        self.feature_history: pd.DataFrame | None = None
        self.indicator_history: pd.DataFrame | None = None
        self.feature_toggle_vars: dict[str, tk.BooleanVar] = {}
        self.forecast_date_var = tk.StringVar(value="Forecast date: —")

        self.currency_mode_var = tk.StringVar(value="local")
        self.currency_button_text = tk.StringVar(value=self._currency_label("local"))
        self.currency_rate_var = tk.StringVar(value=f"{self._currency_rate('usd'):.4f}")

        self.currency_choice_map = {"Local": "local", "USD": "usd", "EUR": "eur"}
        self.currency_display_map = {value: key for key, value in self.currency_choice_map.items()}
        self.price_decimal_places = 2
        self.decimal_option_map = {
            "2 decimals": 2,
            "3 decimals": 3,
            "4 decimals": 4,
        }
        self.decimal_display_map = {value: label for label, value in self.decimal_option_map.items()}
        self.currency_default_var = tk.StringVar(
            value=self.currency_display_map.get(self.currency_mode, "Local")
        )
        self.number_format_var = tk.StringVar(value=self.decimal_display_map[self.price_decimal_places])
        self.show_pnl_var = tk.BooleanVar(value=True)
        self.dark_mode_var = tk.BooleanVar(value=False)
        self.text_widgets: list[tk.Text] = []
        self.dark_mode_enabled = False

        self.position_size_var = tk.IntVar(value=1)
        self.pnl_var = tk.StringVar(value="Expected P&L for 1 share: —")
        self.position_size_var.trace_add("write", self._on_position_size_changed)

        self._busy = False

        self._build_layout(horizons)
        self._apply_theme()
        self.root.after(200, self._initialise_prediction)

    # ------------------------------------------------------------------
    # Layout helpers
    # ------------------------------------------------------------------
    def _resolve_initial_horizon(self, horizons: Iterable[int]) -> HorizonOption:
        for value in horizons:
            for option in self.horizon_options:
                if option.business_days == value:
                    return option
        return self.horizon_options[0]

    def _build_layout(self, horizons: Iterable[int]) -> None:
        self._build_toolbar(list(horizons))
        self._build_notebook()
        self._build_statusbar()
        self._on_currency_mode_changed(self.currency_mode_var.get())

    def _build_toolbar(self, _horizons: list[int]) -> None:
        toolbar = ttk.Frame(self.root, padding=(12, 6))
        toolbar.pack(fill=tk.X)

        self.refresh_button = ttk.Button(toolbar, text="Refresh data", command=self._on_refresh)
        self.refresh_button.pack(side=tk.LEFT, padx=(0, 6))

        self.predict_button = ttk.Button(toolbar, text="Run prediction", command=self._on_predict)
        self.predict_button.pack(side=tk.LEFT, padx=(0, 12))

        ttk.Label(toolbar, text="Horizon").pack(side=tk.LEFT)
        self.horizon_var = tk.StringVar(value=self.selected_horizon_option.label)
        self.horizon_box = ttk.Combobox(
            toolbar,
            width=12,
            state="readonly",
            textvariable=self.horizon_var,
            values=[option.label for option in self.horizon_options],
        )
        self.horizon_box.pack(side=tk.LEFT, padx=(4, 8))
        self.horizon_box.bind("<<ComboboxSelected>>", self._on_horizon_changed)

        self.forecast_label = ttk.Label(toolbar, textvariable=self.forecast_date_var)
        self.forecast_label.pack(side=tk.LEFT, padx=(4, 12))

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
        self.ticker_apply_button.pack(side=tk.LEFT, padx=(4, 4))

        self.ticker_help = ttk.Label(toolbar, text="?", width=2, anchor=tk.CENTER, cursor="question_arrow")
        self.ticker_help.pack(side=tk.LEFT, padx=(0, 12))
        Tooltip(self.ticker_help, "Use official symbols (e.g., RHM.DE, ^GSPC).")

        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)

        self.currency_menu_button = ttk.Menubutton(
            toolbar, textvariable=self.currency_button_text, direction="below"
        )
        self.currency_menu = tk.Menu(self.currency_menu_button, tearoff=False)
        for code in ("local", "usd", "eur"):
            label = self._currency_label(code)
            self.currency_menu.add_radiobutton(
                label=label,
                value=code,
                variable=self.currency_mode_var,
                command=lambda mode=code: self._on_currency_mode_changed(mode),
            )
        self.currency_menu_button.configure(menu=self.currency_menu)
        self.currency_menu_button.pack(side=tk.LEFT, padx=(0, 6))

        self.fx_rate_entry = ttk.Entry(toolbar, width=10, textvariable=self.currency_rate_var, state=tk.DISABLED)
        self.fx_rate_entry.pack(side=tk.LEFT)
        self.fx_rate_entry.bind("<Return>", self._on_currency_rate_submit)

        self.fx_rate_button = ttk.Button(toolbar, text="Update rate", command=self._on_currency_rate_button)
        self.fx_rate_button.pack(side=tk.LEFT, padx=(4, 12))

        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)

        self.progress = ttk.Progressbar(toolbar, mode="indeterminate", length=180)
        self.progress.pack(side=tk.RIGHT)

        self._update_forecast_label()

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
            ("direction", "Direction"),
        ]
        columns = 3
        for idx, (key, label) in enumerate(metric_specs):
            column = idx % columns
            row = idx // columns
            container = ttk.Frame(summary_frame, padding=4)
            container.grid(row=row, column=column, sticky=tk.W)
            ttk.Label(container, text=f"{label}:", font=("TkDefaultFont", 9, "bold")).pack(anchor=tk.W)
            var = tk.StringVar(value="—")
            ttk.Label(container, textvariable=var).pack(anchor=tk.W)
            self.metric_vars[key] = var

        controls_row = (len(metric_specs) + columns - 1) // columns
        controls_frame = ttk.Frame(summary_frame, padding=4)
        controls_frame.grid(row=controls_row, column=0, columnspan=columns, sticky=tk.W)
        ttk.Label(controls_frame, text="Position size:").grid(row=0, column=0, sticky=tk.W)
        self.position_spinbox = ttk.Spinbox(
            controls_frame,
            from_=1,
            to=1_000_000,
            increment=1,
            width=8,
            textvariable=self.position_size_var,
            command=self._recompute_pnl,
        )
        self.position_spinbox.grid(row=0, column=1, padx=(6, 6), sticky=tk.W)
        self.position_spinbox.bind("<FocusOut>", lambda _event: self._recompute_pnl())
        ttk.Label(controls_frame, text="shares").grid(row=0, column=2, sticky=tk.W)

        self.pnl_label = ttk.Label(frame, textvariable=self.pnl_var)
        self._pnl_pack_options = {"anchor": tk.W, "padx": 8, "pady": (0, 12)}
        self.pnl_label.pack(**self._pnl_pack_options)

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
        self.text_widgets.append(self.summary_text)

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
            self.text_widgets.append(widget)

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

        display_box = ttk.LabelFrame(frame, text="Display preferences", padding=8)
        display_box.pack(fill=tk.X, pady=(12, 0))

        currency_row = ttk.Frame(display_box)
        currency_row.pack(fill=tk.X, pady=4)
        ttk.Label(currency_row, text="Currency default:").grid(row=0, column=0, sticky=tk.W)
        self.currency_default_box = ttk.Combobox(
            currency_row,
            state="readonly",
            width=16,
            values=list(self.currency_choice_map.keys()),
            textvariable=self.currency_default_var,
        )
        self.currency_default_box.grid(row=0, column=1, padx=(8, 0), sticky=tk.W)
        self.currency_default_box.bind("<<ComboboxSelected>>", self._on_currency_default_changed)

        number_row = ttk.Frame(display_box)
        number_row.pack(fill=tk.X, pady=4)
        ttk.Label(number_row, text="Number format:").grid(row=0, column=0, sticky=tk.W)
        self.number_format_box = ttk.Combobox(
            number_row,
            state="readonly",
            width=16,
            values=list(self.decimal_option_map.keys()),
            textvariable=self.number_format_var,
        )
        self.number_format_box.grid(row=0, column=1, padx=(8, 0), sticky=tk.W)
        self.number_format_box.bind("<<ComboboxSelected>>", self._on_number_format_changed)

        toggles_preferences = ttk.Frame(display_box)
        toggles_preferences.pack(fill=tk.X, pady=(4, 0))
        self.pnl_toggle = ttk.Checkbutton(
            toggles_preferences,
            text="Show position P&L box",
            variable=self.show_pnl_var,
            command=self._on_pnl_toggle,
        )
        self.pnl_toggle.grid(row=0, column=0, sticky=tk.W)
        self.dark_mode_toggle = ttk.Checkbutton(
            toggles_preferences,
            text="Dark mode",
            variable=self.dark_mode_var,
            command=self._on_dark_mode_toggle,
        )
        self.dark_mode_toggle.grid(row=0, column=1, sticky=tk.W, padx=(12, 0))

        self._update_pnl_visibility()

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
        label = self.horizon_var.get()
        option = self.horizon_option_map.get(label)
        if option is None:
            messagebox.showwarning("Invalid horizon", "Please select a valid horizon value.")
            self.horizon_var.set(self.selected_horizon_option.label)
            return
        self.selected_horizon_option = option
        self._update_forecast_label()
        self._on_predict()

    def _on_currency_default_changed(self, _event: Any | None = None) -> None:
        label = self.currency_default_var.get()
        mode = self.currency_choice_map.get(label, "local")
        if self.currency_mode_var.get() != mode:
            self.currency_mode_var.set(mode)
        self._on_currency_mode_changed(mode)

    def _on_number_format_changed(self, _event: Any | None = None) -> None:
        label = self.number_format_var.get()
        decimals = self.decimal_option_map.get(label, self.price_decimal_places)
        if decimals == self.price_decimal_places:
            return
        self.price_decimal_places = decimals
        display_label = self.decimal_display_map.get(decimals)
        if display_label and display_label != label:
            self.number_format_var.set(display_label)
        self._refresh_numeric_views()

    def _on_pnl_toggle(self) -> None:
        self._update_pnl_visibility()

    def _on_dark_mode_toggle(self) -> None:
        enabled = bool(self.dark_mode_var.get())
        if enabled == self.dark_mode_enabled:
            return
        self.dark_mode_enabled = enabled
        self._apply_theme()
        self._refresh_numeric_views()

    def _on_currency_mode_changed(self, mode: str) -> None:
        if mode not in self.currency_profiles:
            LOGGER.debug("Unsupported currency mode '%s'; defaulting to local", mode)
            mode = "local"
        self.currency_mode = mode
        self.currency_symbol = self._currency_symbol(mode)
        self.currency_button_text.set(self._currency_label(mode))
        if hasattr(self, "currency_default_var"):
            display_label = self.currency_display_map.get(mode, "Local")
            if self.currency_default_var.get() != display_label:
                self.currency_default_var.set(display_label)
        rate = self._currency_rate(mode)
        self.currency_rate_var.set(f"{rate:.4f}")
        if self._busy:
            entry_state = tk.DISABLED
            button_state = tk.DISABLED
        elif mode == "local":
            entry_state = tk.DISABLED
            button_state = tk.DISABLED
        else:
            entry_state = tk.NORMAL
            button_state = tk.NORMAL
        self.fx_rate_entry.configure(state=entry_state)
        self.fx_rate_button.configure(state=button_state)
        if hasattr(self, "metric_vars"):
            self._update_metrics()
        if hasattr(self, "price_ax"):
            self._update_price_chart()
        if hasattr(self, "indicator_tree"):
            self._update_indicator_view()

    def _on_currency_rate_submit(self, _event: Any) -> None:
        self._on_currency_rate_button()

    def _on_currency_rate_button(self) -> None:
        mode = self.currency_mode_var.get()
        if mode == "local":
            return
        raw = self.currency_rate_var.get()
        try:
            rate = float(raw)
        except (TypeError, ValueError):
            messagebox.showwarning("Invalid FX rate", "Please provide a numeric FX conversion rate.")
            self.currency_rate_var.set(f"{self._currency_rate(mode):.4f}")
            return
        if rate <= 0:
            messagebox.showwarning("Invalid FX rate", "FX conversion rate must be positive.")
            self.currency_rate_var.set(f"{self._currency_rate(mode):.4f}")
            return
        self._set_currency_rate(mode, rate)
        self.currency_rate_var.set(f"{self._currency_rate(mode):.4f}")
        self._update_metrics()
        self._update_price_chart()
        self._update_indicator_view()

    def _on_position_size_changed(self, *_args: Any) -> None:
        self._recompute_pnl()

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
        self._style_figure(self.indicator_figure)
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
        prediction = self.application.predict(horizon=self.selected_horizon_option.business_days)
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
        self._update_forecast_label(self.current_prediction.get("target_date"))
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
            self.currency_menu_button,
            self.fx_rate_button,
        ):
            widget.configure(state=state)
        if busy:
            self.fx_rate_entry.configure(state=tk.DISABLED)
            self.fx_rate_button.configure(state=tk.DISABLED)
        else:
            self._on_currency_mode_changed(self.currency_mode_var.get())
        if busy:
            self.progress.start(12)
        else:
            self.progress.stop()
        if status:
            self._set_status(status)

    def _set_status(self, message: str) -> None:
        self.status_var.set(message)

    def _forecast_base_date(self) -> pd.Timestamp:
        prediction = self.current_prediction or {}
        as_of = prediction.get("market_data_as_of") if isinstance(prediction, Mapping) else None
        timestamp = pd.to_datetime(as_of, errors="coerce") if as_of is not None else None
        if pd.notna(timestamp):
            return pd.Timestamp(timestamp).normalize()

        if isinstance(self.price_history, pd.DataFrame) and not self.price_history.empty:
            frame = self.price_history.copy()
            lower_map = {str(column).lower(): column for column in frame.columns}
            if "date" in lower_map:
                dates = pd.to_datetime(frame[lower_map["date"]], errors="coerce").dropna()
                if not dates.empty:
                    return pd.Timestamp(dates.iloc[-1]).normalize()
            index_dates = pd.to_datetime(frame.index, errors="coerce")
            index_series = pd.Series(index_dates).dropna()
            if not index_series.empty:
                return pd.Timestamp(index_series.iloc[-1]).normalize()

        return pd.Timestamp.today().normalize()

    def _compute_forecast_date(self, target_date: Any | None = None) -> pd.Timestamp | None:
        parsed = pd.to_datetime(target_date, errors="coerce") if target_date is not None else None
        if pd.notna(parsed):
            return pd.Timestamp(parsed).normalize()

        option = self.selected_horizon_option
        if option is None:
            return None
        base_date = self._forecast_base_date()
        try:
            offset = pd.tseries.offsets.BusinessDay(option.business_days)
            forecast = (base_date + offset).normalize()
        except Exception:
            forecast = (base_date + pd.to_timedelta(option.business_days, unit="D")).normalize()
        return forecast

    def _update_forecast_label(self, target_date: Any | None = None) -> None:
        display = "Forecast date: —"
        forecast = self._compute_forecast_date(target_date)
        if forecast is not None:
            display = f"Forecast date: {forecast.date().isoformat()}"
        self.forecast_date_var.set(display)

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

        last_close_converted = self._convert_currency(last_close)
        predicted_converted = self._convert_currency(predicted_close)
        change_converted = self._convert_currency(expected_change)

        decimals = self.price_decimal_places
        self.metric_vars["last_close"].set(
            fmt_ccy(last_close_converted, self.currency_symbol, decimals=decimals)
        )

        forecast = self._compute_forecast_date(self.current_prediction.get("target_date"))
        predicted_display = fmt_ccy(predicted_converted, self.currency_symbol, decimals=decimals)
        if forecast is not None:
            forecast_str = forecast.date().isoformat()
            if predicted_display == "—":
                predicted_display = f"— ({forecast_str})"
            else:
                predicted_display = f"{predicted_display} ({forecast_str})"
        self.metric_vars["predicted_close"].set(predicted_display)

        change_display = fmt_ccy(change_converted, self.currency_symbol, decimals=decimals)
        pct_display = fmt_pct(expected_change_pct, show_sign=True)
        if change_display == "—" and pct_display == "—":
            self.metric_vars["expected_change"].set("—")
        elif change_display == "—":
            self.metric_vars["expected_change"].set(pct_display)
        elif pct_display == "—":
            self.metric_vars["expected_change"].set(change_display)
        else:
            self.metric_vars["expected_change"].set(f"{change_display} ({pct_display})")

        prob_up = prediction.get("direction_probability_up")
        prob_down = prediction.get("direction_probability_down")
        prob_up_str = fmt_pct(prob_up, decimals=1)
        prob_down_str = fmt_pct(prob_down, decimals=1)
        parts: list[str] = []
        if prob_up_str != "—":
            parts.append(f"↑ {prob_up_str}")
        if prob_down_str != "—":
            parts.append(f"↓ {prob_down_str}")
        self.metric_vars["direction"].set("   ".join(parts) if parts else "—")

        if explanation and isinstance(explanation, Mapping):
            summary = explanation.get("summary")
            if summary:
                self._set_status(summary)

        self._recompute_pnl()

    def _update_price_chart(self) -> None:
        self.price_ax.clear()
        prediction = self.current_prediction or {}
        forecast = self._compute_forecast_date(
            prediction.get("target_date") or prediction.get("horizon")
        )
        title = "Forecast date: —"
        if forecast is not None:
            title = f"Forecast date: {forecast.date().isoformat()}"
        self.price_ax.set_title(title)
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
            rate = self._currency_rate()
            series = series * rate
            plotted_series = pd.Series(series.to_numpy(), index=pd.Index(x_values)).dropna()
            close_label = "Close (price)" if not self.currency_symbol else f"Close ({self.currency_symbol})"
            self.price_ax.plot(plotted_series.index, plotted_series.values, label=close_label, color="tab:blue")
            if not plotted_series.empty:
                last_x = plotted_series.index[-1]
                last_y = plotted_series.iloc[-1]
                last_display = fmt_ccy(last_y, self.currency_symbol, decimals=self.price_decimal_places)
                self.price_ax.scatter([last_x], [last_y], color="tab:blue", zorder=5)
                self.price_ax.annotate(
                    f"Last: {last_display}",
                    xy=(last_x, last_y),
                    xytext=(8, 0),
                    textcoords="offset points",
                    va="center",
                    ha="left",
                    color="tab:blue",
                )
            predicted_close = _safe_float(prediction.get("predicted_close"))
            if predicted_close is not None:
                converted_prediction = predicted_close * rate
                predicted_label = (
                    "Predicted close (price)"
                    if not self.currency_symbol
                    else f"Predicted close ({self.currency_symbol})"
                )
                self.price_ax.axhline(
                    converted_prediction, color="tab:orange", linestyle="--", label=predicted_label
                )
                if not plotted_series.empty:
                    annotate_x = plotted_series.index[-1]
                elif hasattr(x_values, "iloc"):
                    annotate_x = x_values.iloc[-1]
                else:
                    annotate_x = x_values[-1]
                self.price_ax.annotate(
                    fmt_ccy(
                        converted_prediction,
                        self.currency_symbol,
                        decimals=self.price_decimal_places,
                    ),
                    xy=(annotate_x, converted_prediction),
                    xytext=(8, 0),
                    textcoords="offset points",
                    va="center",
                    ha="left",
                    color="tab:orange",
                )
            ylabel = "Price"
            if self.currency_symbol:
                ylabel = f"Price ({self.currency_symbol})"
            self.price_ax.set_ylabel(ylabel)
            self.price_ax.grid(True, linestyle="--", alpha=0.3)
            self.price_ax.legend(loc="best")
            self.price_ax.xaxis.set_major_formatter(mdates.DateFormatter("%b-%Y"))
        else:
            self.price_ax.text(0.5, 0.5, "Price history unavailable", ha="center", va="center")
        self.price_figure.tight_layout()
        self._style_figure(self.price_figure)
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
                display_value = self._format_price_like_value(column, numeric)
                self.indicator_tree.insert(
                    "",
                    tk.END,
                    iid=str(column),
                    values=(str(column), display_value, str(category)),
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
                    display_value = self._format_price_like_value(name, numeric)
                    self.indicator_tree.insert(
                        "",
                        tk.END,
                        iid=name,
                        values=(name, display_value, "indicator"),
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
        self._style_figure(self.feature_figure)
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
    def _refresh_numeric_views(self) -> None:
        metrics_updated = False
        if hasattr(self, "metric_vars"):
            metrics_updated = True
            self._update_metrics()
        if hasattr(self, "price_ax"):
            self._update_price_chart()
        if hasattr(self, "indicator_tree"):
            self._update_indicator_view()
        if not metrics_updated and hasattr(self, "pnl_label"):
            self._recompute_pnl()

    def _currency_label(self, mode: str | None = None) -> str:
        profile = self.currency_profiles.get(mode or self.currency_mode, {})
        return str(profile.get("label") or "")

    def _currency_symbol(self, mode: str | None = None) -> str:
        profile = self.currency_profiles.get(mode or self.currency_mode, {})
        symbol = str(profile.get("symbol") or "").strip()
        if not symbol:
            label = str(profile.get("label") or "")
            symbol = _detect_currency_symbol(label, fallback="")
        return symbol

    def _currency_rate(self, mode: str | None = None) -> float:
        return float(self.currency_rates.get(mode or self.currency_mode, 1.0))

    def _set_currency_rate(self, mode: str, rate: float) -> None:
        if rate <= 0:
            raise ValueError("FX rate must be positive.")
        if mode not in self.currency_rates:
            raise ValueError(f"Unknown currency mode '{mode}'")
        self.currency_rates[mode] = rate

    def _convert_currency(self, value: Any) -> float | None:
        numeric = _safe_float(value)
        if numeric is None:
            return None
        return numeric * self._currency_rate()

    def _update_pnl_visibility(self) -> None:
        if not hasattr(self, "pnl_label"):
            return
        if self.show_pnl_var.get():
            if not self.pnl_label.winfo_ismapped():
                self.pnl_label.pack(**getattr(self, "_pnl_pack_options", {}))
        else:
            if self.pnl_label.winfo_manager():
                self.pnl_label.pack_forget()

    def _apply_theme(self) -> None:
        enabled = self.dark_mode_enabled
        mpl_style.use("dark_background" if enabled else "default")
        if enabled:
            bg = "#1e1e1e"
            fg = "#f0f0f0"
            field_bg = "#262626"
            accent = "#3a7bd5"
            self.style.theme_use("clam")
            self.style.configure("TFrame", background=bg)
            self.style.configure("TLabelframe", background=bg, foreground=fg)
            self.style.configure("TLabelframe.Label", background=bg, foreground=fg)
            self.style.configure("TLabel", background=bg, foreground=fg)
            self.style.configure("TButton", background=bg, foreground=fg)
            self.style.configure("TCheckbutton", background=bg, foreground=fg)
            self.style.configure("TNotebook", background=bg)
            self.style.configure("TNotebook.Tab", background=bg, foreground=fg)
            self.style.configure(
                "Treeview",
                background=field_bg,
                fieldbackground=field_bg,
                foreground=fg,
                bordercolor=field_bg,
            )
            self.style.configure("Treeview.Heading", background=bg, foreground=fg)
            self.style.map("Treeview", background=[("selected", accent)], foreground=[("selected", fg)])
            self.style.configure("TCombobox", fieldbackground=field_bg, foreground=fg, background=field_bg)
            self.style.map("TCombobox", fieldbackground=[("readonly", field_bg)])
            self.style.configure("TEntry", fieldbackground=field_bg, foreground=fg, background=field_bg)
            self.style.configure("TMenubutton", background=bg, foreground=fg)
            self.style.configure("TSpinbox", background=field_bg, foreground=fg)
            self.style.configure("TProgressbar", background=accent, troughcolor=field_bg)
            root_bg = bg
        else:
            self.style.theme_use(self.default_theme)
            for style_name in (
                "TFrame",
                "TLabelframe",
                "TLabelframe.Label",
                "TLabel",
                "TButton",
                "TCheckbutton",
                "TNotebook",
                "TNotebook.Tab",
                "Treeview",
                "Treeview.Heading",
                "TCombobox",
                "TEntry",
                "TMenubutton",
                "TSpinbox",
                "TProgressbar",
            ):
                self.style.configure(style_name, background="", foreground="", fieldbackground="")
            self.style.map("Treeview", background="", foreground="")
            self.style.map("TCombobox", fieldbackground="")
            field_bg = self.style.lookup("TEntry", "fieldbackground") or "white"
            fg = self.style.lookup("TLabel", "foreground") or "black"
            root_bg = self.style.lookup("TFrame", "background") or "SystemButtonFace"
        self.root.configure(bg=root_bg)
        text_bg = field_bg if enabled else "white"
        text_fg = fg
        for widget in self.text_widgets:
            widget.configure(
                bg=text_bg,
                fg=text_fg,
                insertbackground=text_fg,
                highlightbackground=root_bg,
                highlightcolor=root_bg,
            )
        for figure in (
            getattr(self, "price_figure", None),
            getattr(self, "indicator_figure", None),
            getattr(self, "feature_figure", None),
        ):
            self._style_figure(figure)
        for canvas in (
            getattr(self, "price_canvas", None),
            getattr(self, "indicator_canvas", None),
            getattr(self, "feature_canvas", None),
        ):
            if canvas is not None:
                canvas.draw_idle()

    def _style_figure(self, figure: Figure | None) -> None:
        if figure is None:
            return
        enabled = self.dark_mode_enabled
        fig_bg = "#1e1e1e" if enabled else "white"
        ax_bg = "#242424" if enabled else "white"
        fg = "#f0f0f0" if enabled else "#000000"
        figure.patch.set_facecolor(fig_bg)
        for ax in figure.axes:
            ax.set_facecolor(ax_bg)
            ax.tick_params(colors=fg)
            ax.yaxis.label.set_color(fg)
            ax.xaxis.label.set_color(fg)
            ax.title.set_color(fg)
            for spine in ax.spines.values():
                spine.set_color(fg)
            grid_color = "#444444" if enabled else "#cccccc"
            for grid_line in ax.get_xgridlines() + ax.get_ygridlines():
                grid_line.set_color(grid_color)
            legend = ax.get_legend()
            if legend:
                legend.get_frame().set_facecolor(ax_bg)
                legend.get_frame().set_edgecolor(fg)
                for text in legend.get_texts():
                    text.set_color(fg)

    def _recompute_pnl(self) -> None:
        try:
            size = int(self.position_size_var.get())
        except (tk.TclError, ValueError):
            size = 1
        min_size, max_size = 1, 1_000_000
        clamped = max(min_size, min(max_size, size))
        if clamped != size:
            self.position_size_var.set(clamped)
            return
        size = clamped
        share_label = "share" if size == 1 else "shares"
        prefix = f"Expected P&L for {size:,} {share_label}: "

        prediction = self.current_prediction or {}
        last_raw = _safe_float(prediction.get("last_close"))
        predicted_raw = _safe_float(prediction.get("predicted_close"))
        if last_raw is None or predicted_raw is None:
            self.pnl_var.set(prefix + "—")
            return

        last_converted = self._convert_currency(last_raw)
        predicted_converted = self._convert_currency(predicted_raw)
        if last_converted is None or predicted_converted is None:
            self.pnl_var.set(prefix + "—")
            return

        pnl_value = (predicted_converted - last_converted) * size
        pct_change = _safe_float(prediction.get("expected_change_pct"))
        if pct_change is None and last_raw != 0:
            pct_change = (predicted_raw - last_raw) / last_raw

        pnl_display = fmt_ccy(pnl_value, self.currency_symbol, decimals=self.price_decimal_places)
        pct_display = fmt_pct(pct_change, show_sign=True)
        if pct_display == "—":
            self.pnl_var.set(prefix + pnl_display)
        else:
            self.pnl_var.set(f"{prefix}{pnl_display} ({pct_display})")

    def _format_price_like_value(self, name: Any, value: Any) -> str:
        numeric = _safe_float(value)
        if numeric is None:
            return "—"
        label = str(name).lower()
        if any(keyword in label for keyword in ("price", "close", "open", "high", "low")):
            converted = self._convert_currency(numeric)
            return fmt_ccy(converted, self.currency_symbol, decimals=self.price_decimal_places)
        decimals = self.price_decimal_places
        return f"{numeric:.{decimals}f}"

    def _set_text(self, widget: tk.Text, value: str) -> None:
        widget.configure(state=tk.NORMAL)
        widget.delete("1.0", tk.END)
        widget.insert(tk.END, value)
        widget.configure(state=tk.DISABLED)

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
