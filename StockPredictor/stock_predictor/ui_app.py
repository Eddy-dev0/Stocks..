"""Tkinter desktop application for the stock predictor platform."""

from __future__ import annotations

import logging
import os
import re
import threading
import tkinter as tk
from dataclasses import dataclass
from tkinter import messagebox, ttk
from typing import Any, Callable, Iterable, Mapping

import numpy as np
import pandas as pd
import yfinance as yf
from pandas.tseries.offsets import BDay
from matplotlib import style as mpl_style
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.dates import DateFormatter

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


CURRENCY_SYMBOL_TO_CODE: dict[str, str] = {
    "$": "USD",
    "€": "EUR",
    "£": "GBP",
    "¥": "JPY",
    "₹": "INR",
    "₩": "KRW",
    "₽": "RUB",
    "₪": "ILS",
    "₫": "VND",
    "₱": "PHP",
    "฿": "THB",
    "₦": "NGN",
    "₴": "UAH",
    "₭": "LAK",
    "₲": "PYG",
    "₵": "GHS",
}


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


HORIZON_SUMMARY_PHRASES: dict[str, str] = {
    "Tomorrow": "for tomorrow",
    "1 Week": "for the next week",
    "1 Month": "for the next month",
    "3 Months": "for the next three months",
}


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
        self.root.minsize(1024, 640)
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        self.style = ttk.Style()
        self.light_theme = "vista"
        self.dark_theme = "clam"
        try:
            self.style.theme_use(self.light_theme)
        except tk.TclError:
            self.light_theme = self.dark_theme
            self.style.theme_use(self.dark_theme)
        root_bg = self.style.lookup("TFrame", "background")
        self.root.configure(background=root_bg)
        self._style_targets = (
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
        )
        self._default_style_options = self._capture_style_defaults(self._style_targets)
        self._default_style_maps = {
            "Treeview": {key: list(value) for key, value in self.style.map("Treeview").items()},
            "TCombobox": {
                key: list(value) for key, value in self.style.map("TCombobox").items()
            },
        }

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
        self.horizon_labels = [option.label for option in self.horizon_options]
        self.horizon_map = {
            option.label: {"code": option.code, "offset": option.business_days}
            for option in self.horizon_options
        }
        self.horizon_code_to_label = {option.code: option.label for option in self.horizon_options}
        self.horizon_offset_to_label = {
            option.business_days: option.label for option in self.horizon_options
        }
        self.selected_horizon_label: str = ""
        self.selected_horizon_code: str = ""
        self.selected_horizon_offset: int = 0
        initial_label = self._resolve_initial_horizon_label(horizons)
        self._apply_horizon_selection(initial_label)
        self.current_prediction: dict[str, Any] = {}
        self.price_history: pd.DataFrame | None = None
        self.feature_snapshot: pd.DataFrame | None = None
        self.feature_history: pd.DataFrame | None = None
        self.indicator_history: pd.DataFrame | None = None
        self.price_history_converted: pd.DataFrame | None = None
        self.feature_snapshot_converted: pd.DataFrame | None = None
        self.feature_history_converted: pd.DataFrame | None = None
        self.indicator_history_converted: pd.DataFrame | None = None
        self.feature_toggle_vars: dict[str, tk.BooleanVar] = {}
        self.forecast_date_var = tk.StringVar(value="Forecast date: —")
        self.current_forecast_date: pd.Timestamp | None = None
        self.market_holidays: list[pd.Timestamp] = []

        self.currency_mode_var = tk.StringVar(value="local")
        self.currency_button_text = tk.StringVar(value=self.currency_symbol)
        self.currency_rate_var = tk.StringVar(value=f"{self._currency_rate('usd'):.4f}")
        self._suspend_rate_updates = False

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
        self.apply_theme(self.dark_mode_var.get())
        self.root.after(200, self._initialise_prediction)

    # ------------------------------------------------------------------
    # Layout helpers
    # ------------------------------------------------------------------
    def _capture_style_defaults(self, names: Iterable[str]) -> dict[str, dict[str, str]]:
        defaults: dict[str, dict[str, str]] = {}
        for name in names:
            options: dict[str, str] = {}
            for option in ("background", "foreground", "fieldbackground"):
                value = self.style.lookup(name, option)
                if value:
                    options[option] = value
            if options:
                defaults[name] = options
        return defaults

    def _resolve_initial_horizon_label(self, horizons: Iterable[int]) -> str:
        for value in horizons:
            try:
                numeric = int(value)
            except (TypeError, ValueError):
                continue
            label = self.horizon_offset_to_label.get(numeric)
            if label:
                return label
        return self.horizon_labels[0]

    def _apply_horizon_selection(self, label: str) -> None:
        mapping = self.horizon_map.get(label)
        if mapping is None:
            return
        self.selected_horizon_label = label
        self.selected_horizon_code = str(mapping.get("code") or "")
        try:
            self.selected_horizon_offset = int(mapping.get("offset", 0))
        except (TypeError, ValueError):
            self.selected_horizon_offset = 0

    def _horizon_summary_phrase(self, label: str | None = None) -> str:
        key = label or self.selected_horizon_label
        return HORIZON_SUMMARY_PHRASES.get(key, "").strip()

    def _sync_horizon_from_prediction(self, prediction: Mapping[str, Any] | None) -> None:
        if not isinstance(prediction, Mapping):
            return
        label: str | None = None
        horizon_value = prediction.get("horizon")
        if isinstance(horizon_value, str):
            label = self.horizon_code_to_label.get(horizon_value)
        else:
            try:
                numeric = int(horizon_value)
            except (TypeError, ValueError):
                numeric = None
            if numeric is not None:
                label = self.horizon_offset_to_label.get(numeric)
        if label and label in self.horizon_map:
            self._apply_horizon_selection(label)
            if hasattr(self, "horizon_var") and self.horizon_var.get() != label:
                self.horizon_var.set(label)

    def _resolve_market_holidays(
        self, prediction: Mapping[str, Any] | None = None
    ) -> list[pd.Timestamp]:
        holidays: set[pd.Timestamp] = set()
        sources: list[Any] = []
        active_prediction: Mapping[str, Any] | None = prediction
        if active_prediction is None:
            active_prediction = self.current_prediction if isinstance(self.current_prediction, Mapping) else {}
        if isinstance(active_prediction, Mapping):
            for key in ("market_holidays", "trading_holidays", "holidays"):
                value = active_prediction.get(key)
                if value:
                    sources.append(value)
        metadata = getattr(getattr(self, "application", None), "pipeline", None)
        metadata_mapping = getattr(metadata, "metadata", None)
        if isinstance(metadata_mapping, Mapping):
            for key in ("market_holidays", "trading_holidays", "holidays"):
                value = metadata_mapping.get(key)
                if value:
                    sources.append(value)

        for source in sources:
            for item in self._iter_holiday_values(source):
                try:
                    converted = pd.to_datetime(item, errors="coerce")
                except Exception:
                    continue
                if isinstance(converted, (pd.Series, pd.DatetimeIndex)):
                    iterable = [val for val in converted.dropna()]
                elif isinstance(converted, np.ndarray):
                    iterable = [val for val in converted if pd.notna(val)]
                else:
                    iterable = [converted] if pd.notna(converted) else []
                for val in iterable:
                    try:
                        holidays.add(pd.Timestamp(val).normalize())
                    except Exception:
                        continue
        return sorted(holidays)

    def _build_layout(self, horizons: Iterable[int]) -> None:
        self.toolbar = self._build_toolbar(list(horizons))
        self._build_notebook()
        self.statusbar = self._build_statusbar()
        self._on_currency_mode_changed(self.currency_mode_var.get())

    def _build_toolbar(self, _horizons: list[int]) -> ttk.Frame:
        toolbar = ttk.Frame(self.root, padding=(12, 6))
        toolbar.grid(row=0, column=0, sticky="ew")

        self.horizon_var = tk.StringVar(value=self.selected_horizon_label)
        self.horizon_box = ttk.Combobox(
            toolbar,
            width=12,
            state="readonly",
            textvariable=self.horizon_var,
            values=self.horizon_labels,
        )
        self.horizon_box.bind("<<ComboboxSelected>>", self._on_horizon_changed)

        self.ticker_entry = ttk.Entry(toolbar, width=10, textvariable=self.ticker_var)
        self.ticker_entry.bind("<Return>", self._on_ticker_submitted)
        self.ticker_entry.bind("<FocusOut>", self._on_ticker_focus_out)
        self.ticker_apply_button = ttk.Button(
            toolbar,
            text="Apply",
            command=self._on_ticker_button,
        )
        Tooltip(self.ticker_entry, "Use official symbols (e.g., RHM.DE, ^GSPC).")

        self.position_spinbox = ttk.Spinbox(
            toolbar,
            from_=1,
            to=10_000,
            increment=1,
            width=8,
            textvariable=self.position_size_var,
            command=self._recompute_pnl,
        )
        self.position_spinbox.bind("<FocusOut>", lambda _event: self._recompute_pnl())
        Tooltip(self.position_spinbox, "Number of shares used for P&L estimates.")

        self.currency_menu_button = ttk.Menubutton(
            toolbar,
            textvariable=self.currency_button_text,
            direction="below",
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
        self.currency_menu.add_separator()
        self.currency_menu.add_command(
            label="Update rate",
            command=lambda: self._on_fetch_fx_rate(silent=False),
        )
        self.currency_menu_button.configure(menu=self.currency_menu)
        self.currency_menu_button.configure(width=4)

        self.fx_rate_entry = ttk.Entry(
            toolbar,
            width=8,
            textvariable=self.currency_rate_var,
            state=tk.DISABLED,
        )
        self.fx_rate_entry.bind("<Return>", self._on_currency_rate_submit)

        self.refresh_button = ttk.Button(toolbar, text="Refresh data", command=self._on_refresh)
        self.predict_button = ttk.Button(toolbar, text="Run prediction", command=self._on_predict)

        self.forecast_label = ttk.Label(toolbar, textvariable=self.forecast_date_var)

        widgets_in_order: list[tk.Widget] = [
            self.horizon_box,
            self.ticker_entry,
            self.ticker_apply_button,
            self.position_spinbox,
            self.currency_menu_button,
            self.fx_rate_entry,
            self.refresh_button,
            self.predict_button,
        ]

        for column, widget in enumerate(widgets_in_order):
            widget.grid(row=0, column=column, sticky=tk.W)

        self.forecast_label.grid(row=0, column=len(widgets_in_order), sticky=tk.W)

        for index in range(len(widgets_in_order)):
            toolbar.grid_columnconfigure(index, weight=0)
        toolbar.grid_columnconfigure(len(widgets_in_order), weight=1)

        for child in toolbar.winfo_children():
            child.grid_configure(padx=4, pady=2)

        self._update_forecast_label()

        return toolbar

    def _build_notebook(self) -> None:
        self.notebook = ttk.Notebook(self.root)
        self.notebook.grid(row=1, column=0, sticky="nsew")

        self._build_overview_tab()
        self._build_indicators_tab()
        self._build_explanation_tab()
        self._build_settings_tab()

    def _build_statusbar(self) -> None:
        status_frame = ttk.Frame(self.root, padding=(12, 4))
        status_frame.grid(row=2, column=0, sticky="ew")
        status_frame.grid_columnconfigure(0, weight=1)
        status_frame.grid_columnconfigure(1, weight=0)
        self.status_var = tk.StringVar(value="Ready")
        self.status_label = ttk.Label(status_frame, textvariable=self.status_var, anchor=tk.W)
        self.status_label.grid(row=0, column=0, sticky=tk.W)
        self.progress = ttk.Progressbar(status_frame, mode="indeterminate", length=120)
        self.progress.grid(row=0, column=1, sticky=tk.E, padx=(8, 0))
        return status_frame

    # ------------------------------------------------------------------
    # Tab builders
    # ------------------------------------------------------------------
    def _build_overview_tab(self) -> None:
        frame = ttk.Frame(self.notebook, padding=12)
        self.notebook.add(frame, text="Overview")
        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(0, weight=1)

        overview = ttk.Panedwindow(frame, orient="vertical")
        overview.grid(row=0, column=0, sticky="nsew")
        overview.grid_rowconfigure(0, weight=1)
        overview.grid_columnconfigure(0, weight=1)

        summary_frame = ttk.Frame(overview, padding=(4, 4, 4, 12))
        chart_container = ttk.Frame(overview)
        overview.add(summary_frame, weight=0)
        overview.add(chart_container, weight=1)

        self.metric_vars: dict[str, tk.StringVar] = {}
        metric_specs = [
            ("ticker", "Ticker"),
            ("as_of", "Market data as of"),
            ("last_close", "Last close"),
            ("predicted_close", "Predicted close"),
            ("expected_change", "Expected change"),
            ("direction", "Direction"),
        ]
        for column in range(4):
            weight = 1 if column % 2 == 1 else 0
            summary_frame.grid_columnconfigure(column, weight=weight)
        for idx, (key, label) in enumerate(metric_specs):
            row = idx // 2
            column = (idx % 2) * 2
            caption = ttk.Label(summary_frame, text=f"{label}:", anchor=tk.E)
            caption.grid(row=row, column=column, sticky=tk.E, padx=(4, 8), pady=2)
            var = tk.StringVar(value="—")
            value = ttk.Label(
                summary_frame,
                textvariable=var,
                anchor=tk.E,
                font=("TkDefaultFont", 10, "bold"),
            )
            value.grid(row=row, column=column + 1, sticky=tk.E, padx=(0, 8), pady=2)
            self.metric_vars[key] = var

        self.pnl_label = ttk.Label(
            summary_frame,
            textvariable=self.pnl_var,
            anchor=tk.W,
            font=("TkDefaultFont", 10, "bold"),
        )
        self._pnl_grid_options = {"row": 3, "column": 0, "columnspan": 4, "sticky": tk.W, "pady": (12, 0)}
        self.pnl_label.grid(**self._pnl_grid_options)

        chart_container.grid_rowconfigure(0, weight=1)
        chart_container.grid_columnconfigure(0, weight=1)
        chart_frame = ttk.LabelFrame(chart_container, text="Price history", padding=8)
        chart_frame.grid(row=0, column=0, sticky="nsew")
        chart_frame.grid_rowconfigure(0, weight=1)
        chart_frame.grid_columnconfigure(0, weight=1)
        self.price_figure = Figure(figsize=(8, 4.8), dpi=100, constrained_layout=True)
        self.price_figure.patch.set_facecolor("white")
        self.price_ax = self.price_figure.add_subplot(111)
        self.price_ax.set_facecolor("white")
        self.price_canvas = FigureCanvasTkAgg(self.price_figure, master=chart_frame)
        self.price_canvas_widget = self.price_canvas.get_tk_widget()
        self.price_canvas_widget.grid(row=0, column=0, sticky="nsew")
        self.price_chart_message = ttk.Label(
            chart_frame, text="No data loaded yet", anchor=tk.CENTER, justify=tk.CENTER
        )

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
        if label not in self.horizon_map:
            messagebox.showwarning("Invalid horizon", "Please select a valid horizon value.")
            self.horizon_var.set(self.selected_horizon_label)
            return
        self._apply_horizon_selection(label)
        self._update_forecast_label()
        self._refresh_overview()
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
        self.apply_theme(enabled)
        self._refresh_numeric_views()

    def _on_currency_mode_changed(self, mode: str) -> None:
        previous_mode = getattr(self, "currency_mode", "local")
        if mode not in self.currency_profiles:
            LOGGER.debug("Unsupported currency mode '%s'; defaulting to local", mode)
            mode = "local"
        changed = mode != previous_mode
        self.currency_mode = mode
        self.currency_symbol = self._currency_symbol(mode)
        button_display = self.currency_symbol or self._currency_label(mode)
        self.currency_button_text.set(button_display)
        if hasattr(self, "currency_default_var"):
            display_label = self.currency_display_map.get(mode, "Local")
            if self.currency_default_var.get() != display_label:
                self.currency_default_var.set(display_label)
        if self._busy:
            entry_state = tk.DISABLED
        elif mode == "local":
            entry_state = tk.DISABLED
        else:
            entry_state = tk.NORMAL
        self.fx_rate_entry.configure(state=entry_state)
        rate = self._currency_rate(mode)
        if not changed:
            self._set_currency_rate_var(rate)
            self._on_currency_changed(mode, rate)
            return
        if mode == "local":
            self._set_currency_rate_var(rate)
            self._on_currency_changed(mode, rate)
            return
        updated_rate = self._on_fetch_fx_rate(mode, silent=True, apply=False)
        if updated_rate is None:
            self._set_currency_rate_var(rate)
            self._on_currency_changed(mode, rate)
        else:
            self._on_currency_changed(mode, updated_rate)

    def _on_currency_rate_submit(self, _event: Any) -> None:
        self._apply_manual_fx_rate()

    def _apply_manual_fx_rate(self) -> None:
        mode = self.currency_mode_var.get()
        if mode == "local":
            messagebox.showinfo(
                "Local currency",
                "Local currency does not require a conversion rate.",
            )
            self._set_currency_rate_var(self._currency_rate(mode))
            return
        raw = self.currency_rate_var.get()
        try:
            rate = float(raw)
        except (TypeError, ValueError):
            messagebox.showwarning(
                "Invalid FX rate", "Please provide a numeric FX conversion rate."
            )
            self._set_currency_rate_var(self._currency_rate(mode))
            return
        if rate <= 0:
            messagebox.showwarning("Invalid FX rate", "FX conversion rate must be positive.")
            self._set_currency_rate_var(self._currency_rate(mode))
            return
        self._set_currency_rate(mode, rate)
        self._set_currency_rate_var(self._currency_rate(mode))
        self._on_fx_rate_changed(mode, rate)
        self._set_status(f"FX rate updated to {rate:.4f}.")

    def _on_position_size_changed(self, *_args: Any) -> None:
        self._refresh_overview()

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
        feature_history = (
            self.feature_history_converted
            if isinstance(self.feature_history_converted, pd.DataFrame)
            else self.feature_history
        )
        indicator_history = (
            self.indicator_history_converted
            if isinstance(self.indicator_history_converted, pd.DataFrame)
            else self.indicator_history
        )
        if isinstance(feature_history, pd.DataFrame) and indicator in feature_history.columns:
            series = pd.to_numeric(feature_history[indicator], errors="coerce").dropna()
        elif isinstance(indicator_history, pd.DataFrame):
            columns = {str(column).lower(): column for column in indicator_history.columns}
            indicator_col = columns.get("indicator")
            value_col = columns.get("value")
            date_col = columns.get("date")
            if indicator_col and value_col:
                subset = indicator_history[indicator_history[indicator_col] == indicator]
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
        self.market_holidays = []
        self.current_forecast_date = None
        self.forecast_date_var.set("Forecast date: —")
        self._refresh_overview()
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
        horizon_arg: Any
        if self.selected_horizon_code:
            horizon_arg = self.selected_horizon_code
        else:
            horizon_arg = self.selected_horizon_offset
        prediction = self.application.predict(horizon=horizon_arg)
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
        self._sync_horizon_from_prediction(self.current_prediction)
        self.market_holidays = self._resolve_market_holidays(self.current_prediction)
        self._apply_currency(self._currency_rate())
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
            self.position_spinbox,
            self.currency_menu_button,
        ):
            widget.configure(state=state)
        if busy:
            self.fx_rate_entry.configure(state=tk.DISABLED)
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

    def _compute_forecast_date(
        self, target_date: Any | None = None, *, offset: int | None = None
    ) -> pd.Timestamp | None:
        parsed = pd.to_datetime(target_date, errors="coerce") if target_date is not None else None
        forecast: pd.Timestamp | None = None
        if pd.notna(parsed):
            forecast = pd.Timestamp(parsed).normalize()
        else:
            days = offset if offset is not None else self.selected_horizon_offset
            try:
                days_int = int(days)
            except (TypeError, ValueError):
                days_int = 0
            if days_int > 0:
                base_date = self._forecast_base_date()
                holidays = self.market_holidays or self._resolve_market_holidays()
                if not self.market_holidays and holidays:
                    self.market_holidays = list(holidays)
                try:
                    if holidays:
                        business_offset = BDay(n=days_int, holidays=holidays)
                    else:
                        business_offset = BDay(n=days_int)
                    forecast = (base_date + business_offset).normalize()
                except Exception:
                    forecast = (base_date + pd.to_timedelta(days_int, unit="D")).normalize()
        self.current_forecast_date = forecast
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

        forecast = self.current_forecast_date
        if forecast is None:
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
        ax = self.price_ax
        canvas = self.price_canvas
        canvas_widget = getattr(self, "price_canvas_widget", None)
        empty_label = getattr(self, "price_chart_message", None)

        def show_empty_state() -> None:
            if canvas_widget is not None:
                canvas_widget.grid_remove()
            if empty_label is not None:
                empty_label.grid(row=0, column=0, sticky="nsew")

        ax.clear()
        prediction = self.current_prediction or {}
        forecast = self.current_forecast_date
        if forecast is None:
            fallback_target = prediction.get("target_date") or prediction.get("forecast_date")
            forecast = self._compute_forecast_date(fallback_target)
        title = "Forecast date: —"
        if forecast is not None:
            title = f"Forecast date: {forecast.date().isoformat()}"
        ax.set_title(title)

        frame_source: pd.DataFrame | None = None
        if isinstance(self.price_history_converted, pd.DataFrame) and not self.price_history_converted.empty:
            frame_source = self.price_history_converted
        elif isinstance(self.price_history, pd.DataFrame) and not self.price_history.empty:
            frame_source = self.price_history

        if not isinstance(frame_source, pd.DataFrame) or frame_source.empty:
            show_empty_state()
            self._style_figure(self.price_figure)
            canvas.draw_idle()
            return

        frame = frame_source.copy()
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
            numeric_columns = frame.select_dtypes(include=[np.number]).columns
            if not len(numeric_columns):
                show_empty_state()
                self._style_figure(self.price_figure)
                canvas.draw_idle()
                return
            close_column = numeric_columns[0]

        series = pd.to_numeric(frame[close_column], errors="coerce")
        plotted_series = pd.Series(series.to_numpy(), index=pd.Index(x_values)).dropna()
        if plotted_series.empty:
            show_empty_state()
            self._style_figure(self.price_figure)
            canvas.draw_idle()
            return

        if empty_label is not None:
            empty_label.grid_remove()
        if canvas_widget is not None:
            canvas_widget.grid()

        close_label = "Close (price)" if not self.currency_symbol else f"Close ({self.currency_symbol})"
        ax.plot(plotted_series.index, plotted_series.values, label=close_label, color="tab:blue")

        last_x = plotted_series.index[-1]
        last_y = plotted_series.iloc[-1]
        last_display = fmt_ccy(last_y, self.currency_symbol, decimals=self.price_decimal_places)
        ax.scatter([last_x], [last_y], color="tab:blue", zorder=5)
        ax.annotate(
            f"Last: {last_display}",
            xy=(last_x, last_y),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            ha="left",
            color="tab:blue",
        )

        predicted_close = _safe_float(prediction.get("predicted_close"))
        converted_prediction = self._convert_currency(predicted_close) if predicted_close is not None else None
        if converted_prediction is not None:
            predicted_label = (
                "Predicted close (price)"
                if not self.currency_symbol
                else f"Predicted close ({self.currency_symbol})"
            )
            if forecast is not None:
                line_end = forecast
            else:
                line_end = pd.to_datetime(last_x) + pd.Timedelta(days=1)
            ax.plot(
                [last_x, line_end],
                [converted_prediction, converted_prediction],
                color="tab:orange",
                linestyle="--",
                label=predicted_label,
            )
            annotation_text = fmt_ccy(
                converted_prediction,
                self.currency_symbol,
                decimals=self.price_decimal_places,
            )
            if forecast is not None:
                annotation_text = f"{annotation_text} ({forecast.date().isoformat()})"
            ax.annotate(
                annotation_text,
                xy=(pd.to_datetime(line_end), converted_prediction),
                xytext=(8, 0),
                textcoords="offset points",
                va="center",
                ha="left",
                color="tab:orange",
            )

        ylabel = "Price"
        if self.currency_symbol:
            ylabel = f"Price ({self.currency_symbol})"
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(loc="best")
        ax.xaxis.set_major_formatter(DateFormatter("%b-%Y"))

        self._style_figure(self.price_figure)
        canvas.draw_idle()

    def _update_indicator_view(self) -> None:
        for item in self.indicator_tree.get_children():
            self.indicator_tree.delete(item)
        snapshot_frame = (
            self.feature_snapshot_converted
            if isinstance(self.feature_snapshot_converted, pd.DataFrame)
            else self.feature_snapshot
        )
        if snapshot_frame is not None and not snapshot_frame.empty:
            row = snapshot_frame.iloc[0]
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
        if not self.indicator_tree.get_children():
            indicator_frame = (
                self.indicator_history_converted
                if isinstance(self.indicator_history_converted, pd.DataFrame)
                else self.indicator_history
            )
        else:
            indicator_frame = None
        if not self.indicator_tree.get_children() and isinstance(indicator_frame, pd.DataFrame):
            frame = indicator_frame
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

        summary_raw = explanation.get("summary")
        summary_text = str(summary_raw).strip() if summary_raw else ""
        phrase = self._horizon_summary_phrase()
        if phrase:
            if summary_text:
                summary_text = f"Outlook {phrase}: {summary_text}"
            else:
                summary_text = f"Outlook {phrase}: No explanation available."
        elif not summary_text:
            summary_text = "No explanation available."
        self._set_text(self.summary_text, summary_text)
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
        message = "Explanation unavailable."
        phrase = self._horizon_summary_phrase()
        if phrase:
            message = f"Outlook {phrase}: {message}"
        self._set_text(self.summary_text, message)
        for widget in self.reason_lists.values():
            self._set_text(widget, "No data.")
        for item in self.feature_tree.get_children():
            self.feature_tree.delete(item)
        self._update_feature_chart([])

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def _iter_holiday_values(self, value: Any) -> list[Any]:
        if value is None:
            return []
        if isinstance(value, (str, bytes)):
            return [value]
        if isinstance(value, Mapping):
            return list(value.values())
        if isinstance(value, pd.Series):
            return value.dropna().tolist()
        if isinstance(value, pd.DataFrame):
            return value.dropna().to_numpy().ravel().tolist()
        if isinstance(value, Iterable):
            return list(value)
        return [value]

    def _refresh_overview(self) -> bool:
        refreshed = False
        if hasattr(self, "metric_vars"):
            self._update_metrics()
            refreshed = True
        elif hasattr(self, "pnl_label"):
            self._recompute_pnl()
            refreshed = True
        return refreshed

    def _refresh_numeric_views(self) -> None:
        overview_refreshed = self._refresh_overview()
        if hasattr(self, "price_ax"):
            self._update_price_chart()
        if hasattr(self, "indicator_tree"):
            self._update_indicator_view()
        if not overview_refreshed and hasattr(self, "pnl_label"):
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

    def _set_currency_rate_var(self, rate: float) -> None:
        if self._suspend_rate_updates:
            return
        self._suspend_rate_updates = True
        try:
            self.currency_rate_var.set(f"{float(rate):.4f}")
        finally:
            self._suspend_rate_updates = False

    def _currency_code(self, mode: str) -> str | None:
        profile = self.currency_profiles.get(mode, {})
        label = str(profile.get("label") or "")
        match = re.search(r"\b([A-Z]{3})\b", label.upper())
        if match:
            return match.group(1)
        symbol = str(profile.get("symbol") or "").strip()
        if not symbol:
            symbol = _detect_currency_symbol(label, fallback="")
        if symbol:
            code = CURRENCY_SYMBOL_TO_CODE.get(symbol)
            if code:
                return code
        fallback_map = {"usd": "USD", "eur": "EUR"}
        return fallback_map.get(mode)

    def _resolve_currency_pair(self, target_mode: str) -> str | None:
        if target_mode == "local":
            return None
        base_code_raw = self._currency_code("local")
        target_code_raw = self._currency_code(target_mode)
        base_code = base_code_raw.upper() if isinstance(base_code_raw, str) else None
        target_code = target_code_raw.upper() if isinstance(target_code_raw, str) else None
        if not base_code or not target_code or base_code == target_code:
            return None
        return f"{base_code}{target_code}=X"

    def _fetch_fx_rate(self, mode: str | None = None) -> float | None:
        target_mode = mode or self.currency_mode_var.get()
        if target_mode == "local":
            return 1.0
        pair = self._resolve_currency_pair(target_mode)
        if not pair:
            fallback_rate = self._currency_rate(target_mode)
            LOGGER.warning(
                "No FX pair available for mode '%s'; falling back to stored rate %.6f",
                target_mode,
                fallback_rate,
            )
            return fallback_rate
        previous_rate = self._currency_rate(target_mode)
        try:
            data = yf.download(pair, period="5d", auto_adjust=True, progress=False)
        except Exception as exc:  # pragma: no cover - network errors
            LOGGER.warning(
                "Failed to download FX rate for %s: %s; using stored rate %.6f",
                pair,
                exc,
                previous_rate,
            )
            return previous_rate
        if data.empty:
            LOGGER.warning(
                "Empty FX dataset returned for %s; using stored rate %.6f",
                pair,
                previous_rate,
            )
            return previous_rate
        try:
            last_row = data.iloc[-1]
        except IndexError:  # pragma: no cover - defensive guard
            LOGGER.warning(
                "FX dataset for %s did not contain rows; using stored rate %.6f",
                pair,
                previous_rate,
            )
            return previous_rate
        rate: float | None = None
        for column in ("Adj Close", "Close", "close"):
            if column in last_row:
                rate = _safe_float(last_row[column])
                if rate is not None:
                    break
        if rate is None:
            rate = _safe_float(last_row.squeeze())
        if rate is None or rate <= 0:
            LOGGER.warning(
                "Invalid FX rate %s returned for %s; using stored rate %.6f",
                rate,
                pair,
                previous_rate,
            )
            return previous_rate
        return rate

    def _on_fetch_fx_rate(
        self, mode: str | None = None, *, silent: bool = False, apply: bool = True
    ) -> float | None:
        target_mode = mode or self.currency_mode_var.get()
        rate = self._fetch_fx_rate(target_mode)
        if rate is None:
            if not silent:
                messagebox.showwarning(
                    "FX rate update failed",
                    "Unable to retrieve the latest FX rate. Previous values will be used.",
                )
            return None
        try:
            self._set_currency_rate(target_mode, rate)
        except ValueError as exc:  # pragma: no cover - defensive path
            LOGGER.warning("Rejected FX rate %s for %s: %s", rate, target_mode, exc)
            return None
        self._set_currency_rate_var(rate)
        if apply:
            self._on_fx_rate_changed(target_mode, rate)
        if not silent:
            self._set_status(f"FX rate updated to {rate:.4f}.")
        return rate

    def _apply_currency(self, rate: float) -> None:
        def _convert_frame(frame: pd.DataFrame | None) -> pd.DataFrame | None:
            if not isinstance(frame, pd.DataFrame) or frame.empty:
                return None
            converted = frame.copy()
            numeric_columns: list[str] = []
            for column in converted.columns:
                series = converted[column]
                if pd.api.types.is_numeric_dtype(series):
                    numeric_columns.append(column)
                    continue
                try:
                    coerced = pd.to_numeric(series, errors="coerce")
                except Exception:  # pragma: no cover - defensive conversion
                    continue
                if coerced.isna().all():
                    continue
                converted[column] = coerced
                numeric_columns.append(column)
            if numeric_columns:
                converted.loc[:, numeric_columns] = (
                    converted.loc[:, numeric_columns].astype(float) * float(rate)
                )
            return converted

        self.price_history_converted = _convert_frame(self.price_history)
        self.feature_snapshot_converted = _convert_frame(self.feature_snapshot)
        self.feature_history_converted = _convert_frame(self.feature_history)
        self.indicator_history_converted = _convert_frame(self.indicator_history)

    def _on_currency_changed(self, mode: str, rate: float) -> None:
        self._apply_currency(rate)
        self._refresh_numeric_views()

    def _on_fx_rate_changed(self, mode: str, rate: float) -> None:
        self._apply_currency(rate)
        self._refresh_numeric_views()

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
                self.pnl_label.grid(**getattr(self, "_pnl_grid_options", {}))
        else:
            if self.pnl_label.winfo_manager():
                self.pnl_label.grid_remove()

    def _restore_style_defaults(self) -> None:
        for name in self._style_targets:
            options = self._default_style_options.get(name)
            if options:
                self.style.configure(name, **options)
        treeview_map = self._default_style_maps.get("Treeview")
        if treeview_map is not None:
            self.style.map(
                "Treeview",
                **{key: list(value) for key, value in treeview_map.items()},
            )
        combobox_map = self._default_style_maps.get("TCombobox")
        if combobox_map is not None:
            self.style.map(
                "TCombobox",
                **{key: list(value) for key, value in combobox_map.items()},
            )

    def apply_theme(self, dark: bool) -> None:
        dark_enabled = bool(dark)
        theme_name = self.dark_theme if dark_enabled else self.light_theme
        try:
            self.style.theme_use(theme_name)
        except tk.TclError:
            theme_name = self.dark_theme
            self.style.theme_use(theme_name)
        self.dark_mode_enabled = dark_enabled
        mpl_style.use("dark_background" if dark_enabled else "default")
        if dark_enabled:
            bg = "#1e1e1e"
            fg = "#f0f0f0"
            field_bg = "#262626"
            accent = "#3a7bd5"
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
            self.style.map(
                "Treeview",
                background=[("selected", accent)],
                foreground=[("selected", fg)],
            )
            self.style.configure(
                "TCombobox",
                fieldbackground=field_bg,
                foreground=fg,
                background=field_bg,
            )
            self.style.map("TCombobox", fieldbackground=[("readonly", field_bg)])
            self.style.configure(
                "TEntry",
                fieldbackground=field_bg,
                foreground=fg,
                background=field_bg,
            )
            self.style.configure("TMenubutton", background=bg, foreground=fg)
            self.style.configure("TSpinbox", background=field_bg, foreground=fg)
            self.style.configure("TProgressbar", background=accent, troughcolor=field_bg)
            root_bg = bg
        else:
            self._restore_style_defaults()
            fg = self.style.lookup("TLabel", "foreground") or "#000000"
            field_bg = (
                self.style.lookup("TEntry", "fieldbackground")
                or self.style.lookup("TEntry", "background")
                or "white"
            )
            root_bg = (
                self.style.lookup("TFrame", "background")
                or self.root.cget("background")
                or "SystemButtonFace"
            )
        self.root.configure(background=root_bg)
        text_bg = field_bg
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
