"""Tkinter desktop application for the stock predictor platform."""

from __future__ import annotations

import logging
import math
import os
import re
import threading
import tkinter as tk
from collections import defaultdict
from dataclasses import dataclass
from datetime import timedelta
from zoneinfo import ZoneInfo
from tkinter import messagebox, ttk
from typing import Any, Callable, Iterable, Mapping

import numpy as np
import pandas as pd
import yfinance as yf
from dateutil.relativedelta import relativedelta
from pandas.tseries.offsets import BDay, CustomBusinessDay
from matplotlib import colors as mcolors
from matplotlib import style as mpl_style
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.dates import DateFormatter, date2num
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from stock_predictor.app import StockPredictorApplication
from stock_predictor.core import (
    DEFAULT_PREDICTION_HORIZONS,
    PredictionResult,
    StockPredictorAI,
    TrendFinder,
    TrendInsight,
)
from stock_predictor.core.pipeline import NoPriceDataError, resolve_market_timezone
from stock_predictor.core.features import FEATURE_REGISTRY, FeatureToggles
from stock_predictor.core.preprocessing import derive_price_feature_toggles
from stock_predictor.core.modeling import InsufficientSamplesError
from stock_predictor.core.support_levels import indicator_support_floor

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


CURRENCY_CODE_TO_SYMBOL: dict[str, str] = {
    code: symbol for symbol, code in CURRENCY_SYMBOL_TO_CODE.items()
}


NON_REMOTE_PROVIDER_IDS = {"database", "local", "placeholder", "unknown"}


IMPLEMENTED_FEATURE_GROUPS: set[str] = {
    name for name, spec in FEATURE_REGISTRY.items() if spec.implemented
}


def _normalise_currency_code(value: Any) -> str | None:
    """Return a normalised ISO currency code when possible."""

    if isinstance(value, str):
        candidate = value.strip().upper()
        if re.fullmatch(r"[A-Z]{3}", candidate):
            return candidate
    return None


def _safe_float(value: Any) -> float | None:
    """Best-effort conversion used for rendering numeric values."""

    if isinstance(value, pd.Series):
        # Avoid deprecated implicit conversion from a pandas Series by
        # extracting a representative scalar first.
        candidate = value.dropna()
        if candidate.empty:
            return None
        value = candidate.iloc[0]
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
    summary: str = ""


DEFAULT_HORIZON_PRESETS: tuple[HorizonOption, ...] = (
    HorizonOption("Tomorrow", "1d", 1, "for tomorrow"),
    HorizonOption("1 Week", "1w", 5, "for the next week"),
    HorizonOption("1 Month", "1m", 21, "for the next month"),
    HorizonOption("3 Months", "3m", 63, "for the next three months"),
)


DEFAULT_HORIZON_PRESETS_BY_DAYS: dict[int, HorizonOption] = {
    option.business_days: option for option in DEFAULT_HORIZON_PRESETS
}


_HORIZON_CODE_PATTERN = re.compile(r"^(?P<count>\d+)\s*(?P<unit>[a-zA-Z]+)$")


def _calendar_delta_for_horizon(
    *,
    days: int,
    code: str,
    label: str,
    holidays: Iterable[pd.Timestamp] | None = None,
) -> Any:
    """Return an appropriate calendar delta for the selected horizon."""

    normalized_label = (label or "").strip().lower()
    preset_labels = {option.label.lower() for option in DEFAULT_HORIZON_PRESETS}
    if normalized_label in preset_labels or "trading" in normalized_label:
        if holidays:
            return CustomBusinessDay(n=days, holidays=holidays)
        return BDay(n=days)

    match = _HORIZON_CODE_PATTERN.match(code.strip()) if code else None
    if match:
        count = int(match.group("count"))
        unit = match.group("unit").lower()
        if unit.endswith("w"):
            return timedelta(weeks=count)
        if unit.endswith("m"):
            return relativedelta(months=count)
        if unit.endswith("d"):
            return timedelta(days=count)

    return timedelta(days=days)


def _ensure_future_trading_day(
    base_date: Any, candidate: Any, *, context: str | None = None
) -> pd.Timestamp:
    """Normalise and validate that ``candidate`` is an acceptable forecast date."""

    try:
        base = pd.Timestamp(base_date).normalize()
        forecast = pd.Timestamp(candidate).normalize()
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError("Unable to interpret forecast dates") from exc

    if forecast < base:
        message = "Forecast date precedes the market snapshot"
        if context:
            message = f"{message} for {context}"
        raise ValueError(message)

    if forecast.weekday() >= 5:
        message = "Forecast date falls on a weekend"
        if context:
            message = f"{message} for {context}"
        raise ValueError(message)

    return forecast


def _trading_day_for_horizon(
    base_date: Any,
    *,
    business_days: int,
    holidays: Iterable[pd.Timestamp] | None = None,
    context: str | None = None,
) -> pd.Timestamp:
    """Return the trading-day forecast for the given horizon."""

    if business_days <= 0:
        raise ValueError("Horizon must be a positive number of business days")

    try:
        base = pd.Timestamp(base_date).normalize()
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError("Unable to interpret base forecast date") from exc

    if holidays:
        offset = CustomBusinessDay(n=business_days, holidays=holidays)
    else:
        offset = BDay(n=business_days)
    candidate = (base + offset).normalize()
    return _ensure_future_trading_day(base, candidate, context=context)


def _pluralise(value: int, singular: str, plural: str | None = None) -> str:
    if value == 1:
        return singular
    return plural or f"{singular}s"


def _create_horizon_option(days: int) -> HorizonOption:
    preset = DEFAULT_HORIZON_PRESETS_BY_DAYS.get(days)
    if preset is not None:
        return HorizonOption(preset.label, preset.code, preset.business_days, preset.summary)

    if days <= 0:
        raise ValueError("Horizon must be a positive number of business days.")

    if days % 21 == 0:
        months = days // 21
        label = f"{months} {_pluralise(months, 'Month')}"
        code = f"{months}m"
        summary = f"for the next {months} {_pluralise(months, 'month')}"
    elif days % 5 == 0:
        weeks = days // 5
        label = f"{weeks} {_pluralise(weeks, 'Week')}"
        code = f"{weeks}w"
        summary = f"for the next {weeks} {_pluralise(weeks, 'week')}"
    else:
        label = f"{days} {_pluralise(days, 'Day')}"
        code = f"{days}d"
        summary = f"over the next {days} trading {_pluralise(days, 'day')}"

    return HorizonOption(label, code, days, summary)


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
        self.market_timezone: ZoneInfo = resolve_market_timezone(self.config)
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
            "usd": {"label": "USD $", "symbol": "$", "code": "USD"},
            "eur": {"label": "EUR €", "symbol": "€", "code": "EUR"},
        }
        self.currency_rates: dict[str, float] = {
            "local": 1.0,
            "usd": usd_rate,
            "eur": 1.0,
        }
        self._suspend_rate_updates = False
        self._initialise_currency_profile()
        self.currency_mode: str = "local"
        self.currency_symbol: str = self._currency_symbol("local")

        self.settings_vars: dict[str, tk.StringVar] = {
            "ticker": tk.StringVar(),
            "interval": tk.StringVar(),
            "model": tk.StringVar(),
            "targets": tk.StringVar(),
        }

        horizon_values = self.config.prediction_horizons or DEFAULT_PREDICTION_HORIZONS
        self.horizon_notice_var = tk.StringVar(value="")
        self.horizon_notice_label: ttk.Label | None = None
        self._hidden_horizon_labels: list[str] = []
        self.horizon_options: tuple[HorizonOption, ...] = tuple()
        self.horizon_labels: list[str] = []
        self.horizon_map: dict[str, dict[str, Any]] = {}
        self.horizon_code_to_label: dict[str, str] = {}
        self.horizon_offset_to_label: dict[int, str] = {}
        self.horizon_summary_phrases: dict[str, str] = {}
        self.selected_horizon_label: str = ""
        self.selected_horizon_code: str = ""
        self.selected_horizon_offset: int = 0
        self._configure_horizons(horizon_values)
        self.current_prediction: dict[str, Any] = {}
        self.current_market_timestamp: pd.Timestamp | None = None
        self.market_timestamp_stale: bool = False
        self.current_market_price: float | None = None
        self.price_history: pd.DataFrame | None = None
        self.feature_snapshot: pd.DataFrame | None = None
        self.feature_history: pd.DataFrame | None = None
        self.indicator_history: pd.DataFrame | None = None
        self.price_history_converted: pd.DataFrame | None = None
        self.feature_snapshot_converted: pd.DataFrame | None = None
        self.feature_history_converted: pd.DataFrame | None = None
        self.indicator_history_converted: pd.DataFrame | None = None
        self._indicator_selection_cache: set[str] = set()
        self._indicator_names: list[str] = []
        self._indicator_user_override = False
        self.indicator_toggle_all_button: ttk.Button | None = None
        self.indicator_show_all_button: ttk.Button | None = None
        self._indicator_selector_updating = False
        self.indicator_secondary_ax: Axes | None = None
        self._indicator_extra_axes: list[Axes] = []
        self._indicator_family_colors: dict[str, str] = {}
        self.feature_group_detail_tree: ttk.Treeview | None = None
        self.feature_usage_text: tk.Text | None = None
        self.feature_group_summary_text: tk.Text | None = None
        self.data_source_text: tk.Text | None = None
        self.feature_group_summary_var = tk.StringVar(value="Not available")
        self.feature_usage_summary_var = tk.StringVar(value="No features enabled.")
        self.feature_group_overview_var = tk.StringVar(value="Features used: —")
        self.data_source_detail_var = tk.StringVar(value="Not available")
        self._latest_feature_groups: dict[str, Mapping[str, Any]] = {}
        self._latest_feature_toggles: dict[str, bool] = {}
        self._latest_data_sources: list[Any] = []
        self._latest_feature_usage_entries: list[Any] = []
        self._latest_indicator_report: list[str] = []
        self.feature_toggle_vars: dict[str, tk.BooleanVar] = {}
        self.forecast_date_var = tk.StringVar(value="Forecast date: —")
        self.current_forecast_date: pd.Timestamp | None = None
        self.market_holidays: list[pd.Timestamp] = []

        self.currency_mode_var = tk.StringVar(value="local")
        self.currency_button_text = tk.StringVar(value=self.currency_symbol)
        self.currency_rate_var = tk.StringVar(value=f"{self._currency_rate('usd'):.4f}")

        self.currency_choice_map = {"Local": "local", "USD": "usd", "EUR": "eur"}
        self.currency_display_map = {value: key for key, value in self.currency_choice_map.items()}
        self.chart_type_options: tuple[str, ...] = ("Line", "Candlestick")
        self.chart_type_var = tk.StringVar(value=self.chart_type_options[0])
        self.price_decimal_places = 2
        self.decimal_option_map = {
            "2 decimals": 2,
            "3 decimals": 3,
            "4 decimals": 4,
        }
        self._suspend_expected_low_trace = False
        self.expected_low_multiplier_var = tk.DoubleVar(value=1.0)
        self.expected_low_multiplier = 1.0
        self.expected_low_max_volatility = _safe_float(
            getattr(self.config, "expected_low_max_volatility", 1.0)
        ) or 1.0
        self.expected_low_floor_window = int(
            getattr(self.config, "expected_low_floor_window", 20)
        )
        self.expected_low_multiplier_var.trace_add(
            "write", self._on_expected_low_multiplier_changed
        )

        stop_loss_default = getattr(self.config, "k_stop", 1.0)
        try:
            stop_loss_value = float(stop_loss_default)
        except (TypeError, ValueError):
            stop_loss_value = 1.0
        if not np.isfinite(stop_loss_value) or stop_loss_value <= 0:
            stop_loss_value = 1.0
        self._suspend_stop_loss_trace = False
        self.stop_loss_multiplier_var = tk.DoubleVar(value=stop_loss_value)
        self.stop_loss_multiplier = stop_loss_value
        self.stop_loss_display_var = tk.StringVar(value=f"{stop_loss_value:.2f}×")
        self.stop_loss_multiplier_var.trace_add(
            "write", self._on_stop_loss_multiplier_changed
        )

        self.trend_finder: TrendFinder | None = None
        self.trend_results: list[TrendInsight] = []
        self.trend_horizon_var = tk.StringVar(value=self.selected_horizon_label)
        self.trend_status_var = tk.StringVar(
            value="Select a horizon and click “Find Opportunities”."
        )
        self._trend_busy = False
        self.trend_tree: ttk.Treeview | None = None
        self.trend_placeholder: ttk.Label | None = None
        self.trend_progress: ttk.Progressbar | None = None
        self._trend_placeholder_default = "Run a scan to discover new opportunities."
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
        self.sentiment_label_var = tk.StringVar(value="Sentiment unavailable")
        self.sentiment_score_var = tk.StringVar(value="—")

        self._busy = False
        self._availability_log_state: dict[str, bool] = {}

        self._build_layout(horizon_values)
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

    @staticmethod
    def _normalise_horizon_values(horizons: Iterable[Any] | None) -> list[int]:
        values: list[int] = []
        if horizons is None:
            return values
        for raw in horizons:
            if raw is None:
                continue
            try:
                numeric = int(raw)
            except (TypeError, ValueError):
                continue
            if numeric <= 0:
                continue
            if numeric not in values:
                values.append(numeric)
        values.sort()
        return values

    def _summary_for_option(self, option: HorizonOption) -> str:
        summary = (option.summary or "").strip()
        if summary:
            return summary
        days = option.business_days
        if days <= 0:
            return ""
        return f"over the next {days} trading {_pluralise(days, 'day')}"

    def _update_horizon_notice(self) -> None:
        hidden = [label for label in self._hidden_horizon_labels if label]
        if hidden:
            hidden_text = ", ".join(hidden)
            message = f"Hiding horizons without trained models: {hidden_text}."
        else:
            message = ""
        self.horizon_notice_var.set(message)
        if self.horizon_notice_label is not None:
            if message:
                self.horizon_notice_label.grid()
            else:
                self.horizon_notice_label.grid_remove()

    def _configure_horizons(self, horizons: Iterable[Any] | None) -> None:
        resolved = self._normalise_horizon_values(horizons)
        if not resolved:
            resolved = list(DEFAULT_PREDICTION_HORIZONS)

        options: list[HorizonOption] = []
        for days in resolved:
            try:
                option = _create_horizon_option(days)
            except ValueError:
                continue
            options.append(option)

        if not options:
            fallback = DEFAULT_HORIZON_PRESETS[0]
            options = [
                HorizonOption(
                    fallback.label,
                    fallback.code,
                    fallback.business_days,
                    fallback.summary,
                )
            ]
            resolved = [options[0].business_days]

        self.horizon_options = tuple(options)
        self.horizon_labels = [option.label for option in options]
        self.horizon_map = {
            option.label: {"code": option.code, "offset": option.business_days}
            for option in options
        }
        self.horizon_code_to_label = {
            option.code: option.label for option in options if option.code
        }
        self.horizon_offset_to_label = {
            option.business_days: option.label for option in options
        }
        self.horizon_summary_phrases = {
            option.label: self._summary_for_option(option) for option in options
        }

        resolved_tuple = tuple(resolved)
        if resolved_tuple and resolved_tuple != self.config.prediction_horizons:
            try:
                self.config.prediction_horizons = resolved_tuple
            except Exception:
                LOGGER.debug("Unable to update configured horizons to %s", resolved_tuple)

        self._hidden_horizon_labels = [
            preset.label
            for preset in DEFAULT_HORIZON_PRESETS
            if preset.business_days not in resolved
        ]
        self._update_horizon_notice()

        preferred = self.selected_horizon_label
        if preferred not in self.horizon_map:
            if self.selected_horizon_offset in self.horizon_offset_to_label:
                preferred = self.horizon_offset_to_label[self.selected_horizon_offset]
            else:
                preferred = self._resolve_initial_horizon_label(resolved)
        if preferred:
            self._apply_horizon_selection(preferred)
        elif self.horizon_labels:
            self._apply_horizon_selection(self.horizon_labels[0])

        if hasattr(self, "horizon_box") and self.horizon_box is not None:
            self.horizon_box.configure(values=self.horizon_labels)
        if hasattr(self, "horizon_var") and self.horizon_var.get() != self.selected_horizon_label:
            self.horizon_var.set(self.selected_horizon_label)
        if hasattr(self, "trend_horizon_box") and self.trend_horizon_box is not None:
            self.trend_horizon_box.configure(values=self.horizon_labels)
        if hasattr(self, "trend_horizon_var") and self.trend_horizon_var.get() != self.selected_horizon_label:
            self.trend_horizon_var.set(self.selected_horizon_label)

    def _resolve_initial_horizon_label(self, horizons: Iterable[int]) -> str:
        if not self.horizon_labels:
            return ""
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
        if hasattr(self, "trend_horizon_var") and self.trend_horizon_var.get() != label:
            self.trend_horizon_var.set(label)

    def _horizon_summary_phrase(self, label: str | None = None) -> str:
        key = label or self.selected_horizon_label
        summary = self.horizon_summary_phrases.get(key, "").strip()
        if summary:
            return summary
        mapping = self.horizon_map.get(key)
        if mapping is None:
            return ""
        try:
            offset = int(mapping.get("offset", 0))
        except (TypeError, ValueError):
            offset = 0
        if offset <= 0:
            return ""
        return f"over the next {offset} trading {_pluralise(offset, 'day')}"

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

        self.chart_type_box = ttk.Combobox(
            toolbar,
            width=14,
            state="readonly",
            textvariable=self.chart_type_var,
            values=self.chart_type_options,
        )
        self.chart_type_box.bind("<<ComboboxSelected>>", self._on_chart_type_changed)
        Tooltip(self.chart_type_box, "Choose the price chart style.")

        self.refresh_button = ttk.Button(toolbar, text="Refresh data", command=self._on_refresh)
        self.predict_button = ttk.Button(toolbar, text="Run prediction", command=self._on_predict)

        self.forecast_label = ttk.Label(toolbar, textvariable=self.forecast_date_var)

        widgets_in_order: list[tk.Widget] = [
            self.horizon_box,
            self.ticker_entry,
            self.ticker_apply_button,
            self.position_spinbox,
            self.currency_menu_button,
            self.chart_type_box,
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

        total_columns = len(widgets_in_order) + 1
        self.horizon_notice_label = ttk.Label(
            toolbar,
            textvariable=self.horizon_notice_var,
            wraplength=520,
            justify=tk.LEFT,
        )
        self.horizon_notice_label.grid(
            row=1,
            column=0,
            columnspan=total_columns,
            sticky=tk.W,
            padx=4,
            pady=(0, 2),
        )
        self._update_horizon_notice()

        self._update_forecast_label()

        return toolbar

    def _build_notebook(self) -> None:
        self.notebook = ttk.Notebook(self.root)
        self.notebook.grid(row=1, column=0, sticky="nsew")

        self._build_overview_tab()
        self._build_trend_finder_tab()
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

    def _refresh_settings_labels(self) -> None:
        if not self.settings_vars:
            return

        targets = ", ".join(self.config.prediction_targets) if self.config.prediction_targets else "—"
        self.settings_vars["ticker"].set(f"Ticker: {self.config.ticker}")
        self.settings_vars["interval"].set(f"Interval: {self.config.interval}")
        self.settings_vars["model"].set(f"Model: {self.config.model_type}")
        self.settings_vars["targets"].set(f"Prediction targets: {targets}")

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
        stop_loss_var = tk.StringVar(value="—")
        metric_specs: list[tuple[str, str, tk.StringVar | None]] = [
            ("ticker", "Ticker", None),
            ("as_of", "Market data as of", None),
            ("last_close", "Last price", None),
            ("predicted_close", "Predicted close", None),
            ("expected_low", "Expected low", None),
            (
                "stop_loss",
                "Recommended Stop-Loss Price",
                stop_loss_var,
            ),
            ("expected_change", "Expected change", None),
            ("direction", "Direction", None),
        ]
        for column in range(4):
            weight = 1 if column % 2 == 1 else 0
            summary_frame.grid_columnconfigure(column, weight=weight)
        for idx, (key, label, var) in enumerate(metric_specs):
            row = idx // 2
            column = (idx % 2) * 2
            caption = ttk.Label(summary_frame, text=f"{label}:", anchor=tk.E)
            caption.grid(row=row, column=column, sticky=tk.E, padx=(4, 8), pady=2)
            metric_var = var or tk.StringVar(value="—")
            value = ttk.Label(
                summary_frame,
                textvariable=metric_var,
                anchor=tk.E,
                font=("TkDefaultFont", 10, "bold"),
            )
            value.grid(row=row, column=column + 1, sticky=tk.E, padx=(0, 8), pady=2)
            self.metric_vars[key] = metric_var

        self.stop_loss_var = stop_loss_var

        metric_rows = math.ceil(len(metric_specs) / 2)
        sentiment_row = metric_rows
        sentiment_frame = ttk.LabelFrame(
            summary_frame, text="News sentiment", padding=6
        )
        sentiment_frame.grid(
            row=sentiment_row, column=0, columnspan=4, sticky=tk.EW, pady=(8, 0)
        )
        sentiment_frame.grid_columnconfigure(0, weight=1)
        self.sentiment_status_label = ttk.Label(
            sentiment_frame,
            textvariable=self.sentiment_label_var,
            anchor=tk.W,
            font=("TkDefaultFont", 10, "bold"),
        )
        self.sentiment_status_label.grid(row=0, column=0, sticky=tk.W)
        self.sentiment_score_label = ttk.Label(
            sentiment_frame,
            textvariable=self.sentiment_score_var,
            anchor=tk.W,
        )
        self.sentiment_score_label.grid(row=1, column=0, sticky=tk.W, pady=(2, 0))

        self.pnl_label = ttk.Label(
            summary_frame,
            textvariable=self.pnl_var,
            anchor=tk.W,
            font=("TkDefaultFont", 10, "bold"),
        )
        self._pnl_grid_options = {
            "row": sentiment_row + 1,
            "column": 0,
            "columnspan": 4,
            "sticky": tk.W,
            "pady": (12, 0),
        }
        self.pnl_label.grid(**self._pnl_grid_options)

        feature_group_overview_row = self._pnl_grid_options["row"] + 1
        self._feature_group_overview_grid_options = {
            "row": feature_group_overview_row,
            "column": 0,
            "columnspan": 4,
            "sticky": tk.W,
            "pady": (6, 0),
        }
        self.feature_group_overview_label = ttk.Label(
            summary_frame,
            textvariable=self.feature_group_overview_var,
            anchor=tk.W,
            justify=tk.LEFT,
            wraplength=520,
        )
        self.feature_group_overview_label.grid(**self._feature_group_overview_grid_options)

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

    def _build_trend_finder_tab(self) -> None:
        frame = ttk.Frame(self.notebook, padding=12)
        self.notebook.add(frame, text="Trend Finder")
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_rowconfigure(2, weight=1)

        controls = ttk.Frame(frame)
        controls.grid(row=0, column=0, sticky="ew")
        controls.grid_columnconfigure(2, weight=1)

        caption = ttk.Label(controls, text="Horizon:")
        caption.grid(row=0, column=0, sticky=tk.W, padx=(0, 6))

        self.trend_horizon_box = ttk.Combobox(
            controls,
            width=12,
            state="readonly",
            values=self.horizon_labels,
            textvariable=self.trend_horizon_var,
        )
        self.trend_horizon_box.grid(row=0, column=1, sticky=tk.W)

        self.trend_scan_button = ttk.Button(
            controls, text="Find Opportunities", command=self._on_trend_scan
        )
        self.trend_scan_button.grid(row=0, column=2, sticky=tk.W, padx=(12, 0))

        self.trend_progress = ttk.Progressbar(controls, mode="determinate", length=120)
        self.trend_progress.grid(row=0, column=3, sticky=tk.W, padx=(12, 0))
        self.trend_progress.grid_remove()

        status = ttk.Label(frame, textvariable=self.trend_status_var, anchor=tk.W)
        status.grid(row=1, column=0, sticky="ew", pady=(8, 4))

        table_frame = ttk.Frame(frame)
        table_frame.grid(row=2, column=0, sticky="nsew")
        table_frame.grid_columnconfigure(0, weight=1)
        table_frame.grid_rowconfigure(0, weight=1)

        columns = (
            "rank",
            "ticker",
            "confidence",
            "score",
            "technical",
            "macro",
            "sentiment",
        )
        self.trend_tree = ttk.Treeview(
            table_frame,
            columns=columns,
            show="headings",
            height=12,
            selectmode="browse",
        )
        headings = {
            "rank": "#",
            "ticker": "Ticker",
            "confidence": "Confidence",
            "score": "Composite",
            "technical": "Technical",
            "macro": "Macro",
            "sentiment": "Sentiment",
        }
        for column, title in headings.items():
            self.trend_tree.heading(column, text=title)
            anchor = tk.CENTER if column in {"ticker", "rank"} else tk.E
            if column == "rank":
                width = 60
            elif column == "ticker":
                width = 90
            else:
                width = 120
            self.trend_tree.column(column, anchor=anchor, width=width, stretch=True)

        trend_scrollbar = ttk.Scrollbar(
            table_frame, orient=tk.VERTICAL, command=self.trend_tree.yview
        )
        trend_scrollbar.grid(row=0, column=1, sticky="ns")
        self.trend_tree.configure(yscrollcommand=trend_scrollbar.set)
        self.trend_tree.grid(row=0, column=0, sticky="nsew")

        self.trend_placeholder = ttk.Label(
            table_frame,
            text=self._trend_placeholder_default,
            anchor=tk.CENTER,
            justify=tk.CENTER,
        )
        self.trend_placeholder.grid(row=0, column=0, sticky="nsew")
        self.trend_tree.grid_remove()

    def _create_scrolling_text(
        self, parent: tk.Widget, height: int = 5, wrap: str = tk.WORD
    ) -> tuple[ttk.Frame, tk.Text]:
        """Create a read-only text widget with a vertical scrollbar."""

        container = ttk.Frame(parent)
        container.grid_columnconfigure(0, weight=1)
        text = tk.Text(container, height=height, wrap=wrap, state=tk.DISABLED)
        text.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(container, orient=tk.VERTICAL, command=text.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        text.configure(yscrollcommand=scrollbar.set)
        self.text_widgets.append(text)
        return container, text

    def _set_text_content(self, widget: tk.Text | None, content: str) -> None:
        if widget is None:
            return
        widget.configure(state=tk.NORMAL)
        widget.delete("1.0", tk.END)
        widget.insert(tk.END, content or "—")
        widget.configure(state=tk.DISABLED)

    def _build_indicators_tab(self) -> None:
        frame = ttk.Frame(self.notebook, padding=12)
        self.notebook.add(frame, text="Indicators")
        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(0, weight=3)
        frame.grid_columnconfigure(1, weight=1)

        chart_frame = ttk.LabelFrame(frame, text="Price & indicators", padding=8)
        chart_frame.grid(row=0, column=0, sticky="nsew")
        chart_frame.grid_rowconfigure(0, weight=1)
        chart_frame.grid_columnconfigure(0, weight=1)

        self.indicator_price_figure = Figure(figsize=(7.5, 4.5), dpi=100, constrained_layout=True)
        self.indicator_price_figure.patch.set_facecolor("white")
        self.indicator_price_ax = self.indicator_price_figure.add_subplot(111)
        self.indicator_price_ax.set_facecolor("white")
        self.indicator_price_canvas = FigureCanvasTkAgg(self.indicator_price_figure, master=chart_frame)
        self.indicator_price_canvas_widget = self.indicator_price_canvas.get_tk_widget()
        self.indicator_price_canvas_widget.grid(row=0, column=0, sticky="nsew")

        self.indicator_chart_message = ttk.Label(
            chart_frame,
            text="No indicators loaded yet",
            anchor=tk.CENTER,
            justify=tk.CENTER,
        )
        self.indicator_chart_message.grid(row=0, column=0, sticky="nsew")
        self.indicator_price_canvas_widget.grid_remove()

        sidebar = ttk.Frame(frame, padding=(12, 0, 0, 0))
        sidebar.grid(row=0, column=1, sticky="nsew")
        sidebar.grid_rowconfigure(1, weight=1)
        sidebar.grid_rowconfigure(3, weight=1)
        sidebar.grid_columnconfigure(0, weight=1)

        selector_box = ttk.LabelFrame(sidebar, text="Indicator visibility", padding=8)
        selector_box.grid(row=0, column=0, sticky="nsew")
        selector_box.grid_columnconfigure(0, weight=1)
        selector_box.grid_rowconfigure(1, weight=1)

        toggle_frame = ttk.Frame(selector_box)
        toggle_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 6))
        toggle_frame.grid_columnconfigure(0, weight=1)
        toggle_frame.grid_columnconfigure(1, weight=1)
        self.indicator_toggle_all_button = ttk.Button(
            toggle_frame,
            text="Select all indicators",
            command=self._on_indicator_toggle_all,
        )
        self.indicator_toggle_all_button.grid(row=0, column=0, sticky=tk.W)
        self.indicator_toggle_all_button.configure(state=tk.DISABLED)

        self.indicator_show_all_button = ttk.Button(
            toggle_frame,
            text="Show all indicators",
            command=self._on_indicator_show_all,
        )
        self.indicator_show_all_button.grid(row=0, column=1, sticky=tk.E)
        self.indicator_show_all_button.configure(state=tk.DISABLED)

        self.indicator_listbox = tk.Listbox(
            selector_box,
            selectmode=tk.MULTIPLE,
            exportselection=False,
            height=12,
        )
        self.indicator_listbox.grid(row=1, column=0, sticky="nsew")
        selector_scrollbar = ttk.Scrollbar(
            selector_box, orient=tk.VERTICAL, command=self.indicator_listbox.yview
        )
        selector_scrollbar.grid(row=1, column=1, sticky="ns")
        self.indicator_listbox.configure(yscrollcommand=selector_scrollbar.set)
        self.indicator_listbox.bind("<<ListboxSelect>>", self._on_indicator_selection_changed)

        info_box = ttk.LabelFrame(sidebar, text="Insights", padding=8)
        info_box.grid(row=1, column=0, sticky="nsew", pady=(12, 0))
        info_box.grid_columnconfigure(1, weight=1)

        self.indicator_info_vars = {
            "total_indicators": tk.StringVar(value="0"),
            "data_sources": tk.StringVar(value="0"),
            "confidence": tk.StringVar(value="—"),
        }
        info_specs = (
            ("total_indicators", "Indicators tracked"),
            ("data_sources", "Online sources"),
            ("confidence", "Confidence"),
        )
        for row, (key, label) in enumerate(info_specs):
            caption = ttk.Label(info_box, text=f"{label}:")
            caption.grid(row=row, column=0, sticky=tk.W, padx=(0, 6), pady=2)
            value = ttk.Label(
                info_box,
                textvariable=self.indicator_info_vars[key],
                anchor=tk.W,
                justify=tk.LEFT,
                wraplength=260,
            )
            value.grid(row=row, column=1, sticky=tk.EW, pady=2)

        usage_box = ttk.LabelFrame(sidebar, text="Feature usage", padding=8)
        usage_box.grid(row=2, column=0, sticky="nsew", pady=(12, 0))
        usage_box.grid_columnconfigure(0, weight=1)
        usage_box.grid_rowconfigure(0, weight=1)
        usage_container, self.feature_usage_text = self._create_scrolling_text(
            usage_box, height=6
        )
        usage_container.grid(row=0, column=0, sticky="nsew")

        model_inputs_box = ttk.LabelFrame(sidebar, text="Model inputs", padding=8)
        model_inputs_box.grid(row=3, column=0, sticky="nsew", pady=(12, 0))
        model_inputs_box.grid_columnconfigure(0, weight=1)
        model_inputs_box.grid_rowconfigure(1, weight=1)
        model_inputs_box.grid_rowconfigure(3, weight=1)
        model_inputs_box.grid_rowconfigure(5, weight=1)
        ttk.Label(model_inputs_box, text="Feature groups used:").grid(
            row=0, column=0, sticky=tk.W
        )
        summary_container, self.feature_group_summary_text = self._create_scrolling_text(
            model_inputs_box, height=6
        )
        summary_container.grid(row=1, column=0, sticky="nsew", pady=(2, 8))
        ttk.Label(model_inputs_box, text="Group breakdown:").grid(
            row=2, column=0, sticky=tk.W
        )
        feature_tree_frame = ttk.Frame(model_inputs_box)
        feature_tree_frame.grid(row=3, column=0, sticky="nsew")
        feature_tree_frame.grid_columnconfigure(0, weight=1)
        self.feature_group_detail_tree = ttk.Treeview(
            feature_tree_frame,
            columns=("group", "status", "count", "highlights"),
            show="headings",
            height=6,
        )
        self.feature_group_detail_tree.heading("group", text="Group")
        self.feature_group_detail_tree.heading("status", text="Status")
        self.feature_group_detail_tree.heading("count", text="# series")
        self.feature_group_detail_tree.heading("highlights", text="Top features")
        self.feature_group_detail_tree.column("group", width=90, anchor=tk.W)
        self.feature_group_detail_tree.column("status", width=90, anchor=tk.W)
        self.feature_group_detail_tree.column("count", width=70, anchor=tk.CENTER)
        self.feature_group_detail_tree.column("highlights", width=180, anchor=tk.W)
        self.feature_group_detail_tree.grid(row=0, column=0, sticky="nsew")
        feature_tree_scroll = ttk.Scrollbar(
            feature_tree_frame,
            orient=tk.VERTICAL,
            command=self.feature_group_detail_tree.yview,
        )
        feature_tree_scroll.grid(row=0, column=1, sticky="ns")
        self.feature_group_detail_tree.configure(yscrollcommand=feature_tree_scroll.set)
        data_source_row = 4
        ttk.Label(model_inputs_box, text="Data sources:").grid(
            row=data_source_row, column=0, sticky=tk.W
        )
        data_source_container, self.data_source_text = self._create_scrolling_text(
            model_inputs_box, height=6
        )
        data_source_container.grid(
            row=data_source_row + 1, column=0, sticky="nsew", pady=(8, 0)
        )

        details_button = ttk.Button(
            model_inputs_box, text="Details…", command=self._open_feature_details_dialog
        )
        details_button.grid(row=data_source_row + 2, column=0, sticky=tk.E, pady=(12, 0))

        self._set_text_content(self.feature_usage_text, self.feature_usage_summary_var.get())
        self._set_text_content(
            self.feature_group_summary_text, self.feature_group_summary_var.get()
        )
        self._set_text_content(self.data_source_text, self.data_source_detail_var.get())

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
        self._refresh_settings_labels()
        ttk.Label(info_box, textvariable=self.settings_vars["ticker"]).pack(anchor=tk.W)
        ttk.Label(info_box, textvariable=self.settings_vars["interval"]).pack(anchor=tk.W)
        ttk.Label(info_box, textvariable=self.settings_vars["model"]).pack(anchor=tk.W)
        ttk.Label(info_box, textvariable=self.settings_vars["targets"]).pack(anchor=tk.W)

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

        expected_low_row = ttk.Frame(display_box)
        expected_low_row.pack(fill=tk.X, pady=4)
        ttk.Label(expected_low_row, text="Expected-low sensitivity:").grid(
            row=0, column=0, sticky=tk.W
        )
        self.expected_low_spinbox = ttk.Spinbox(
            expected_low_row,
            from_=0.0,
            to=5.0,
            increment=0.1,
            format="%.1f",
            width=6,
            textvariable=self.expected_low_multiplier_var,
        )
        self.expected_low_spinbox.grid(row=0, column=1, padx=(8, 0), sticky=tk.W)
        Tooltip(
            self.expected_low_spinbox,
            "Higher values subtract more volatility from the prediction to find the expected low.",
        )

        stop_loss_row = ttk.Frame(display_box)
        stop_loss_row.pack(fill=tk.X, pady=4)
        ttk.Label(stop_loss_row, text="Stop-loss multiplier:").grid(row=0, column=0, sticky=tk.W)
        stop_loss_row.columnconfigure(1, weight=1)
        self.stop_loss_slider = ttk.Scale(
            stop_loss_row,
            from_=0.1,
            to=5.0,
            orient=tk.HORIZONTAL,
            variable=self.stop_loss_multiplier_var,
        )
        self.stop_loss_slider.grid(row=0, column=1, padx=(8, 8), sticky=tk.EW)
        self.stop_loss_value_label = ttk.Label(
            stop_loss_row, textvariable=self.stop_loss_display_var, width=6, anchor=tk.W
        )
        self.stop_loss_value_label.grid(row=0, column=2, sticky=tk.W)
        Tooltip(
            self.stop_loss_slider,
            "Scales the predicted volatility when computing protective stop-loss levels.",
        )

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

        self.feature_toggle_vars.clear()
        toggle_names = [
            name
            for name in sorted(self.config.feature_toggles)
            if name in IMPLEMENTED_FEATURE_GROUPS
        ]
        for idx, name in enumerate(toggle_names):
            enabled = bool(self.config.feature_toggles.get(name, False))
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
        if hasattr(self, "price_ax"):
            self._update_price_chart()
        self._on_predict()

    def _on_trend_scan(self) -> None:
        if self._trend_busy:
            return
        label = self.trend_horizon_var.get() or self.selected_horizon_label
        mapping = self.horizon_map.get(label)
        if mapping is None:
            messagebox.showwarning(
                "Invalid horizon", "Please select a valid horizon value for the scan."
            )
            self.trend_horizon_var.set(self.selected_horizon_label)
            return
        try:
            horizon_value = int(mapping.get("offset", 0))
        except (TypeError, ValueError):
            horizon_value = 0
        if horizon_value <= 0:
            messagebox.showwarning(
                "Unsupported horizon",
                "The selected horizon is not supported for trend scanning.",
            )
            return
        summary = self._horizon_summary_phrase(label)
        if summary:
            status = f"Scanning universe for Top 10 opportunities {summary}…"
        else:
            status = "Scanning universe for Top 10 opportunities…"
        self._set_trend_busy(True, status)
        finder = self._get_trend_finder()

        if self.trend_progress is not None:
            self.trend_progress.configure(maximum=max(len(finder.universe), 1), value=0)

        def worker() -> None:
            try:
                def progress_callback(current: int, total: int, message: str) -> None:
                    self.root.after(
                        0,
                        lambda curr=current, tot=total, msg=message: self._update_trend_progress(
                            curr, tot, msg
                        ),
                    )

                results = finder.scan(
                    horizon=horizon_value,
                    limit=10,
                    progress_callback=progress_callback,
                )
            except Exception as exc:  # pragma: no cover - optional providers may fail
                LOGGER.exception("Trend scan failed: %s", exc)
                self.root.after(0, lambda err=exc: self._on_trend_scan_failure(err))
            else:
                self.root.after(
                    0,
                    lambda payload=(label, results): self._on_trend_scan_success(
                        payload[0], payload[1]
                    ),
                )

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

    def _on_trend_scan_success(self, label: str, results: list[TrendInsight]) -> None:
        self.trend_results = list(results)
        target = 10
        if results:
            summary = self._horizon_summary_phrase(label)
            if summary:
                message = f"Top {target} opportunities {summary} ready."
            else:
                message = f"Top {target} opportunities ready."
        else:
            message = "No opportunities found for the selected horizon."
        self._refresh_trend_table(
            "No opportunities were identified during the latest scan." if not results else None
        )
        self._set_trend_busy(False, message)

    def _on_trend_scan_failure(self, exc: Exception) -> None:
        self._set_trend_busy(False, "Trend scan failed.")
        messagebox.showerror("Trend scan failed", str(exc))

    def _set_trend_busy(self, busy: bool, status: str | None = None) -> None:
        self._trend_busy = busy
        if status:
            self.trend_status_var.set(status)
        if hasattr(self, "trend_scan_button") and self.trend_scan_button:
            state = tk.DISABLED if busy else tk.NORMAL
            self.trend_scan_button.configure(state=state)
        if hasattr(self, "trend_horizon_box") and self.trend_horizon_box:
            if busy:
                self.trend_horizon_box.configure(state="disabled")
            else:
                self.trend_horizon_box.configure(state="readonly")
        if self.trend_progress is not None:
            if busy:
                maximum = self.trend_progress.cget("maximum")
                try:
                    maximum_int = int(maximum)
                except (TypeError, ValueError):  # pragma: no cover - defensive
                    maximum_int = 1
                self.trend_progress.configure(value=0, maximum=max(maximum_int, 1))
                self.trend_progress.grid()
            else:
                self.trend_progress.configure(value=0, maximum=1)
                self.trend_progress.grid_remove()

    def _update_trend_progress(self, current: int, total: int, status: str) -> None:
        maximum = max(int(total), 1)
        value = max(0, min(int(current), maximum))
        if self.trend_progress is not None:
            self.trend_progress.configure(maximum=maximum, value=value)
        if status:
            percent = 0
            try:
                percent = int(round((value / maximum) * 100)) if maximum else 0
            except Exception:  # pragma: no cover - defensive guard
                percent = 0
            if maximum > 0:
                display = f"{status} – {percent}% complete"
            else:
                display = status
            self.trend_status_var.set(display)

    def _refresh_trend_table(self, empty_message: str | None = None) -> None:
        if self.trend_tree is None:
            return
        for item in self.trend_tree.get_children():
            self.trend_tree.delete(item)
        if not self.trend_results:
            if self.trend_placeholder is not None:
                if empty_message:
                    self.trend_placeholder.configure(text=empty_message)
                else:
                    self.trend_placeholder.configure(text=self._trend_placeholder_default)
                self.trend_placeholder.grid()
            self.trend_tree.grid_remove()
            return
        if self.trend_placeholder is not None:
            self.trend_placeholder.grid_remove()
        self.trend_tree.grid(row=0, column=0, sticky="nsew")
        sorted_results = self._sorted_trend_results()
        for rank, insight in enumerate(sorted_results[:10], start=1):
            confidence = self._trend_confidence_value(insight)
            confidence_display = fmt_pct(confidence, decimals=1) if confidence is not None else "—"
            values = (
                rank,
                insight.ticker,
                confidence_display,
                self._format_trend_score(insight.composite_score),
                self._format_trend_score(insight.technical_score),
                self._format_trend_score(getattr(insight, "macro_score", None)),
                self._format_trend_score(insight.sentiment_score),
            )
            self.trend_tree.insert("", tk.END, values=values)

    def _format_trend_score(self, value: float | None) -> str:
        numeric = _safe_float(value)
        if numeric is None:
            return "—"
        scaled = max(-1.0, min(1.0, float(numeric)))
        percentile = (scaled + 1.0) / 2.0 * 100.0
        return f"{percentile:0.1f}%"

    def _trend_confidence_value(self, insight: TrendInsight) -> float | None:
        metadata = insight.metadata if isinstance(insight, TrendInsight) else {}
        if not isinstance(metadata, Mapping):
            metadata = {}
        confidence_block = metadata.get("confidence") if isinstance(metadata, Mapping) else None

        confidence_value: Any | None = None
        if isinstance(confidence_block, Mapping):
            for key in (
                "confidence_rank",
                "direction_confidence",
                "signal_confluence_score",
                "confluence_confidence",
            ):
                if confidence_value is None:
                    confidence_value = confidence_block.get(key)

        if confidence_value is None:
            confidence_value = getattr(insight, "confidence_rank", None)

        return _safe_float(confidence_value)

    def _sorted_trend_results(self) -> list[TrendInsight]:
        def sort_key(insight: TrendInsight) -> tuple[float, float]:
            confidence = self._trend_confidence_value(insight)
            composite = _safe_float(getattr(insight, "composite_score", None))
            confidence_key = confidence if confidence is not None else -math.inf
            composite_key = composite if composite is not None else -math.inf
            return (confidence_key, composite_key)

        return sorted(self.trend_results, key=sort_key, reverse=True)

    def _get_trend_finder(self) -> TrendFinder:
        if self.trend_finder is None:
            self.trend_finder = TrendFinder(self.config)
        else:
            self.trend_finder.update_base_config(self.config)
        return self.trend_finder

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

    def _on_chart_type_changed(self, _event: Any | None = None) -> None:
        value = self.chart_type_var.get()
        if value not in self.chart_type_options:
            self.chart_type_var.set(self.chart_type_options[0])
        self._update_price_chart()

    def _on_expected_low_multiplier_changed(self, *_args: Any) -> None:
        if getattr(self, "_suspend_expected_low_trace", False):
            return
        try:
            raw_value = float(self.expected_low_multiplier_var.get())
        except (tk.TclError, TypeError, ValueError):
            raw_value = self.expected_low_multiplier
        if not np.isfinite(raw_value):
            raw_value = self.expected_low_multiplier
        clipped = max(0.0, min(5.0, raw_value))
        if abs(clipped - raw_value) > 1e-6:
            self._suspend_expected_low_trace = True
            self.expected_low_multiplier_var.set(clipped)
            self._suspend_expected_low_trace = False
        if abs(clipped - self.expected_low_multiplier) <= 1e-6:
            return
        self.expected_low_multiplier = clipped
        self._refresh_numeric_views()

    def _on_stop_loss_multiplier_changed(self, *_args: Any) -> None:
        if getattr(self, "_suspend_stop_loss_trace", False):
            return
        try:
            raw_value = float(self.stop_loss_multiplier_var.get())
        except (tk.TclError, TypeError, ValueError):
            raw_value = self.stop_loss_multiplier
        if not np.isfinite(raw_value):
            raw_value = self.stop_loss_multiplier
        clipped = max(0.1, min(5.0, raw_value))
        if abs(clipped - raw_value) > 1e-6:
            self._suspend_stop_loss_trace = True
            self.stop_loss_multiplier_var.set(clipped)
            self._suspend_stop_loss_trace = False
        if abs(clipped - self.stop_loss_multiplier) <= 1e-6:
            return
        self.stop_loss_multiplier = clipped
        if hasattr(self, "stop_loss_display_var"):
            self.stop_loss_display_var.set(f"{clipped:.2f}×")
        self._sync_stop_loss_multiplier()

    def _on_pnl_toggle(self) -> None:
        self._update_pnl_visibility()

    def _on_dark_mode_toggle(self) -> None:
        enabled = bool(self.dark_mode_var.get())
        if enabled == self.dark_mode_enabled:
            return
        self.apply_theme(enabled)
        self._refresh_numeric_views()

    def _sync_stop_loss_multiplier(self) -> None:
        value = float(self.stop_loss_multiplier)
        config_obj = getattr(self, "config", None)
        if config_obj is not None:
            config_obj.k_stop = value
        application = getattr(self, "application", None)
        application_config = getattr(application, "config", None)
        if application_config is not None:
            application_config.k_stop = value
        pipeline = getattr(application, "pipeline", None)
        pipeline_config = getattr(pipeline, "config", None)
        if pipeline_config is not None:
            pipeline_config.k_stop = value

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

    def _on_indicator_selection_changed(self, _event: Any | None = None) -> None:
        if self._indicator_selector_updating:
            return
        selections = self._current_indicator_selection()
        self._indicator_selection_cache = set(selections)
        self._indicator_user_override = True
        self._update_indicator_toggle_button_state(self._indicator_names, self._indicator_selection_cache)
        self._update_indicator_chart(selections)

    def _on_indicator_show_all(self) -> None:
        if self._indicator_selector_updating or not hasattr(self, "indicator_listbox"):
            return
        names = list(self._indicator_names)
        if not names:
            return
        _, price_index = self._prepare_price_series_for_indicators()
        if price_index.empty:
            self._set_status("No price data available to visualise indicators.")
            return

        available: list[tuple[int, str]] = []
        for idx, name in enumerate(names):
            series = self._extract_indicator_series(name)
            if series is None or series.empty:
                continue
            aligned = self._align_indicator_to_price_index(series, price_index)
            if aligned is None or aligned.empty:
                if isinstance(series.index, pd.DatetimeIndex) and not price_index.empty:
                    price_start, price_end = price_index.min(), price_index.max()
                    series_index = series.index
                    price_tz = getattr(price_index, "tz", None)
                    series_tz = getattr(series_index, "tz", None)

                    if price_tz != series_tz:
                        try:
                            series_index = series_index.tz_localize(None)
                        except (TypeError, AttributeError, ValueError):
                            try:
                                series_index = series_index.tz_convert(None)
                            except Exception:
                                continue
                        try:
                            price_index = price_index.tz_localize(None)
                        except (TypeError, AttributeError, ValueError):
                            try:
                                price_index = price_index.tz_convert(None)
                            except Exception:
                                continue
                        price_start, price_end = price_index.min(), price_index.max()

                    if getattr(series_index, "tz", None) != getattr(price_index, "tz", None):
                        continue

                    series = series.copy()
                    series.index = series_index
                    series = series.loc[(series_index >= price_start) & (series_index <= price_end)]
                    aligned = self._align_indicator_to_price_index(series, price_index)
            if aligned is not None and not aligned.empty:
                available.append((idx, name))

        if not available:
            self.indicator_listbox.selection_clear(0, tk.END)
            self._indicator_selection_cache = set()
            self._indicator_user_override = True
            self._update_indicator_toggle_button_state(names, set())
            self._update_indicator_chart(set())
            self._set_status("No indicators available for this symbol and date range.")
            return

        self._indicator_selector_updating = True
        try:
            self.indicator_listbox.selection_clear(0, tk.END)
            for index, _ in available:
                self.indicator_listbox.selection_set(index)
        finally:
            self._indicator_selector_updating = False

        selections: set[str] = {name for _, name in available}
        self._indicator_selection_cache = selections
        self._indicator_user_override = True
        self._update_indicator_toggle_button_state(names, selections)
        self._update_indicator_chart(selections)
        self._set_status(
            f"Showing all {len(selections)} indicators available for the current view."
        )

    def _on_indicator_toggle_all(self) -> None:
        if self._indicator_selector_updating or not hasattr(self, "indicator_listbox"):
            return
        names = list(self._indicator_names)
        if not names:
            return
        current = set(self._current_indicator_selection())
        select_all = len(current) != len(names)
        self._indicator_selector_updating = True
        try:
            if select_all:
                self.indicator_listbox.selection_set(0, tk.END)
                selections: set[str] = set(names)
            else:
                self.indicator_listbox.selection_clear(0, tk.END)
                selections = set()
        finally:
            self._indicator_selector_updating = False
        self._indicator_selection_cache = selections
        self._indicator_user_override = True
        self._update_indicator_toggle_button_state(names, selections)
        self._update_indicator_chart(selections)

    def _current_indicator_selection(self) -> list[str]:
        if not hasattr(self, "indicator_listbox"):
            return []
        try:
            indices = self.indicator_listbox.curselection()
        except Exception:
            return []
        return [self.indicator_listbox.get(idx) for idx in indices]

    def _collect_indicator_names(self) -> list[str]:
        names: set[str] = set()
        fetcher = getattr(getattr(self.application, "pipeline", None), "fetcher", None)
        if fetcher is not None and hasattr(fetcher, "get_indicator_columns"):
            try:
                stored_columns = fetcher.get_indicator_columns()
            except Exception as exc:  # pragma: no cover - defensive for optional stack
                LOGGER.debug("Failed to load stored indicator column metadata: %s", exc)
                stored_columns = []
            for item in stored_columns:
                label = str(item).strip()
                if label:
                    names.add(label)
        metadata = getattr(getattr(self.application, "pipeline", None), "metadata", None)
        category_map: Mapping[str, Any] | None = None
        if isinstance(metadata, Mapping):
            raw_names = metadata.get("indicator_columns")
            if isinstance(raw_names, (list, tuple, set)):
                for item in raw_names:
                    label = str(item).strip()
                    if label:
                        names.add(label)
            category_map = metadata.get("feature_categories") if metadata else None
            if isinstance(category_map, Mapping):
                for column, category in category_map.items():
                    category_label = str(category).lower()
                    if "indicator" in category_label or "oscillator" in category_label:
                        label = str(column).strip()
                        if label:
                            names.add(label)

        feature_history = (
            self.feature_history_converted
            if isinstance(self.feature_history_converted, pd.DataFrame)
            else self.feature_history
        )
        if not names and isinstance(feature_history, pd.DataFrame):
            excluded_tokens = {"open", "high", "low", "close", "volume", "adjclose", "adj_close"}
            for column in feature_history.columns:
                label = str(column).strip()
                if not label:
                    continue
                if label.lower() in excluded_tokens:
                    continue
                value = feature_history[column].iloc[-1] if len(feature_history) else None
                if _safe_float(value) is None:
                    continue
                names.add(label)

        indicator_history = (
            self.indicator_history_converted
            if isinstance(self.indicator_history_converted, pd.DataFrame)
            else self.indicator_history
        )
        if isinstance(indicator_history, pd.DataFrame):
            column_map = {str(col).lower(): col for col in indicator_history.columns}
            indicator_col = column_map.get("indicator")
            if indicator_col:
                for value in indicator_history[indicator_col].dropna().unique():
                    label = str(value).strip()
                    if label:
                        names.add(label)

        return sorted(names, key=str.lower)

    def _populate_indicator_selector(self, names: list[str], selections: set[str]) -> None:
        if not hasattr(self, "indicator_listbox"):
            return
        self._indicator_names = list(names)
        self._indicator_selector_updating = True
        try:
            self.indicator_listbox.delete(0, tk.END)
            for index, name in enumerate(names):
                self.indicator_listbox.insert(tk.END, name)
                if name in selections:
                    self.indicator_listbox.selection_set(index)
        finally:
            self._indicator_selector_updating = False
        self._update_indicator_toggle_button_state(self._indicator_names, selections)

    def _update_indicator_toggle_button_state(
        self,
        names: Iterable[str] | None = None,
        selections: Iterable[str] | None = None,
    ) -> None:
        button = getattr(self, "indicator_toggle_all_button", None)
        show_all_button = getattr(self, "indicator_show_all_button", None)
        if button is None:
            return
        if names is not None:
            self._indicator_names = list(names)
        available = list(self._indicator_names)
        if selections is None:
            selection_set = set(self._indicator_selection_cache)
        else:
            selection_set = set(selections)
        if available:
            selection_set &= set(available)
        if not available:
            button.configure(text="Select all indicators", state=tk.DISABLED)
            if show_all_button is not None:
                show_all_button.configure(state=tk.DISABLED)
            return
        if show_all_button is not None:
            show_all_button.configure(state=tk.NORMAL)
        button.configure(
            text="Deselect all indicators"
            if len(selection_set) == len(available)
            else "Select all indicators",
            state=tk.NORMAL,
        )

    def _update_indicator_info_panel(self, indicator_names: list[str]) -> None:
        if isinstance(self.current_prediction, PredictionResult):
            prediction = self.current_prediction.to_dict()
        elif isinstance(self.current_prediction, Mapping):
            prediction = self.current_prediction
        else:
            prediction = {}

        total: int | None = None
        usage_reported = False
        indicator_report: list[str] = []
        indicators_used = prediction.get("indicators_used") if isinstance(prediction, Mapping) else None
        if isinstance(indicators_used, Iterable) and not isinstance(indicators_used, (str, bytes)):
            for entry in indicators_used:
                label: str | None = None
                if isinstance(entry, Mapping):
                    label = entry.get("name") or entry.get("id") or entry.get("indicator")
                else:
                    label = str(entry)
                if label is None:
                    continue
                clean_label = str(label).strip()
                if clean_label:
                    indicator_report.append(clean_label)
            if indicator_report:
                usage_reported = True
                total = len(set(indicator_report))

        if total is None:
            summary_entries = prediction.get("feature_usage_summary") if isinstance(prediction, Mapping) else None
            if isinstance(summary_entries, Iterable) and not isinstance(
                summary_entries, (str, bytes)
            ):
                counts: list[int] = []
                for entry in summary_entries:
                    value = entry.get("count") if isinstance(entry, Mapping) else getattr(entry, "count", None)
                    try:
                        counts.append(int(value))
                    except (TypeError, ValueError):
                        continue
                if counts:
                    usage_reported = True
                    total = sum(counts)

        if total is None and not usage_reported:
            self.indicator_info_vars["total_indicators"].set("Not reported")
        else:
            self.indicator_info_vars["total_indicators"].set(str(total or 0))
        feature_groups = self._resolve_feature_groups()
        feature_toggles = self._normalise_feature_toggles(
            prediction.get("feature_toggles") if isinstance(prediction, Mapping) else None
        )
        pipeline_meta = getattr(getattr(self.application, "pipeline", None), "metadata", None)
        if not feature_toggles and isinstance(pipeline_meta, Mapping):
            raw_toggles = pipeline_meta.get("feature_toggles")
            feature_toggles = self._normalise_feature_toggles(raw_toggles)

        def _extend_sources(candidate: Any) -> None:
            if candidate is None:
                return
            if isinstance(candidate, Mapping):
                values = list(candidate.values()) if candidate else []
                if values:
                    source_entries.extend(values)
                else:
                    source_entries.extend(list(candidate))
                return
            if isinstance(candidate, Iterable) and not isinstance(candidate, (str, bytes)):
                source_entries.extend(list(candidate))
                return
            source_entries.append(candidate)

        source_entries: list[Any] = []
        if isinstance(prediction, Mapping):
            for key in ("data_sources", "sources", "data_providers"):
                raw_sources = prediction.get(key)
                if raw_sources:
                    _extend_sources(raw_sources)
                    break

        pipeline = getattr(self.application, "pipeline", None)
        metadata = getattr(pipeline, "metadata", None)
        if not source_entries and isinstance(metadata, Mapping):
            _extend_sources(metadata.get("data_sources"))

        if not source_entries and getattr(getattr(pipeline, "fetcher", None), "get_data_sources", None):
            try:
                fetched = pipeline.fetcher.get_data_sources()
            except Exception:  # pragma: no cover - defensive network call
                fetched = []
            _extend_sources(fetched)

        provider_ids: set[str] = set()
        for entry in source_entries:
            candidate: Any = None
            if isinstance(entry, Mapping):
                candidate = (
                    entry.get("id")
                    or entry.get("provider")
                    or entry.get("provider_id")
                )
            else:
                candidate = getattr(entry, "provider_id", None) or getattr(entry, "id", None)
            if candidate is None and isinstance(entry, str):
                candidate = entry
            if candidate is None and entry is not None and not isinstance(entry, (str, bytes)):
                candidate = str(entry)
            if candidate is None:
                continue
            normalized = str(candidate).strip().lower()
            if normalized and normalized not in NON_REMOTE_PROVIDER_IDS:
                provider_ids.add(normalized)

        if not provider_ids and source_entries:
            fallback = {
                value
                for entry in source_entries
                if isinstance(entry, str)
                and (value := str(entry).strip().lower())
                and value not in NON_REMOTE_PROVIDER_IDS
            }
            provider_ids.update(fallback)

        self.indicator_info_vars["data_sources"].set(str(len(provider_ids)))
        self.data_source_detail_var.set(self._data_source_summary_text(source_entries))
        self._latest_data_sources = list(source_entries)
        self._latest_feature_groups = feature_groups or {}
        self._latest_feature_toggles = feature_toggles or {}

        raw_usage_summary = (
            prediction.get("feature_usage_summary") if isinstance(prediction, Mapping) else None
        )
        if isinstance(raw_usage_summary, Iterable) and not isinstance(
            raw_usage_summary, (str, bytes)
        ):
            self._latest_feature_usage_entries = list(raw_usage_summary)
        else:
            self._latest_feature_usage_entries = []

        self.feature_group_summary_var.set(
            self._feature_group_summary_text(feature_groups, feature_toggles)
        )
        self.feature_usage_summary_var.set(
            self._feature_usage_summary_text(
                raw_usage_summary,
                feature_toggles,
            )
        )
        self.feature_group_overview_var.set(
            self._feature_group_overview_text(
                self._resolve_executed_feature_groups(prediction, feature_groups),
                feature_toggles,
            )
        )
        if indicator_report:
            self._latest_indicator_report = sorted(set(indicator_report))
        else:
            cleaned = [str(name).strip() for name in self._indicator_names if str(name).strip()]
            self._latest_indicator_report = sorted(set(cleaned))

        self._set_text_content(self.feature_usage_text, self.feature_usage_summary_var.get())
        self._set_text_content(
            self.feature_group_summary_text, self.feature_group_summary_var.get()
        )
        self._set_text_content(self.data_source_text, self.data_source_detail_var.get())
        self._refresh_feature_group_table(feature_groups, feature_toggles)

        confidence_display = self._compute_confidence_metric()
        self.indicator_info_vars["confidence"].set(confidence_display)

    def _render_feature_details_text(self) -> str:
        sections: list[str] = []

        overview = self.feature_group_overview_var.get().strip()
        if overview:
            sections.append("Overview\n" + overview)

        usage_text = self.feature_usage_summary_var.get().strip()
        if usage_text:
            sections.append("Feature usage\n" + usage_text)

        feature_lines: list[str] = []
        reported_groups: set[str] = set()
        feature_groups = self._latest_feature_groups or {}
        feature_toggles = self._latest_feature_toggles or {}
        for name, summary in sorted(feature_groups.items()):
            reported_groups.add(name)
            status_label = self._feature_group_status_label(name, summary, feature_toggles)
            configured = summary.get("configured")
            highlights_count, highlights_preview = self._feature_group_highlights(summary)
            description = summary.get("description") or summary.get("note")
            categories = summary.get("categories")
            detail_bits: list[str] = []
            if configured is not None:
                detail_bits.append("enabled" if configured else "disabled")
            elif name in feature_toggles:
                detail_bits.append("enabled" if feature_toggles[name] else "disabled")

            raw_status = str(summary.get("status") or "").replace("_", " ").strip()
            if raw_status and raw_status.lower() not in {status_label.lower()}:
                detail_bits.append(raw_status)

            if categories:
                category_list = [str(item).strip() for item in categories if str(item).strip()]
                if category_list:
                    detail_bits.append("Categories: " + ", ".join(category_list))

            if highlights_count is not None:
                detail_bits.append(f"{highlights_count} series")
            if highlights_preview:
                detail_bits.append(f"Top features: {highlights_preview}")

            line = f"• {name}: {status_label}"
            if detail_bits:
                line += " (" + "; ".join(detail_bits) + ")"
            if description:
                line += f"\n  {description}"
            feature_lines.append(line)

        for name, enabled in sorted(feature_toggles.items()):
            if name in reported_groups:
                continue
            label = "Enabled" if enabled else "Disabled"
            feature_lines.append(f"• {name}: {label} (no execution report)")

        if feature_lines:
            sections.append("Feature groups\n" + "\n".join(feature_lines))
        else:
            sections.append("Feature groups\nNo feature groups reported.")

        indicator_lines = [f"• {name}" for name in self._latest_indicator_report] or [
            "No indicator usage reported."
        ]
        sections.append("Indicators tracked\n" + "\n".join(indicator_lines))

        data_sources_text = self._data_source_summary_text(self._latest_data_sources)
        sections.append("Data sources\n" + data_sources_text)

        return "\n\n".join(sections)

    def _open_feature_details_dialog(self) -> None:
        dialog = tk.Toplevel(self.root)
        dialog.title("Model inputs and data sources")
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.geometry("720x520")
        dialog.grid_columnconfigure(0, weight=1)
        dialog.grid_rowconfigure(0, weight=1)

        frame = ttk.Frame(dialog, padding=12)
        frame.grid(row=0, column=0, sticky="nsew")
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_rowconfigure(0, weight=1)

        container, text_widget = self._create_scrolling_text(frame, height=22)
        container.grid(row=0, column=0, sticky="nsew")
        self._set_text_content(text_widget, self._render_feature_details_text())

        button_frame = ttk.Frame(frame)
        button_frame.grid(row=1, column=0, sticky=tk.E, pady=(12, 0))
        ttk.Button(button_frame, text="Close", command=dialog.destroy).grid(
            row=0, column=0, sticky=tk.E
        )

    def _normalise_feature_toggles(
        self, toggles: Mapping[str, Any] | None
    ) -> dict[str, bool]:
        if not isinstance(toggles, Mapping):
            return {}
        normalised: dict[str, bool] = {}
        for name, value in toggles.items():
            key = str(name).strip()
            if not key:
                continue
            normalised[key] = bool(value)
        return normalised

    def _resolve_feature_groups(self) -> dict[str, Mapping[str, Any]]:
        """Return the feature group metadata from the latest prediction or pipeline."""

        if isinstance(self.current_prediction, PredictionResult):
            prediction_meta: Mapping[str, Any] = self.current_prediction.meta
        elif isinstance(self.current_prediction, Mapping):
            prediction_meta = self.current_prediction
        else:
            prediction_meta = {}

        block = (
            prediction_meta.get("feature_groups")
            if isinstance(prediction_meta, Mapping)
            else None
        )
        if isinstance(block, Mapping):
            return dict(block)

        pipeline = getattr(self.application, "pipeline", None)
        metadata = getattr(pipeline, "metadata", None)
        if isinstance(metadata, Mapping):
            groups = metadata.get("feature_groups")
            if isinstance(groups, Mapping):
                return dict(groups)

        return {}

    def _resolve_executed_feature_groups(
        self,
        prediction: Mapping[str, Any] | None,
        feature_groups: Mapping[str, Mapping[str, Any]] | None,
    ) -> list[str]:
        """Return the executed feature groups for the active prediction."""

        raw_candidates: Iterable[Any] | None = None
        if isinstance(prediction, Mapping):
            raw_candidates = (
                prediction.get("feature_groups_used")
                or prediction.get("executed_feature_groups")
                or prediction.get("used_feature_groups")
            )

        candidates: list[str] = []
        seen: set[str] = set()
        if isinstance(raw_candidates, Mapping):
            raw_candidates = raw_candidates.keys()
        if isinstance(raw_candidates, Iterable) and not isinstance(
            raw_candidates, (str, bytes)
        ):
            for name in raw_candidates:
                label = str(name).strip()
                if label and label not in seen:
                    seen.add(label)
                    candidates.append(label)

        if not candidates and isinstance(feature_groups, Mapping):
            for name, summary in feature_groups.items():
                if isinstance(summary, Mapping) and summary.get("executed"):
                    label = str(name).strip()
                    if label and label not in seen:
                        seen.add(label)
                        candidates.append(label)

        return candidates

    def _feature_group_summary_text(
        self,
        feature_groups: Mapping[str, Mapping[str, Any]] | None,
        feature_toggles: Mapping[str, bool] | None,
    ) -> str:
        """Render a human-readable summary of which feature groups were used."""

        if not feature_groups:
            if feature_toggles:
                lines = []
                for name, enabled in sorted(feature_toggles.items()):
                    label = "enabled" if enabled else "disabled"
                    lines.append(f"• {name}: {label} (no execution report)")
                if lines:
                    return "\n".join(lines)
            return "Model did not report feature usage."

        lines = []
        for name, summary in sorted(feature_groups.items()):
            executed = bool(summary.get("executed"))
            configured = summary.get("configured")
            status = str(summary.get("status") or ("executed" if executed else "skipped"))
            status_label = status.replace("_", " ")
            marker = "✅" if executed else "•"

            detail_bits: list[str] = []
            if configured is not None:
                detail_bits.append("enabled" if configured else "disabled")
            elif feature_toggles and name in feature_toggles:
                detail_bits.append("enabled" if feature_toggles[name] else "disabled")

            categories = summary.get("categories")
            if categories:
                category_list = list(categories)
                trimmed = ", ".join(category_list[:3])
                if len(category_list) > 3:
                    trimmed += ", …"
                detail_bits.append(trimmed)

            description = summary.get("description")
            detail_text = "; ".join(detail_bits)
            label = f"{marker} {name}: {status_label}"
            if detail_text:
                label += f" ({detail_text})"
            if description and executed:
                label += f" – {description}"
            lines.append(label)

        return "\n".join(lines) if lines else "Model did not report feature usage."

    def _feature_group_overview_text(
        self,
        executed_feature_groups: Iterable[Any] | None,
        feature_toggles: Mapping[str, bool] | None,
    ) -> str:
        labels: list[str] = []
        seen: set[str] = set()
        if executed_feature_groups:
            for name in executed_feature_groups:
                label = str(name).strip()
                if label and label not in seen:
                    seen.add(label)
                    labels.append(label)

        if labels:
            return "Features used: " + ", ".join(labels)

        if feature_toggles is not None:
            flags = [bool(flag) for _, flag in feature_toggles.items()]
            if not flags or not any(flags):
                return "Features used: Baseline (no optional feature groups enabled)"

        return "Features used: not reported"

    def _feature_usage_summary_text(
        self,
        usage_summary: Iterable[Any] | None,
        feature_toggles: Mapping[str, bool] | None,
    ) -> str:
        lines: list[str] = []
        if isinstance(usage_summary, Iterable) and not isinstance(
            usage_summary, (str, bytes)
        ):
            for entry in usage_summary:
                if isinstance(entry, Mapping):
                    name = entry.get("group_name") or entry.get("name")
                    count = entry.get("count")
                else:
                    name = getattr(entry, "group_name", None) or getattr(
                        entry, "name", None
                    )
                    count = getattr(entry, "count", None)
                label = str(name).strip() if name is not None else ""
                if not label:
                    continue
                try:
                    count_value = int(count)
                except (TypeError, ValueError):
                    count_value = 0
                lines.append(f"• {label} ({count_value} signals)")

        if lines:
            return "\n".join(lines)

        if feature_toggles is not None and not any(feature_toggles.values()):
            return "No features enabled."

        if usage_summary is not None:
            return "No features enabled."

        return "Feature usage not reported."

    def _feature_group_highlights(
        self, summary: Mapping[str, Any] | None
    ) -> tuple[int | None, str]:
        if not isinstance(summary, Mapping):
            return None, ""
        columns = summary.get("top_features") or summary.get("columns") or []
        if isinstance(columns, Mapping):
            columns = list(columns.keys())
        if isinstance(columns, (str, bytes)):
            columns = [columns]
        if not isinstance(columns, Iterable) or isinstance(columns, (str, bytes)):
            columns = [] if not isinstance(columns, list) else columns
        column_names = [str(col).strip() for col in columns if str(col).strip()]
        column_count = len(column_names)
        preview_items = column_names[:3]
        preview = ", ".join(preview_items)
        if column_count > len(preview_items):
            preview = f"{preview}, …" if preview else "…"
        return column_count, preview

    def _feature_group_status_label(
        self,
        name: str,
        summary: Mapping[str, Any] | None,
        feature_toggles: Mapping[str, bool] | None,
    ) -> str:
        if not isinstance(summary, Mapping):
            return "—"
        executed = bool(summary.get("executed"))
        if executed:
            return "Active"
        configured = summary.get("configured")
        if configured is False or (
            feature_toggles and name in feature_toggles and not feature_toggles[name]
        ):
            return "Disabled"
        status = str(summary.get("status") or "").replace("_", " ").strip()
        return status or "Skipped"

    def _refresh_feature_group_table(
        self,
        feature_groups: Mapping[str, Mapping[str, Any]] | None,
        feature_toggles: Mapping[str, bool] | None,
    ) -> None:
        tree = getattr(self, "feature_group_detail_tree", None)
        if tree is None:
            return
        for item in tree.get_children():
            tree.delete(item)

        rows: list[tuple[str, str, str, str]] = []
        if isinstance(feature_groups, Mapping) and feature_groups:
            for name, summary in sorted(feature_groups.items()):
                status = self._feature_group_status_label(name, summary, feature_toggles)
                column_count, preview = self._feature_group_highlights(summary)
                count_display = "—" if column_count is None else str(column_count)
                rows.append((name, status, count_display, preview or "—"))
        elif feature_toggles:
            for name, enabled in sorted(feature_toggles.items()):
                status = "Enabled" if enabled else "Disabled"
                rows.append((name, status + " (no report)", "—", "—"))

        if not rows:
            rows.append(("—", "No feature data", "—", "—"))

        for values in rows:
            tree.insert("", tk.END, values=values)

    def _data_source_summary_text(self, source_entries: Iterable[Any]) -> str:
        """Render the list of external data providers used for the forecast."""

        lines: list[str] = []
        for entry in source_entries:
            if isinstance(entry, Mapping):
                provider = (
                    entry.get("id")
                    or entry.get("provider")
                    or entry.get("provider_id")
                    or "unknown"
                )
                description = entry.get("description")
                datasets = entry.get("datasets")
                detail_parts: list[str] = []
                if datasets:
                    dataset_list = [str(item) for item in datasets if item]
                    if dataset_list:
                        detail_parts.append(", ".join(dataset_list))
                if description:
                    detail_parts.append(str(description))
                detail = " – ".join(detail_parts)
                lines.append(f"• {provider}{(': ' + detail) if detail else ''}")
            else:
                value = str(entry).strip()
                if value:
                    lines.append(f"• {value}")

        if not lines:
            return "No external data sources reported."
        return "\n".join(lines)

    def _compute_confidence_metric(self) -> str:
        prediction = self.current_prediction if isinstance(self.current_prediction, Mapping) else {}
        confluence_block = prediction.get("signal_confluence") if isinstance(prediction, Mapping) else None
        confluence_passed = bool(confluence_block.get("passed")) if isinstance(confluence_block, Mapping) else False

        note_fragments: list[str] = []
        if isinstance(confluence_block, Mapping) and not confluence_passed:
            note_fragments.append("Confluence weak; probability scaled")
        confidence_note = prediction.get("confidence_note") if isinstance(prediction, Mapping) else None
        if confidence_note:
            note_fragments.append(str(confidence_note))

        gated_confidence = _safe_float(prediction.get("confluence_confidence"))
        if gated_confidence is not None:
            display = fmt_pct(gated_confidence, decimals=1)
            if note_fragments:
                display = f"{display} ({'; '.join(note_fragments)})"
            return display

        historical_confidence = _safe_float(prediction.get("historical_confidence"))
        if historical_confidence is not None:
            display = fmt_pct(historical_confidence, decimals=1)
            if note_fragments:
                display = f"{display} ({'; '.join(note_fragments)})"
            return display

        confidence = None
        confidence_block = prediction.get("confidence") if isinstance(prediction, Mapping) else None
        if isinstance(confidence_block, Mapping):
            numeric_values = [
                _safe_float(value)
                for value in confidence_block.values()
                if _safe_float(value) is not None
            ]
            if numeric_values:
                confidence = max(numeric_values)

        if confidence is None:
            up = _safe_float(prediction.get("direction_probability_up"))
            down = _safe_float(prediction.get("direction_probability_down"))
            target_hit = _safe_float(prediction.get("target_hit_probability"))
            candidates = [value for value in (up, down, target_hit) if value is not None]
            if candidates:
                confidence = max(candidates)

        if confidence is None:
            return "—"

        display = fmt_pct(confidence, decimals=1)
        if note_fragments:
            display = f"{display} ({'; '.join(note_fragments)})"
        return display

    def _extract_indicator_series(self, indicator: str) -> pd.Series | None:
        target_key = str(indicator).strip().lower()
        feature_history = (
            self.feature_history_converted
            if isinstance(self.feature_history_converted, pd.DataFrame)
            else self.feature_history
        )
        if isinstance(feature_history, pd.DataFrame) and not feature_history.empty:
            frame = feature_history.copy()
            columns = {str(col).lower(): col for col in frame.columns}
            col_name = columns.get(target_key)
            if col_name is not None:
                date_col = columns.get("date")
                if date_col is not None:
                    frame[date_col] = pd.to_datetime(frame[date_col], errors="coerce")
                    frame = frame.dropna(subset=[date_col]).sort_values(date_col)
                    index = pd.DatetimeIndex(frame[date_col])
                else:
                    index = pd.to_datetime(frame.index, errors="coerce")
                    valid_mask = pd.notna(index)
                    frame = frame.loc[valid_mask]
                    index = pd.DatetimeIndex(index[valid_mask])
                series = pd.to_numeric(frame[col_name], errors="coerce")
                series = pd.Series(series.values, index=index)
                series = series.dropna()
                if not series.empty:
                    return series

        indicator_history = (
            self.indicator_history_converted
            if isinstance(self.indicator_history_converted, pd.DataFrame)
            else self.indicator_history
        )
        if isinstance(indicator_history, pd.DataFrame) and not indicator_history.empty:
            columns = {str(column).lower(): column for column in indicator_history.columns}
            indicator_col = columns.get("indicator")
            value_col = columns.get("value")
            date_col = columns.get("date")
            if indicator_col and value_col:
                lower_names = {
                    str(value).strip().lower(): str(value).strip()
                    for value in indicator_history[indicator_col].dropna().unique()
                }
                if target_key not in lower_names:
                    return None
                subset = indicator_history[
                    indicator_history[indicator_col]
                    .astype(str)
                    .str.strip()
                    .str.lower()
                    == target_key
                ].copy()
                if date_col:
                    subset[date_col] = pd.to_datetime(subset[date_col], errors="coerce")
                values = pd.to_numeric(subset[value_col], errors="coerce")
                if values.empty:
                    return None
                mask = pd.notna(values)
                if date_col:
                    mask &= pd.notna(subset[date_col])
                subset = subset.loc[mask]
                values = values.loc[mask]
                if values.empty:
                    return None
                if date_col:
                    subset = subset.sort_values(date_col)
                    values = values.loc[subset.index]
                    index = pd.DatetimeIndex(subset[date_col])
                else:
                    index = pd.RangeIndex(start=0, stop=len(values))
                series = pd.Series(values.to_numpy(), index=index)
                if isinstance(series.index, pd.DatetimeIndex):
                    series = series.loc[~series.index.duplicated(keep="last")].sort_index()
                return series
        return None

    def _indicator_family_color(self, family: str) -> str:
        palette = (
            "tab:orange",
            "tab:green",
            "tab:red",
            "tab:purple",
            "tab:brown",
            "tab:pink",
            "tab:olive",
            "tab:cyan",
            "tab:gray",
            "tab:blue",
        )
        if family not in self._indicator_family_colors:
            index = len(self._indicator_family_colors) % len(palette)
            self._indicator_family_colors[family] = palette[index]
        return self._indicator_family_colors[family]

    def _prepare_price_series_for_indicators(
        self,
    ) -> tuple[pd.Series | None, pd.DatetimeIndex]:
        price_frame = (
            self.price_history_converted
            if isinstance(self.price_history_converted, pd.DataFrame)
            else self.price_history
        )
        if not isinstance(price_frame, pd.DataFrame) or price_frame.empty:
            return None, pd.DatetimeIndex([])

        frame = price_frame.copy()
        columns = {str(col).lower(): col for col in frame.columns}
        date_col = columns.get("date")
        if date_col:
            frame[date_col] = pd.to_datetime(frame[date_col], errors="coerce")
            frame = frame.dropna(subset=[date_col]).sort_values(date_col)
            index = pd.DatetimeIndex(frame[date_col])
        else:
            index = pd.to_datetime(frame.index, errors="coerce")
            valid_mask = pd.notna(index)
            frame = frame.loc[valid_mask]
            index = pd.DatetimeIndex(index[valid_mask])

        if index.empty:
            return None, pd.DatetimeIndex([])

        close_col = None
        for candidate in ("close", "adjclose", "adj_close"):
            close_col = columns.get(candidate)
            if close_col:
                break
        if close_col is None:
            numeric_cols = frame.select_dtypes(include=[np.number]).columns
            if len(numeric_cols):
                close_col = numeric_cols[0]
        if close_col is None:
            return None, pd.DatetimeIndex([])

        price_series = pd.to_numeric(frame[close_col], errors="coerce")
        price_series.index = index
        price_series = price_series.dropna()
        if price_series.empty:
            return None, pd.DatetimeIndex([])

        price_index = pd.DatetimeIndex(price_series.index)
        if price_index.has_duplicates:
            unique_mask = ~price_index.duplicated(keep="last")
            price_series = price_series.loc[unique_mask]
            price_index = pd.DatetimeIndex(price_series.index)
        price_series = price_series.loc[price_index].sort_index()
        price_index = pd.DatetimeIndex(price_series.index)
        return price_series, price_index

    def _align_indicator_to_price_index(
        self, series: pd.Series | None, price_index: pd.DatetimeIndex
    ) -> pd.Series | None:
        if series is None or not isinstance(series, pd.Series) or series.empty:
            return None
        if price_index.empty:
            return None

        cleaned = pd.to_numeric(series, errors="coerce")
        cleaned = cleaned.dropna()
        if cleaned.empty:
            return None

        if isinstance(cleaned.index, pd.DatetimeIndex):
            index = cleaned.index
            valid_mask = pd.notna(index)
            if not valid_mask.all():
                cleaned = cleaned.loc[valid_mask]
                index = index[valid_mask]
            try:
                index = index.tz_localize(None)
            except (TypeError, AttributeError, ValueError):
                try:
                    index = index.tz_convert(None)
                except Exception:
                    pass
            cleaned.index = pd.DatetimeIndex(index)
            if cleaned.empty:
                return None
            dedup_mask = ~cleaned.index.duplicated(keep="last")
            if not dedup_mask.all():
                cleaned = cleaned.loc[dedup_mask]
            cleaned = cleaned.sort_index()
            aligned = cleaned.reindex(price_index).dropna()
            if aligned.empty:
                return None
            return aligned

        trimmed = cleaned.copy()
        if len(trimmed) >= len(price_index):
            trimmed = trimmed.iloc[-len(price_index):]
            aligned_index = price_index
        else:
            aligned_index = price_index[-len(trimmed):]
        trimmed = trimmed.copy()
        trimmed.index = aligned_index
        return trimmed

    def _update_indicator_chart(self, selections: Iterable[str] | None = None) -> None:
        if not hasattr(self, "indicator_price_ax"):
            return
        ax = self.indicator_price_ax
        canvas = getattr(self, "indicator_price_canvas", None)
        widget = getattr(self, "indicator_price_canvas_widget", None)
        empty_label = getattr(self, "indicator_chart_message", None)

        def show_empty_state() -> None:
            if widget is not None:
                widget.grid_remove()
            if empty_label is not None:
                empty_label.grid(row=0, column=0, sticky="nsew")

        ax.clear()
        if self.indicator_secondary_ax is not None:
            try:
                self.indicator_secondary_ax.remove()
            except Exception:  # pragma: no cover - defensive cleanup
                pass
            self.indicator_secondary_ax = None
        if getattr(self, "_indicator_extra_axes", None):
            for extra_ax in list(self._indicator_extra_axes):
                try:
                    extra_ax.remove()
                except Exception:  # pragma: no cover - defensive cleanup
                    pass
            self._indicator_extra_axes = []

        price_series, price_index = self._prepare_price_series_for_indicators()
        if price_series is None or price_series.empty or price_index.empty:
            show_empty_state()
            if canvas is not None:
                self._style_figure(self.indicator_price_figure)
                canvas.draw_idle()
            return

        price_start = price_index.min()
        price_end = price_index.max()

        if empty_label is not None:
            empty_label.grid_remove()
        if widget is not None:
            widget.grid()

        (price_line,) = ax.plot(
            price_series.index, price_series.values, label="Price", color="tab:blue"
        )
        ylabel = "Price"
        if self.currency_symbol:
            ylabel = f"Price ({self.currency_symbol})"
        ax.set_ylabel(ylabel)
        ax.set_title(f"{self.config.ticker} price and indicators")
        ax.grid(True, linestyle="--", alpha=0.3)

        if pd.notna(price_start) and pd.notna(price_end):
            ax.set_xlim(price_start, price_end)

        def align_indicator_series(series: pd.Series) -> pd.Series | None:
            return self._align_indicator_to_price_index(series, price_index)

        if selections is None:
            selection_set = set(self._indicator_selection_cache)
        else:
            selection_set = {str(name) for name in selections}
        selected_names: list[str] = []
        if selection_set:
            ordered = self._indicator_names or list(selection_set)
            selected_names = [name for name in ordered if name in selection_set]
            for name in selection_set:
                if name not in selected_names:
                    selected_names.append(name)

        legend_entries: dict[str, Any] = {}
        legend_order: list[str] = []

        def register_legend(name: str, artist: Any | None) -> None:
            if artist is None:
                return
            if name not in legend_entries:
                legend_entries[name] = artist
                legend_order.append(name)

        register_legend("Price", price_line)

        pipeline = getattr(self.application, "pipeline", None)
        metadata = getattr(pipeline, "metadata", None)
        normalized_categories: dict[str, str] = {}
        if isinstance(metadata, Mapping):
            raw_categories = metadata.get("feature_categories")
            if isinstance(raw_categories, Mapping):
                normalized_categories = {
                    str(key).strip().lower(): str(value).strip()
                    for key, value in raw_categories.items()
                }

        @dataclass(slots=True)
        class IndicatorRenderItem:
            name: str
            family: str
            role: str
            style: str
            axis_role: str
            series: pd.Series
            color: str

        oscillator_keywords = (
            "rsi",
            "stoch",
            "momentum",
            "osc",
            "adx",
            "di",
            "obv",
            "roc",
            "ppo",
            "mfi",
            "cci",
        )

        def classify_indicator(name: str, category_label: str) -> tuple[str, str, str, str]:
            lower = name.lower()
            category_lower = category_label.lower()

            if "macd" in lower:
                period_match = re.search(r"(\d+)", lower)
                family = f"macd_{period_match.group(1)}" if period_match else "macd"
                if "hist" in lower or "histogram" in lower:
                    return family, "histogram", "histogram", "oscillator"
                return family, "macd_line", "line", "oscillator"

            if "bollinger" in lower or lower.startswith("bb_") or "bb " in lower:
                period_match = re.search(r"(\d+)", lower)
                family = (
                    f"bollinger_{period_match.group(1)}"
                    if period_match
                    else "bollinger"
                )
                if "bandwidth" in lower or "percent" in lower:
                    return family, "derived", "line", "oscillator"
                if any(token in lower for token in ("upper", "high", "top")):
                    return family, "upper", "band", "price"
                if any(token in lower for token in ("lower", "low", "bottom")):
                    return family, "lower", "band", "price"
                if any(token in lower for token in ("middle", "mid")):
                    return family, "middle", "line", "price"
                return family, "bollinger", "line", "price"

            if "supertrend" in lower:
                period_match = re.search(r"(\d+)", lower)
                family = (
                    f"supertrend_{period_match.group(1)}"
                    if period_match
                    else "supertrend"
                )
                if "direction" in lower:
                    return family, "direction", "zone", "price"
                return family, "trend", "zone", "price"

            if "volume" in lower:
                return name, "volume", "line", "indicator"

            if any(
                token in lower
                for token in ("ema", "sma", "hma", "wma", "vwma", "moving_average")
            ) or lower.endswith("_ma"):
                return name, "average", "line", "price"

            if (
                "oscillator" in category_lower
                or any(token in lower for token in oscillator_keywords)
            ):
                return name, "oscillator", "line", "oscillator"

            if "atr" in lower or "volatility" in category_lower:
                return name, "volatility", "line", "indicator"

            return name, "value", "line", "indicator"

        families: dict[str, list[IndicatorRenderItem]] = defaultdict(list)
        missing_names: set[str] = set()

        for name in selected_names:
            series = self._extract_indicator_series(name)
            if series is None or series.empty:
                missing_names.add(name)
                continue
            aligned_series = align_indicator_series(series)
            if aligned_series is None or aligned_series.empty:
                missing_names.add(name)
                continue
            aligned_series.name = name
            category_label = normalized_categories.get(name.lower(), "")
            family, role, style, axis_role = classify_indicator(name, category_label)
            color = self._indicator_family_color(family)
            families[family].append(
                IndicatorRenderItem(
                    name=name,
                    family=family,
                    role=role,
                    style=style,
                    axis_role=axis_role,
                    series=aligned_series,
                    color=color,
                )
            )

        axes_by_role: dict[str, Axes] = {"price": ax}
        oscillator_zero_added = False

        def blend_color(base: str, target: str, weight: float, *, alpha: float | None = None) -> tuple[float, float, float, float]:
            weight = min(max(weight, 0.0), 1.0)
            base_rgb = np.array(mcolors.to_rgb(base))
            target_rgb = np.array(mcolors.to_rgb(target))
            mixed = (1 - weight) * base_rgb + weight * target_rgb
            rgba = (*mixed, 1.0)
            if alpha is not None:
                rgba = (rgba[0], rgba[1], rgba[2], alpha)
            return rgba

        def get_axis(role: str) -> Axes:
            nonlocal oscillator_zero_added
            if role == "price":
                return ax
            if role == "indicator":
                indicator_ax = axes_by_role.get("indicator")
                if indicator_ax is None:
                    indicator_ax = ax.twinx()
                    indicator_ax.set_ylabel("Indicator value")
                    indicator_ax.grid(False)
                    axes_by_role["indicator"] = indicator_ax
                    self._indicator_extra_axes.append(indicator_ax)
                    self.indicator_secondary_ax = indicator_ax
                    if "oscillator" in axes_by_role:
                        axes_by_role["oscillator"].spines["right"].set_position(("axes", 1.1))
                return indicator_ax
            if role == "oscillator":
                oscillator_ax = axes_by_role.get("oscillator")
                if oscillator_ax is None:
                    oscillator_ax = ax.twinx()
                    oscillator_ax.set_ylabel("Oscillator value")
                    oscillator_ax.grid(False)
                    axes_by_role["oscillator"] = oscillator_ax
                    self._indicator_extra_axes.append(oscillator_ax)
                    if "indicator" in axes_by_role:
                        oscillator_ax.spines["right"].set_position(("axes", 1.1))
                    else:
                        oscillator_ax.spines["right"].set_position(("axes", 1.0))
                if not oscillator_zero_added:
                    oscillator_ax.axhline(0, color="#808080", linestyle=":", linewidth=0.8, alpha=0.6)
                    oscillator_zero_added = True
                return oscillator_ax
            return get_axis("indicator")

        for family, items in families.items():
            plotted: set[str] = set()

            band_members = [item for item in items if item.style == "band"]
            if band_members:
                upper = next((item for item in band_members if item.role == "upper"), None)
                lower = next((item for item in band_members if item.role == "lower"), None)
                axis = get_axis("price")
                if upper is not None and lower is not None:
                    aligned_upper, aligned_lower = upper.series.align(lower.series, join="inner")
                    aligned_upper = aligned_upper.dropna()
                    aligned_lower = aligned_lower.dropna()
                    common_index = aligned_upper.index.intersection(aligned_lower.index)
                    if not common_index.empty:
                        axis.fill_between(
                            common_index,
                            aligned_upper.reindex(common_index),
                            aligned_lower.reindex(common_index),
                            color=blend_color(upper.color, "white", 0.5, alpha=0.15),
                            label="_nolegend_",
                        )
                    for member in (upper, lower):
                        line, = axis.plot(
                            member.series.index,
                            member.series.values,
                            color=member.color,
                            linestyle="--",
                            label=member.name,
                        )
                        register_legend(member.name, line)
                        plotted.add(member.name)
                else:
                    for member in band_members:
                        line, = axis.plot(
                            member.series.index,
                            member.series.values,
                            color=member.color,
                            linestyle="--",
                            label=member.name,
                        )
                        register_legend(member.name, line)
                        plotted.add(member.name)

            trend_member = next((item for item in items if item.role == "trend"), None)
            direction_member = next((item for item in items if item.role == "direction"), None)
            if trend_member is not None:
                axis = get_axis("price")
                line, = axis.plot(
                    trend_member.series.index,
                    trend_member.series.values,
                    color=trend_member.color,
                    linestyle="--",
                    label=trend_member.name,
                )
                register_legend(trend_member.name, line)
                plotted.add(trend_member.name)
                if direction_member is not None:
                    direction_series = direction_member.series.reindex(trend_member.series.index)
                    price_aligned = price_series.reindex(trend_member.series.index)
                    zone_frame = pd.DataFrame(
                        {
                            "trend": trend_member.series,
                            "direction": direction_series,
                            "price": price_aligned,
                        }
                    ).dropna()
                    if not zone_frame.empty:
                        bullish_mask = zone_frame["direction"] >= 0
                        bearish_mask = zone_frame["direction"] < 0
                        bullish_fill = axis.fill_between(
                            zone_frame.index,
                            zone_frame["price"],
                            zone_frame["trend"],
                            where=bullish_mask,
                            color=blend_color(direction_member.color, "white", 0.4, alpha=0.18),
                            interpolate=True,
                            label=direction_member.name,
                        )
                        axis.fill_between(
                            zone_frame.index,
                            zone_frame["price"],
                            zone_frame["trend"],
                            where=bearish_mask,
                            color=blend_color(direction_member.color, "black", 0.25, alpha=0.18),
                            interpolate=True,
                            label="_nolegend_",
                        )
                        register_legend(direction_member.name, bullish_fill)
                        plotted.add(direction_member.name)

            for item in items:
                if item.name in plotted:
                    continue
                axis = get_axis(item.axis_role)
                if item.style == "histogram":
                    x_values = item.series.index
                    if len(x_values) > 1:
                        numeric_index = date2num(x_values.to_pydatetime())
                        width = float(np.diff(numeric_index).mean()) * 0.8
                    else:
                        width = 0.6
                    positive = item.series[item.series >= 0]
                    negative = item.series[item.series < 0]
                    if not positive.empty:
                        axis.bar(
                            positive.index,
                            positive.values,
                            width=width,
                            color=blend_color(item.color, "white", 0.25, alpha=0.65),
                            align="center",
                        )
                    if not negative.empty:
                        axis.bar(
                            negative.index,
                            negative.values,
                            width=width,
                            color=blend_color(item.color, "black", 0.3, alpha=0.65),
                            align="center",
                        )
                    legend_patch = Rectangle(
                        (0, 0),
                        1,
                        1,
                        facecolor=blend_color(item.color, "white", 0.25, alpha=0.8),
                        edgecolor="none",
                    )
                    register_legend(item.name, legend_patch)
                else:
                    linestyle = "--" if item.axis_role != "price" else "-"
                    line, = axis.plot(
                        item.series.index,
                        item.series.values,
                        color=item.color,
                        linestyle=linestyle,
                        label=item.name,
                    )
                    register_legend(item.name, line)

        if pd.notna(price_start) and pd.notna(price_end):
            for axis_obj in axes_by_role.values():
                axis_obj.set_xlim(price_start, price_end)

        if "indicator" not in axes_by_role and "oscillator" in axes_by_role:
            self.indicator_secondary_ax = axes_by_role["oscillator"]

        if legend_order:
            handles = [legend_entries[label] for label in legend_order]
            ax.legend(handles, legend_order, loc="best")

        if missing_names:
            self._indicator_selection_cache.difference_update(missing_names)
            if hasattr(self, "indicator_listbox"):
                self._indicator_selector_updating = True
                try:
                    values = self.indicator_listbox.get(0, tk.END)
                    for index, value in enumerate(values):
                        if value in missing_names:
                            self.indicator_listbox.selection_clear(index)
                finally:
                    self._indicator_selector_updating = False
            self._update_indicator_toggle_button_state(
                self._indicator_names, self._indicator_selection_cache
            )

        ax.set_xlabel("Date")
        ax.xaxis.set_major_formatter(DateFormatter("%b-%Y"))

        if canvas is not None:
            self._style_figure(self.indicator_price_figure)
            canvas.draw_idle()

    def _on_feature_toggle_changed(self) -> None:
        if self._busy:
            return
        toggles: dict[str, bool] = {}
        for name in sorted(IMPLEMENTED_FEATURE_GROUPS):
            if name in self.feature_toggle_vars:
                toggles[name] = bool(self.feature_toggle_vars[name].get())
            else:
                toggles[name] = bool(self.config.feature_toggles.get(name, False))
        toggles_obj = FeatureToggles.from_any(toggles, defaults=self.config.feature_toggles)
        self.config.feature_toggles = toggles_obj
        self.config.price_feature_toggles = derive_price_feature_toggles(toggles_obj)
        self.application.pipeline = StockPredictorAI(self.config)
        status_message = "Feature toggles updated."
        self._set_busy(True, "Recomputing indicators for updated feature toggles…")
        try:
            indicators_built = self.application.pipeline.refresh_indicators(force=True)
            indicators: pd.DataFrame | None = None
            try:
                indicators = self.application.pipeline.fetcher.fetch_indicator_data()
            except Exception as exc:  # pragma: no cover - optional data source
                LOGGER.debug("Indicator dataset unavailable after refresh: %s", exc)
            self.indicator_history = indicators if isinstance(indicators, pd.DataFrame) else None
            self.indicator_history_converted = None
            self._indicator_selection_cache = set()
            self._indicator_user_override = False
            self._update_indicator_view()
            if indicators_built:
                status_message = f"Feature toggles updated. {indicators_built} indicators refreshed."
            else:
                status_message = "Feature toggles updated. No indicators available for this selection."
        except Exception as exc:  # pragma: no cover - defensive UI update
            LOGGER.exception("Failed to refresh indicators after toggling features: %s", exc)
            status_message = "Feature toggles updated. Re-run prediction to apply changes."
        finally:
            self._set_busy(False, status_message)

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
        self.market_timezone = resolve_market_timezone(self.config)
        self.root.title(f"Stock Predictor – {self.config.ticker}")
        self.ticker_var.set(self.config.ticker)
        self._refresh_settings_labels()
        self._configure_horizons(self.config.prediction_horizons)
        self.market_holidays = []
        self.current_forecast_date = None
        self.forecast_date_var.set("Forecast date: —")
        self.trend_results = []
        self.trend_status_var.set("Select a horizon and click “Find Opportunities”.")
        self._refresh_trend_table()
        self._refresh_overview()
        self._run_async(self._refresh_and_predict, f"Loading data for {self.config.ticker}…")

    # ------------------------------------------------------------------
    # Core actions
    # ------------------------------------------------------------------
    def _initialise_prediction(self) -> None:
        self._on_predict()

    def _refresh_and_predict(self) -> dict[str, Any]:
        self.application.refresh_data(force=True)
        return self._predict_payload(refresh=True)

    def _predict_only(self) -> dict[str, Any]:
        return self._predict_payload(refresh=True)

    def _log_prediction_unavailability(
        self,
        *,
        horizon: Any,
        status: str | None,
        reason: str | None,
        message: str | None = None,
    ) -> None:
        if status != "no_data":
            return
        key = f"{horizon}:{reason}"
        seen = self._availability_log_state.get(key, False)
        log_fn = LOGGER.warning if not seen else LOGGER.info
        log_fn(
            "Prediction unavailable for %s (status=%s, reason=%s)%s",
            self.config.ticker,
            status,
            reason,
            f": {message}" if message else "",
        )
        self._availability_log_state[key] = True

    def _predict_payload(self, *, refresh: bool = False) -> dict[str, Any]:
        horizon_arg: Any
        if self.selected_horizon_code:
            horizon_arg = self.selected_horizon_code
        else:
            horizon_arg = self.selected_horizon_offset
        self._sync_stop_loss_multiplier()
        prediction = self.application.predict(horizon=horizon_arg, refresh=refresh)
        raw_payload: Mapping[str, Any] | None = None
        try:
            if hasattr(prediction, "to_dict"):
                raw_payload = prediction.to_dict()
        except Exception as exc:  # pragma: no cover - defensive guard for optional metadata
            LOGGER.debug("Failed to serialise prediction for UI payload: %s", exc)
        if raw_payload is None and isinstance(prediction, Mapping):
            raw_payload = prediction

        status = None
        reason = None
        message = None
        if isinstance(raw_payload, Mapping):
            status = raw_payload.get("status")
            reason = raw_payload.get("reason")
            message = raw_payload.get("message")
        payload_horizon = raw_payload.get("horizon") if isinstance(raw_payload, Mapping) else None
        if status == "no_data":
            friendly_message = (
                message
                or f"Not enough historical data to generate predictions for {self.config.ticker} (horizon {payload_horizon or horizon_arg}) yet."
            )
            LOGGER.info(friendly_message)
            self._log_prediction_unavailability(
                horizon=horizon_arg,
                status=status,
                reason=reason,
                message=friendly_message,
            )
            return {
                "prediction": {
                    "status": status,
                    "reason": reason,
                    "message": friendly_message,
                    "sample_counts": raw_payload.get("sample_counts", {}) if isinstance(raw_payload, Mapping) else {},
                    "missing_targets": raw_payload.get("missing_targets", {}) if isinstance(raw_payload, Mapping) else {},
                    "checked_at": raw_payload.get("checked_at") if isinstance(raw_payload, Mapping) else None,
                },
                "snapshot": None,
                "feature_history": None,
                "price_history": None,
                "indicator_history": None,
                "horizons": self.config.prediction_horizons,
                "stop_loss_multiplier": self.stop_loss_multiplier,
                "raw": raw_payload,
                "message": friendly_message,
            }
        if status == "error":
            friendly_message = message or "An error occurred while generating predictions."
            LOGGER.error(
                "Prediction failed for %s (horizon %s): %s",
                self.config.ticker,
                payload_horizon or horizon_arg,
                friendly_message,
                exc_info=True,
            )
            return {
                "prediction": {
                    "status": status,
                    "reason": reason,
                    "message": friendly_message,
                },
                "snapshot": None,
                "feature_history": None,
                "price_history": None,
                "indicator_history": None,
                "horizons": self.config.prediction_horizons,
                "stop_loss_multiplier": self.stop_loss_multiplier,
                "raw": raw_payload,
                "message": friendly_message,
            }
        prediction_meta: Mapping[str, Any] | None = None
        if isinstance(prediction, PredictionResult):
            prediction_meta = prediction.meta
        metadata = prediction_meta or self.application.pipeline.metadata
        snapshot = metadata.get("latest_features") if isinstance(metadata, Mapping) else None
        horizon_values: Iterable[Any] | None = None
        if isinstance(metadata, Mapping):
            horizon_values = metadata.get("horizons")
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
            "horizons": horizon_values if horizon_values is not None else self.config.prediction_horizons,
            "stop_loss_multiplier": self.stop_loss_multiplier,
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
            except NoPriceDataError as exc:
                LOGGER.warning("No price data available for %s: %s", exc.ticker, exc)
                self.root.after(0, lambda err=exc: self._on_no_price_data_error(err))
            except InsufficientSamplesError as exc:
                LOGGER.info("Insufficient samples for prediction: %s", exc)
                self.root.after(0, lambda err=exc: self._on_insufficient_samples(err))
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

        horizons_payload = payload.get("horizons")
        if horizons_payload is not None:
            self._configure_horizons(horizons_payload)

        prediction = payload.get("prediction", {})
        snapshot = payload.get("snapshot")
        feature_history = payload.get("feature_history")
        price_history = payload.get("price_history")
        indicator_history = payload.get("indicator_history")

        status = None
        reason = None
        message = None
        if isinstance(prediction, Mapping):
            status = prediction.get("status")
            reason = prediction.get("reason")
            message = prediction.get("message")
        elif isinstance(prediction, PredictionResult):
            status = prediction.status
            reason = prediction.reason
            message = prediction.message
        if status == "no_data":
            summary = f"Prediction unavailable (status={status}, reason={reason or 'unknown'})"
            LOGGER.warning(summary)
            user_message = (
                message
                if message and "Horizon" not in str(message)
                else f"Not enough historical data to generate predictions for {self.config.ticker} yet."
            )
            LOGGER.info(user_message)
            self.current_prediction = {}
            self.feature_snapshot = None
            self.feature_history = None
            self.price_history = None
            self.indicator_history = None
            self._set_busy(False, user_message)
            messagebox.showinfo("Predictions unavailable", user_message)
            return
        if status == "error":
            message = message or "An error occurred while generating predictions."
            LOGGER.error(
                "Prediction failed for %s (horizon %s): %s",
                self.config.ticker,
                prediction.get("horizon") if isinstance(prediction, Mapping) else "?",
                message,
                exc_info=True,
            )
            self._set_busy(False, message)
            messagebox.showerror("Prediction failed", message)
            return

        is_prediction_mapping = isinstance(prediction, (Mapping, PredictionResult))
        self.current_prediction = prediction if is_prediction_mapping else {}
        prediction_timestamp = None
        if isinstance(self.current_prediction, (Mapping, PredictionResult)):
            prediction_timestamp = self._localize_market_timestamp(
                self.current_prediction.get("market_data_as_of")
                or self.current_prediction.get("as_of")
            )
        if snapshot is None and isinstance(feature_history, pd.DataFrame) and not feature_history.empty:
            snapshot = feature_history.iloc[[-1]]
        self.feature_snapshot = snapshot if isinstance(snapshot, pd.DataFrame) else None
        self.feature_history = feature_history if isinstance(feature_history, pd.DataFrame) else None
        self.price_history = price_history if isinstance(price_history, pd.DataFrame) else None
        self.indicator_history = indicator_history if isinstance(indicator_history, pd.DataFrame) else None
        self._update_local_currency_profile(self.price_history)
        self._sync_horizon_from_prediction(self.current_prediction)
        self.market_holidays = self._resolve_market_holidays(self.current_prediction)
        latest_price, latest_timestamp = self._extract_latest_price_point()
        localized_price_timestamp = self._localize_market_timestamp(latest_timestamp)
        self.current_market_timestamp = localized_price_timestamp or prediction_timestamp
        self.market_timestamp_stale = localized_price_timestamp is None
        self.current_market_price = latest_price
        if self.current_market_price is None:
            self.current_market_price = _safe_float(
                self.current_prediction.get("last_price")
                or self.current_prediction.get("last_close")
            )
        self._apply_currency(mode=self.currency_mode)
        self._update_forecast_label(self.current_prediction.get("target_date"))
        self._update_metrics()
        self._update_price_chart()
        self._update_indicator_view()
        self._update_explanation()
        self._set_busy(False, "Prediction updated.")

    def _on_async_failure(self, exc: Exception) -> None:
        self._set_busy(False, "An error occurred.")
        messagebox.showerror("Prediction failed", str(exc))

    def _on_no_price_data_error(self, exc: NoPriceDataError) -> None:
        ticker = getattr(exc, "ticker", self.config.ticker)
        detail = str(exc)
        status = detail if detail else f"No data available for {ticker}."
        self._set_busy(False, status)
        messagebox.showwarning("No price data", detail)

    def _on_insufficient_samples(self, exc: InsufficientSamplesError) -> None:
        message = "Not enough historical data to generate predictions yet"
        detail = str(exc).strip()
        self._set_busy(False, message)
        display_message = f"{message}\n\n{detail}" if detail and detail != message else message
        messagebox.showinfo("Predictions unavailable", display_message)

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
        if not busy:
            self._on_currency_mode_changed(self.currency_mode_var.get())
        if busy:
            self.progress.start(12)
        else:
            self.progress.stop()
        if status:
            self._set_status(status)

    def _set_status(self, message: str) -> None:
        self.status_var.set(message)

    def _now(self) -> pd.Timestamp:
        return pd.Timestamp.now(tz=self.market_timezone)

    def _today(self) -> pd.Timestamp:
        return self._now().normalize()

    def _resolve_market_timestamp(self) -> pd.Timestamp | None:
        latest_timestamp: pd.Timestamp | None = None
        has_history = isinstance(self.price_history, pd.DataFrame) and not self.price_history.empty
        if has_history:
            _, latest_timestamp = self._extract_latest_price_point(self.price_history)
        elif self.current_market_timestamp is not None:
            latest_timestamp = self.current_market_timestamp

        localized = self._localize_market_timestamp(latest_timestamp)
        self.market_timestamp_stale = localized is None or not has_history
        if localized is not None:
            self.current_market_timestamp = localized
        return localized

    def _extract_latest_price_point(
        self, frame: pd.DataFrame | None = None
    ) -> tuple[float | None, pd.Timestamp | None]:
        if frame is None:
            frame = self.price_history

        if not isinstance(frame, pd.DataFrame) or frame.empty:
            return None, None

        columns = {str(column).lower(): column for column in frame.columns}
        timestamp_series: pd.Series | pd.Index | None = None
        if "date" in columns:
            timestamp_series = pd.to_datetime(frame[columns["date"]], errors="coerce")
        else:
            timestamp_series = pd.to_datetime(frame.index, errors="coerce")

        valid_mask = pd.notna(timestamp_series)
        if isinstance(timestamp_series, pd.Series):
            timestamps = pd.DatetimeIndex(timestamp_series[valid_mask])
            data_frame = frame.loc[valid_mask]
        else:
            timestamps = pd.DatetimeIndex(timestamp_series[valid_mask])
            data_frame = frame.loc[valid_mask]

        if timestamps.empty:
            return None, None

        price_col = None
        for key in ("close", "adj close", "adjclose", "adj_close", "last", "price"):
            if key in columns:
                price_col = columns[key]
                break

        if price_col is None:
            numeric_cols = [
                col for col in data_frame.columns if pd.api.types.is_numeric_dtype(data_frame[col])
            ]
            price_col = numeric_cols[0] if numeric_cols else None

        if price_col is None:
            return None, timestamps[-1]

        prices = pd.to_numeric(data_frame[price_col], errors="coerce")
        price_series = pd.Series(prices.values, index=timestamps).dropna()
        if price_series.empty:
            return None, timestamps[-1]

        return float(price_series.iloc[-1]), price_series.index[-1]

    def _localize_market_timestamp(self, value: Any) -> pd.Timestamp | None:
        if value is None:
            return None
        try:
            parsed = pd.to_datetime(value, errors="coerce")
        except Exception:  # pragma: no cover - defensive
            return None
        if pd.isna(parsed):
            return None
        ts = pd.Timestamp(parsed)
        tz = getattr(self, "market_timezone", None)
        if tz is None:
            return ts
        if ts.tzinfo is None:
            try:
                return ts.tz_localize(tz)
            except Exception:  # pragma: no cover - defensive fallback
                try:
                    return ts.tz_localize("UTC").tz_convert(tz)
                except Exception:
                    return None
        try:
            return ts.tz_convert(tz)
        except Exception:  # pragma: no cover - defensive fallback
            return ts.tz_localize(tz)

    def _forecast_base_date(self) -> pd.Timestamp:
        timestamp = self._resolve_market_timestamp()
        if timestamp is not None:
            return timestamp.normalize()

        if isinstance(self.price_history, pd.DataFrame) and not self.price_history.empty:
            frame = self.price_history.copy()
            lower_map = {str(column).lower(): column for column in frame.columns}
            if "date" in lower_map:
                dates = pd.to_datetime(frame[lower_map["date"]], errors="coerce").dropna()
                if not dates.empty:
                    localized = self._localize_market_timestamp(dates.iloc[-1])
                    if localized is not None:
                        return localized.normalize()
            index_dates = pd.to_datetime(frame.index, errors="coerce")
            index_series = pd.Series(index_dates).dropna()
            if not index_series.empty:
                localized = self._localize_market_timestamp(index_series.iloc[-1])
                if localized is not None:
                    return localized.normalize()

        return self._today()

    def _compute_forecast_date(
        self, target_date: Any | None = None, *, offset: int | None = None
    ) -> pd.Timestamp | None:
        base_date = self._forecast_base_date()
        parsed = (
            self._localize_market_timestamp(target_date)
            if target_date is not None
            else None
        )
        holidays = self.market_holidays or self._resolve_market_holidays()
        if not self.market_holidays and holidays:
            self.market_holidays = list(holidays)

        forecast: pd.Timestamp | None = None
        if parsed is not None:
            try:
                forecast = _ensure_future_trading_day(
                    base_date,
                    parsed.normalize(),
                    context=str(target_date),
                )
            except ValueError as exc:
                LOGGER.warning("Discarding invalid forecast date %s: %s", target_date, exc)
                forecast = None
        else:
            days = offset if offset is not None else self.selected_horizon_offset
            try:
                days_int = int(days)
            except (TypeError, ValueError):
                days_int = 0
            if days_int > 0:
                try:
                    forecast = _trading_day_for_horizon(
                        base_date,
                        business_days=days_int,
                        holidays=holidays,
                        context=self.selected_horizon_label or self.selected_horizon_code,
                    )
                except ValueError as exc:
                    LOGGER.warning(
                        "Unable to compute forecast date for %s business days from %s: %s",
                        days_int,
                        base_date.date().isoformat(),
                        exc,
                    )
                    forecast = None

        self.current_forecast_date = forecast
        return forecast

    def _update_forecast_label(self, target_date: Any | None = None) -> None:
        display = "Forecast date: —"
        forecast = self._compute_forecast_date(target_date)
        if forecast is not None:
            local_forecast = (
                forecast.tz_convert(self.market_timezone)
                if forecast.tzinfo
                else forecast.tz_localize(self.market_timezone)
            )
            display = f"Forecast date: {local_forecast.date().isoformat()}"
        self.forecast_date_var.set(display)

    # ------------------------------------------------------------------
    # UI updates
    # ------------------------------------------------------------------
    def _compute_expected_low(
        self,
        prediction: Mapping[str, Any] | None,
        *,
        multiplier: float | None = None,
    ) -> float | None:
        if not isinstance(prediction, Mapping):
            return None

        def _extract_lower(block: Mapping[str, Any] | None) -> float | None:
            if not isinstance(block, Mapping):
                return None
            preferred_keys = ("lower", "lower_bound", "low", "p10", "10%", "10", "0.1")
            for key in preferred_keys:
                if key in block:
                    candidate = _safe_float(block.get(key))
                    if candidate is not None:
                        return candidate
            numeric_values: list[float] = []
            for value in block.values():
                numeric = _safe_float(value)
                if numeric is not None:
                    numeric_values.append(numeric)
            if numeric_values:
                return float(min(numeric_values))
            return None

        def _max_drawdown_from_history(frame: pd.DataFrame | None) -> float | None:
            if frame is None or frame.empty:
                return None
            lower_columns = {column.lower(): column for column in frame.columns}
            close_column = lower_columns.get("close") or lower_columns.get("adj close")
            if close_column is None:
                return None
            closes = pd.to_numeric(frame[close_column], errors="coerce").dropna()
            if closes.empty:
                return None
            running_max = closes.cummax()
            drawdowns = closes / running_max - 1.0
            if drawdowns.empty:
                return None
            return float(abs(drawdowns.min()))

        def _trailing_low_from_history(frame: pd.DataFrame | None, window: int) -> float | None:
            if frame is None or frame.empty or window <= 0:
                return None
            lower_columns = {column.lower(): column for column in frame.columns}
            low_column = (
                lower_columns.get("low")
                or lower_columns.get("l")
                or lower_columns.get("close")
                or lower_columns.get("adj close")
            )
            if low_column is None:
                return None
            lows = pd.to_numeric(frame[low_column], errors="coerce").dropna()
            if lows.empty:
                return None
            if len(lows) > window:
                lows = lows.tail(window)
            return float(lows.min())

        intervals = prediction.get("prediction_intervals")
        if isinstance(intervals, Mapping):
            close_interval = intervals.get("close")
            lower = _extract_lower(close_interval)
            if lower is None:
                for candidate in intervals.values():
                    lower = _extract_lower(candidate)
                    if lower is not None:
                        break
            if lower is not None:
                return lower

        quantiles = prediction.get("quantile_forecasts")
        if isinstance(quantiles, Mapping):
            close_quantiles = quantiles.get("close")
            lower = _extract_lower(close_quantiles)
            if lower is None:
                for candidate in quantiles.values():
                    lower = _extract_lower(candidate)
                    if lower is not None:
                        break
            if lower is not None:
                return lower

        indicator_low = _safe_float(prediction.get("indicator_expected_low"))
        if indicator_low is None:
            indicator_low = self._indicator_expected_low_from_local_data()

        trailing_window = int(
            _safe_float(prediction.get("trailing_low_window"))
            or self.expected_low_floor_window
        )
        trailing_low_prediction = _safe_float(prediction.get("trailing_low"))

        frame_source: pd.DataFrame | None = None
        price_history_converted = getattr(self, "price_history_converted", None)
        if isinstance(price_history_converted, pd.DataFrame) and not price_history_converted.empty:
            frame_source = price_history_converted
        else:
            price_history = getattr(self, "price_history", None)
            if isinstance(price_history, pd.DataFrame) and not price_history.empty:
                frame_source = price_history

        trailing_low_history = _trailing_low_from_history(frame_source, trailing_window)
        trailing_low_candidates = [
            value
            for value in (trailing_low_prediction, trailing_low_history)
            if value is not None
        ]
        trailing_low_value = max(trailing_low_candidates) if trailing_low_candidates else None

        floor_candidates: list[float] = []
        if indicator_low is not None and indicator_low > 0:
            floor_candidates.append(indicator_low)
        if trailing_low_value is not None and trailing_low_value > 0:
            floor_candidates.append(trailing_low_value)
        floor_value = max(floor_candidates) if floor_candidates else None

        predicted_close = _safe_float(prediction.get("predicted_close"))
        predicted_volatility = _safe_float(prediction.get("predicted_volatility"))
        fallback = _safe_float(prediction.get("expected_low"))
        if predicted_close is None:
            return floor_value if floor_value is not None else fallback
        if predicted_volatility is None:
            base_low = fallback if fallback is not None else predicted_close
            if floor_value is not None:
                base_low = max(base_low, floor_value)
            return float(base_low)
        scale = multiplier if multiplier is not None else self.expected_low_multiplier
        try:
            scale_value = float(scale)
        except (TypeError, ValueError):
            scale_value = self.expected_low_multiplier
        if not np.isfinite(scale_value):
            scale_value = self.expected_low_multiplier
        scale_value = max(0.0, scale_value)

        volatility_pct = abs(float(predicted_volatility))
        if volatility_pct > 1.0 and volatility_pct <= 100:
            LOGGER.warning(
                "Predicted volatility %.4f interpreted as percentage; expected fractional (e.g., 0.02).",
                volatility_pct,
            )
            volatility_pct /= 100.0

        max_volatility = self.expected_low_max_volatility if self.expected_low_max_volatility > 0 else 1.0
        historical_cap = _safe_float(prediction.get("max_drawdown_fraction"))
        if historical_cap is None:
            historical_cap = _max_drawdown_from_history(frame_source)
        cap_candidates = [value for value in (max_volatility, historical_cap) if value and value > 0]
        volatility_cap = min(cap_candidates) if cap_candidates else max_volatility
        if volatility_pct > volatility_cap:
            LOGGER.warning(
                "Predicted volatility %.4f exceeds cap %.4f; clipping.",
                volatility_pct,
                volatility_cap,
            )
            volatility_pct = volatility_cap

        delta = predicted_close * volatility_pct * scale_value
        if not np.isfinite(delta):
            delta = 0.0
        expected_low = predicted_close - delta
        expected_low = max(0.0, expected_low)
        if floor_value is not None:
            expected_low = max(expected_low, floor_value)
        return float(expected_low)

    def _indicator_expected_low_from_local_data(self) -> float | None:
        frames: list[pd.DataFrame] = []
        snapshot = getattr(self, "feature_snapshot", None)
        indicators = getattr(self, "indicator_history", None)
        if isinstance(snapshot, pd.DataFrame) and not snapshot.empty:
            frames.append(snapshot)
        if isinstance(indicators, pd.DataFrame) and not indicators.empty:
            frames.append(indicators)

        best_value: float | None = None
        for frame in frames:
            indicator_value, _ = indicator_support_floor(frame)
            numeric = _safe_float(indicator_value)
            if numeric is None or numeric <= 0:
                continue
            if best_value is None or numeric < best_value:
                best_value = numeric
        return best_value

    def _reference_last_price(
        self, prediction: Mapping[str, Any] | None = None
    ) -> tuple[float | None, float | None]:
        """Return the raw and converted reference price for calculations."""

        data = prediction if isinstance(prediction, Mapping) else {}
        raw_value: float | None = None
        converted: float | None = None

        if isinstance(self.price_history, pd.DataFrame) and not self.price_history.empty:
            raw_value, _ = self._extract_latest_price_point(self.price_history)
            converted_frame = getattr(self, "price_history_converted", None)
            if isinstance(converted_frame, pd.DataFrame) and not converted_frame.empty:
                converted, _ = self._extract_latest_price_point(converted_frame)
        else:
            raw_value = _safe_float(data.get("last_price"))
            if raw_value is None:
                raw_value = _safe_float(data.get("last_close"))
            if raw_value is None:
                raw_value = self.current_market_price

        if converted is None:
            converted = self._convert_currency(raw_value)
        return raw_value, converted

    def _derive_prediction_metrics(
        self, prediction: Mapping[str, Any] | None = None
    ) -> dict[str, float | None]:
        """Normalise prediction fields around a shared anchor price."""

        anchor_raw, anchor_converted = self._reference_last_price(prediction)
        predicted_raw = _safe_float(prediction.get("predicted_close")) if isinstance(
            prediction, Mapping
        ) else None
        predicted_converted = self._convert_currency(predicted_raw)

        change_raw = _safe_float(prediction.get("expected_change")) if isinstance(
            prediction, Mapping
        ) else None
        pct_change = _safe_float(prediction.get("expected_change_pct")) if isinstance(
            prediction, Mapping
        ) else None

        if predicted_raw is not None and anchor_raw is not None:
            change_raw = predicted_raw - anchor_raw
            pct_change = (change_raw / anchor_raw) if anchor_raw != 0 else None
        elif change_raw is not None and anchor_raw not in (None, 0) and pct_change is None:
            pct_change = change_raw / anchor_raw

        change_converted = self._convert_currency(change_raw)

        return {
            "anchor_raw": anchor_raw,
            "anchor_converted": anchor_converted,
            "predicted_raw": predicted_raw,
            "predicted_converted": predicted_converted,
            "change_raw": change_raw,
            "change_converted": change_converted,
            "pct_change": pct_change,
        }

    def _extract_sentiment_from_row(
        self, row: Mapping[str, Any] | pd.Series | None
    ) -> tuple[float | None, float | None]:
        if row is None:
            return None, None

        def getter(key: str) -> Any:
            if hasattr(row, "get"):
                try:
                    return row.get(key)
                except Exception:  # pragma: no cover - defensive
                    return None
            try:
                return row[key]
            except Exception:  # pragma: no cover - defensive
                return None

        avg: float | None = None
        for key in ("Sentiment_Avg", "sentiment", "Score", "score", "sentiment_score"):
            avg = _safe_float(getter(key))
            if avg is not None:
                break
        trend = _safe_float(getter("Sentiment_7d"))
        return avg, trend

    def _resolve_sentiment_snapshot(
        self, prediction: Mapping[str, Any] | None = None
    ) -> tuple[float | None, float | None]:
        candidates: list[tuple[float | None, float | None]] = []

        if isinstance(prediction, Mapping):
            candidates.append(self._extract_sentiment_from_row(prediction))

        for frame in (self.feature_snapshot, self.feature_history):
            if isinstance(frame, pd.DataFrame) and not frame.empty:
                candidates.append(self._extract_sentiment_from_row(frame.iloc[-1]))

        for avg, trend in candidates:
            if avg is not None or trend is not None:
                return avg, trend
        return None, None

    def _format_sentiment_status(
        self, avg: float | None, trend: float | None
    ) -> tuple[str, str]:
        if avg is None and trend is None:
            return "Sentiment unavailable", "—"

        label = "Neutral"
        if avg is not None:
            if avg >= 0.15:
                label = "Positive"
            elif avg <= -0.15:
                label = "Negative"

        parts: list[str] = []
        if avg is not None:
            parts.append(f"Avg: {avg:+.2f}")
        if trend is not None:
            parts.append(f"7d: {trend:+.2f}")
        score = " | ".join(parts) if parts else "—"
        return label, score

    def _update_metrics(self) -> None:
        prediction = self.current_prediction or {}
        explanation = prediction.get("explanation") if isinstance(prediction, Mapping) else None
        self.metric_vars["ticker"].set(str(prediction.get("ticker") or self.config.ticker))
        localized_as_of = self._resolve_market_timestamp()
        as_of_display = (
            localized_as_of.strftime("%Y-%m-%d %H:%M %Z") if localized_as_of is not None else "—"
        )
        self.metric_vars["as_of"].set(as_of_display)

        if self.market_timestamp_stale:
            status_message = "Market data timestamp unavailable or stale."
        else:
            status_message = (
                f"Market data as of {as_of_display}"
                if as_of_display != "—"
                else "Market data timestamp unavailable."
            )

        confluence_info = prediction.get("signal_confluence") if isinstance(prediction, Mapping) else None
        if isinstance(confluence_info, Mapping):
            if confluence_info.get("passed"):
                gate_note = "Signal confluence confirmed; targets and alerts are active."
            else:
                gate_note = "Signal confluence not met; target price and alerts are gated."
            status_message = f"{status_message} • {gate_note}" if status_message else gate_note

        derived_metrics = self._derive_prediction_metrics(prediction)
        last_close_converted = derived_metrics["anchor_converted"]
        predicted_converted = derived_metrics["predicted_converted"]
        change_converted = derived_metrics["change_converted"]

        decimals = self.price_decimal_places
        self.metric_vars["last_close"].set(
            fmt_ccy(last_close_converted, self.currency_symbol, decimals=decimals)
        )

        forecast = self.current_forecast_date
        if forecast is None:
            forecast = self._compute_forecast_date(self.current_prediction.get("target_date"))
        predicted_display = fmt_ccy(predicted_converted, self.currency_symbol, decimals=decimals)
        if forecast is not None:
            local_forecast = forecast.tz_convert(self.market_timezone) if forecast.tzinfo else forecast.tz_localize(self.market_timezone)
            forecast_str = local_forecast.date().isoformat()
            if predicted_display == "—":
                predicted_display = f"— ({forecast_str})"
            else:
                predicted_display = f"{predicted_display} ({forecast_str})"
        self.metric_vars["predicted_close"].set(predicted_display)

        expected_low_value = self._compute_expected_low(prediction, multiplier=self.expected_low_multiplier)
        expected_low_converted = self._convert_currency(expected_low_value)
        self.metric_vars["expected_low"].set(
            fmt_ccy(expected_low_converted, self.currency_symbol, decimals=decimals)
        )

        stop_loss_value = prediction.get("stop_loss") if isinstance(prediction, Mapping) else None
        stop_loss_converted = self._convert_currency(stop_loss_value)
        stop_loss_display = fmt_ccy(stop_loss_converted, self.currency_symbol, decimals=decimals)
        stop_loss_var = getattr(self, "stop_loss_var", self.metric_vars.get("stop_loss"))
        if stop_loss_var is not None:
            stop_loss_var.set(stop_loss_display)

        change_display = fmt_ccy(change_converted, self.currency_symbol, decimals=decimals)
        pct_display = fmt_pct(derived_metrics["pct_change"], show_sign=True)
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
        hit_prob = prediction.get("target_hit_probability")
        prob_up_str = fmt_pct(prob_up, decimals=1)
        prob_down_str = fmt_pct(prob_down, decimals=1)
        hit_prob_str = fmt_pct(hit_prob, decimals=1)
        parts: list[str] = []
        if prob_up_str != "—":
            parts.append(f"↑ {prob_up_str}")
        if prob_down_str != "—":
            parts.append(f"↓ {prob_down_str}")
        if hit_prob_str != "—":
            parts.append(f"🎯 {hit_prob_str}")
        self.metric_vars["direction"].set("   ".join(parts) if parts else "—")

        sentiment_error = prediction.get("sentiment_error") if isinstance(prediction, Mapping) else None
        avg_sentiment, trend_sentiment = self._resolve_sentiment_snapshot(prediction)
        if sentiment_error:
            self.sentiment_label_var.set("Sentiment fetch failed")
            self.sentiment_score_var.set(str(sentiment_error))
            status_message = (
                f"{status_message} • Sentiment unavailable: {sentiment_error}"
                if status_message
                else f"Sentiment unavailable: {sentiment_error}"
            )
        else:
            sentiment_label, sentiment_score = self._format_sentiment_status(
                avg_sentiment, trend_sentiment
            )
            self.sentiment_label_var.set(sentiment_label)
            self.sentiment_score_var.set(sentiment_score)

        if explanation and isinstance(explanation, Mapping):
            summary = explanation.get("summary")
            if summary:
                status_message = f"{status_message} • {summary}" if status_message else summary

        if status_message:
            self._set_status(status_message)

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

        try:
            chart_type_value = str(self.chart_type_var.get())
        except Exception:
            chart_type_value = "Line"
        chart_type = chart_type_value.strip().lower()
        use_candlestick = chart_type == "candlestick"

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
        normalised_map = {
            re.sub(r"[^a-z]", "", str(column).lower()): column for column in frame.columns
        }

        def resolve_column(*candidates: str) -> str | None:
            for candidate in candidates:
                key = str(candidate).lower()
                column = lower_map.get(key)
                if column is not None:
                    return column
                normalised = re.sub(r"[^a-z]", "", key)
                column = normalised_map.get(normalised)
                if column is not None:
                    return column
            return None

        date_column = resolve_column("date")
        if date_column is not None:
            frame[date_column] = pd.to_datetime(frame[date_column], errors="coerce")
            frame = frame.dropna(subset=[date_column])
            frame = frame.sort_values(date_column)
            x_index = pd.DatetimeIndex(frame[date_column])
        else:
            frame.index = pd.to_datetime(frame.index, errors="coerce")
            valid_mask = pd.notna(frame.index)
            frame = frame.loc[valid_mask]
            frame = frame.sort_index()
            x_index = pd.DatetimeIndex(frame.index)

        if x_index.empty:
            show_empty_state()
            self._style_figure(self.price_figure)
            canvas.draw_idle()
            return

        frame = frame.copy()
        frame.index = x_index
        frame = frame.loc[~frame.index.duplicated(keep="last")]
        x_index = pd.DatetimeIndex(frame.index)

        close_column = resolve_column("close", "adjclose", "adj_close")
        if close_column is None:
            numeric_columns = frame.select_dtypes(include=[np.number]).columns
            if not len(numeric_columns):
                show_empty_state()
                self._style_figure(self.price_figure)
                canvas.draw_idle()
                return
            close_column = numeric_columns[0]

        series = pd.to_numeric(frame[close_column], errors="coerce")
        plotted_series = series.dropna()

        if plotted_series.empty:
            show_empty_state()
            self._style_figure(self.price_figure)
            canvas.draw_idle()
            return

        legend_handles: list[Any] = []
        candlestick_frame: pd.DataFrame | None = None
        if use_candlestick:
            ohlc_columns = {
                "open": resolve_column("open", "open_price", "o"),
                "high": resolve_column("high", "high_price", "h"),
                "low": resolve_column("low", "low_price", "l"),
                "close": close_column,
            }
            if all(ohlc_columns.values()):
                candlestick_frame = pd.DataFrame(
                    {
                        key: pd.to_numeric(frame[column], errors="coerce")
                        for key, column in ohlc_columns.items()
                    },
                    index=frame.index,
                )
                candlestick_frame = candlestick_frame.dropna().sort_index()
                if candlestick_frame.empty:
                    candlestick_frame = None
                else:
                    plotted_series = candlestick_frame["close"]
            else:
                candlestick_frame = None
            if candlestick_frame is None:
                use_candlestick = False

        if empty_label is not None:
            empty_label.grid_remove()
        if canvas_widget is not None:
            canvas_widget.grid()

        last_x = plotted_series.index[-1]
        last_y = float(plotted_series.iloc[-1])
        _, reference_last_converted = self._reference_last_price(prediction)
        display_last_value = reference_last_converted if reference_last_converted is not None else last_y
        last_display = fmt_ccy(
            display_last_value, self.currency_symbol, decimals=self.price_decimal_places
        )

        try:
            prediction_line_end = (
                pd.to_datetime(forecast)
                if forecast is not None
                else pd.to_datetime(last_x) + pd.Timedelta(days=1)
            )
        except Exception:
            prediction_line_end = pd.to_datetime(last_x) + pd.Timedelta(days=1)

        if use_candlestick and candlestick_frame is not None:
            date_numbers = date2num(candlestick_frame.index.to_pydatetime())
            if len(date_numbers) > 1:
                unique_steps = np.diff(np.unique(date_numbers))
                base_width = float(unique_steps.min()) if len(unique_steps) else 1.0
            else:
                base_width = 1.0
            candle_width = max(min(base_width * 0.6, 0.8), 0.15)
            for date_num, (open_, high_, low_, close_) in zip(
                date_numbers, candlestick_frame.itertuples(index=False, name=None)
            ):
                color = "tab:green" if close_ >= open_ else "tab:red"
                ax.vlines(date_num, low_, high_, color=color, linewidth=1)
                body_bottom = min(open_, close_)
                body_height = max(abs(close_ - open_), 1e-9)
                rect = Rectangle(
                    (date_num - candle_width / 2, body_bottom),
                    candle_width,
                    body_height,
                    facecolor=color,
                    edgecolor=color,
                    alpha=0.9,
                )
                ax.add_patch(rect)
            ax.scatter([date_numbers[-1]], [display_last_value], color="tab:blue", zorder=5)
            ax.annotate(
                f"Last: {last_display}",
                xy=(date_numbers[-1], display_last_value),
                xytext=(8, 0),
                textcoords="offset points",
                va="center",
                ha="left",
                color="tab:blue",
            )
            legend_handles.extend(
                [
                    Line2D([0], [0], color="tab:green", linewidth=3, label="Up day"),
                    Line2D([0], [0], color="tab:red", linewidth=3, label="Down day"),
                ]
            )
            last_x_value = date_numbers[-1]
        else:
            close_label = (
                "Close (price)"
                if not self.currency_symbol
                else f"Close ({self.currency_symbol})"
            )
            (line_handle,) = ax.plot(
                plotted_series.index,
                plotted_series.values,
                label=close_label,
                color="tab:blue",
            )
            legend_handles.append(line_handle)
            ax.scatter([last_x], [display_last_value], color="tab:blue", zorder=5)
            ax.annotate(
                f"Last: {last_display}",
                xy=(last_x, display_last_value),
                xytext=(8, 0),
                textcoords="offset points",
                va="center",
                ha="left",
                color="tab:blue",
            )
            last_x_value = last_x

        predicted_close = _safe_float(prediction.get("predicted_close"))
        converted_prediction = self._convert_currency(predicted_close) if predicted_close is not None else None
        opportunity_patch_handle: Rectangle | None = None
        if converted_prediction is not None:
            predicted_label = (
                "Predicted close (price)"
                if not self.currency_symbol
                else f"Predicted close ({self.currency_symbol})"
            )
            if use_candlestick and candlestick_frame is not None:
                last_x_num = float(last_x_value)
                line_end_num = float(date2num(pd.to_datetime(prediction_line_end)))
                (predicted_handle,) = ax.plot(
                    [last_x_num, line_end_num],
                    [converted_prediction, converted_prediction],
                    color="tab:orange",
                    linestyle="--",
                    label=predicted_label,
                )
                legend_handles.append(predicted_handle)
                annotation_xy = (line_end_num, converted_prediction)
            else:
                (predicted_handle,) = ax.plot(
                    [last_x, prediction_line_end],
                    [converted_prediction, converted_prediction],
                    color="tab:orange",
                    linestyle="--",
                    label=predicted_label,
                )
                legend_handles.append(predicted_handle)
                annotation_xy = (pd.to_datetime(prediction_line_end), converted_prediction)
            annotation_text = fmt_ccy(
                converted_prediction,
                self.currency_symbol,
                decimals=self.price_decimal_places,
            )
            if forecast is not None:
                annotation_text = f"{annotation_text} ({forecast.date().isoformat()})"
            ax.annotate(
                annotation_text,
                xy=annotation_xy,
                xytext=(8, 0),
                textcoords="offset points",
                va="center",
                ha="left",
                color="tab:orange",
            )

        expected_low_value = self._compute_expected_low(prediction, multiplier=self.expected_low_multiplier)
        expected_low_converted = (
            self._convert_currency(expected_low_value) if expected_low_value is not None else None
        )
        zone_color = "#fcd34d"
        expected_color = "#b45309"
        if expected_low_converted is not None:
            expected_label = (
                "Expected low"
                if not self.currency_symbol
                else f"Expected low ({self.currency_symbol})"
            )
            if use_candlestick and candlestick_frame is not None:
                last_x_num = float(last_x_value)
                line_end_num = float(date2num(pd.to_datetime(prediction_line_end)))
                (expected_handle,) = ax.plot(
                    [last_x_num, line_end_num],
                    [expected_low_converted, expected_low_converted],
                    color=expected_color,
                    linestyle=":",
                    linewidth=1.6,
                    label=expected_label,
                )
                annotation_xy = (line_end_num, expected_low_converted)
            else:
                (expected_handle,) = ax.plot(
                    [last_x, prediction_line_end],
                    [expected_low_converted, expected_low_converted],
                    color=expected_color,
                    linestyle=":",
                    linewidth=1.6,
                    label=expected_label,
                )
                annotation_xy = (pd.to_datetime(prediction_line_end), expected_low_converted)
            legend_handles.append(expected_handle)
            annotation_text = fmt_ccy(
                expected_low_converted,
                self.currency_symbol,
                decimals=self.price_decimal_places,
            )
            annotation_text = f"{annotation_text} (expected low)"
            ax.annotate(
                annotation_text,
                xy=annotation_xy,
                xytext=(8, -10),
                textcoords="offset points",
                va="top",
                ha="left",
                color=expected_color,
            )

            zone_reference = (
                converted_prediction if converted_prediction is not None else display_last_value
            )
            if zone_reference is not None:
                lower, upper = sorted([expected_low_converted, float(zone_reference)])
                if abs(upper - lower) > 1e-9:
                    ax.axhspan(
                        lower,
                        upper,
                        color=zone_color,
                        alpha=0.15,
                        zorder=0.5,
                        label="_nolegend_",
                    )
                    opportunity_patch_handle = Rectangle(
                        (0, 0),
                        1,
                        1,
                        facecolor=zone_color,
                        alpha=0.25,
                        edgecolor="none",
                        label="Opportunity zone",
                    )

        if opportunity_patch_handle is not None:
            legend_handles.append(opportunity_patch_handle)

        stop_loss_value = prediction.get("stop_loss") if isinstance(prediction, Mapping) else None
        stop_loss_converted = self._convert_currency(stop_loss_value)
        stop_loss_numeric = _safe_float(stop_loss_converted)
        if stop_loss_numeric is not None:
            stop_color = "#dc2626"
            stop_label = (
                "Stop-loss" if not self.currency_symbol else f"Stop-loss ({self.currency_symbol})"
            )
            if use_candlestick and candlestick_frame is not None:
                last_x_num = float(last_x_value)
                line_end_num = float(date2num(pd.to_datetime(prediction_line_end)))
                (stop_handle,) = ax.plot(
                    [last_x_num, line_end_num],
                    [stop_loss_numeric, stop_loss_numeric],
                    color=stop_color,
                    linestyle="--",
                    linewidth=1.6,
                    label=stop_label,
                )
                scatter_x = line_end_num
            else:
                (stop_handle,) = ax.plot(
                    [last_x, prediction_line_end],
                    [stop_loss_numeric, stop_loss_numeric],
                    color=stop_color,
                    linestyle="--",
                    linewidth=1.6,
                    label=stop_label,
                )
                scatter_x = pd.to_datetime(prediction_line_end)
            legend_handles.append(stop_handle)
            ax.scatter(
                [scatter_x],
                [stop_loss_numeric],
                color=stop_color,
                edgecolors="white",
                linewidth=0.8,
                s=36,
                zorder=6,
            )
            annotation_text = fmt_ccy(
                stop_loss_numeric,
                self.currency_symbol,
                decimals=self.price_decimal_places,
            )
            annotation_text = f"{annotation_text} (stop-loss)"
            ax.annotate(
                annotation_text,
                xy=(scatter_x, stop_loss_numeric),
                xytext=(8, 10),
                textcoords="offset points",
                va="bottom",
                ha="left",
                color=stop_color,
            )
            current_ylim = ax.get_ylim()
            lower_bound = min(current_ylim[0], stop_loss_numeric)
            if lower_bound < stop_loss_numeric - 1e-9:
                ax.axhspan(
                    lower_bound,
                    stop_loss_numeric,
                    color="#fecaca",
                    alpha=0.25,
                    zorder=0.1,
                    label="_nolegend_",
                )

        ylabel = "Price"
        if self.currency_symbol:
            ylabel = f"Price ({self.currency_symbol})"
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", alpha=0.3)
        if legend_handles:
            ax.legend(handles=legend_handles, loc="best")
        if use_candlestick and candlestick_frame is not None:
            ax.xaxis_date()
        ax.xaxis.set_major_formatter(DateFormatter("%b-%Y"))

        self._style_figure(self.price_figure)
        canvas.draw_idle()

    def _update_indicator_view(self) -> None:
        indicator_names = self._collect_indicator_names()
        selections = set(self._indicator_selection_cache)
        if selections:
            selections &= set(indicator_names)
        if not selections and not self._indicator_user_override:
            selections.update(indicator_names[: min(3, len(indicator_names))])
        self._populate_indicator_selector(indicator_names, selections)
        self._indicator_selection_cache = set(selections)
        if not indicator_names:
            self._indicator_user_override = False
        self._update_indicator_toggle_button_state(indicator_names, selections)
        self._update_indicator_info_panel(indicator_names)
        self._update_indicator_chart(selections)

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
        if hasattr(self, "indicator_price_ax"):
            self._update_indicator_view()
        if not overview_refreshed and hasattr(self, "pnl_label"):
            self._recompute_pnl()

    def _initialise_currency_profile(self) -> None:
        detected_code = self._detect_local_currency_code()
        if not detected_code:
            detected_code = self._extract_currency_code(
                self.currency_profiles.get("local"), mode="local"
            )
        if detected_code:
            self._set_local_currency_code(detected_code)

    def _detect_local_currency_code(self) -> str | None:
        explicit = _normalise_currency_code(
            os.environ.get("STOCK_PREDICTOR_UI_BASE_CURRENCY_CODE")
        )
        if explicit:
            return explicit

        pipeline = getattr(self.application, "pipeline", None)
        metadata = getattr(pipeline, "metadata", None)
        if isinstance(metadata, Mapping):
            for key in ("price_currency", "currency", "financial_currency", "quote_currency"):
                candidate = _normalise_currency_code(metadata.get(key))
                if candidate:
                    return candidate

        code = self._infer_currency_from_frame(getattr(self, "price_history", None))
        if code:
            return code

        fetcher = getattr(pipeline, "fetcher", None)
        if fetcher and hasattr(fetcher, "fetch_price_data"):
            try:
                cached_prices = fetcher.fetch_price_data(force=False)
            except Exception as exc:  # pragma: no cover - cached data optional
                LOGGER.debug(
                    "Unable to retrieve cached price data for currency detection: %s",
                    exc,
                )
            else:
                code = self._infer_currency_from_frame(cached_prices)
                if code:
                    return code

        try:
            ticker = yf.Ticker(self.config.ticker)
            fast_info = getattr(ticker, "fast_info", None)
            if fast_info:
                if isinstance(fast_info, Mapping):
                    code = _normalise_currency_code(
                        fast_info.get("currency") or fast_info.get("quote_currency")
                    )
                else:
                    code = _normalise_currency_code(getattr(fast_info, "currency", None))
                if code:
                    return code
            info = getattr(ticker, "info", None)
            if isinstance(info, Mapping):
                for key in ("currency", "financialCurrency", "quoteCurrency"):
                    code = _normalise_currency_code(info.get(key))
                    if code:
                        return code
        except Exception as exc:  # pragma: no cover - yfinance metadata optional
            LOGGER.debug("Failed to resolve ticker currency via yfinance: %s", exc)

        return None

    @staticmethod
    def _infer_currency_from_frame(frame: pd.DataFrame | None) -> str | None:
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            return None
        if hasattr(frame, "attrs"):
            for key in ("currency", "Currency"):
                code = _normalise_currency_code(frame.attrs.get(key))
                if code:
                    return code
        for column in ("Currency", "currency", "CURRENCY"):
            if column in frame.columns:
                series = frame[column].dropna()
                if not series.empty:
                    code = _normalise_currency_code(series.iloc[0])
                    if code:
                        return code
        return None

    def _set_local_currency_code(self, code: str) -> None:
        normalised = _normalise_currency_code(code)
        if not normalised:
            return
        profile = self.currency_profiles.setdefault("local", {})
        previous = _normalise_currency_code(profile.get("code"))
        if previous == normalised:
            return
        profile["code"] = normalised
        label = str(profile.get("label") or "")
        if not label.strip() or label.strip().lower() == "local":
            profile["label"] = normalised
        symbol = str(profile.get("symbol") or "").strip()
        if not symbol:
            profile["symbol"] = CURRENCY_CODE_TO_SYMBOL.get(normalised, symbol)
        LOGGER.info("Detected local currency code: %s", normalised)
        if getattr(self, "currency_mode", "local") == "local":
            self.currency_symbol = self._currency_symbol("local")
            if hasattr(self, "currency_button_text"):
                display = self.currency_symbol or self._currency_label("local")
                self.currency_button_text.set(display)

    def _update_local_currency_profile(self, price_history: pd.DataFrame | None) -> None:
        code = self._infer_currency_from_frame(price_history)
        if code:
            self._set_local_currency_code(code)

    def _extract_currency_code(
        self, profile: Mapping[str, Any] | None, *, mode: str | None = None
    ) -> str | None:
        if not isinstance(profile, Mapping):
            return None
        code = _normalise_currency_code(profile.get("code"))
        if code:
            return code
        label = str(profile.get("label") or "")
        match = re.search(r"\b([A-Z]{3})\b", label.upper())
        if match:
            return match.group(1)
        symbol = str(profile.get("symbol") or "").strip()
        if not symbol:
            symbol = _detect_currency_symbol(label, fallback="")
        if symbol:
            mapped = CURRENCY_SYMBOL_TO_CODE.get(symbol)
            if mapped:
                return mapped
        fallback_map = {"usd": "USD", "eur": "EUR"}
        if mode:
            return fallback_map.get(mode.lower())
        return None

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
        code = self._extract_currency_code(profile, mode=mode)
        if code:
            return code
        if mode == "local":
            detected = self._detect_local_currency_code()
            if detected:
                self._set_local_currency_code(detected)
                return detected
        return None

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
            base_code = self._currency_code("local")
            target_code = self._currency_code(target_mode)
            base_norm = base_code.upper() if isinstance(base_code, str) else None
            target_norm = target_code.upper() if isinstance(target_code, str) else None
            if base_norm and target_norm and base_norm == target_norm:
                return 1.0
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
            self._set_status(f"FX rate automatically updated to {rate:.4f}.")
        return rate

    def _apply_currency(self, rate: float | None = None, *, mode: str | None = None) -> None:
        target_mode = mode or self.currency_mode
        resolved_rate = float(rate) if rate is not None else self._currency_rate(target_mode)
        if resolved_rate <= 0:
            resolved_rate = 1.0

        def _convert_frame(frame: pd.DataFrame | None) -> pd.DataFrame | None:
            if not isinstance(frame, pd.DataFrame) or frame.empty:
                return None
            converted = frame.copy()
            numeric_columns: list[str] = []
            excluded_tokens = {
                "volume",
                "volumes",
                "shares",
                "sharecount",
                "openinterest",
                "contract",
                "contracts",
            }
            for column in converted.columns:
                series = converted[column]
                name_normalised = re.sub(r"[^a-z0-9]+", "", str(column).lower())
                if name_normalised and any(token in name_normalised for token in excluded_tokens):
                    continue
                if pd.api.types.is_datetime64_any_dtype(series) or pd.api.types.is_timedelta64_dtype(series):
                    continue
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
                    converted.loc[:, numeric_columns].astype(float) * float(resolved_rate)
                )
            return converted

        self.price_history_converted = _convert_frame(self.price_history)
        self.feature_snapshot_converted = _convert_frame(self.feature_snapshot)
        self.feature_history_converted = _convert_frame(self.feature_history)
        self.indicator_history_converted = _convert_frame(self.indicator_history)

    def _on_currency_changed(self, mode: str, rate: float) -> None:
        self._apply_currency(rate, mode=mode)
        self._refresh_numeric_views()

    def _on_fx_rate_changed(self, mode: str, rate: float) -> None:
        self._apply_currency(rate, mode=mode)
        self._refresh_numeric_views()

    def _convert_currency(self, value: Any, mode: str | None = None) -> float | None:
        numeric = _safe_float(value)
        if numeric is None:
            return None
        rate = self._currency_rate(mode)
        return numeric * rate

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
            getattr(self, "indicator_price_figure", None),
            getattr(self, "feature_figure", None),
        ):
            self._style_figure(figure)
        for canvas in (
            getattr(self, "price_canvas", None),
            getattr(self, "indicator_price_canvas", None),
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
        derived_metrics = self._derive_prediction_metrics(prediction)
        anchor_raw = derived_metrics["anchor_raw"]
        anchor_converted = derived_metrics["anchor_converted"]
        predicted_raw = derived_metrics["predicted_raw"]
        predicted_converted = derived_metrics["predicted_converted"]

        if anchor_raw is None or predicted_raw is None:
            self.pnl_var.set(prefix + "—")
            return

        if anchor_converted is None or predicted_converted is None:
            self.pnl_var.set(prefix + "—")
            return

        pnl_value = (predicted_converted - anchor_converted) * size
        pct_change = derived_metrics["pct_change"]
        if pct_change is None and anchor_raw not in (None, 0):
            pct_change = (predicted_raw - anchor_raw) / anchor_raw

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
