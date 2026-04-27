"""Screener-only Tkinter UI for StockPredictor."""

from __future__ import annotations

import tkinter as tk
from datetime import datetime
from tkinter import ttk
from types import SimpleNamespace
from typing import Any
import logging

from stock_predictor.screener.market_data.symbol_universe import SymbolUniverseService
from stock_predictor.screener.pattern_engine.types import Candle
from stock_predictor.screener.services import ScreenerService


SCREENER_PATTERN_CHOICES: tuple[str, ...] = (
    "Double Bottom",
    "Double Top",
    "Triple Bottom",
    "Triple Top",
    "Head and Shoulders",
    "Inverted Head and Shoulders",
    "Ascending Triangle",
    "Descending Triangle",
    "Pennant",
    "Flag",
    "Bearish Flag",
    "Channel",
    "Channel Up",
    "Channel Down",
    "Cup and Handle",
    "Diamond",
)


class YahooMarketDataProvider:
    """Market data adapter used by the desktop screener for broad scans."""
    serial_scan = False

    def __init__(self) -> None:
        self._failed_symbols_until: dict[str, datetime] = {}
        self._failure_cooldown_seconds = 30 * 60
        logging.getLogger("yfinance").setLevel(logging.CRITICAL)
        logging.getLogger("yfinance.shared").setLevel(logging.CRITICAL)

    def get_universe(self, market_type: str) -> list[str]:
        universe = SymbolUniverseService().get_universe(market_type)
        return [item.symbol for item in universe]

    def get_historical_bars(self, symbol: str, timeframe: str, lookback: int) -> list[Candle]:
        # Loaded lazily so launching the UI does not require yfinance unless scanning.
        import pandas as pd
        import yfinance as yf
        from datetime import timedelta

        interval = "1h" if timeframe == "1h" else timeframe
        period = "6mo" if lookback >= 300 else "3mo"
        normalized_symbol = symbol.strip().upper().replace(".", "-")
        blocked_until = self._failed_symbols_until.get(normalized_symbol)
        now = datetime.utcnow()
        if blocked_until is not None and blocked_until > now:
            return []
        if blocked_until is not None and blocked_until <= now:
            self._failed_symbols_until.pop(normalized_symbol, None)
        try:
            if hasattr(yf, "download"):
                frame = yf.download(
                    tickers=normalized_symbol,
                    period=period,
                    interval=interval,
                    auto_adjust=False,
                    progress=False,
                    threads=False,
                )
            else:
                frame = yf.Ticker(normalized_symbol).history(
                    period=period,
                    interval=interval,
                    auto_adjust=False,
                )
        except Exception:
            self._failed_symbols_until[normalized_symbol] = datetime.utcnow() + timedelta(
                seconds=self._failure_cooldown_seconds
            )
            return []
        if frame is None or frame.empty:
            self._failed_symbols_until[normalized_symbol] = datetime.utcnow() + timedelta(
                seconds=self._failure_cooldown_seconds
            )
            return []

        normalized = frame.reset_index()
        if "Datetime" in normalized.columns:
            normalized = normalized.rename(columns={"Datetime": "timestamp"})
        elif "Date" in normalized.columns:
            normalized = normalized.rename(columns={"Date": "timestamp"})

        out: list[Candle] = []
        for row in normalized.itertuples(index=False):
            timestamp = pd.to_datetime(getattr(row, "timestamp", None), errors="coerce")
            if pd.isna(timestamp):
                continue
            try:
                out.append(
                    Candle(
                        timestamp=timestamp.to_pydatetime(),
                        open=float(getattr(row, "Open")),
                        high=float(getattr(row, "High")),
                        low=float(getattr(row, "Low")),
                        close=float(getattr(row, "Close")),
                        volume=float(getattr(row, "Volume", 0.0)),
                    )
                )
            except Exception:
                continue
        return out


class StockPredictorDesktopApp:
    """Desktop application reduced to the screener workflow only."""

    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Stock Predictor – Screener")
        self.root.geometry("1220x760")
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        self.config = SimpleNamespace(ticker="AAPL")
        self.ticker_var = tk.StringVar(value=self.config.ticker)
        self.screener_status_var = tk.StringVar(
            value="Select patterns and click 'Scan now' to run a market-wide 1h scan."
        )
        self.screener_progress_var = tk.StringVar(value="")
        self.screener_last_scan_var = tk.StringVar(value="Last scan: —")
        self.selected_pattern_var = tk.StringVar(value=SCREENER_PATTERN_CHOICES[0])
        self.screener_row_symbol_map: dict[str, str] = {}
        self.screener_pattern_buttons: dict[str, tk.Button] = {}
        self.screener_service = ScreenerService(YahooMarketDataProvider())

        self._build_screener_tab()

    def _set_status(self, message: str) -> None:
        self.screener_status_var.set(message)

    def _on_refresh(self) -> None:
        self._update_screener_view(force_refresh_data=True)

    def _build_screener_tab(self) -> None:
        frame = ttk.Frame(self.root, padding=12)
        frame.grid(row=0, column=0, sticky="nsew")
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_rowconfigure(4, weight=1)

        ttk.Label(frame, text="Screener", font=("TkDefaultFont", 14, "bold")).grid(
            row=0, column=0, sticky="w", pady=(0, 4)
        )
        ttk.Label(
            frame,
            text="Market-wide 1h pattern screener for stocks/futures.",
        ).grid(row=1, column=0, sticky="w", pady=(0, 8))

        control_row = ttk.Frame(frame)
        control_row.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        control_row.grid_columnconfigure(0, weight=1)

        ttk.Label(control_row, text="Pattern").grid(row=0, column=0, sticky="w", pady=(0, 6))

        pattern_grid = ttk.Frame(control_row)
        pattern_grid.grid(row=1, column=0, sticky="ew")
        columns = 8
        for idx, name in enumerate(SCREENER_PATTERN_CHOICES):
            row_idx = idx // columns
            col_idx = idx % columns
            pattern_grid.grid_columnconfigure(col_idx, weight=1)
            button = tk.Button(
                pattern_grid,
                text=name,
                command=lambda pattern=name: self._select_pattern(pattern),
                width=18,
                height=2,
                relief="ridge",
                borderwidth=2,
                font=("TkDefaultFont", 9, "bold"),
                wraplength=125,
                justify="center",
            )
            button.grid(row=row_idx, column=col_idx, padx=4, pady=4, sticky="nsew")
            self.screener_pattern_buttons[name] = button

        action_row = ttk.Frame(control_row)
        action_row.grid(row=2, column=0, sticky="e", pady=(8, 0))
        ttk.Button(action_row, text="Scan now", command=self._on_refresh).grid(row=0, column=0)
        ttk.Label(action_row, textvariable=self.screener_progress_var).grid(row=0, column=1, sticky="w", padx=(10, 0))
        self._refresh_pattern_button_styles()

        status_row = ttk.Frame(frame)
        status_row.grid(row=3, column=0, sticky="ew", pady=(0, 6))
        status_row.grid_columnconfigure(0, weight=1)
        ttk.Label(status_row, textvariable=self.screener_status_var).grid(row=0, column=0, sticky="w")
        ttk.Label(status_row, textvariable=self.screener_last_scan_var).grid(row=0, column=1, sticky="e")

        table_frame = ttk.Frame(frame)
        table_frame.grid(row=4, column=0, sticky="nsew")
        table_frame.grid_columnconfigure(0, weight=1)
        table_frame.grid_rowconfigure(0, weight=1)

        columns = (
            "name",
            "symbol",
            "pattern",
            "status",
            "direction",
            "confidence",
            "quality",
            "close",
            "signal",
        )
        self.screener_tree = ttk.Treeview(table_frame, columns=columns, show="headings")
        labels = {
            "name": "Name",
            "symbol": "Symbol",
            "pattern": "Pattern",
            "status": "Status",
            "direction": "Direction",
            "confidence": "Confidence",
            "quality": "Trade Quality",
            "close": "Close",
            "signal": "Signal Time",
        }
        for column in columns:
            self.screener_tree.heading(column, text=labels[column])
            self.screener_tree.column(column, width=130 if column not in {"name", "pattern"} else 220)

        self.screener_tree.grid(row=0, column=0, sticky="nsew")
        self.screener_tree.bind("<<TreeviewSelect>>", self._on_screener_row_selected)

        self.screener_placeholder = ttk.Label(
            table_frame,
            text="No scan results yet.",
            anchor="center",
        )
        self.screener_placeholder.grid(row=0, column=0, sticky="nsew")

    def _update_screener_view(self, force_refresh_data: bool = True) -> None:
        if self.screener_tree is None:
            return

        for item in self.screener_tree.get_children():
            self.screener_tree.delete(item)
        self.screener_row_symbol_map.clear()

        pattern = self._get_selected_pattern()
        if not pattern:
            self.screener_status_var.set("Please select a pattern.")
            return

        self.screener_progress_var.set("0 von 0 Aktien gescannt")

        def _on_progress(done_count: int, total: int, _message: str) -> None:
            self.screener_progress_var.set(f"{done_count} von {total} Aktien gescannt")
            if self.root is not None:
                self.root.update_idletasks()

        rows = self.screener_service.scan_market(
            pattern,
            "stock",
            timeframe="1h",
            force_refresh_data=force_refresh_data,
            progress_callback=_on_progress,
        )

        total_scanned = self._extract_total_scanned(self.screener_progress_var.get())
        if total_scanned is None:
            total_scanned = len(rows)
            self.screener_progress_var.set(f"{total_scanned} von {total_scanned} Aktien gescannt")

        debug = self.screener_service.get_last_debug_stats()

        if not rows:
            self.screener_status_var.set(
                f"No active patterns found. Debug: {debug.scannedSymbols} scanned, "
                f"{debug.symbolsWithData} with data, {debug.symbolsWithEnoughCandles} enough candles, "
                f"{debug.pipeline.get('rawDetections', 0)} candidates, "
                f"{debug.pipeline.get('afterConfidenceFilter', 0)} after confidence filter."
            )
            if self.screener_placeholder is not None:
                self.screener_placeholder.configure(text="No matches for selected filters. Open debug stats in status line.")
                self.screener_placeholder.grid()
            self.screener_tree.grid_remove()
            return

        for row in rows:
            quality = row.get("tradeQuality") or {}
            quality_text = f"{quality.get('rating', '—')} {quality.get('successes', 0)}/{quality.get('occurrences', 0)}"
            iid = self.screener_tree.insert(
                "",
                "end",
                values=(
                    row.get("name", "—"),
                    row.get("symbol", "—"),
                    row.get("patternType", "—"),
                    row.get("status", "—"),
                    row.get("direction", "—"),
                    f"{float(row.get('confidence', 0.0)):.1f}%",
                    quality_text,
                    f"{float(row.get('close', 0.0)):.2f}",
                    row.get("signalTime") or row.get("detectedAt") or "—",
                ),
            )
            symbol = str(row.get("symbol") or "").upper().strip()
            if symbol:
                self.screener_row_symbol_map[iid] = symbol

        self.screener_status_var.set(
            f"Detected {len(rows)} symbols matching {pattern} on the 1h timeframe. "
            f"Pipeline raw/active/displayed: {debug.pipeline.get('rawDetections', 0)}/"
            f"{debug.pipeline.get('activeDetections', 0)}/{debug.pipeline.get('displayedResults', 0)}."
        )
        self.screener_progress_var.set(f"{len(rows)} Treffer, {total_scanned} Aktien gescannt")
        self.screener_last_scan_var.set(
            f"Last scan: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}"
        )
        self.screener_tree.grid()
        if self.screener_placeholder is not None:
            self.screener_placeholder.grid_remove()

    def _select_pattern(self, pattern: str) -> None:
        if pattern not in SCREENER_PATTERN_CHOICES:
            return
        self.selected_pattern_var.set(pattern)
        self._refresh_pattern_button_styles()

    def _get_selected_pattern(self) -> str:
        if hasattr(self, "selected_pattern_var") and self.selected_pattern_var is not None:
            pattern = str(self.selected_pattern_var.get()).strip()
            if pattern:
                return pattern
        if hasattr(self, "screener_pattern_listbox") and self.screener_pattern_listbox is not None:
            selection = self.screener_pattern_listbox.curselection()
            if selection:
                return str(self.screener_pattern_listbox.get(selection[0])).strip()
        return ""

    def _refresh_pattern_button_styles(self) -> None:
        selected_pattern = self.selected_pattern_var.get()
        for pattern, button in self.screener_pattern_buttons.items():
            is_selected = pattern == selected_pattern
            button.configure(
                bg="#1d4ed8" if is_selected else "#f3f4f6",
                fg="white" if is_selected else "#111827",
                activebackground="#1e40af" if is_selected else "#e5e7eb",
                activeforeground="white" if is_selected else "#111827",
                highlightbackground="#1d4ed8" if is_selected else "#d1d5db",
                highlightthickness=2,
            )

    @staticmethod
    def _extract_total_scanned(progress_text: str) -> int | None:
        parts = progress_text.strip().split()
        if len(parts) >= 3 and parts[1] == "von":
            try:
                return int(parts[2])
            except ValueError:
                return None
        return None

    def _on_screener_row_selected(self, _event: tk.Event | None) -> None:
        if self.screener_tree is None or self.ticker_var is None:
            return
        selected = self.screener_tree.selection()
        if not selected:
            return
        iid = selected[0]
        symbol = self.screener_row_symbol_map.get(iid)
        if not symbol:
            values = self.screener_tree.item(iid, "values")
            if len(values) >= 2:
                symbol = str(values[1]).upper().strip()
        if not symbol:
            return

        self.ticker_var.set(symbol)
        if hasattr(self.config, "ticker"):
            self.config.ticker = symbol
        self._on_refresh()

    def run(self) -> None:
        self.root.mainloop()


def run_tkinter_app() -> None:
    """Launch the desktop screener application."""

    app = StockPredictorDesktopApp()
    app.run()


if __name__ == "__main__":
    run_tkinter_app()
