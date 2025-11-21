"""Helpers for deriving tactical buy zones from market data."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .config import PredictorConfig
from .data_fetcher import DataFetcher
from .indicator_bundle import compute_indicators
from .support_levels import indicator_support_floor


def _safe_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric):
        return None
    return float(numeric)


@dataclass(frozen=True)
class IndicatorConfirmation:
    """Represents the state of an indicator-driven confirmation signal."""

    confirmed: bool
    value: float | None = None
    threshold: float | None = None
    detail: str | None = None


@dataclass(frozen=True)
class BuyZoneResult:
    """Container describing the computed buy zone for a ticker."""

    ticker: str
    window_start: datetime | None
    window_end: datetime | None
    lower_bound: float | None
    upper_bound: float | None
    support_level: float | None
    last_close: float | None
    confirmations: Mapping[str, IndicatorConfirmation] = field(default_factory=dict)
    support_components: Mapping[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON serialisable representation."""

        def _iso(value: datetime | None) -> str | None:
            if value is None:
                return None
            return value.isoformat(timespec="seconds")

        confirmation_payload: dict[str, dict[str, Any]] = {}
        for name, confirmation in self.confirmations.items():
            payload: dict[str, Any] = {"confirmed": bool(confirmation.confirmed)}
            if confirmation.value is not None:
                payload["value"] = confirmation.value
            if confirmation.threshold is not None:
                payload["threshold"] = confirmation.threshold
            if confirmation.detail:
                payload["detail"] = confirmation.detail
            confirmation_payload[name] = payload

        return {
            "ticker": self.ticker,
            "window": {"start": _iso(self.window_start), "end": _iso(self.window_end)},
            "price_bounds": {
                "lower": self.lower_bound,
                "upper": self.upper_bound,
                "support": self.support_level,
                "last_close": self.last_close,
            },
            "confirmations": confirmation_payload,
            "support_components": dict(self.support_components),
        }


class BuyZoneAnalyzer:
    """Derive a probabilistic buy zone using recent indicators."""

    def __init__(self, config: PredictorConfig, *, fetcher: DataFetcher | None = None) -> None:
        self.config = config
        self.fetcher = fetcher or DataFetcher(config)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def compute(self, *, refresh: bool = False, lookback_days: int = 30) -> BuyZoneResult:
        """Compute a tactical buy zone for the configured ticker."""

        if refresh:
            self.fetcher.refresh_all(force=True)

        price_df = self.fetcher.fetch_price_data()
        if price_df is None or price_df.empty:
            raise ValueError(f"No price data available for {self.config.ticker}")

        cleaned_df, date_series = self._prepare_price_frame(price_df)
        latest_date = date_series.iloc[-1] if not date_series.empty else None
        earliest_date = date_series.iloc[0] if not date_series.empty else None

        window_end = latest_date.to_pydatetime() if latest_date is not None else None
        window_start = None
        if latest_date is not None and earliest_date is not None:
            candidate = latest_date - timedelta(days=int(max(1, lookback_days)))
            window_start = max(candidate, earliest_date).to_pydatetime()

        indicators = compute_indicators(cleaned_df)
        latest_indicators = indicators.iloc[-1] if not indicators.empty else pd.Series()

        support_level, support_components = indicator_support_floor(indicators)
        last_close = _safe_float(cleaned_df["Close"].iloc[-1])

        atr_value = _safe_float(latest_indicators.get("ATR_14"))
        lower_bound, upper_bound = self._price_bounds(last_close, support_level, atr_value)
        confirmations = self._build_confirmations(
            latest_indicators, last_close, support_level
        )

        return BuyZoneResult(
            ticker=self.config.ticker,
            window_start=window_start,
            window_end=window_end,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            support_level=_safe_float(support_level),
            last_close=last_close,
            confirmations=confirmations,
            support_components={k: float(v) for k, v in support_components.items()},
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _prepare_price_frame(
        self, price_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.Series]:
        df = price_df.copy()
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df = df.dropna(subset=["Date"]).sort_values("Date")
            date_series = df["Date"].reset_index(drop=True)
        else:
            df = df.sort_index()
            date_series = pd.to_datetime(df.index.to_series(), errors="coerce").dropna()
            if not date_series.empty:
                df = df.loc[date_series.index]

        df = df.reset_index(drop=True)
        return df, date_series

    def _price_bounds(
        self, last_close: float | None, support_level: float | None, atr: float | None
    ) -> tuple[float | None, float | None]:
        buffer = atr if atr is not None and atr > 0 else None
        if buffer is None and last_close is not None:
            buffer = abs(last_close) * 0.02

        if support_level is not None and buffer is not None:
            lower_bound = max(0.0, support_level - 0.25 * buffer)
            upper_bound = support_level + buffer
            return float(lower_bound), float(upper_bound)

        if support_level is not None:
            return float(max(0.0, support_level)), float(max(0.0, support_level))

        if last_close is None:
            return None, None

        lower = last_close - buffer if buffer is not None else last_close
        upper = last_close + buffer if buffer is not None else last_close
        return float(max(0.0, lower)), float(max(0.0, upper))

    def _build_confirmations(
        self,
        indicators: Mapping[str, Any],
        last_close: float | None,
        support_level: float | None,
    ) -> dict[str, IndicatorConfirmation]:
        confirmations: dict[str, IndicatorConfirmation] = {}

        rsi_value = _safe_float(indicators.get("RSI_14"))
        if rsi_value is not None:
            confirmations["rsi_oversold"] = IndicatorConfirmation(
                confirmed=rsi_value < 40,
                value=rsi_value,
                threshold=40.0,
                detail="RSI below 40 often signals oversold conditions.",
            )

        macd_hist = _safe_float(indicators.get("MACD_12_26_9_Hist"))
        if macd_hist is not None:
            confirmations["macd_bullish"] = IndicatorConfirmation(
                confirmed=macd_hist > 0,
                value=macd_hist,
                threshold=0.0,
                detail="Positive MACD histogram suggests bullish momentum.",
            )

        supertrend_direction = None
        for name, value in indicators.items():
            if str(name).startswith("Supertrend_Direction_"):
                supertrend_direction = _safe_float(value)
                break
        if supertrend_direction is not None:
            confirmations["supertrend_bullish"] = IndicatorConfirmation(
                confirmed=supertrend_direction > 0,
                value=supertrend_direction,
                threshold=0.0,
                detail="Upward Supertrend direction supports bullish bias.",
            )

        support_price = _safe_float(support_level)
        if support_price is not None and last_close is not None:
            tolerance = support_price * 0.03
            confirmations["near_support"] = IndicatorConfirmation(
                confirmed=last_close <= support_price + tolerance,
                value=last_close,
                threshold=support_price + tolerance,
                detail="Price is within 3% of the nearest support level.",
            )

        return confirmations


__all__ = ["BuyZoneAnalyzer", "BuyZoneResult", "IndicatorConfirmation"]
