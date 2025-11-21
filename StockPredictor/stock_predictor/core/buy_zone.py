"""Helpers for deriving tactical buy zones from market data."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .config import BuyZoneConfirmationSettings, PredictorConfig
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

        indicator_result = compute_indicators(cleaned_df)
        indicators = indicator_result.dataframe
        latest_indicators = indicators.iloc[-1] if not indicators.empty else pd.Series()

        support_level, support_components = indicator_support_floor(indicators)
        last_close = _safe_float(cleaned_df["Close"].iloc[-1])

        atr_value = _safe_float(latest_indicators.get("ATR_14"))
        lower_bound, upper_bound = self._price_bounds(last_close, support_level, atr_value)
        confirmations = self._build_confirmations(
            indicators, last_close, support_level
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
        indicators: pd.DataFrame,
        last_close: float | None,
        support_level: float | None,
    ) -> dict[str, IndicatorConfirmation]:
        confirmations: dict[str, IndicatorConfirmation] = {}
        latest = indicators.iloc[-1] if not indicators.empty else pd.Series(dtype=float)

        settings = getattr(self.config, "buy_zone", BuyZoneConfirmationSettings())
        if isinstance(settings, Mapping):
            settings = BuyZoneConfirmationSettings.from_mapping(settings)
        elif not isinstance(settings, BuyZoneConfirmationSettings):
            settings = BuyZoneConfirmationSettings()

        rsi_value = _safe_float(latest.get("RSI_14"))
        if settings.enable_rsi and rsi_value is not None:
            confirmations["rsi_oversold"] = IndicatorConfirmation(
                confirmed=rsi_value < settings.rsi_threshold,
                value=rsi_value,
                threshold=settings.rsi_threshold,
                detail=f"RSI below {settings.rsi_threshold:.1f} often signals oversold conditions.",
            )

        macd_hist = _safe_float(latest.get("MACD_12_26_9_Hist"))
        if settings.enable_macd and macd_hist is not None:
            confirmations["macd_bullish"] = IndicatorConfirmation(
                confirmed=macd_hist > settings.macd_hist_threshold,
                value=macd_hist,
                threshold=settings.macd_hist_threshold,
                detail="Positive MACD histogram suggests bullish momentum.",
            )

        bb_lower = _safe_float(latest.get("BB_Lower_20"))
        bb_upper = _safe_float(latest.get("BB_Upper_20"))
        proximity_pct = max(0.0, float(settings.bollinger_proximity_pct))

        def _pct_distance(value: float | None, reference: float | None) -> float | None:
            if value is None or reference is None or reference == 0:
                return None
            return (value - reference) / abs(reference)

        lower_distance = _pct_distance(last_close, bb_lower)
        upper_distance = _pct_distance(bb_upper, last_close)
        if settings.enable_bollinger and last_close is not None:
            confirmations["bollinger_lower_proximity"] = IndicatorConfirmation(
                confirmed=lower_distance is not None and lower_distance <= proximity_pct,
                value=(lower_distance * 100) if lower_distance is not None else None,
                threshold=proximity_pct * 100,
                detail=(
                    f"Price is within {proximity_pct * 100:.1f}% of the lower Bollinger band"
                    if lower_distance is not None
                    else "Lower Bollinger band proximity unavailable."
                ),
            )
            confirmations["bollinger_upper_headroom"] = IndicatorConfirmation(
                confirmed=upper_distance is not None and upper_distance >= proximity_pct,
                value=(upper_distance * 100) if upper_distance is not None else None,
                threshold=proximity_pct * 100,
                detail=(
                    f"At least {proximity_pct * 100:.1f}% room to the upper band"
                    if upper_distance is not None
                    else "Upper Bollinger band proximity unavailable."
                ),
            )

        atr_value = _safe_float(latest.get("ATR_14"))
        atr_fraction = (atr_value / last_close) if atr_value is not None and last_close else None
        if settings.enable_volatility and atr_fraction is not None:
            confirmations["volatility_contained"] = IndicatorConfirmation(
                confirmed=atr_fraction <= settings.max_atr_fraction_of_price,
                value=atr_fraction,
                threshold=settings.max_atr_fraction_of_price,
                detail=(
                    "ATR as a fraction of price remains within the configured buffer,"
                    " suggesting manageable volatility."
                ),
            )

        supertrend_direction = None
        for name in indicators.columns:
            if str(name).startswith("Supertrend_Direction_"):
                supertrend_direction = _safe_float(latest.get(name))
                break
        if supertrend_direction is not None:
            confirmations["supertrend_bullish"] = IndicatorConfirmation(
                confirmed=supertrend_direction > 0,
                value=supertrend_direction,
                threshold=0.0,
                detail="Upward Supertrend direction supports bullish bias.",
            )

        mfi_column = next((col for col in indicators.columns if str(col).startswith("MFI_")), None)
        mfi_value = _safe_float(latest.get(mfi_column)) if mfi_column else None
        obv_series = (
            pd.to_numeric(indicators.get("OBV"), errors="coerce") if "OBV" in indicators else pd.Series(dtype=float)
        )
        obv_change: float | None = None
        obv_confirmation = False
        if settings.enable_volume and not obv_series.empty:
            lookback = max(2, int(settings.obv_lookback))
            recent_obv = obv_series.dropna().tail(lookback)
            if len(recent_obv) >= 2:
                obv_change = float(recent_obv.iloc[-1] - recent_obv.iloc[0])
                obv_confirmation = obv_change > 0

        mfi_confirmation = (
            settings.enable_volume
            and mfi_value is not None
            and mfi_value >= settings.mfi_threshold
        )
        if settings.enable_volume and (mfi_value is not None or obv_change is not None):
            confirmations["volume_inflows"] = IndicatorConfirmation(
                confirmed=bool(mfi_confirmation or obv_confirmation),
                value=mfi_value if mfi_value is not None else obv_change,
                threshold=settings.mfi_threshold if mfi_value is not None else 0.0,
                detail=(
                    "Rising OBV and supportive MFI hint at accumulation."
                    if mfi_confirmation and obv_confirmation
                    else (
                        "Money Flow Index is above the configured inflow threshold."
                        if mfi_confirmation
                        else "On-balance volume is trending higher over the recent window."
                    )
                ),
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
