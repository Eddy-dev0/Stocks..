"""Feature engineering utilities for the stock predictor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

import numpy as np
import pandas as pd

from ..sentiment import aggregate_daily_sentiment, attach_sentiment


@dataclass(slots=True)
class FeatureResult:
    """Container holding processed features and metadata."""

    features: pd.DataFrame
    targets: Dict[int, Dict[str, pd.Series]]
    metadata: Dict[str, object]


class FeatureAssembler:
    """Build enriched feature sets from heterogeneous market inputs.

    The assembler orchestrates a modular collection of feature builders that
    generate technical indicators, Elliott wave descriptors, lightweight
    fundamentals, sentiment aggregates, and macro-style factors derived from
    historical price series.
    """

    SUPPORTED_SETS = {
        "technical",
        "elliott",
        "fundamental",
        "sentiment",
        "macro",
    }

    def __init__(self, enabled_sets: Iterable[str], horizons: Iterable[int] | None = None) -> None:
        self.enabled_sets = {name.lower() for name in enabled_sets if name}
        self.enabled_sets.intersection_update(self.SUPPORTED_SETS)
        if not self.enabled_sets:
            self.enabled_sets = {"technical"}
        if horizons is None:
            horizons = (1,)
        self.horizons = _normalise_horizons(horizons)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def build(
        self,
        price_df: pd.DataFrame,
        news_df: pd.DataFrame | None,
        sentiment_enabled: bool,
    ) -> FeatureResult:
        if price_df.empty:
            raise ValueError("Price dataframe cannot be empty when building features.")

        processed = price_df.copy()
        processed = _ensure_datetime_index(processed)

        feature_blocks: list[pd.DataFrame] = []
        metadata: Dict[str, object] = {}

        if "technical" in self.enabled_sets:
            feature_blocks.append(_build_technical_features(processed))
        if "elliott" in self.enabled_sets:
            feature_blocks.append(_build_elliott_wave_descriptors(processed))
        if "fundamental" in self.enabled_sets:
            feature_blocks.append(_build_fundamental_proxies(processed))
        if "macro" in self.enabled_sets:
            feature_blocks.append(_build_macro_context(processed))

        if sentiment_enabled and news_df is not None and not news_df.empty:
            sentiment = _prepare_sentiment_features(news_df)
            if "sentiment" in self.enabled_sets:
                feature_blocks.append(sentiment)
            metadata["sentiment_daily"] = sentiment
        else:
            metadata["sentiment_daily"] = pd.DataFrame(columns=["Date", "sentiment"])

        if not feature_blocks:
            raise RuntimeError("No feature blocks were generated. Check configuration.")

        merged = processed[["Date", "Close"]].copy()
        for block in feature_blocks:
            merged = merged.merge(block, on="Date", how="left")

        merged = merged.sort_values("Date").reset_index(drop=True)
        merged = merged.ffill().bfill()

        merged["Close_Current"] = merged["Close"]

        targets = _generate_targets(merged, self.horizons)

        feature_columns = [col for col in merged.columns if col not in {"Date", "Close"}]
        metadata.update(
            {
                "feature_columns": feature_columns,
                "latest_features": merged.iloc[[-1]][feature_columns],
                "latest_close": float(merged.iloc[-1]["Close"]),
                "latest_date": pd.to_datetime(merged.iloc[-1]["Date"]),
                "horizons": self.horizons,
                "target_dates": _estimate_target_dates(merged["Date"], self.horizons),
            }
        )

        features = merged[feature_columns]
        return FeatureResult(features=features, targets=targets, metadata=metadata)


# ----------------------------------------------------------------------
# Feature blocks
# ----------------------------------------------------------------------

def _ensure_datetime_index(price_df: pd.DataFrame) -> pd.DataFrame:
    df = price_df.copy()
    if "Date" not in df.columns:
        raise ValueError("Input price data must contain a 'Date' column.")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])

    numeric_candidates = [
        "Open",
        "High",
        "Low",
        "Close",
        "Adj Close",
        "Volume",
    ]
    for column in numeric_candidates:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=["Close"])
    df = df.sort_values("Date")
    return df


def _build_technical_features(price_df: pd.DataFrame) -> pd.DataFrame:
    df = price_df.copy()
    df = df.sort_values("Date")

    df["Return_1d"] = df["Close"].pct_change()
    df["LogReturn_1d"] = np.log(df["Close"].replace(0, np.nan)).diff()
    df["SMA_5"] = df["Close"].rolling(window=5, min_periods=1).mean()
    df["SMA_20"] = df["Close"].rolling(window=20, min_periods=1).mean()
    df["EMA_12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["EMA_26"] = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = df["EMA_12"] - df["EMA_26"]
    df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["RSI_14"] = _compute_rsi(df["Close"], window=14)
    df["Bollinger_Mid"] = df["Close"].rolling(window=20, min_periods=1).mean()
    df["Bollinger_Std"] = df["Close"].rolling(window=20, min_periods=1).std()
    df["Bollinger_Upper"] = df["Bollinger_Mid"] + 2 * df["Bollinger_Std"]
    df["Bollinger_Lower"] = df["Bollinger_Mid"] - 2 * df["Bollinger_Std"]
    df["Volume_Change"] = df.get("Volume", pd.Series(index=df.index)).pct_change()
    df["ATR_14"] = _compute_atr(df, window=14)

    columns = [
        "Date",
        "Return_1d",
        "LogReturn_1d",
        "SMA_5",
        "SMA_20",
        "EMA_12",
        "EMA_26",
        "MACD",
        "Signal",
        "RSI_14",
        "Bollinger_Mid",
        "Bollinger_Std",
        "Bollinger_Upper",
        "Bollinger_Lower",
        "Volume_Change",
        "ATR_14",
    ]
    return df[columns]


def _compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def _compute_atr(price_df: pd.DataFrame, window: int = 14) -> pd.Series:
    high = price_df.get("High", price_df["Close"])
    low = price_df.get("Low", price_df["Close"])
    close = price_df["Close"].shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - close).abs(),
            (low - close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(window=window, min_periods=1).mean()
    return atr


def _build_elliott_wave_descriptors(price_df: pd.DataFrame) -> pd.DataFrame:
    df = price_df[["Date", "Close"]].copy()
    df["Swing_High"] = df["Close"].rolling(window=5, center=True).max()
    df["Swing_Low"] = df["Close"].rolling(window=5, center=True).min()
    df["Wave_Strength"] = (df["Swing_High"] - df["Swing_Low"]) / df["Close"].replace(0, np.nan)
    df["Impulse"] = df["Close"].diff(3)
    df["Corrective"] = df["Close"].diff().rolling(window=3, min_periods=1).sum()
    df["Wave_Oscillator"] = df["Impulse"].rolling(window=5, min_periods=1).mean()
    df = df.fillna(0.0)
    return df[["Date", "Swing_High", "Swing_Low", "Wave_Strength", "Impulse", "Corrective", "Wave_Oscillator"]]


def _build_fundamental_proxies(price_df: pd.DataFrame) -> pd.DataFrame:
    df = price_df.copy()
    df["Price_to_SMA20"] = df["Close"] / df["Close"].rolling(window=20, min_periods=1).mean()
    df["Price_to_SMA200"] = df["Close"] / df["Close"].rolling(window=200, min_periods=1).mean()
    df["Volume_Trend"] = df.get("Volume", pd.Series(index=df.index)).rolling(window=30, min_periods=1).mean()
    df["Liquidity_Ratio"] = df.get("Volume", pd.Series(index=df.index)) / df["Close"].replace(0, np.nan)
    df["Momentum_12"] = df["Close"].pct_change(periods=252)
    df = df.fillna(0.0)
    return df[[
        "Date",
        "Price_to_SMA20",
        "Price_to_SMA200",
        "Volume_Trend",
        "Liquidity_Ratio",
        "Momentum_12",
    ]]


def _build_macro_context(price_df: pd.DataFrame) -> pd.DataFrame:
    df = price_df.copy()
    df["Volatility_21"] = price_df["Close"].pct_change().rolling(window=21, min_periods=1).std()
    df["Volatility_63"] = price_df["Close"].pct_change().rolling(window=63, min_periods=1).std()
    df["Trend_Slope"] = _rolling_linear_trend(price_df["Close"], window=21)
    df["Trend_Curvature"] = _rolling_linear_trend(price_df["Close"], window=63)
    df["Return_Correlation"] = price_df["Close"].pct_change().rolling(window=63, min_periods=1).corr(price_df["Close"].pct_change().shift(1))
    df = df.fillna(0.0)
    return df[[
        "Date",
        "Volatility_21",
        "Volatility_63",
        "Trend_Slope",
        "Trend_Curvature",
        "Return_Correlation",
    ]]


def _rolling_linear_trend(series: pd.Series, window: int) -> pd.Series:
    if window <= 1:
        return pd.Series(0.0, index=series.index)
    slopes = []
    x = np.arange(window)
    for i in range(len(series)):
        if i + 1 < window:
            slopes.append(np.nan)
            continue
        window_values = series.iloc[i + 1 - window : i + 1].values
        if np.isnan(window_values).all():
            slopes.append(np.nan)
            continue
        coeffs = np.polyfit(x, window_values, deg=2)
        slopes.append(coeffs[1])
    return pd.Series(slopes, index=series.index)


def _prepare_sentiment_features(news_df: pd.DataFrame) -> pd.DataFrame:
    scored = attach_sentiment(news_df)
    aggregated = aggregate_daily_sentiment(scored)
    if aggregated.empty:
        return pd.DataFrame({"Date": [], "sentiment": []})
    aggregated = aggregated.rename(columns={"sentiment": "Sentiment_Avg"})
    aggregated["Sentiment_Change"] = aggregated["Sentiment_Avg"].diff().fillna(0.0)
    return aggregated


# ----------------------------------------------------------------------
# Target generation
# ----------------------------------------------------------------------

def _generate_targets(merged: pd.DataFrame, horizons: Iterable[int]) -> Dict[int, Dict[str, pd.Series]]:
    merged = merged.copy()
    merged["Daily_Return"] = merged["Close"].pct_change()

    targets_by_horizon: Dict[int, Dict[str, pd.Series]] = {}
    for horizon in horizons:
        future_close = merged["Close"].shift(-horizon)
        future_return = (future_close - merged["Close"]) / merged["Close"]
        direction = (future_return > 0).astype(float)
        direction[future_return.isna()] = np.nan
        volatility = (
            merged["Daily_Return"].rolling(window=horizon, min_periods=1).std().shift(-horizon)
        )

        horizon_targets: Dict[str, pd.Series] = {
            "close": future_close,
            "return": future_return,
            "direction": direction,
            "volatility": volatility,
        }

        valid_targets: Dict[str, pd.Series] = {}
        for name, series in horizon_targets.items():
            cleaned = series.dropna()
            if cleaned.empty:
                continue
            valid_targets[name] = series
        if valid_targets:
            targets_by_horizon[horizon] = valid_targets

    return targets_by_horizon


def _estimate_target_dates(dates: pd.Series, horizons: Iterable[int]) -> Dict[int, pd.Timestamp]:
    timestamps = pd.to_datetime(dates).dropna().sort_values()
    if timestamps.empty:
        return {}
    latest = timestamps.iloc[-1]
    result: Dict[int, pd.Timestamp] = {}
    for horizon in horizons:
        try:
            offset = pd.tseries.offsets.BDay(int(horizon))
        except Exception:  # pragma: no cover - defensive
            offset = pd.Timedelta(days=int(horizon))
        result[int(horizon)] = latest + offset
    return result


def _normalise_horizons(horizons: Iterable[int]) -> tuple[int, ...]:
    unique: list[int] = []
    for raw in horizons:
        try:
            value = int(raw)
        except (TypeError, ValueError):
            continue
        if value <= 0:
            continue
        if value not in unique:
            unique.append(value)
    if not unique:
        unique = [1]
    unique.sort()
    return tuple(unique)
