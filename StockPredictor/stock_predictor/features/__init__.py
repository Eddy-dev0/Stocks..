"""Feature engineering utilities for the stock predictor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, Optional

import numpy as np
import pandas as pd

from ..sentiment import aggregate_daily_sentiment, attach_sentiment


@dataclass(slots=True)
class FeatureResult:
    """Container holding processed features and metadata."""

    features: pd.DataFrame
    targets: Dict[int, Dict[str, pd.Series]]
    metadata: Dict[str, object]


@dataclass(slots=True)
class FeatureBlock:
    """Bundle of features that belong to the same semantic category."""

    frame: pd.DataFrame
    category: str
    column_categories: Optional[Dict[str, str]] = None

    def iter_categories(self) -> Iterator[tuple[str, str]]:
        mapping = self.column_categories
        if mapping is None:
            mapping = {col: self.category for col in self.frame.columns if col != "Date"}
        return ((column, category) for column, category in mapping.items() if column != "Date")


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

        feature_blocks: list[FeatureBlock] = []
        metadata: Dict[str, object] = {}
        feature_categories: Dict[str, str] = {}

        if "technical" in self.enabled_sets:
            block = _build_technical_features(processed)
            if block is not None:
                feature_blocks.append(block)
        if "elliott" in self.enabled_sets:
            block = _build_elliott_wave_descriptors(processed)
            if block is not None:
                feature_blocks.append(block)
        if "fundamental" in self.enabled_sets:
            for block in _build_fundamental_blocks(processed):
                if block is not None:
                    feature_blocks.append(block)
        if "macro" in self.enabled_sets:
            for block in _build_macro_blocks(processed):
                if block is not None:
                    feature_blocks.append(block)

        if sentiment_enabled and news_df is not None and not news_df.empty:
            sentiment_block = _build_sentiment_features(news_df)
            if "sentiment" in self.enabled_sets and sentiment_block is not None:
                feature_blocks.append(sentiment_block)
            metadata["sentiment_daily"] = (
                sentiment_block.frame if sentiment_block is not None else _empty_sentiment_frame()
            )
        else:
            metadata["sentiment_daily"] = _empty_sentiment_frame()

        if not feature_blocks:
            raise RuntimeError("No feature blocks were generated. Check configuration.")

        merged = processed[["Date", "Close"]].copy()
        for block in feature_blocks:
            merged = merged.merge(block.frame, on="Date", how="left")
            for column, category in block.iter_categories():
                feature_categories[column] = category

        merged = merged.sort_values("Date").reset_index(drop=True)
        merged = merged.replace([np.inf, -np.inf], np.nan)
        merged = merged.ffill().bfill()
        numeric_cols = merged.select_dtypes(include=[np.number]).columns
        merged[numeric_cols] = merged[numeric_cols].fillna(0.0)

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
                "feature_categories": feature_categories,
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


def _build_technical_features(price_df: pd.DataFrame) -> FeatureBlock | None:
    if price_df.empty:
        return None

    df = price_df.sort_values("Date").reset_index(drop=True)
    close = pd.to_numeric(df["Close"], errors="coerce")
    volume = _get_numeric_series(df, "Volume", default=np.nan)
    high = _get_numeric_series(df, "High", default=np.nan).fillna(close)
    low = _get_numeric_series(df, "Low", default=np.nan).fillna(close)

    features: Dict[str, pd.Series] = {}

    features["Return_1d"] = close.pct_change()
    features["LogReturn_1d"] = np.log(close.replace(0, np.nan)).diff()

    sma_windows = (5, 10, 20, 50, 100, 200)
    ema_windows = (10, 20, 50, 100, 200)
    for window in sma_windows:
        sma = close.rolling(window=window, min_periods=1).mean()
        features[f"SMA_{window}"] = sma
        features[f"Price_to_SMA_{window}"] = _safe_divide(close, sma)

    for span in ema_windows:
        ema = close.ewm(span=span, adjust=False, min_periods=1).mean()
        features[f"EMA_{span}"] = ema
        features[f"Price_to_EMA_{span}"] = _safe_divide(close, ema)

    macd_configs = ((12, 26, 9), (8, 17, 9), (5, 35, 5))
    for fast, slow, signal in macd_configs:
        macd_line, signal_line, hist = _compute_macd(close, fast, slow, signal)
        prefix = f"MACD_{fast}_{slow}_{signal}"
        features[f"{prefix}_Line"] = macd_line
        features[f"{prefix}_Signal"] = signal_line
        features[f"{prefix}_Hist"] = hist

    for window in (7, 14, 21):
        features[f"RSI_{window}"] = _compute_rsi(close, window=window)

    for window, num_std in ((20, 2.0), (50, 2.0)):
        mid, upper, lower, bandwidth, percent_b = _bollinger(close, window, num_std)
        prefix = f"Bollinger_{window}"
        features[f"{prefix}_Mid"] = mid
        features[f"{prefix}_Upper"] = upper
        features[f"{prefix}_Lower"] = lower
        features[f"{prefix}_Bandwidth"] = bandwidth
        features[f"{prefix}_PercentB"] = percent_b

    stochastic_k, stochastic_d = _compute_stochastic(close, high, low, window=14, smooth_k=3, smooth_d=3)
    features["Stochastic_%K"] = stochastic_k
    features["Stochastic_%D"] = stochastic_d

    for window in (5, 10, 20, 63, 126):
        features[f"ROC_{window}"] = close.pct_change(periods=window)

    for window in (7, 14, 21):
        features[f"ATR_{window}"] = _compute_atr(df.assign(High=high, Low=low, Close=close), window=window)

    volume_change = volume.pct_change()
    features["Volume_Change"] = volume_change
    if not volume.isna().all():
        for window in (10, 20, 50):
            rolling_vol = volume.rolling(window=window, min_periods=1).mean()
            rolling_std = volume.rolling(window=window, min_periods=1).std()
            features[f"Volume_SMA_{window}"] = rolling_vol
            features[f"Volume_EMA_{window}"] = volume.ewm(span=window, adjust=False, min_periods=1).mean()
            features[f"Volume_ZScore_{window}"] = (volume - rolling_vol) / rolling_std.replace(0, np.nan)
        price_volume = _safe_divide(volume, close)
        features["Volume_to_Price"] = price_volume

    feature_frame = pd.DataFrame({"Date": df["Date"]})
    for name, series in features.items():
        feature_frame[name] = series

    return FeatureBlock(feature_frame, category="technical")


def _get_numeric_series(df: pd.DataFrame, column: str, default: float = np.nan) -> pd.Series:
    if column in df:
        return pd.to_numeric(df[column], errors="coerce")
    return pd.Series(default, index=df.index, dtype="float64")


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0, np.nan)


def _compute_macd(
    close: pd.Series, fast: int, slow: int, signal: int
) -> tuple[pd.Series, pd.Series, pd.Series]:
    fast_ema = close.ewm(span=fast, adjust=False, min_periods=1).mean()
    slow_ema = close.ewm(span=slow, adjust=False, min_periods=1).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=1).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def _bollinger(
    series: pd.Series, window: int, num_std: float
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    mid = series.rolling(window=window, min_periods=1).mean()
    std = series.rolling(window=window, min_periods=1).std(ddof=0)
    upper = mid + num_std * std
    lower = mid - num_std * std
    bandwidth = (upper - lower) / mid.replace(0, np.nan)
    percent_b = (series - lower) / (upper - lower).replace(0, np.nan)
    return mid, upper, lower, bandwidth, percent_b


def _compute_stochastic(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    window: int = 14,
    smooth_k: int = 3,
    smooth_d: int = 3,
) -> tuple[pd.Series, pd.Series]:
    lowest_low = low.rolling(window=window, min_periods=1).min()
    highest_high = high.rolling(window=window, min_periods=1).max()
    raw_k = 100 * _safe_divide(close - lowest_low, (highest_high - lowest_low))
    smooth_k_series = raw_k.rolling(window=smooth_k, min_periods=1).mean()
    smooth_d_series = smooth_k_series.rolling(window=smooth_d, min_periods=1).mean()
    return smooth_k_series, smooth_d_series


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


def _build_elliott_wave_descriptors(price_df: pd.DataFrame) -> FeatureBlock | None:
    if "Close" not in price_df:
        return None

    df = price_df[["Date", "Close"]].copy()
    df = df.sort_values("Date").reset_index(drop=True)
    close = pd.to_numeric(df["Close"], errors="coerce")

    df["Swing_High"] = close.rolling(window=5, center=True, min_periods=1).max()
    df["Swing_Low"] = close.rolling(window=5, center=True, min_periods=1).min()
    df["Wave_Strength"] = _safe_divide(df["Swing_High"] - df["Swing_Low"], close.abs())
    df["Impulse"] = close.diff(3)
    df["Corrective"] = close.diff().rolling(window=3, min_periods=1).sum()
    df["Wave_Oscillator"] = df["Impulse"].rolling(window=5, min_periods=1).mean()

    df["Wave_Phase_Code"] = np.select(
        [df["Impulse"] > 0, df["Impulse"] < 0],
        [1.0, -1.0],
        default=0.0,
    )
    try:
        df["Wave_Strength_Bucket"] = pd.qcut(
            df["Wave_Strength"].rank(method="first"),
            q=4,
            labels=False,
        ).astype(float)
    except ValueError:
        df["Wave_Strength_Bucket"] = np.nan

    df = df.fillna(0.0)

    columns = [
        "Date",
        "Swing_High",
        "Swing_Low",
        "Wave_Strength",
        "Impulse",
        "Corrective",
        "Wave_Oscillator",
        "Wave_Phase_Code",
        "Wave_Strength_Bucket",
    ]

    column_categories = {
        "Swing_High": "elliott",
        "Swing_Low": "elliott",
        "Wave_Strength": "elliott",
        "Impulse": "elliott",
        "Corrective": "elliott",
        "Wave_Oscillator": "elliott",
        "Wave_Phase_Code": "elliott_categorical",
        "Wave_Strength_Bucket": "elliott_categorical",
    }

    return FeatureBlock(df[columns], category="elliott", column_categories=column_categories)


def _build_fundamental_blocks(price_df: pd.DataFrame) -> list[FeatureBlock | None]:
    return [
        _build_fundamental_proxies(price_df),
        _build_fundamental_metrics(price_df),
    ]


def _build_fundamental_proxies(price_df: pd.DataFrame) -> FeatureBlock | None:
    if price_df.empty:
        return None

    df = price_df.copy()
    close = pd.to_numeric(df["Close"], errors="coerce")
    volume = _get_numeric_series(df, "Volume", default=np.nan)

    proxies = pd.DataFrame({"Date": df["Date"]})
    proxies["Price_to_SMA20"] = _safe_divide(close, close.rolling(window=20, min_periods=1).mean())
    proxies["Price_to_SMA200"] = _safe_divide(close, close.rolling(window=200, min_periods=1).mean())
    proxies["Volume_Trend_30"] = volume.rolling(window=30, min_periods=1).mean()
    proxies["Liquidity_Ratio"] = _safe_divide(volume, close)
    proxies["Momentum_252"] = close.pct_change(periods=252)
    proxies = proxies.fillna(0.0)

    return FeatureBlock(proxies, category="fundamental")


def _build_fundamental_metrics(price_df: pd.DataFrame) -> FeatureBlock | None:
    metric_cols = [col for col in price_df.columns if col.startswith("Fundamental_")]
    if not metric_cols:
        return None

    df = price_df[["Date"] + metric_cols].copy()
    metrics = pd.DataFrame({"Date": df["Date"]})
    column_categories: Dict[str, str] = {}

    for column in metric_cols:
        series = pd.to_numeric(df[column], errors="coerce")
        base_name = column.replace("Fundamental_", "")
        latest_col = f"Fund_{base_name}_Latest"
        metrics[latest_col] = series.ffill()
        column_categories[latest_col] = "fundamental"

        pct_change_col = f"Fund_{base_name}_PctChange_63"
        metrics[pct_change_col] = series.ffill().pct_change(periods=63)
        column_categories[pct_change_col] = "fundamental_trend"

        zscore_col = f"Fund_{base_name}_ZScore_252"
        rolling_mean = series.ffill().rolling(window=252, min_periods=5).mean()
        rolling_std = series.ffill().rolling(window=252, min_periods=5).std()
        metrics[zscore_col] = (series.ffill() - rolling_mean) / rolling_std.replace(0, np.nan)
        column_categories[zscore_col] = "fundamental_trend"

    metrics = metrics.fillna(0.0)
    return FeatureBlock(metrics, category="fundamental", column_categories=column_categories)


def _build_macro_blocks(price_df: pd.DataFrame) -> list[FeatureBlock | None]:
    return [
        _build_macro_context(price_df),
        _build_macro_benchmarks(price_df),
        _build_cross_sectional_betas(price_df),
    ]


def _build_macro_context(price_df: pd.DataFrame) -> FeatureBlock | None:
    if "Close" not in price_df:
        return None

    df = price_df.sort_values("Date").reset_index(drop=True)
    returns = pd.to_numeric(df["Close"], errors="coerce").pct_change()

    macro = pd.DataFrame({"Date": df["Date"]})
    macro["Volatility_21"] = returns.rolling(window=21, min_periods=5).std()
    macro["Volatility_63"] = returns.rolling(window=63, min_periods=10).std()
    macro["Trend_Slope_21"] = _rolling_linear_trend(df["Close"], window=21)
    macro["Trend_Curvature_63"] = _rolling_linear_trend(df["Close"], window=63)
    macro["Return_Corr_63"] = returns.rolling(window=63, min_periods=10).corr(returns.shift(1))
    macro = macro.fillna(0.0)

    column_categories = {
        "Volatility_21": "macro_volatility",
        "Volatility_63": "macro_volatility",
        "Trend_Slope_21": "macro_trend",
        "Trend_Curvature_63": "macro_trend",
        "Return_Corr_63": "macro_correlation",
    }

    return FeatureBlock(macro, category="macro", column_categories=column_categories)


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


def _build_macro_benchmarks(price_df: pd.DataFrame) -> FeatureBlock | None:
    base_close_raw = price_df.get("Close")
    if base_close_raw is None:
        return None
    base_close = pd.to_numeric(base_close_raw, errors="coerce")

    benchmarks = {
        "^GSPC": "sp500",
        "^VIX": "vix",
    }

    features: Dict[str, pd.Series] = {}
    date_series = price_df.get("Date")
    for symbol, name in benchmarks.items():
        benchmark = _extract_benchmark_series(price_df, symbol)
        if benchmark is None:
            continue
        benchmark_returns = benchmark.pct_change()
        symbol_prefix = name.upper()
        features[f"{symbol_prefix}_Return"] = benchmark_returns
        features[f"{symbol_prefix}_RollingCorr_21"] = (
            base_close.pct_change().rolling(window=21, min_periods=5).corr(benchmark_returns)
        )
        features[f"{symbol_prefix}_Relative_Return"] = base_close.pct_change() - benchmark_returns
        features[f"{symbol_prefix}_Price_Ratio"] = _safe_divide(base_close, benchmark)

    if not features:
        return None

    feature_frame = pd.DataFrame({"Date": date_series})
    for name, series in features.items():
        feature_frame[name] = series

    column_categories = {col: "macro_benchmark" for col in feature_frame.columns if col != "Date"}
    return FeatureBlock(feature_frame.fillna(0.0), category="macro", column_categories=column_categories)


def _build_cross_sectional_betas(price_df: pd.DataFrame) -> FeatureBlock | None:
    base_close_raw = price_df.get("Close")
    if base_close_raw is None:
        return None
    base_close = pd.to_numeric(base_close_raw, errors="coerce")

    base_returns = base_close.pct_change()
    benchmarks = {
        "^GSPC": "Beta_SP500",
        "^VIX": "Beta_VIX",
    }

    beta_features: Dict[str, pd.Series] = {}
    for symbol, label in benchmarks.items():
        benchmark = _extract_benchmark_series(price_df, symbol)
        if benchmark is None:
            continue
        benchmark_returns = benchmark.pct_change()
        for window in (21, 63, 126):
            cov = base_returns.rolling(window=window, min_periods=5).cov(benchmark_returns)
            var = benchmark_returns.rolling(window=window, min_periods=5).var()
            beta_features[f"{label}_{window}"] = _safe_divide(cov, var)

    if not beta_features:
        return None

    frame = pd.DataFrame({"Date": price_df["Date"]})
    for name, series in beta_features.items():
        frame[name] = series

    column_categories = {col: "macro_beta" for col in frame.columns if col != "Date"}
    return FeatureBlock(frame.fillna(0.0), category="macro", column_categories=column_categories)


def _extract_benchmark_series(price_df: pd.DataFrame, symbol: str) -> pd.Series | None:
    normalized_symbol = symbol.replace("^", "")
    candidates = []
    possible_names = [
        f"Close_{symbol}",
        f"Close_{normalized_symbol}",
        f"{symbol}_Close",
        f"{normalized_symbol}_Close",
        f"Adj Close_{symbol}",
        f"Adj Close_{normalized_symbol}",
        f"{symbol}_Adj Close",
        f"{normalized_symbol}_Adj Close",
    ]
    for name in possible_names:
        if name in price_df.columns:
            candidates.append(name)
    if not candidates:
        for column in price_df.columns:
            if normalized_symbol in column and "close" in column.lower():
                candidates.append(column)
    if not candidates:
        return None

    series = pd.to_numeric(price_df[candidates[0]], errors="coerce")
    return series


def _build_sentiment_features(news_df: pd.DataFrame) -> FeatureBlock | None:
    scored = attach_sentiment(news_df)
    aggregated = aggregate_daily_sentiment(scored)
    if aggregated.empty:
        return None

    aggregated = aggregated.rename(columns={"sentiment": "Sentiment_Avg"})
    aggregated["Sentiment_Change"] = aggregated["Sentiment_Avg"].diff()
    aggregated["Sentiment_7d"] = aggregated["Sentiment_Avg"].rolling(window=7, min_periods=1).mean()
    aggregated["Sentiment_30d"] = aggregated["Sentiment_Avg"].rolling(window=30, min_periods=1).mean()
    aggregated["Sentiment_ZScore_30"] = (
        aggregated["Sentiment_Avg"] - aggregated["Sentiment_Avg"].rolling(window=30, min_periods=5).mean()
    ) / aggregated["Sentiment_Avg"].rolling(window=30, min_periods=5).std().replace(0, np.nan)
    aggregated = aggregated.fillna(0.0)

    column_categories = {
        "Sentiment_Avg": "sentiment",
        "Sentiment_Change": "sentiment_trend",
        "Sentiment_7d": "sentiment_trend",
        "Sentiment_30d": "sentiment_trend",
        "Sentiment_ZScore_30": "sentiment_trend",
    }

    return FeatureBlock(aggregated, category="sentiment", column_categories=column_categories)


def _empty_sentiment_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "Date",
            "Sentiment_Avg",
            "Sentiment_Change",
            "Sentiment_7d",
            "Sentiment_30d",
            "Sentiment_ZScore_30",
        ]
    )


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
