"""Lorentzian classification strategy implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..indicators import IndicatorInputs
from ..indicators.trend import supertrend
from ..indicators.volatility import average_true_range


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=1).mean()


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def _macd_hist(close: pd.Series) -> pd.Series:
    ema12 = _ema(close, 12)
    ema26 = _ema(close, 26)
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False, min_periods=1).mean()
    return macd - signal


@dataclass(slots=True)
class LorentzianClassificationStrategy:
    """Combine Lorentzian classification with EMA200 and Supertrend filters."""

    ema_period: int = 200
    atr_period: int = 14
    atr_multiplier: float = 1.5
    supertrend_period: int = 10
    supertrend_multiplier: float = 3.0
    lookback: int = 50

    def compute(
        self,
        inputs: IndicatorInputs,
        *,
        backtest: bool = False,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        close = inputs.close

        ema200 = _ema(close, self.ema_period)
        atr_df = average_true_range(inputs, period=self.atr_period)
        atr = atr_df.iloc[:, 0] if not atr_df.empty else pd.Series(0.0, index=close.index)
        st_df = supertrend(
            inputs,
            period=self.supertrend_period,
            multiplier=self.supertrend_multiplier,
        )
        st_line = st_df.iloc[:, 0] if not st_df.empty else pd.Series(np.nan, index=close.index)

        rsi = _rsi(close, period=14)
        macd_hist = _macd_hist(close)
        trend = (close - ema200) / atr.replace(0, np.nan)
        features = pd.concat(
            [
                (rsi - 50) / 50,
                macd_hist / atr.replace(0, np.nan),
                trend,
            ],
            axis=1,
        ).fillna(0.0)
        features.columns = ["rsi_norm", "macd_norm", "trend_norm"]

        bullish_target = np.array([1.0, 1.0, 1.0])
        bearish_target = np.array([-1.0, -1.0, -1.0])

        distance_bull = np.log1p(np.abs(features - bullish_target)).sum(axis=1)
        distance_bear = np.log1p(np.abs(features - bearish_target)).sum(axis=1)
        score = distance_bear - distance_bull

        bullish_filter = close > ema200
        bearish_filter = close < ema200
        st_bull = close > st_line
        st_bear = close < st_line

        signal = pd.Series(0.0, index=close.index)
        signal[(score > 0.1) & bullish_filter & st_bull] = 1.0
        signal[(score < -0.1) & bearish_filter & st_bear] = -1.0

        stop_loss = close - atr * self.atr_multiplier
        stop_loss = stop_loss.where(signal >= 0, close + atr * self.atr_multiplier)

        output = pd.DataFrame(
            {
                "Lorentzian_Score": score,
                "Lorentzian_Signal": signal,
                "Lorentzian_Stop": stop_loss,
            },
            index=close.index,
        )

        metrics: dict[str, Any] = {}
        if backtest:
            returns = close.pct_change(fill_method=None).shift(-1)
            strategy_returns = returns * signal
            metrics = {
                "hit_rate": float((strategy_returns > 0).mean()),
                "avg_return": float(strategy_returns.mean()),
                "trade_count": int((signal != 0).sum()),
            }

        return output, metrics


__all__ = ["LorentzianClassificationStrategy"]
