import numpy as np
import pandas as pd

from stock_predictor.core.features import FeatureAssembler
from stock_predictor.core.indicator_bundle import compute_indicators
from stock_predictor.core.indicators import (
    IndicatorInputs,
    anchored_vwap,
    liquidity_proxies,
    supertrend,
)


def _sample_price_frame(rows: int = 60) -> pd.DataFrame:
    dates = pd.date_range("2023-01-01", periods=rows, freq="D")
    base = np.linspace(100, 130, rows)
    data = {
        "Date": dates,
        "Open": base + 0.5,
        "High": base + 1.0,
        "Low": base - 1.0,
        "Close": base + np.sin(np.linspace(0, 3.14, rows)),
        "Volume": np.linspace(1_000_000, 1_500_000, rows),
    }
    return pd.DataFrame(data)


def test_supertrend_outputs_columns() -> None:
    price_df = _sample_price_frame()
    inputs = IndicatorInputs(
        high=price_df["High"].set_axis(price_df["Date"]),
        low=price_df["Low"].set_axis(price_df["Date"]),
        close=price_df["Close"].set_axis(price_df["Date"]),
        volume=price_df["Volume"].set_axis(price_df["Date"]),
    )

    result = supertrend(inputs, period=7, multiplier=2.0)
    assert "Supertrend_7" in result
    assert "Supertrend_Direction_7" in result
    assert result["Supertrend_7"].notna().any()


def test_anchored_vwap_uses_anchor_date() -> None:
    price_df = _sample_price_frame()
    inputs = IndicatorInputs(
        high=price_df["High"].set_axis(price_df["Date"]),
        low=price_df["Low"].set_axis(price_df["Date"]),
        close=price_df["Close"].set_axis(price_df["Date"]),
        volume=price_df["Volume"].set_axis(price_df["Date"]),
    )
    anchor_date = price_df.loc[20, "Date"]
    anchored = anchored_vwap(inputs, anchor=anchor_date)
    column = anchored.columns[0]
    before_anchor = anchored.loc[price_df.loc[0:19, "Date"], column]
    after_anchor = anchored.loc[price_df.loc[20:, "Date"], column]

    assert before_anchor.isna().all()
    assert after_anchor.notna().any()


def test_compute_indicators_respects_config_override() -> None:
    price_df = _sample_price_frame()
    result = compute_indicators(price_df, {"atr": {"period": 5}})
    assert "ATR_5" in result.dataframe.columns


def test_feature_assembler_uses_indicator_config() -> None:
    price_df = _sample_price_frame()
    assembler = FeatureAssembler(
        feature_toggles={"technical": True},
        horizons=(1,),
        technical_indicator_config={"supertrend": {"period": 5}},
    )
    features = assembler.build(price_df, news_df=None, sentiment_enabled=False)
    assert any(col.startswith("Supertrend_5") for col in features.features.columns)


def test_liquidity_proxies_generate_columns() -> None:
    price_df = _sample_price_frame()
    inputs = IndicatorInputs(
        high=price_df["High"].set_axis(price_df["Date"]),
        low=price_df["Low"].set_axis(price_df["Date"]),
        close=price_df["Close"].set_axis(price_df["Date"]),
        volume=price_df["Volume"].set_axis(price_df["Date"]),
    )
    liquidity = liquidity_proxies(inputs, window=10)
    expected = {
        "Liquidity_DollarVolume_10",
        "Liquidity_VolumeVolatility_10",
        "Liquidity_Turnover_10",
        "Liquidity_ImpactProxy_10",
        "Sentiment_VolumeCorrelation_10",
    }
    assert expected.issubset(set(liquidity.columns))
