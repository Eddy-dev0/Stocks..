import numpy as np
import pandas as pd

from stock_predictor.core.features import FeatureAssembler
from stock_predictor.core.indicator_bundle import compute_indicators
from stock_predictor.core.indicators import (
    IndicatorInputs,
    accumulation_distribution_line,
    anchored_vwap,
    chaikin_accumulation_distribution,
    liquidity_proxies,
    momentum,
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


def test_feature_assembler_handles_date_column_and_index() -> None:
    price_df = _sample_price_frame()
    price_df = price_df.set_index("Date")
    price_df["Date"] = price_df.index

    assembler = FeatureAssembler(
        feature_toggles={"technical": True, "volume_liquidity": True, "macro": True},
        horizons=(1,),
    )

    result = assembler.build(price_df, news_df=None, sentiment_enabled=False)

    assert not result.features.empty
    expected_date = pd.to_datetime(price_df["Date"], utc=True).iloc[-1]
    assert result.metadata["latest_date"] == expected_date


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


def test_cci_uses_talib_when_available(monkeypatch) -> None:
    price_df = _sample_price_frame()
    inputs = IndicatorInputs(
        high=price_df["High"].set_axis(price_df["Date"]),
        low=price_df["Low"].set_axis(price_df["Date"]),
        close=price_df["Close"].set_axis(price_df["Date"]),
    )

    expected = pd.Series(np.linspace(-1, 1, len(price_df)), index=inputs.close.index, name="CCI_20")

    class _FakeTalib:
        @staticmethod
        def CCI(high, low, close, timeperiod=20):
            assert timeperiod == 20
            pd.testing.assert_series_equal(high, inputs.high)
            pd.testing.assert_series_equal(low, inputs.low)
            pd.testing.assert_series_equal(close, inputs.close)
            return expected

    monkeypatch.setattr(momentum, "TA_LIB_AVAILABLE", True)
    monkeypatch.setattr(momentum, "talib", _FakeTalib())

    cci = momentum.commodity_channel_index(inputs, period=20)
    pd.testing.assert_series_equal(cci["CCI_20"], expected)


def test_cci_manual_calculation_matches_reference(monkeypatch) -> None:
    price_df = _sample_price_frame()
    inputs = IndicatorInputs(
        high=price_df["High"].set_axis(price_df["Date"]),
        low=price_df["Low"].set_axis(price_df["Date"]),
        close=price_df["Close"].set_axis(price_df["Date"]),
    )

    monkeypatch.setattr(momentum, "TA_LIB_AVAILABLE", False)
    monkeypatch.setattr(momentum, "talib", None)

    result = momentum.commodity_channel_index(inputs, period=5)["CCI_5"]

    typical_price = (inputs.high + inputs.low + inputs.close) / 3
    rolling_mean = typical_price.rolling(window=5, min_periods=1).mean()
    rolling_dev = typical_price.rolling(window=5, min_periods=1).apply(
        lambda window: np.mean(np.abs(window - window.mean())), raw=False
    )
    expected = (typical_price - rolling_mean) / (0.015 * rolling_dev.replace(0, np.nan))
    expected.name = "CCI_5"

    pd.testing.assert_series_equal(result, expected)


def test_indicator_bundle_exposes_configurable_mas_and_cci(monkeypatch) -> None:
    price_df = _sample_price_frame(rows=120)

    monkeypatch.setattr(momentum, "TA_LIB_AVAILABLE", False)
    monkeypatch.setattr(momentum, "talib", None)

    config = {
        "moving_averages": {"sma_periods": (10, 20, 100), "ema_periods": (10, 26, 100)},
        "cci": {"period": 30},
    }

    indicators = compute_indicators(price_df, config).dataframe

    assert {"SMA_10", "SMA_100", "EMA_10", "EMA_100", "CCI_30"}.issubset(indicators.columns)


def test_features_propagate_indicator_columns(monkeypatch) -> None:
    price_df = _sample_price_frame(rows=90)

    monkeypatch.setattr(momentum, "TA_LIB_AVAILABLE", False)
    monkeypatch.setattr(momentum, "talib", None)

    assembler = FeatureAssembler(
        feature_toggles={"technical": True},
        horizons=(1,),
        technical_indicator_config={"cci": {"period": 15}},
    )
    result = assembler.build(price_df, news_df=None, sentiment_enabled=False)

    columns = set(result.features.columns)
    assert "CCI_15" in columns
    assert any(col.startswith("Price_to_SMA_10") or col.startswith("Price_to_SMA_100") for col in columns)


def test_accumulation_distribution_matches_manual_calculation() -> None:
    price_df = _sample_price_frame(rows=30)
    inputs = IndicatorInputs(
        high=price_df["High"].set_axis(price_df["Date"]),
        low=price_df["Low"].set_axis(price_df["Date"]),
        close=price_df["Close"].set_axis(price_df["Date"]),
        volume=price_df["Volume"].set_axis(price_df["Date"]),
    )

    adl = accumulation_distribution_line(inputs)["ADL"]

    base = price_df["Close"] - price_df["Low"]
    mfm = (base - (price_df["High"] - price_df["Close"])) / (price_df["High"] - price_df["Low"])
    money_flow_volume = mfm * price_df["Volume"]
    expected = money_flow_volume.cumsum().set_axis(price_df["Date"])
    expected.name = "ADL"

    pd.testing.assert_series_equal(adl, expected)


def test_chaikin_ad_added_via_config_and_matches_reference() -> None:
    price_df = _sample_price_frame(rows=25)
    inputs = IndicatorInputs(
        high=price_df["High"].set_axis(price_df["Date"]),
        low=price_df["Low"].set_axis(price_df["Date"]),
        close=price_df["Close"].set_axis(price_df["Date"]),
        volume=price_df["Volume"].set_axis(price_df["Date"]),
    )

    adl = accumulation_distribution_line(inputs)["ADL"]
    expected = chaikin_accumulation_distribution(inputs, short_period=2, long_period=5)

    indicators = compute_indicators(
        price_df,
        {"adl": {"chaikin_enabled": True, "short_period": 2, "long_period": 5}},
    ).dataframe

    pd.testing.assert_series_equal(indicators["ADL"], adl)
    pd.testing.assert_series_equal(indicators["Chaikin_AD_2_5"], expected.iloc[:, 0])


def test_adl_handles_missing_volume_gracefully() -> None:
    price_df = _sample_price_frame(rows=15).drop(columns=["Volume"])
    indicators = compute_indicators(price_df).dataframe

    assert "ADL" in indicators.columns
    assert indicators["ADL"].isna().all()
