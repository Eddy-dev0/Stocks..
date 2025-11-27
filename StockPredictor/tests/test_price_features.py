from datetime import datetime, timedelta
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stock_predictor.core.preprocessing import compute_price_features


def _base_price_frame(rows: int = 10) -> pd.DataFrame:
    start = datetime(2024, 1, 1)
    dates = [start + timedelta(days=i) for i in range(rows)]
    close = np.linspace(100.0, 110.0, rows)
    return pd.DataFrame(
        {
            "Date": dates,
            "Open": close + 0.5,
            "High": close + 1.0,
            "Low": close - 1.0,
            "Close": close,
            "Adj Close": close,
            "Volume": np.linspace(1_000_000, 1_200_000, rows),
        }
    )


def test_compute_price_features_adds_volume_signals():
    price_df = _base_price_frame()
    features = compute_price_features(price_df)

    expected_columns = {
        "Volume_OBV",
        "VWMomentum_5",
        "VWMomentum_10",
        "Orderflow_Imbalance_5",
        "Orderflow_Imbalance_20",
        "Tick_Imbalance_5",
        "Tick_Imbalance_20",
    }

    assert expected_columns.issubset(features.columns)
    assert features["Volume_OBV"].iloc[-1] != 0


def test_price_feature_toggles_disable_optional_blocks():
    price_df = _base_price_frame()
    features = compute_price_features(
        price_df,
        feature_toggles={
            "obv": False,
            "volume_weighted_momentum": False,
            "orderflow": False,
            "macro_merge": False,
        },
    )

    blocked_columns = [
        "Volume_OBV",
        "VWMomentum_5",
        "Orderflow_Imbalance_5",
        "Macro_VIX_Close",
    ]

    for column in blocked_columns:
        assert column not in features.columns


def test_macro_series_are_merged_when_available():
    price_df = _base_price_frame()
    macro_df = pd.DataFrame(
        {
            "Date": price_df["Date"],
            "Close_^VIX": np.linspace(15.0, 18.0, len(price_df)),
            "DXY_Close": np.linspace(100.0, 101.0, len(price_df)),
            "Adj Close_^TNX": np.linspace(3.8, 4.0, len(price_df)),
        }
    )

    features = compute_price_features(
        price_df,
        macro_df=macro_df,
        macro_symbols=["^VIX", "DXY", "^TNX"],
    )

    expected = {
        "Macro_VIX_Close",
        "Macro_VIX_Return",
        "Macro_DXY_Close",
        "Macro_DXY_Return",
        "Macro_TNX_Close",
        "Macro_TNX_Return",
    }

    assert expected.issubset(features.columns)
    assert np.isclose(features.loc[1, "Macro_VIX_Close"], macro_df.loc[1, "Close_^VIX"])
