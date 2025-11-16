from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stock_predictor.core import PredictorConfig, StockPredictorAI


def _price_frame(rows: int = 80) -> pd.DataFrame:
    dates = pd.date_range("2022-01-01", periods=rows, freq="B")
    base = np.linspace(100.0, 120.0, rows)
    return pd.DataFrame(
        {
            "Date": dates,
            "Open": base + 0.5,
            "High": base + 1.0,
            "Low": base - 1.0,
            "Close": base,
            "Adj Close": base,
            "Volume": np.linspace(1_000_000, 1_200_000, rows),
        }
    )


def _macro_indicator_rows(dates: pd.Series) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    start = 4000.0
    for idx, date in enumerate(dates):
        records.append(
            {
                "Date": date,
                "Indicator": "macro:^GSPC",
                "Value": start + idx,
                "Category": "macro",
            }
        )
    return pd.DataFrame(records)


class _MacroStaticFetcher:
    def __init__(self, price_df: pd.DataFrame, macro_df: pd.DataFrame) -> None:
        self._price_df = price_df
        self._macro_df = macro_df

    def fetch_price_data(self, force: bool = False) -> pd.DataFrame:
        return self._price_df.copy()

    def fetch_news_data(self, force: bool = False) -> pd.DataFrame:
        return pd.DataFrame()

    def fetch_indicator_data(self, category: str | None = None) -> pd.DataFrame:
        if category == "macro":
            return self._macro_df.copy()
        return pd.DataFrame()

    def fetch_fundamentals(self) -> pd.DataFrame:
        return pd.DataFrame()

    def get_data_sources(self) -> list[str]:
        return ["dummy"]


def test_macro_features_include_benchmarks_and_betas(tmp_path: Path) -> None:
    price_df = _price_frame(90)
    macro_rows = _macro_indicator_rows(price_df["Date"])
    fetcher = _MacroStaticFetcher(price_df, macro_rows)

    config = PredictorConfig(
        ticker="TEST",
        data_dir=tmp_path,
        models_dir=tmp_path / "models",
        sentiment=False,
        prediction_targets=("direction",),
        prediction_horizons=(1,),
    )
    config.ensure_directories()

    pipeline = StockPredictorAI(config)
    pipeline.fetcher = fetcher

    features, *_ = pipeline.prepare_features(price_df=price_df)

    expected_sp500 = {
        "SP500_Return",
        "SP500_RollingCorr_21",
        "SP500_Relative_Return",
        "SP500_Price_Ratio",
    }
    for column in expected_sp500:
        assert column in features.columns

    for window in (21, 63, 126):
        assert f"Beta_SP500_{window}" in features.columns
