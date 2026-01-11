from __future__ import annotations

from datetime import date

import pandas as pd

from stock_predictor.core.config import PredictorConfig
from stock_predictor.core.pipeline import MarketDataETL


def test_apply_asof_limit_trims_future_rows() -> None:
    config = PredictorConfig(ticker="AAPL")
    etl = MarketDataETL(config)

    frame = pd.DataFrame(
        {
            "Date": pd.to_datetime(
                ["2024-04-29", "2024-05-01", "2024-05-03"]
            ),
            "Close": [100.0, 101.0, 102.0],
        }
    )

    trimmed = etl._apply_asof_limit(frame, date(2024, 5, 1))

    assert trimmed["Date"].max().date() <= date(2024, 5, 1)
