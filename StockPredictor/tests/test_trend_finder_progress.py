from datetime import date
from pathlib import Path
import sys

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stock_predictor.core.config import PredictorConfig
from stock_predictor.core.trend_finder import TrendFinder
from stock_predictor.core.pipeline import MarketDataETL


class SimpleAI:
    """Deterministic AI stub for exercising TrendFinder progress callbacks."""

    def __init__(self, config: PredictorConfig, *, horizon: int | None = None) -> None:
        self.config = config
        self.horizon = horizon
        self.metadata: dict[str, object] = {}

    def prepare_features(self):
        features = pd.DataFrame(
            [
                {
                    "tech_indicator": 0.5,
                    "fundamental_metric": 0.25,
                    "sentiment_signal": 0.1,
                }
            ]
        )
        self.metadata = {
            "latest_features": features,
            "feature_categories": {
                "tech_indicator": "technical indicator",
                "fundamental_metric": "fundamental metric",
                "sentiment_signal": "sentiment gauge",
            },
        }
        return features, {}, {}

    def predict(self, horizon: int | None = None):
        return {"expected_change_pct": 0.02}


def test_trend_finder_reports_progress(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Every ticker processed should trigger the provided progress callback."""

    monkeypatch.setattr(MarketDataETL, "refresh_prices", lambda *args, **kwargs: None)

    database_path = tmp_path / "prices.db"
    config = PredictorConfig(
        ticker="INIT",
        start_date=date(2024, 1, 1),
        interval="1d",
        data_dir=tmp_path,
        database_url=f"sqlite:///{database_path}",
        sentiment=False,
        prediction_horizons=(5,),
    )
    config.ensure_directories()

    trend = TrendFinder(config, ai_factory=SimpleAI)

    universe = ["AAA", "BBB", "CCC"]
    progress: list[tuple[int, int, str]] = []

    def callback(current: int, total: int, status: str) -> None:
        progress.append((current, total, status))

    results = trend.scan(universe=universe, horizon=5, limit=5, progress_callback=callback)

    assert len(progress) == len(universe)
    assert [step for step, _, _ in progress] == list(range(1, len(universe) + 1))
    assert all(total == len(universe) for _, total, _ in progress)
    assert all(universe[idx] in status for idx, (_, _, status) in enumerate(progress))
    assert results, "TrendFinder should return insights when stubs supply data."
