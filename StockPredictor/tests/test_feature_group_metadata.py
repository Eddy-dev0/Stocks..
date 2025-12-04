from pathlib import Path
import sys
import types

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stock_predictor.core import PredictorConfig, StockPredictorAI
from stock_predictor.core.features import FeatureToggles


class _StubFetcher:
    def __init__(self, price_df: pd.DataFrame) -> None:
        self._price_df = price_df

    def fetch_price_data(self, force: bool = False) -> pd.DataFrame:  # pragma: no cover - passthrough
        return self._price_df.copy()

    def fetch_news_data(self, force: bool = False) -> pd.DataFrame:  # pragma: no cover - passthrough
        return pd.DataFrame()

    def fetch_fundamentals(self, force: bool = False) -> pd.DataFrame:  # pragma: no cover - passthrough
        return pd.DataFrame()

    def get_data_sources(self) -> list[str]:  # pragma: no cover - passthrough
        return ["stub"]

    def refresh_all(self, force: bool = False) -> dict[str, str]:  # pragma: no cover - passthrough
        return {}


class _ConstantModel:
    """Simple regression model used to avoid training in tests."""

    def __init__(self) -> None:
        self.named_steps: dict[str, object] = {}

    def predict(self, _features):  # pragma: no cover - deterministic
        return [0.0]


def _synthetic_price_frame() -> pd.DataFrame:
    dates = pd.date_range("2024-01-02", periods=5, freq="B")
    close = pd.Series(range(100, 105), index=dates)
    return pd.DataFrame({"Date": dates, "Close": close.values})


def _install_feature_metadata_stub(
    predictor: StockPredictorAI, price_df: pd.DataFrame, executed_groups: set[str]
) -> None:
    def _prepare(self, price_df=None, news_df=None):
        working = price_df.copy() if price_df is not None else self.fetcher.fetch_price_data()
        working["Date"] = pd.to_datetime(working["Date"], errors="coerce")
        working = working.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
        latest_row = working.iloc[-1]
        latest_date = pd.to_datetime(latest_row["Date"])
        latest_close = float(latest_row.get("Close", 0.0))
        features = pd.DataFrame({"bias": [1.0] * len(working)})
        latest_features = features.iloc[[-1]].copy()
        horizons = tuple(self.config.prediction_horizons)
        target_dates = {
            int(h): latest_date + pd.tseries.offsets.BDay(int(h)) for h in horizons
        }

        executed_list = [str(name) for name in sorted(executed_groups)]
        group_metadata: dict[str, dict[str, object]] = {}
        for name, enabled in self.config.feature_toggles.items():
            label = str(name)
            executed = label in executed_list
            group_metadata[label] = {
                "configured": bool(enabled),
                "executed": executed,
                "dependencies": [],
                "implemented": True,
                "description": "",
                "columns": ["bias"] if executed else [],
                "categories": [],
                "status": "executed" if executed else ("disabled" if not enabled else "skipped_no_data"),
            }

        metadata = {
            "latest_features": latest_features,
            "raw_feature_columns": list(latest_features.columns),
            "feature_columns": list(latest_features.columns),
            "latest_close": latest_close,
            "latest_date": latest_date,
            "horizons": horizons,
            "target_dates": target_dates,
            "feature_groups": group_metadata,
            "executed_feature_groups": executed_list,
            "feature_toggles": self.config.feature_toggles.asdict(),
        }
        self.metadata = metadata
        self.preprocessor_templates = {}
        return features, {}, {}

    predictor.prepare_features = types.MethodType(_prepare, predictor)


def test_prediction_respects_feature_toggle_configuration(tmp_path: Path) -> None:
    price_df = _synthetic_price_frame()
    toggles = FeatureToggles.from_any(
        {
            "technical": True,
            "sentiment": False,
            "macro": False,
            "fundamental": False,
            "volume_liquidity": False,
            "elliott": False,
        }
    )
    config = PredictorConfig(
        ticker="TGL",
        sentiment=False,
        prediction_targets=("close",),
        prediction_horizons=(1,),
        data_dir=tmp_path,
        models_dir=tmp_path / "models",
        model_params={"global": {}},
        feature_toggles=toggles,
    )
    config.ensure_directories()

    predictor = StockPredictorAI(config)
    predictor.fetcher = _StubFetcher(price_df)
    predictor.models[("close", 1)] = _ConstantModel()

    _install_feature_metadata_stub(predictor, price_df, executed_groups={"technical"})

    result = predictor.predict(targets=("close",), horizon=1)

    assert result.get("executed_feature_groups") == ["technical"]
    feature_group_meta = result.get("feature_groups", {})
    assert feature_group_meta.get("sentiment", {}).get("configured") is False
    warnings = result.get("warnings") or []
    assert not any("feature group" in message.lower() for message in warnings)


def test_prediction_warns_on_feature_toggle_mismatch(tmp_path: Path) -> None:
    price_df = _synthetic_price_frame()
    toggles = FeatureToggles.from_any(
        {
            "technical": False,
            "macro": True,
            "sentiment": False,
            "fundamental": False,
            "volume_liquidity": False,
            "elliott": False,
        }
    )
    config = PredictorConfig(
        ticker="MIS",
        sentiment=False,
        prediction_targets=("close",),
        prediction_horizons=(1,),
        data_dir=tmp_path,
        models_dir=tmp_path / "models",
        model_params={"global": {}},
        feature_toggles=toggles,
    )
    config.ensure_directories()

    predictor = StockPredictorAI(config)
    predictor.fetcher = _StubFetcher(price_df)
    predictor.models[("close", 1)] = _ConstantModel()

    _install_feature_metadata_stub(predictor, price_df, executed_groups={"technical"})

    result = predictor.predict(targets=("close",), horizon=1)

    warnings = result.get("warnings") or []
    assert any("disabled via feature_toggles" in message for message in warnings)
    assert any("enabled but not reported" in message for message in warnings)
    assert "technical" in (result.get("executed_feature_groups") or [])
