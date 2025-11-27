import math

from stock_predictor.core.config import PredictorConfig
from stock_predictor.core.modeling.main import StockPredictorAI
from stock_predictor.providers.fundamentals import normalize_fundamentals_payload


def test_normalize_fundamentals_payload_populates_sector_baseline():
    payload = {
        "trailingPE": 25.0,
        "earningsQuarterlyGrowth": 0.12,
        "debtToEquity": 0.8,
        "trailingEps": 5.0,
        "sector": "Technology",
    }

    snapshot = normalize_fundamentals_payload("TEST", payload)

    assert snapshot.pe_ratio == 25.0
    assert snapshot.earnings_growth == 0.12
    assert snapshot.debt_to_equity == 0.8
    assert snapshot.eps == 5.0
    assert snapshot.sector_pe == 24.0
    assert math.isclose(snapshot.sector_growth or 0.0, 0.10, rel_tol=1e-6)


def test_confidence_penalty_reduces_for_extreme_valuations():
    config = PredictorConfig(ticker="TEST")
    predictor = StockPredictorAI(config)
    predictor.metadata["latest_close"] = 100.0
    predictor.metadata["fundamental_snapshot"] = {
        "pe_ratio": 30.0,
        "sector_pe": 15.0,
        "debt_to_equity": 2.5,
        "sector_debt_to_equity": 1.0,
        "earnings_growth": 0.02,
        "sector_growth": 0.08,
        "eps": 5.0,
    }

    adjusted, fundamentals = predictor._apply_fundamental_penalty(0.85, 0.5)

    assert adjusted < 0.85
    assert fundamentals["valuation_gap"] > 1.5
    assert fundamentals.get("leverage_flag") == 1.0
