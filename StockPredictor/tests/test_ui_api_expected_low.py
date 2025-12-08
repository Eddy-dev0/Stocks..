from __future__ import annotations

from pathlib import Path
import sys

import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ui.api.app import create_app


class _DummyFetcher:
    def __init__(self, last_price: float):
        self.last_price = last_price

    def fetch_live_price(self, *, force: bool = True):
        return self.last_price, None


class _DummyPipeline:
    def __init__(self, last_price: float):
        self.fetcher = _DummyFetcher(last_price)

    def live_price_snapshot(
        self,
        *,
        horizon=None,
    ):
        base_price = self.fetcher.last_price
        predicted_close = base_price * 1.05
        expected_low = base_price * 0.8
        stop_loss = expected_low
        prob_up = 0.6
        prob_down = 0.4

        return {
            "ticker": "ABC",
            "market_time": None,
            "last_price": base_price,
            "predicted_close": predicted_close,
            "expected_change_pct": 0.05,
            "expected_low": expected_low,
            "stop_loss": stop_loss,
            "probabilities": {"up": prob_up, "down": prob_down},
        }


class _DummyApplication:
    def __init__(self, last_price: float):
        self.pipeline = _DummyPipeline(last_price)
        self.config = None


def _build_app(monkeypatch, last_price: float = 100.0) -> TestClient:
    app = create_app()
    app.router.on_startup.clear()
    app.router.on_shutdown.clear()

    def _fake_from_environment(*_args, **_kwargs):
        return _DummyApplication(last_price)

    monkeypatch.setattr(
        "stock_predictor.app.StockPredictorApplication.from_environment",
        _fake_from_environment,
    )

    return TestClient(app)


def test_expected_low_never_exceeds_last_price(monkeypatch):
    client = _build_app(monkeypatch, last_price=100.0)

    response = client.post(
        "/live-price/abc",
        json={},
    )
    assert response.status_code == 200
    payload = response.json()["price"]
    assert payload["expected_low"] == pytest.approx(80.0)
    assert payload["expected_low"] <= payload["last_price"]
