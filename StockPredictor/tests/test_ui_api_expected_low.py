from __future__ import annotations

from pathlib import Path
import sys

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

    positive_response = client.post(
        "/live-price/abc",
        json={"expected_low_pct_model": 0.1},
    )
    assert positive_response.status_code == 200
    positive_payload = positive_response.json()["price"]
    assert positive_payload["expected_low"] == 90.0
    assert positive_payload["expected_low"] <= positive_payload["last_price"]

    negative_response = client.post(
        "/live-price/abc",
        json={"expected_low_pct_model": -0.2},
    )
    assert negative_response.status_code == 200
    negative_payload = negative_response.json()["price"]
    assert negative_payload["expected_low"] == negative_payload["last_price"]
    assert negative_payload["expected_low"] <= negative_payload["last_price"]
