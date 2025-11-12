"""Tests for configuration helpers."""

from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stock_predictor.core.config import PredictorConfig


def make_config(**overrides):
    params = {"ticker": "AAPL"}
    params.update(overrides)
    return PredictorConfig(**params)


def test_resolve_horizon_accepts_string_code_for_day():
    config = make_config()
    assert config.resolve_horizon("1d") == 1


def test_resolve_horizon_accepts_string_code_for_week():
    config = make_config()
    assert config.resolve_horizon("1w") == 5


def test_resolve_horizon_accepts_string_code_for_month_multiplier():
    config = make_config()
    assert config.resolve_horizon("3m") == 63


def test_resolve_horizon_rejects_unknown_code():
    config = make_config()
    with pytest.raises(ValueError):
        config.resolve_horizon("2x")
