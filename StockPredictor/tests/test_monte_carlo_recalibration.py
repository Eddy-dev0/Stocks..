from __future__ import annotations

import math
from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stock_predictor.core.modeling.main import _recalibrated_log_drift  # noqa: E402  pylint: disable=wrong-import-position


def test_monte_carlo_recalibration_uses_log_return_drift() -> None:
    drift = _recalibrated_log_drift(105.0, 100.0, 10)

    assert drift is not None
    assert drift == pytest.approx(math.log(1.05) / 10.0)
    assert drift < 0.01  # Regression: avoid multi-unit drift values for modest moves


def test_monte_carlo_recalibration_guards_invalid_prices() -> None:
    drift = _recalibrated_log_drift(-5.0, 100.0, 10)

    assert drift is None
