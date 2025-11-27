"""Lightweight checks for rendering helpers in the Streamlit frontend."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ui.frontend import app


def test_beta_guidance_summary_formats_values() -> None:
    beta_block = {
        "sp500": {"label": "S&P 500", "value": 1.65, "window": 21, "risk_level": "high"},
        "vix": {"label": "VIX", "value": 0.55, "window": 63, "risk_level": "defensive"},
    }

    summaries = app._summarise_beta_guidance(beta_block)

    assert "S&P 500 beta 1.65 (21-day window) – high volatility" in summaries
    assert "VIX beta 0.55 (63-day window) – defensive / low sensitivity" in summaries
