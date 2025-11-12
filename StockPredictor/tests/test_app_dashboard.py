"""Tests for dashboard launch orchestration."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stock_predictor.app import StockPredictorApplication


class DashboardLaunchTests(TestCase):
    """Ensure the dashboard launch handles platform differences."""

    def test_windows_dashboard_shutdown_uses_terminate(self) -> None:
        app = StockPredictorApplication.__new__(StockPredictorApplication)
        app.config = SimpleNamespace(ticker="TEST")

        api_process = MagicMock()
        api_process.wait.return_value = None

        with patch("stock_predictor.app.Path.exists", return_value=True), patch(
            "stock_predictor.app.subprocess.Popen", return_value=api_process
        ), patch("stock_predictor.app.subprocess.run", side_effect=KeyboardInterrupt), patch(
            "stock_predictor.app.platform.system", return_value="Windows"
        ):
            result = app.launch_dashboard()

        self.assertEqual(result, 0)
        api_process.terminate.assert_called_once()
        api_process.send_signal.assert_not_called()
        api_process.wait.assert_called_once_with(timeout=10)
        api_process.kill.assert_not_called()


if __name__ == "__main__":  # pragma: no cover - test harness
    import unittest

    unittest.main()
