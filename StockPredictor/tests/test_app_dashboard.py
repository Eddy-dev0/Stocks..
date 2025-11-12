"""Tests for dashboard launch orchestration."""

from __future__ import annotations

import os
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
        api_process.is_alive.side_effect = [True, False]

        with patch("stock_predictor.app.Process", return_value=api_process) as mock_process, patch(
            "stock_predictor.app.run_streamlit_app", side_effect=KeyboardInterrupt
        ) as mock_streamlit, patch(
            "stock_predictor.app.platform.system", return_value="Windows"
        ), patch.dict("stock_predictor.app.os.environ", {}, clear=True):
            result = app.launch_dashboard()

        self.assertEqual(result, 0)
        api_process.start.assert_called_once()
        api_process.terminate.assert_called_once()
        api_process.kill.assert_not_called()
        api_process.join.assert_called_once_with(timeout=10)

        mock_process.assert_called_once()
        _, process_kwargs = mock_process.call_args
        self.assertTrue(process_kwargs["daemon"])
        self.assertEqual(process_kwargs["kwargs"]["host"], "127.0.0.1")
        self.assertEqual(process_kwargs["kwargs"]["port"], 8000)

        env = process_kwargs["kwargs"]["env"]
        self.assertIsInstance(env, dict)
        self.assertEqual(env["PYTHONPATH"].split(os.pathsep)[0], str(PROJECT_ROOT))

        mock_streamlit.assert_called_once()
        _, streamlit_kwargs = mock_streamlit.call_args
        self.assertEqual(streamlit_kwargs["port"], 8501)
        self.assertEqual(streamlit_kwargs["env"], env)


if __name__ == "__main__":  # pragma: no cover - test harness
    import unittest

    unittest.main()
