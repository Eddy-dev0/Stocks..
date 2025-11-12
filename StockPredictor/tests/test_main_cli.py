"""Tests for the unified CLI launcher."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest import TestCase
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import main as cli_main  # pylint: disable=wrong-import-position


class ParseArgsTests(TestCase):
    """Verify the CLI argument parsing behaviour."""

    def test_defaults(self) -> None:
        args = cli_main.parse_args([])
        self.assertEqual(args.mode, "tk")
        self.assertEqual(args.host, os.getenv("STOCK_PREDICTOR_API_HOST", "127.0.0.1"))
        self.assertEqual(args.port, int(os.getenv("STOCK_PREDICTOR_API_PORT", "8000")))
        self.assertEqual(
            args.dash_port,
            int(os.getenv("STOCK_PREDICTOR_DASHBOARD_PORT", "8501")),
        )

    def test_custom_values(self) -> None:
        args = cli_main.parse_args(
            [
                "--mode",
                "api",
                "--host",
                "0.0.0.0",
                "--port",
                "9000",
                "--dash-port",
                "9001",
                "--log-level",
                "debug",
                "--no-train",
                "--no-refresh",
            ]
        )
        self.assertEqual(args.mode, "api")
        self.assertEqual(args.host, "0.0.0.0")
        self.assertEqual(args.port, 9000)
        self.assertEqual(args.dash_port, 9001)
        self.assertEqual(args.log_level.lower(), "debug")
        self.assertTrue(args.no_train)
        self.assertTrue(args.no_refresh)


class MainDispatchTests(TestCase):
    """Ensure the correct launch routine is invoked for each mode."""

    def setUp(self) -> None:
        patcher = patch("main._configure_logging")
        self.addCleanup(patcher.stop)
        patcher.start()

        deps_patcher = patch("main._ensure_dependencies", return_value=True)
        self.addCleanup(deps_patcher.stop)
        deps_patcher.start()

    def test_tk_mode_dispatch(self) -> None:
        with patch("main.run_tkinter_app") as mock_tk:
            exit_code = cli_main.main(["--mode", "tk"])

        self.assertEqual(exit_code, 0)
        mock_tk.assert_called_once()

    def test_api_mode_dispatch(self) -> None:
        with patch("main.run_api") as mock_api:
            exit_code = cli_main.main(["--mode", "api", "--host", "0.0.0.0", "--port", "8100"])

        self.assertEqual(exit_code, 0)
        mock_api.assert_called_once_with(host="0.0.0.0", port=8100)

    def test_dash_mode_dispatch(self) -> None:
        with patch("main.run_streamlit_app", return_value=0) as mock_dash:
            exit_code = cli_main.main(["--mode", "dash", "--dash-port", "8600"])

        self.assertEqual(exit_code, 0)
        _, kwargs = mock_dash.call_args
        self.assertEqual(kwargs["port"], 8600)

    def test_full_mode_dispatch(self) -> None:
        fake_process = MagicMock()
        fake_process.poll.return_value = None
        fake_process.wait.return_value = 0

        with patch("main._spawn_streamlit_process", return_value=fake_process) as mock_spawn, patch(
            "main._start_api_thread"
        ) as mock_thread, patch("main.run_tkinter_app") as mock_tk:
            exit_code = cli_main.main(["--mode", "full"])

        self.assertEqual(exit_code, 0)
        mock_thread.assert_called_once()
        mock_spawn.assert_called_once()
        mock_tk.assert_called_once()
        fake_process.terminate.assert_called_once()


if __name__ == "__main__":  # pragma: no cover - test harness
    import unittest

    unittest.main()
