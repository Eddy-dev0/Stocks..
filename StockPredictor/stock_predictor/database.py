"""SQLite-backed experiment tracking for stock predictor runs."""

from __future__ import annotations

import json
import sqlite3
import time
from typing import Any, Dict

from .config import PredictorConfig

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    model_type TEXT NOT NULL,
    target TEXT NOT NULL,
    run_type TEXT NOT NULL,
    started_at REAL NOT NULL,
    finished_at REAL NOT NULL,
    parameters TEXT,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL,
    name TEXT NOT NULL,
    value REAL,
    context TEXT,
    FOREIGN KEY(run_id) REFERENCES runs(id)
);
"""


class ExperimentTracker:
    """Persist model training and evaluation metadata in SQLite."""

    def __init__(self, config: PredictorConfig) -> None:
        self.config = config
        self.path = config.database_path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with sqlite3.connect(self.path) as conn:
            conn.executescript(SCHEMA)

    def log_run(
        self,
        *,
        target: str,
        run_type: str,
        parameters: Dict[str, Any] | None,
        metrics: Dict[str, float] | None,
        context: Dict[str, Any] | None = None,
        notes: str | None = None,
    ) -> int:
        started = time.time()
        finished = started
        params_json = json.dumps(parameters or {}, default=str)
        notes_text = notes or ""
        context_json = json.dumps(context or {}, default=str)

        with sqlite3.connect(self.path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO runs (ticker, model_type, target, run_type, started_at, finished_at, parameters, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    self.config.ticker,
                    self.config.model_type,
                    target,
                    run_type,
                    started,
                    finished,
                    params_json,
                    notes_text,
                ),
            )
            run_id = int(cursor.lastrowid)
            if metrics:
                for name, value in metrics.items():
                    cursor.execute(
                        """
                        INSERT INTO metrics (run_id, name, value, context)
                        VALUES (?, ?, ?, ?)
                        """,
                        (run_id, name, float(value), context_json),
                    )
            conn.commit()
        return run_id

    def list_runs(self) -> list[dict[str, Any]]:
        with sqlite3.connect(self.path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT id, ticker, model_type, target, run_type, started_at, finished_at FROM runs ORDER BY id DESC"
            ).fetchall()
            return [dict(row) for row in rows]

    def get_metrics(self, run_id: int) -> Dict[str, float]:
        with sqlite3.connect(self.path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT name, value FROM metrics WHERE run_id = ?",
                (run_id,),
            ).fetchall()
            return {row["name"]: row["value"] for row in rows}
