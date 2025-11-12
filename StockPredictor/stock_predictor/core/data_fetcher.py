"""Stub data fetcher used in the test environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import pandas as pd


@dataclass(slots=True)
class DataFetcher:  # pragma: no cover - simple compatibility shim
    """Minimal stub that satisfies the modeling module imports."""

    sources: Mapping[str, str] | None = None

    def fetch(self, symbols: Iterable[str]) -> pd.DataFrame:
        raise NotImplementedError("Data fetching is not available in the lightweight test harness.")
