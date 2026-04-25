from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any


@dataclass
class BacktestCache:
    ttl_minutes: int = 120
    _store: dict[str, tuple[datetime, Any]] = field(default_factory=dict)

    def get(self, key: str) -> Any | None:
        item = self._store.get(key)
        if not item:
            return None
        ts, value = item
        if datetime.utcnow() - ts > timedelta(minutes=self.ttl_minutes):
            self._store.pop(key, None)
            return None
        return value

    def set(self, key: str, value: Any) -> None:
        self._store[key] = (datetime.utcnow(), value)
