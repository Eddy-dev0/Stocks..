from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Iterable, TypeVar

T = TypeVar("T")
R = TypeVar("R")


def parallel_map(items: Iterable[T], fn: Callable[[T], R], max_workers: int = 6) -> list[R]:
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        return list(pool.map(fn, items))
