"""Small infrastructure helpers."""

from __future__ import annotations

import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass


@dataclass
class Timing:
    """Result of :func:`timer`; ``seconds`` is filled in when the ``with`` block exits."""

    label: str
    seconds: float = 0.0


@contextmanager
def timer(label: str = "block") -> Generator[Timing, None, None]:
    """Measure the wall-clock time of a ``with`` block, even if it raises.

    Example:
        >>> with timer("fit") as t:
        ...     _ = sum(range(10))
        >>> t.seconds >= 0
        True
    """
    result = Timing(label)
    start = time.perf_counter()
    try:
        yield result
    finally:
        result.seconds = time.perf_counter() - start
