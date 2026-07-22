"""Runtime measurements for validation and benchmark studies."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter
from typing import Any, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class BenchmarkResult:
    """Value and wall-clock runtime returned by a benchmark call."""

    value: Any
    runtime_seconds: float


def benchmark_call(function: Callable[..., T], *args, **kwargs) -> BenchmarkResult:
    """Execute a callable once and measure elapsed wall-clock time."""
    start = perf_counter()
    value = function(*args, **kwargs)
    elapsed = perf_counter() - start
    return BenchmarkResult(value=value, runtime_seconds=elapsed)
