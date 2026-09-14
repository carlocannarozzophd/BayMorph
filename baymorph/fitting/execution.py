"""Backend-neutral sampler execution configuration."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any


_AUTO_PROCESS_LIMIT = 10
_AUTO_RESERVED_CPUS = 2
_WORKER_THREAD_LIMITER = None

_THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


@dataclass(frozen=True)
class SamplerExecution:
    """Resolved computational execution settings for one sampler run."""

    requested_processes: str | int
    processes: int
    progress: bool

    def as_dict(self) -> dict[str, Any]:
        """Return execution settings suitable for provenance."""
        return {
            "requested_processes": self.requested_processes,
            "processes": self.processes,
            "progress": self.progress,
        }


def available_cpu_count() -> int:
    """Return the CPU count visible to the current Python process."""
    count = os.cpu_count()

    if count is None:
        return 1

    return max(
        1,
        int(count),
    )


def resolve_process_count(
    value: str | int,
) -> int:
    """Resolve one user-facing process-count specification."""
    if isinstance(
        value,
        bool,
    ):
        raise TypeError(
            "Sampler process count must be 'auto' or a positive integer."
        )

    if isinstance(
        value,
        int,
    ):
        if value < 1:
            raise ValueError(
                "Sampler process count must be at least 1."
            )

        return value

    normalized = str(
        value
    ).strip().lower()

    if normalized != "auto":
        raise ValueError(
            "Sampler process count must be 'auto' or a positive integer."
        )

    available = available_cpu_count()

    if available <= _AUTO_RESERVED_CPUS:
        return available

    return min(
        _AUTO_PROCESS_LIMIT,
        available - _AUTO_RESERVED_CPUS,
    )


def resolve_sampler_execution(
    config: Mapping[str, Any] | None = None,
) -> SamplerExecution:
    """Validate and resolve backend-neutral sampler execution settings."""
    raw = (
        {}
        if config is None
        else dict(config)
    )

    unknown = sorted(
        set(raw)
        - {
            "processes",
            "progress",
        }
    )

    if unknown:
        names = ", ".join(
            unknown
        )
        raise ValueError(
            "Unknown sampler execution option(s): "
            f"{names}."
        )

    requested_processes = raw.get(
        "processes",
        "auto",
    )

    progress = raw.get(
        "progress",
        True,
    )

    if not isinstance(
        progress,
        bool,
    ):
        raise TypeError(
            "Sampler execution progress must be a boolean."
        )

    return SamplerExecution(
        requested_processes=requested_processes,
        processes=resolve_process_count(
            requested_processes
        ),
        progress=progress,
    )


def _configure_worker_threads() -> None:
    """Restrict numerical-library threading inside one worker process."""
    global _WORKER_THREAD_LIMITER

    for name in _THREAD_ENVIRONMENT:
        os.environ[name] = "1"

    from threadpoolctl import threadpool_limits

    _WORKER_THREAD_LIMITER = threadpool_limits(
        limits=1
    )


def _configure_sampler_worker(
    initializer: Callable[..., None],
    initargs: tuple[Any, ...],
) -> None:
    """Configure worker threads and run a sampler-specific initializer."""
    _configure_worker_threads()
    initializer(*initargs)


@contextmanager
def sampler_pool(
    processes: int,
    *,
    initializer: Callable[..., None] | None = None,
    initargs: tuple[Any, ...] = (),
) -> Iterator[Any | None]:
    """Provide one cleanly managed local sampler process pool."""
    processes = resolve_process_count(
        processes
    )

    if initializer is None and initargs:
        raise ValueError(
            "sampler_pool initargs require an initializer."
        )

    if initializer is not None and not callable(initializer):
        raise TypeError(
            "sampler_pool initializer must be callable."
        )

    if processes == 1:
        yield None
        return

    try:
        import multiprocess as mp
    except ImportError as exc:
        raise ImportError(
            "Parallel BayMorph inference requires the multiprocess package."
        ) from exc

    if initializer is None:
        pool = mp.Pool(
            processes=processes,
            initializer=_configure_worker_threads,
        )
    else:
        pool = mp.Pool(
            processes=processes,
            initializer=_configure_sampler_worker,
            initargs=(
                initializer,
                tuple(initargs),
            ),
        )

    try:
        yield pool
    except BaseException:
        pool.terminate()
        pool.join()
        raise
    else:
        pool.close()
        pool.join()


__all__ = [
    "SamplerExecution",
    "available_cpu_count",
    "resolve_process_count",
    "resolve_sampler_execution",
    "sampler_pool",
]
