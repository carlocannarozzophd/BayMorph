"""Runtime measurements for validation and benchmark studies."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter
from typing import Any, TypeVar

import numpy as np

from baymorph.fitting.problem import InferenceProblem
from baymorph.validation.run import ValidationRun

T = TypeVar("T")


@dataclass(frozen=True)
class BenchmarkResult:
    """Value and wall-clock runtime returned by a benchmark call."""

    value: Any
    runtime_seconds: float


class TimedInferenceProblem(InferenceProblem):
    """Instrument an inference problem without changing its mathematics.

    Notes
    -----
    Timing counters are intended for serial benchmark runs. They are not
    process-safe and must not be used for multiprocessing or MPI call
    accounting.
    """

    def __init__(
        self,
        problem: InferenceProblem,
    ) -> None:
        if not isinstance(problem, InferenceProblem):
            raise TypeError(
                "problem must be an InferenceProblem instance."
            )

        self.problem = problem
        self.reset()

    def __getattr__(
        self,
        name: str,
    ):
        """Delegate problem-specific attributes to the wrapped problem."""
        return getattr(
            self.problem,
            name,
        )

    @property
    def ndim(self) -> int:
        """Return the wrapped problem dimensionality."""
        return self.problem.ndim

    @property
    def parameter_names(self) -> list[str]:
        """Return wrapped parameter names in sampling order."""
        return self.problem.parameter_names

    def prior_transform(
        self,
        cube: np.ndarray,
    ) -> np.ndarray:
        """Evaluate and time one prior transformation."""
        start = perf_counter()

        try:
            return self.problem.prior_transform(
                cube
            )
        finally:
            self.prior_transform_calls += 1
            self.prior_transform_seconds += (
                perf_counter() - start
            )

    def log_likelihood(
        self,
        theta: np.ndarray,
    ) -> float:
        """Evaluate and time one likelihood call."""
        start = perf_counter()

        try:
            return float(
                self.problem.log_likelihood(
                    theta
                )
            )
        finally:
            self.likelihood_calls += 1
            self.likelihood_seconds += (
                perf_counter() - start
            )

    def reset(self) -> None:
        """Reset all timing and call counters."""
        self.likelihood_calls = 0
        self.likelihood_seconds = 0.0
        self.prior_transform_calls = 0
        self.prior_transform_seconds = 0.0

    @property
    def metrics(
        self,
    ) -> dict[str, float | int]:
        """Return serializable timing diagnostics."""
        mean_likelihood_seconds = (
            self.likelihood_seconds
            / self.likelihood_calls
            if self.likelihood_calls > 0
            else 0.0
        )

        return {
            "likelihood_calls": self.likelihood_calls,
            "likelihood_seconds": self.likelihood_seconds,
            "mean_likelihood_seconds": mean_likelihood_seconds,
            "prior_transform_calls": self.prior_transform_calls,
            "prior_transform_seconds": self.prior_transform_seconds,
        }


def benchmark_call(
    function: Callable[..., T],
    *args,
    **kwargs,
) -> BenchmarkResult:
    """Execute a callable once and measure elapsed wall-clock time."""
    start = perf_counter()

    value = function(
        *args,
        **kwargs,
    )

    elapsed = (
        perf_counter()
        - start
    )

    return BenchmarkResult(
        value=value,
        runtime_seconds=elapsed,
    )


@dataclass(frozen=True)
class ComparisonReport:
    """Comparison between two runs of the same validation case."""

    problem_name: str
    reference_backend: str
    candidate_backend: str
    logz_difference: float
    posterior_mean_distance: float
    likelihood_call_ratio: float
    runtime_ratio: float


def compare_results(
    reference: ValidationRun,
    candidate: ValidationRun,
) -> ComparisonReport:
    """Compare two production-backend runs of the same validation case."""
    if reference.case_name != candidate.case_name:
        raise ValueError(
            "Validation runs must refer to the same problem."
        )

    if (
        reference.result.samples.shape[1]
        != candidate.result.samples.shape[1]
    ):
        raise ValueError(
            "Validation runs must have matching dimensionality."
        )

    likelihood_call_ratio = _safe_ratio(
        _required_likelihood_calls(
            candidate
        ),
        _required_likelihood_calls(
            reference
        ),
    )
    runtime_ratio = _safe_ratio(
        candidate.runtime_seconds,
        reference.runtime_seconds,
    )

    return ComparisonReport(
        problem_name=reference.case_name,
        reference_backend=reference.backend_name,
        candidate_backend=candidate.backend_name,
        logz_difference=float(
            _required_logz(
                candidate
            )
            - _required_logz(
                reference
            )
        ),
        posterior_mean_distance=float(
            np.linalg.norm(
                _posterior_mean(
                    candidate
                )
                - _posterior_mean(
                    reference
                )
            )
        ),
        likelihood_call_ratio=likelihood_call_ratio,
        runtime_ratio=runtime_ratio,
    )


def _posterior_mean(
    run: ValidationRun,
) -> np.ndarray:
    """Return the weighted posterior mean."""
    return np.average(
        run.result.samples,
        axis=0,
        weights=run.result.weights,
    )


def _required_logz(
    run: ValidationRun,
) -> float:
    """Return evidence, requiring it to exist."""
    value = run.result.logz

    if value is None:
        raise ValueError(
            "Validation comparison requires Bayesian evidence."
        )

    return float(
        value
    )


def _required_likelihood_calls(
    run: ValidationRun,
) -> int:
    """Return likelihood calls, requiring the diagnostic to exist."""
    calls = run.likelihood_calls

    if calls is None:
        raise ValueError(
            "Validation comparison requires likelihood-call diagnostics."
        )

    return calls


def _safe_ratio(
    numerator: float,
    denominator: float,
) -> float:
    """Return a ratio, preserving an infinite zero-denominator case."""
    if denominator == 0.0:
        if numerator == 0.0:
            return 1.0

        return float(
            np.inf
        )

    return float(
        numerator
        / denominator
    )
