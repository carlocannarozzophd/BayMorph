"""Tests for backend-neutral inference timing instrumentation."""

from __future__ import annotations

import numpy as np

from baymorph.fitting.parameters import (
    Parameter,
    ParameterSet,
)
from baymorph.fitting.problem import InferenceProblem
from baymorph.validation.benchmark import TimedInferenceProblem


class InstrumentedProblem(InferenceProblem):
    """Minimal inference problem with delegated attributes."""

    def __init__(self) -> None:
        self.parameters = ParameterSet(
            [
                Parameter(
                    "x",
                    0.0,
                    1.0,
                )
            ]
        )

    @property
    def ndim(self) -> int:
        return 1

    @property
    def parameter_names(self) -> list[str]:
        return ["x"]

    def prior_transform(
        self,
        cube: np.ndarray,
    ) -> np.ndarray:
        return np.asarray(
            cube,
            dtype=float,
        )

    def log_likelihood(
        self,
        theta: np.ndarray,
    ) -> float:
        return float(
            -0.5
            * np.asarray(
                theta,
                dtype=float,
            )[0] ** 2
        )


def test_timed_problem_preserves_problem_contract():
    base = InstrumentedProblem()
    problem = TimedInferenceProblem(
        base
    )

    transformed = problem.prior_transform(
        np.asarray(
            [0.25]
        )
    )

    logl = problem.log_likelihood(
        transformed
    )

    assert transformed == np.asarray(
        [0.25]
    )

    assert logl == -0.03125

    assert problem.ndim == 1
    assert problem.parameter_names == ["x"]

    assert (
        problem.parameters
        is base.parameters
    )


def test_timed_problem_counts_calls_and_runtime():
    problem = TimedInferenceProblem(
        InstrumentedProblem()
    )

    problem.prior_transform(
        np.asarray(
            [0.5]
        )
    )

    problem.log_likelihood(
        np.asarray(
            [0.5]
        )
    )

    problem.log_likelihood(
        np.asarray(
            [0.25]
        )
    )

    metrics = problem.metrics

    assert metrics[
        "prior_transform_calls"
    ] == 1

    assert metrics[
        "likelihood_calls"
    ] == 2

    assert metrics[
        "prior_transform_seconds"
    ] >= 0.0

    assert metrics[
        "likelihood_seconds"
    ] >= 0.0

    assert metrics[
        "mean_likelihood_seconds"
    ] >= 0.0


def test_timed_problem_reset():
    problem = TimedInferenceProblem(
        InstrumentedProblem()
    )

    problem.log_likelihood(
        np.asarray(
            [0.5]
        )
    )

    problem.reset()

    assert problem.metrics == {
        "likelihood_calls": 0,
        "likelihood_seconds": 0.0,
        "mean_likelihood_seconds": 0.0,
        "prior_transform_calls": 0,
        "prior_transform_seconds": 0.0,
    }
