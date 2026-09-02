"""Tests for the scientific validation framework."""

from __future__ import annotations

import numpy as np
import pytest

from baymorph.fitting.parameters import (
    Parameter,
    ParameterSet,
)
from baymorph.validation.reference import (
    CallableInferenceProblem,
    ReferenceValidationCase,
)
from validation import (
    ValidationResult,
    compare_results,
)


def make_problem() -> ReferenceValidationCase:
    """Create a simple one-dimensional validation problem."""
    parameters = ParameterSet(
        [
            Parameter(
                "x",
                -2.0,
                2.0,
            )
        ]
    )

    return ReferenceValidationCase(
        name="quadratic",
        problem=CallableInferenceProblem(
            parameters=parameters,
            log_likelihood=lambda values: -0.5
            * float(
                values[0] ** 2
            ),
        ),
        reference_logz=-1.0,
        reference_mean=np.asarray(
            [
                0.0
            ]
        ),
        reference_covariance=np.asarray(
            [
                [
                    1.0
                ]
            ]
        ),
    )


def make_result(
    *,
    backend_name: str,
    logz: float,
    ncall: int,
    runtime_seconds: float,
    offset: float = 0.0,
) -> ValidationResult:
    """Create a small weighted validation result."""
    return ValidationResult(
        problem_name="quadratic",
        backend_name=backend_name,
        seed=42,
        config={
            "nlive": 10,
        },
        samples=np.asarray(
            [
                [
                    -1.0 + offset
                ],
                [
                    1.0 + offset
                ],
            ]
        ),
        weights=np.asarray(
            [
                1.0,
                1.0,
            ]
        ),
        logl=np.asarray(
            [
                -0.5,
                -0.5,
            ]
        ),
        logz=logz,
        logzerr=0.1,
        ncall=ncall,
        runtime_seconds=runtime_seconds,
    )


def test_reference_case_owns_canonical_inference_problem():
    problem = make_problem()

    assert problem.problem.ndim == 1
    assert problem.problem.log_likelihood(
        np.asarray(
            [
                1.0
            ]
        )
    ) == pytest.approx(
        -0.5
    )




def test_validation_result_normalizes_weights():
    result = make_result(
        backend_name="native",
        logz=-1.0,
        ncall=100,
        runtime_seconds=2.0,
    )

    np.testing.assert_allclose(
        result.weights,
        np.asarray(
            [
                0.5,
                0.5,
            ]
        ),
    )
    assert result.nsamples == 2
    assert result.ndim == 1


def test_validation_result_computes_posterior_statistics():
    result = make_result(
        backend_name="native",
        logz=-1.0,
        ncall=100,
        runtime_seconds=2.0,
    )

    np.testing.assert_allclose(
        result.posterior_mean,
        np.asarray(
            [
                0.0
            ]
        ),
    )
    np.testing.assert_allclose(
        result.posterior_covariance,
        np.asarray(
            [
                [
                    1.0
                ]
            ]
        ),
    )


def test_validation_result_summary_is_serializable():
    result = make_result(
        backend_name="native",
        logz=-1.0,
        ncall=100,
        runtime_seconds=2.0,
    )

    summary = result.summary()

    assert summary["problem_name"] == "quadratic"
    assert summary["backend_name"] == "native"
    assert summary["posterior_mean"] == [
        0.0
    ]
    assert summary["nsamples"] == 2


def test_compare_results_reports_backend_differences():
    reference = make_result(
        backend_name="dynesty",
        logz=-1.1,
        ncall=200,
        runtime_seconds=4.0,
    )
    candidate = make_result(
        backend_name="native",
        logz=-1.0,
        ncall=100,
        runtime_seconds=2.0,
        offset=0.2,
    )

    report = compare_results(
        reference,
        candidate,
    )

    assert report.problem_name == "quadratic"
    assert report.reference_backend == "dynesty"
    assert report.candidate_backend == "native"
    assert report.logz_difference == pytest.approx(
        0.1
    )
    assert report.posterior_mean_distance == pytest.approx(
        0.2
    )
    assert report.likelihood_call_ratio == pytest.approx(
        0.5
    )
    assert report.runtime_ratio == pytest.approx(
        0.5
    )


def test_compare_results_rejects_different_problems():
    reference = make_result(
        backend_name="dynesty",
        logz=-1.0,
        ncall=100,
        runtime_seconds=1.0,
    )
    candidate = make_result(
        backend_name="native",
        logz=-1.0,
        ncall=100,
        runtime_seconds=1.0,
    )

    object.__setattr__(
        candidate,
        "problem_name",
        "different",
    )

    with pytest.raises(
        ValueError,
        match="same problem",
    ):
        compare_results(
            reference,
            candidate,
        )
