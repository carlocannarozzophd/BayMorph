"""Tests for canonical reference-validation cases."""

import numpy as np
import pytest

from baymorph.fitting.parameters import (
    Parameter,
    ParameterSet,
)
from baymorph.fitting.problem import InferenceProblem
from baymorph.validation.reference import (
    CallableInferenceProblem,
    ReferenceValidationCase,
)


def make_problem() -> CallableInferenceProblem:
    parameters = ParameterSet(
        [
            Parameter(
                "x",
                -2.0,
                2.0,
            ),
            Parameter(
                "y",
                0.0,
                4.0,
            ),
        ]
    )

    return CallableInferenceProblem(
        parameters=parameters,
        log_likelihood=lambda values: (
            -0.5
            * float(
                values @ values
            )
        ),
    )


def test_callable_inference_problem_implements_canonical_contract():
    problem = make_problem()

    assert isinstance(
        problem,
        InferenceProblem,
    )
    assert problem.ndim == 2
    assert problem.parameter_names == [
        "x",
        "y",
    ]

    transformed = problem.prior_transform(
        np.asarray(
            [
                0.25,
                0.75,
            ]
        )
    )

    assert transformed == pytest.approx(
        np.asarray(
            [
                -1.0,
                3.0,
            ]
        )
    )

    assert problem.log_likelihood(
        np.asarray(
            [
                1.0,
                2.0,
            ]
        )
    ) == pytest.approx(
        -2.5
    )


def test_reference_validation_case_preserves_inference_problem():
    problem = make_problem()

    case = ReferenceValidationCase(
        name="gaussian_2d",
        problem=problem,
        reference_logz=-1.25,
        reference_mean=np.asarray(
            [
                0.0,
                2.0,
            ]
        ),
        reference_covariance=np.asarray(
            [
                [1.0, 0.2],
                [0.2, 0.5],
            ]
        ),
        metadata={
            "experiment_id": "MAT-TEST",
        },
    )

    assert case.problem is problem
    assert case.name == "gaussian_2d"
    assert case.reference_logz == pytest.approx(
        -1.25
    )
    assert case.metadata[
        "experiment_id"
    ] == "MAT-TEST"


def test_reference_validation_case_requires_matching_reference_shapes():
    problem = make_problem()

    with pytest.raises(
        ValueError,
        match="reference_mean shape",
    ):
        ReferenceValidationCase(
            name="invalid",
            problem=problem,
            reference_mean=np.asarray(
                [
                    0.0,
                ]
            ),
            reference_covariance=np.eye(
                2
            ),
        )


def test_reference_validation_case_requires_mean_and_covariance_together():
    with pytest.raises(
        ValueError,
        match="provided together",
    ):
        ReferenceValidationCase(
            name="invalid",
            problem=make_problem(),
            reference_mean=np.zeros(
                2
            ),
        )
