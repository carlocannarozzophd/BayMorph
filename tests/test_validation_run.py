"""Tests for canonical scientific-validation execution records."""

from __future__ import annotations

import numpy as np
import pytest

from baymorph.fitting.backend import InferenceBackend
from baymorph.fitting.parameters import (
    Parameter,
    ParameterSet,
)
from baymorph.results import (
    EvidenceEstimate,
    InferenceResult,
    PosteriorSamples,
)
from baymorph.validation.reference import (
    CallableInferenceProblem,
    ReferenceValidationCase,
)
from baymorph.validation.run import (
    ValidationRun,
    execute_validation_case,
)


def make_case() -> ReferenceValidationCase:
    parameters = ParameterSet(
        [
            Parameter(
                "x",
                -2.0,
                2.0,
            )
        ]
    )

    problem = CallableInferenceProblem(
        parameters=parameters,
        log_likelihood=lambda values: (
            -0.5
            * float(
                values[0] ** 2
            )
        ),
    )

    return ReferenceValidationCase(
        name="quadratic",
        problem=problem,
        reference_logz=-1.0,
        reference_mean=np.asarray(
            [
                0.0,
            ]
        ),
        reference_covariance=np.asarray(
            [
                [
                    1.0,
                ]
            ]
        ),
    )


class DeterministicBackend(
    InferenceBackend
):
    """Small production-contract backend used only by these tests."""

    @property
    def name(self) -> str:
        """Return the deterministic test-backend name."""
        return "deterministic"

    def fit(
        self,
        *,
        progress_label: str | None = None,
    ) -> InferenceResult:
        """Return a deterministic canonical inference result."""
        del progress_label

        return InferenceResult(
            posterior=PosteriorSamples(
                samples=np.asarray(
                    [
                        [
                            -1.0,
                        ],
                        [
                            1.0,
                        ],
                    ]
                ),
                weights=np.asarray(
                    [
                        0.5,
                        0.5,
                    ]
                ),
                parameter_names=[
                    "x",
                ],
                log_likelihood=np.asarray(
                    [
                        -0.5,
                        -0.5,
                    ]
                ),
            ),
            evidence=EvidenceEstimate(
                logz=-1.0,
                uncertainty=0.1,
                method="deterministic",
            ),
            diagnostics={
                "backend": self.name,
                "calls": 2,
            },
            provenance={
                "backend": self.name,
            },
        )


def test_execute_validation_case_uses_production_backend_contract():
    case = make_case()

    backend = DeterministicBackend(
        problem=case.problem,
        config={
            "seed": 42,
        },
    )

    run = execute_validation_case(
        case,
        backend,
        seed=42,
    )

    assert isinstance(
        run,
        ValidationRun,
    )
    assert isinstance(
        run.result,
        InferenceResult,
    )

    assert run.case_name == "quadratic"
    assert run.backend_name == "deterministic"
    assert run.seed == 42
    assert run.config == {
        "seed": 42,
    }
    assert run.runtime_seconds >= 0.0
    assert run.likelihood_calls == 2

    assert run.result.logz == pytest.approx(
        -1.0
    )


def test_execute_validation_case_requires_exact_case_problem():
    case = make_case()
    other_case = make_case()

    backend = DeterministicBackend(
        problem=other_case.problem,
    )

    with pytest.raises(
        ValueError,
        match="validation case problem",
    ):
        execute_validation_case(
            case,
            backend,
        )


def test_validation_run_allows_missing_likelihood_call_diagnostic():
    case = make_case()

    result = InferenceResult(
        posterior=PosteriorSamples(
            samples=np.asarray(
                [
                    [
                        0.0,
                    ]
                ]
            ),
            weights=np.asarray(
                [
                    1.0,
                ]
            ),
            parameter_names=[
                "x",
            ],
            log_likelihood=np.asarray(
                [
                    0.0,
                ]
            ),
        ),
        diagnostics={},
    )

    run = ValidationRun(
        case_name=case.name,
        backend_name="test",
        result=result,
        runtime_seconds=0.0,
    )

    assert run.likelihood_calls is None


def test_validation_run_does_not_duplicate_inference_products():
    case = make_case()

    backend = DeterministicBackend(
        problem=case.problem,
    )

    run = execute_validation_case(
        case,
        backend,
    )

    assert not hasattr(
        run,
        "samples",
    )
    assert not hasattr(
        run,
        "weights",
    )
    assert not hasattr(
        run,
        "logz",
    )
    assert not hasattr(
        run,
        "logzerr",
    )
