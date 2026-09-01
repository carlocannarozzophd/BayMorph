"""Tests for backend-neutral Dynesty inference problems."""

import numpy as np
import pytest

from baymorph.fitting.dynesty import DynestyFitter
from baymorph.fitting.problem import InferenceProblem


class QuadraticProblem(InferenceProblem):
    """Minimal backend-neutral problem for sampler tests."""

    @property
    def ndim(self) -> int:
        return 2

    @property
    def parameter_names(self) -> list[str]:
        return ["x", "y"]

    def prior_transform(
        self,
        cube: np.ndarray,
    ) -> np.ndarray:
        return 2.0 * np.asarray(
            cube,
            dtype=float,
        ) - 1.0

    def log_likelihood(
        self,
        theta: np.ndarray,
    ) -> float:
        values = np.asarray(
            theta,
            dtype=float,
        )
        return float(
            -0.5 * np.sum(values**2)
        )


def test_dynesty_accepts_inference_problem():
    fitter = DynestyFitter(
        problem=QuadraticProblem()
    )

    assert fitter.ndim == 2
    assert fitter.parameter_names == [
        "x",
        "y",
    ]

    transformed = fitter.prior_transform(
        np.array(
            [
                0.25,
                0.75,
            ]
        )
    )

    assert np.allclose(
        transformed,
        np.array(
            [
                -0.5,
                0.5,
            ]
        ),
    )

    assert np.isclose(
        fitter.loglikelihood(
            np.array(
                [
                    1.0,
                    2.0,
                ]
            )
        ),
        -2.5,
    )


def test_dynesty_problem_without_image_has_no_render_model():
    fitter = DynestyFitter(
        problem=QuadraticProblem()
    )

    with pytest.raises(
        AttributeError,
        match="does not expose one model image",
    ):
        fitter.render_model(
            np.zeros(2)
        )


def test_dynesty_requires_inference_problem():
    with pytest.raises(
        TypeError,
        match="problem must be an InferenceProblem",
    ):
        DynestyFitter(
            problem=object(),
        )
