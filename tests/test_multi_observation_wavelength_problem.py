"""Inference-problem closure tests for wavelength-dependent coupling."""

from __future__ import annotations

import numpy as np
import pytest

from baymorph.fitting.multi_observation import (
    MultiObservationParameterLayout,
    MultiObservationParameterSpec,
)
from baymorph.fitting.multi_observation_problem import (
    MultiObservationInferenceProblem,
)
from baymorph.fitting.priors import UniformPrior
from baymorph.fitting.representations import RepresentationFactory
from baymorph.likelihood import GaussianImageLikelihood
from baymorph.likelihood.multi_observation import MultiObservationLikelihood


class ConstantRepresentation:
    """Minimal observation representation for inference closure tests."""

    def __init__(
        self,
        *,
        amplitude: float,
    ) -> None:
        self.amplitude = float(
            amplitude
        )

    def render(
        self,
        shape,
    ) -> np.ndarray:
        """Render a constant image."""
        return np.full(
            shape,
            self.amplitude,
            dtype=float,
        )


def _factory(
    *,
    amplitude,
):
    return ConstantRepresentation(
        amplitude=amplitude,
    )


def _problem() -> MultiObservationInferenceProblem:
    observations = (
        "blue",
        "red",
    )

    half_span = (
        0.5
        * np.log(2.0)
    )

    layout = MultiObservationParameterLayout(
        observations=observations,
        specifications=[
            MultiObservationParameterSpec.wavelength(
                "amplitude",
                prior=UniformPrior(
                    lower=0.0,
                    upper=2.0,
                ),
            ),
        ],
        wavelength_coordinates={
            "blue": -half_span,
            "red": half_span,
        },
    )

    representation_factory = RepresentationFactory(
        factory=_factory,
        fixed_kwargs={
            "blue": {},
            "red": {},
        },
    )

    likelihood = MultiObservationLikelihood(
        {
            "blue": GaussianImageLikelihood(
                data=np.zeros(
                    (5, 5),
                    dtype=float,
                ),
                noise=np.ones(
                    (5, 5),
                    dtype=float,
                ),
            ),
            "red": GaussianImageLikelihood(
                data=np.zeros(
                    (7, 4),
                    dtype=float,
                ),
                noise=np.ones(
                    (7, 4),
                    dtype=float,
                ),
            ),
        }
    )

    return MultiObservationInferenceProblem(
        representation_factory=representation_factory,
        parameter_layout=layout,
        likelihood=likelihood,
    )


def test_wavelength_problem_exposes_backend_neutral_sampling_state():
    problem = _problem()

    assert problem.ndim == 3

    assert problem.parameter_names == [
        "amplitude__rstar",
        "amplitude__z__blue",
        "amplitude__z__red",
    ]

    theta = problem.prior_transform(
        np.asarray(
            [
                0.5,
                0.25,
                0.75,
            ],
            dtype=float,
        )
    )

    assert theta.shape == (
        problem.ndim,
    )

    assert np.all(
        np.isfinite(
            theta
        )
    )


def test_wavelength_problem_routes_renders_and_scores():
    problem = _problem()

    theta = problem.prior_transform(
        np.asarray(
            [
                0.5,
                0.25,
                0.75,
            ],
            dtype=float,
        )
    )

    rendered = problem.render_models(
        theta
    )

    assert rendered["blue"].shape == (
        5,
        5,
    )

    assert rendered["red"].shape == (
        7,
        4,
    )

    assert not np.allclose(
        rendered["blue"][0, 0],
        rendered["red"][0, 0],
    )

    assert np.isfinite(
        problem.log_likelihood(
            theta
        )
    )


def test_wavelength_problem_exact_sharing_limit_closes_through_renderer():
    problem = _problem()

    theta = np.asarray(
        [
            1.0,
            -1.0,
            1.0,
        ],
        dtype=float,
    )

    rendered = problem.render_models(
        theta
    )

    assert rendered[
        "blue"
    ][
        0,
        0,
    ] == pytest.approx(
        rendered[
            "red"
        ][
            0,
            0,
        ]
    )
