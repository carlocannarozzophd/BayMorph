"""Tests for the backend-neutral multi-band inference problem."""

import numpy as np
import pytest

from baymorph.fitting.multiband import (
    MultiBandParameterLayout,
    MultiBandParameterSpec,
)
from baymorph.fitting.multiband_problem import MultiBandInferenceProblem
from baymorph.fitting.parameters import Parameter
from baymorph.fitting.representations import RepresentationFactory
from baymorph.likelihood import GaussianImageLikelihood
from baymorph.likelihood.multiband import MultiBandLikelihood
from baymorph.models import SersicModel


def make_base_factory(
    *,
    flux,
    re,
    fixed_n,
):
    """Build one Sérsic representation."""
    return SersicModel(
        flux=flux,
        re=re,
        n=fixed_n,
        q=0.8,
        theta=0.0,
        x0=7.0,
        y0=7.0,
    )


def make_problem() -> MultiBandInferenceProblem:
    layout = MultiBandParameterLayout(
        bands=["g", "r"],
        specifications=[
            MultiBandParameterSpec.shared_parameter(
                Parameter(
                    name="re",
                    lower=2.0,
                    upper=10.0,
                )
            ),
            MultiBandParameterSpec.band_specific(
                name="flux",
                parameters={
                    "g": Parameter(
                        name="flux",
                        lower=100.0,
                        upper=1000.0,
                    ),
                    "r": Parameter(
                        name="flux",
                        lower=200.0,
                        upper=2000.0,
                    ),
                },
            ),
        ],
    )

    likelihood = MultiBandLikelihood(
        {
            "g": GaussianImageLikelihood(
                data=np.zeros((15, 15)),
                noise=np.ones((15, 15)),
            ),
            "r": GaussianImageLikelihood(
                data=np.zeros((15, 15)),
                noise=np.ones((15, 15)),
            ),
        }
    )

    representation_factory = RepresentationFactory(
        factory=make_base_factory,
        fixed_kwargs={
            "g": {
                "fixed_n": 1.5,
            },
            "r": {
                "fixed_n": 2.5,
            },
        },
    )

    return MultiBandInferenceProblem(
        representation_factory=representation_factory,
        parameter_layout=layout,
        likelihood=likelihood,
    )


def test_multiband_problem_exposes_global_parameters():
    problem = make_problem()

    assert problem.ndim == 3
    assert problem.parameter_names == [
        "re",
        "flux__g",
        "flux__r",
    ]


def test_multiband_problem_routes_and_renders_each_band():
    problem = make_problem()

    rendered = problem.render_models(
        np.array(
            [
                5.0,
                500.0,
                1200.0,
            ]
        )
    )

    assert rendered["g"].shape == (15, 15)
    assert rendered["r"].shape == (15, 15)

    assert not np.allclose(
        rendered["g"],
        rendered["r"],
    )


def test_multiband_problem_evaluates_joint_likelihood():
    problem = make_problem()

    value = problem.log_likelihood(
        np.array(
            [
                5.0,
                500.0,
                1200.0,
            ]
        )
    )

    assert np.isfinite(value)


def test_multiband_problem_rejects_representation_band_mismatch():
    problem = make_problem()

    representation_factory = RepresentationFactory(
        factory=make_base_factory,
        fixed_kwargs={
            "g": {
                "fixed_n": 2.0,
            },
        },
    )

    with pytest.raises(
        ValueError,
        match="must match",
    ):
        MultiBandInferenceProblem(
            representation_factory=representation_factory,
            parameter_layout=problem.parameter_layout,
            likelihood=problem.likelihood,
        )


def test_multiband_problem_rejects_likelihood_band_mismatch():
    layout = MultiBandParameterLayout(
        bands=["g"],
        specifications=[
            MultiBandParameterSpec.shared_parameter(
                Parameter(
                    name="re",
                    lower=1.0,
                    upper=10.0,
                )
            )
        ],
    )

    representation_factory = RepresentationFactory(
        factory=make_base_factory,
        fixed_kwargs={
            "g": {
                "fixed_n": 2.0,
            },
        },
    )

    likelihood = MultiBandLikelihood(
        {
            "r": GaussianImageLikelihood(
                data=np.zeros((3, 3)),
                noise=np.ones((3, 3)),
            )
        }
    )

    with pytest.raises(
        ValueError,
        match="bands must match",
    ):
        MultiBandInferenceProblem(
            representation_factory=representation_factory,
            parameter_layout=layout,
            likelihood=likelihood,
        )
