"""Tests for the backend-neutral single-image inference problem."""

import numpy as np
import pytest

from baymorph.core import make_constant_noise_map
from baymorph.fitting.image_problem import ImageInferenceProblem
from baymorph.fitting.parameters import Parameter, ParameterSet
from baymorph.likelihood import GaussianImageLikelihood
from baymorph.models import SersicModel


def make_problem() -> ImageInferenceProblem:
    data = SersicModel(
        flux=1000.0,
        re=8.0,
        n=2.0,
        q=0.8,
        theta=0.0,
        x0=15.0,
        y0=15.0,
    ).render(
        shape=(30, 30)
    )

    likelihood = GaussianImageLikelihood(
        data=data,
        noise=make_constant_noise_map(
            data.shape,
            sigma=1.0,
        ),
    )

    parameters = ParameterSet(
        [
            Parameter(
                name="flux",
                lower=100.0,
                upper=2000.0,
            ),
            Parameter(
                name="re",
                lower=2.0,
                upper=20.0,
            ),
        ]
    )

    def model_factory(
        *,
        flux,
        re,
    ):
        return SersicModel(
            flux=flux,
            re=re,
            n=2.0,
            q=0.8,
            theta=0.0,
            x0=15.0,
            y0=15.0,
        )

    return ImageInferenceProblem(
        model_factory=model_factory,
        parameters=parameters,
        likelihood=likelihood,
    )


def test_image_problem_exposes_sampler_interface():
    problem = make_problem()

    assert problem.ndim == 2
    assert problem.parameter_names == [
        "flux",
        "re",
    ]

    transformed = problem.prior_transform(
        np.array(
            [
                0.5,
                0.5,
            ]
        )
    )

    assert np.allclose(
        transformed,
        np.array(
            [
                1050.0,
                11.0,
            ]
        ),
    )


def test_image_problem_renders_named_parameters():
    problem = make_problem()

    rendered = problem.render_model(
        np.array(
            [
                1000.0,
                8.0,
            ]
        )
    )

    expected = SersicModel(
        flux=1000.0,
        re=8.0,
        n=2.0,
        q=0.8,
        theta=0.0,
        x0=15.0,
        y0=15.0,
    ).render(
        shape=(30, 30)
    )

    assert np.allclose(
        rendered,
        expected,
    )


def test_image_problem_evaluates_likelihood():
    problem = make_problem()

    value = problem.log_likelihood(
        np.array(
            [
                1000.0,
                8.0,
            ]
        )
    )

    assert np.isfinite(value)


def test_image_problem_rejects_invalid_unit_cube_values():
    problem = make_problem()

    with pytest.raises(
        ValueError,
        match="between zero and one",
    ):
        problem.prior_transform(
            np.array(
                [
                    -0.1,
                    0.5,
                ]
            )
        )


def test_image_problem_rejects_wrong_parameter_dimension():
    problem = make_problem()

    with pytest.raises(
        ValueError,
        match="one value per parameter",
    ):
        problem.render_model(
            np.array(
                [
                    1000.0,
                ]
            )
        )
