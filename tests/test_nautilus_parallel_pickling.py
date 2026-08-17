"""Tests for pickle-safe Nautilus parallel single-image inference."""

from __future__ import annotations

import pickle

import numpy as np

from baymorph.fitting import Parameter, ParameterSet
from baymorph.fitting.image_problem import ImageInferenceProblem
from baymorph.fitting.nautilus import NautilusFitter
from baymorph.likelihood import GaussianImageLikelihood
from baymorph.pipeline.real_fit import _make_model_factory
from baymorph.simulation.psf import GaussianPSF


def test_single_band_nautilus_callables_are_pickle_safe() -> None:
    """Production prior and likelihood callables must support multiprocessing."""
    shape = (
        21,
        21,
    )

    factory = _make_model_factory(
        "single_sersic",
        GaussianPSF(
            fwhm=2.5
        ),
        background_cfg={
            "model": "plane",
        },
    )

    parameters = ParameterSet(
        [
            Parameter("flux", 100.0, 1000.0),
            Parameter("re", 1.0, 10.0),
            Parameter("n", 0.5, 5.0),
            Parameter("q", 0.4, 1.0),
            Parameter("theta", -1.0, 1.0),
            Parameter("x0", 8.0, 12.0),
            Parameter("y0", 8.0, 12.0),
            Parameter("background_level", -5.0, 5.0),
            Parameter("background_gradient_x", -0.1, 0.1),
            Parameter("background_gradient_y", -0.1, 0.1),
        ]
    )

    likelihood = GaussianImageLikelihood(
        data=np.zeros(
            shape,
            dtype=float,
        ),
        noise=np.ones(
            shape,
            dtype=float,
        ),
    )

    problem = ImageInferenceProblem(
        model_factory=factory,
        parameters=parameters,
        likelihood=likelihood,
    )

    fitter = NautilusFitter(
        problem=problem
    )

    pickle.loads(
        pickle.dumps(
            fitter.prior_transform
        )
    )

    restored_likelihood = pickle.loads(
        pickle.dumps(
            fitter.loglikelihood
        )
    )

    theta = np.array(
        [
            500.0,
            4.0,
            2.0,
            0.8,
            0.0,
            10.0,
            10.0,
            0.0,
            0.01,
            -0.01,
        ],
        dtype=float,
    )

    assert np.isfinite(
        restored_likelihood(
            theta
        )
    )
