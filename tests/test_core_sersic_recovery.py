"""Controlled recovery of a resolved depleted core."""

from __future__ import annotations

import numpy as np

from baymorph.fitting import (
    Parameter,
    ParameterSet,
    make_core_sersic_psf_model,
    make_sampler,
)
from baymorph.fitting.image_problem import ImageInferenceProblem
from baymorph.likelihood import GaussianImageLikelihood
from baymorph.simulation.psf import GaussianPSF


def test_core_sersic_sampler_recovers_break_radius_and_slope() -> None:
    """The posterior of rb and gamma covers the values of a simulated core."""
    rng = np.random.default_rng(20261003)
    shape = (41, 41)
    noise_sigma = 0.5
    psf = GaussianPSF(fwhm=1.5)
    fixed = dict(
        flux=20000.0,
        re=8.0,
        n=4.0,
        alpha=2.0,
        q=0.9,
        theta=0.3,
        x0=20.2,
        y0=19.8,
    )
    truth = {"rb": 3.0, "gamma": 0.2}

    noiseless = make_core_sersic_psf_model(**fixed, **truth, psf=psf).render(
        shape
    )
    data = noiseless + rng.normal(0.0, noise_sigma, shape)
    likelihood = GaussianImageLikelihood(
        data=data,
        noise=np.full(shape, noise_sigma),
    )
    parameters = ParameterSet(
        [
            Parameter(name="rb", lower=0.5, upper=8.0),
            Parameter(name="gamma", lower=0.0, upper=0.9),
        ]
    )

    def model_factory(*, rb: float, gamma: float):
        return make_core_sersic_psf_model(**fixed, rb=rb, gamma=gamma, psf=psf)

    problem = ImageInferenceProblem(
        model_factory=model_factory,
        parameters=parameters,
        likelihood=likelihood,
    )
    result = make_sampler(
        {
            "name": "dynesty",
            "mode": "static",
            "init": {
                "nlive": 60,
                "sample": "rwalk",
                "rstate": np.random.default_rng(12345),
            },
            "execution": {"processes": 1, "progress": False},
            "run": {"dlogz": 0.5},
        },
        problem=problem,
    ).fit()

    for name, value in truth.items():
        summary = result.summary(name)
        width = 0.5 * (summary["q84"] - summary["q16"])

        assert abs(summary["median"] - value) <= 4.0 * width

    assert np.isfinite(result.logz)
