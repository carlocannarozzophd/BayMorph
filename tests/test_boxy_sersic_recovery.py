"""Controlled recovery of the isophote shape of a boxy Sérsic profile."""

from __future__ import annotations

import numpy as np

from baymorph.fitting import (
    Parameter,
    ParameterSet,
    make_boxy_sersic_psf_model,
    make_sampler,
)
from baymorph.fitting.image_problem import ImageInferenceProblem
from baymorph.likelihood import GaussianImageLikelihood
from baymorph.simulation.psf import GaussianPSF


def test_boxy_sersic_sampler_recovers_shape_and_index() -> None:
    """The posterior of c0 and n covers the values of a simulated image."""
    rng = np.random.default_rng(20261002)
    shape = (31, 31)
    noise_sigma = 1.0
    psf = GaussianPSF(fwhm=2.5)
    fixed = dict(
        flux=3000.0, re=5.0, q=0.6, theta=0.3, x0=15.2, y0=14.8
    )
    truth = {"c0": 0.6, "n": 2.0}

    noiseless = make_boxy_sersic_psf_model(**fixed, **truth, psf=psf).render(
        shape
    )
    data = noiseless + rng.normal(0.0, noise_sigma, shape)
    likelihood = GaussianImageLikelihood(
        data=data,
        noise=np.full(shape, noise_sigma),
    )
    parameters = ParameterSet(
        [
            Parameter(name="c0", lower=-0.5, upper=1.0),
            Parameter(name="n", lower=1.0, upper=4.0),
        ]
    )

    def model_factory(*, c0: float, n: float):
        return make_boxy_sersic_psf_model(**fixed, c0=c0, n=n, psf=psf)

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
