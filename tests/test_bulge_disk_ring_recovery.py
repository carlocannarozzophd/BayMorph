"""Controlled recovery of the radius and the width of a Gaussian ring."""

from __future__ import annotations

import numpy as np

from baymorph.fitting import (
    Parameter,
    ParameterSet,
    make_bulge_disk_ring_psf_model,
    make_sampler,
)
from baymorph.fitting.image_problem import ImageInferenceProblem
from baymorph.likelihood import GaussianImageLikelihood
from baymorph.simulation.psf import GaussianPSF


def test_bulge_disk_ring_sampler_recovers_the_ring() -> None:
    """The posterior of r_ring and sigma_ring covers the values of a ring."""
    rng = np.random.default_rng(20261006)
    shape = (61, 61)
    noise_sigma = 0.5
    psf = GaussianPSF(fwhm=2.5)
    fixed = dict(
        flux_bulge=1500.0,
        re_bulge=2.0,
        n_bulge=3.0,
        q_bulge=0.9,
        theta_bulge=0.2,
        flux_disk=15000.0,
        re_disk=12.0,
        q_disk=0.8,
        theta_disk=0.2,
        flux_ring=3000.0,
        q_ring=0.7,
        theta_ring=0.4,
        x0=30.2,
        y0=29.8,
    )
    truth = {"r_ring": 12.0, "sigma_ring": 2.5}

    noiseless = make_bulge_disk_ring_psf_model(
        **fixed, **truth, psf=psf
    ).render(shape)
    data = noiseless + rng.normal(0.0, noise_sigma, shape)
    likelihood = GaussianImageLikelihood(
        data=data,
        noise=np.full(shape, noise_sigma),
    )
    parameters = ParameterSet(
        [
            Parameter(name="r_ring", lower=6.0, upper=18.0),
            Parameter(name="sigma_ring", lower=2.0, upper=5.0),
        ]
    )

    def model_factory(*, r_ring: float, sigma_ring: float):
        return make_bulge_disk_ring_psf_model(
            **fixed, r_ring=r_ring, sigma_ring=sigma_ring, psf=psf
        )

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
                "sample": "unif",
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
