"""Controlled recovery of the length and the axis ratio of a Ferrers bar."""

from __future__ import annotations

import numpy as np

from baymorph.fitting import (
    Parameter,
    ParameterSet,
    make_bulge_disk_bar_psf_model,
    make_sampler,
)
from baymorph.fitting.image_problem import ImageInferenceProblem
from baymorph.likelihood import GaussianImageLikelihood
from baymorph.simulation.psf import GaussianPSF


def test_bulge_disk_bar_sampler_recovers_the_bar() -> None:
    """The posterior of a_bar and q_bar covers the values of a bar."""
    rng = np.random.default_rng(20261005)
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
        flux_bar=4000.0,
        alpha_bar=2.0,
        c0_bar=0.0,
        theta_bar=0.7,
        x0=30.2,
        y0=29.8,
    )
    truth = {"a_bar": 11.0, "q_bar": 0.3}

    noiseless = make_bulge_disk_bar_psf_model(
        **fixed, **truth, psf=psf
    ).render(shape)
    data = noiseless + rng.normal(0.0, noise_sigma, shape)
    likelihood = GaussianImageLikelihood(
        data=data,
        noise=np.full(shape, noise_sigma),
    )
    parameters = ParameterSet(
        [
            Parameter(name="a_bar", lower=6.0, upper=16.0),
            Parameter(name="q_bar", lower=0.2, upper=0.6),
        ]
    )

    def model_factory(*, a_bar: float, q_bar: float):
        return make_bulge_disk_bar_psf_model(
            **fixed, a_bar=a_bar, q_bar=q_bar, psf=psf
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
