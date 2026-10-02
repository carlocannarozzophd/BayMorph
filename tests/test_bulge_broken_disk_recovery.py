"""Controlled recovery of the break of a broken exponential disc."""

from __future__ import annotations

import numpy as np

from baymorph.fitting import (
    Parameter,
    ParameterSet,
    make_bulge_broken_disk_psf_model,
    make_sampler,
)
from baymorph.fitting.image_problem import ImageInferenceProblem
from baymorph.likelihood import GaussianImageLikelihood
from baymorph.simulation.psf import GaussianPSF


def test_bulge_broken_disk_sampler_recovers_the_break() -> None:
    """The posterior of h_out and rb covers the values of a type II disc."""
    rng = np.random.default_rng(20261004)
    shape = (61, 61)
    noise_sigma = 0.5
    psf = GaussianPSF(fwhm=2.5)
    fixed = dict(
        flux_bulge=1500.0,
        re_bulge=2.0,
        n_bulge=3.0,
        q_bulge=0.9,
        theta_bulge=0.2,
        flux_disk=20000.0,
        h_in_disk=8.0,
        wb_disk=1.0,
        q_disk=0.7,
        theta_disk=0.2,
        x0=30.2,
        y0=29.8,
    )
    truth = {"h_out_disk": 4.0, "rb_disk": 14.0}

    noiseless = make_bulge_broken_disk_psf_model(
        **fixed, **truth, psf=psf
    ).render(shape)
    data = noiseless + rng.normal(0.0, noise_sigma, shape)
    likelihood = GaussianImageLikelihood(
        data=data,
        noise=np.full(shape, noise_sigma),
    )
    parameters = ParameterSet(
        [
            Parameter(name="h_out_disk", lower=2.0, upper=12.0),
            Parameter(name="rb_disk", lower=6.0, upper=24.0),
        ]
    )

    def model_factory(*, h_out_disk: float, rb_disk: float):
        return make_bulge_broken_disk_psf_model(
            **fixed, h_out_disk=h_out_disk, rb_disk=rb_disk, psf=psf
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
