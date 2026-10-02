"""Controlled recovery of the winding and the phase of two spiral arms."""

from __future__ import annotations

import numpy as np

from baymorph.fitting import (
    Parameter,
    ParameterSet,
    make_bulge_disk_spiral_psf_model,
    make_sampler,
)
from baymorph.fitting.image_problem import ImageInferenceProblem
from baymorph.likelihood import GaussianImageLikelihood
from baymorph.simulation.psf import GaussianPSF


def test_bulge_disk_spiral_sampler_recovers_the_arms() -> None:
    """The posterior of winding_spiral and phase_spiral covers their values."""
    rng = np.random.default_rng(20261007)
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
        re_disk=12.0,
        q_disk=0.8,
        theta_disk=0.4,
        amplitude_spiral=0.6,
        r_in_spiral=3.0,
        x0=30.2,
        y0=29.8,
    )
    truth = {"winding_spiral": 1.5, "phase_spiral": 1.2}

    noiseless = make_bulge_disk_spiral_psf_model(
        **fixed, **truth, psf=psf
    ).render(shape)
    data = noiseless + rng.normal(0.0, noise_sigma, shape)
    likelihood = GaussianImageLikelihood(
        data=data,
        noise=np.full(shape, noise_sigma),
    )
    parameters = ParameterSet(
        [
            Parameter(name="winding_spiral", lower=-2.0, upper=2.0),
            Parameter(name="phase_spiral", lower=0.0, upper=np.pi),
        ]
    )

    def model_factory(*, winding_spiral: float, phase_spiral: float):
        return make_bulge_disk_spiral_psf_model(
            **fixed,
            winding_spiral=winding_spiral,
            phase_spiral=phase_spiral,
            psf=psf,
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
