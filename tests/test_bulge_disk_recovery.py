"""Controlled recovery test for bulge-disc inference."""

from __future__ import annotations

import numpy as np
import pytest

from baymorph.fitting import (
    Parameter,
    ParameterSet,
    make_bulge_disk_psf_sky_model,
    make_sampler,
)
from baymorph.fitting.image_problem import ImageInferenceProblem
from baymorph.likelihood import GaussianImageLikelihood
from baymorph.simulation.psf import GaussianPSF


def test_bulge_disk_recovers_principal_component_parameters() -> None:
    """A controlled fit should recover bulge and disc fluxes."""
    rng = np.random.default_rng(20260806)

    shape = (25, 25)
    noise_sigma = 1.5
    psf = GaussianPSF(fwhm=2.5)

    truth = {
        "flux_bulge": 250.0,
        "flux_disk": 600.0,
    }

    fixed_parameters = {
        "re_bulge": 2.5,
        "n_bulge": 3.0,
        "q_bulge": 0.85,
        "theta_bulge": 0.0,
        "re_disk": 7.0,
        "q_disk": 0.65,
        "theta_disk": 0.0,
        "x0": 12.0,
        "y0": 12.0,
        "sky": 0.0,
    }

    truth_model = make_bulge_disk_psf_sky_model(
        **truth,
        **fixed_parameters,
        psf=psf,
    )

    noiseless = truth_model.render(shape)

    data = noiseless + rng.normal(
        loc=0.0,
        scale=noise_sigma,
        size=shape,
    )

    likelihood = GaussianImageLikelihood(
        data=data,
        noise=np.full(
            shape,
            noise_sigma,
            dtype=float,
        ),
    )

    parameters = ParameterSet(
        [
            Parameter(
                name="flux_bulge",
                lower=100.0,
                upper=450.0,
            ),
            Parameter(
                name="flux_disk",
                lower=350.0,
                upper=850.0,
            ),
        ]
    )

    def model_factory(
        *,
        flux_bulge: float,
        flux_disk: float,
    ):
        return make_bulge_disk_psf_sky_model(
            flux_bulge=flux_bulge,
            flux_disk=flux_disk,
            **fixed_parameters,
            psf=psf,
        )

    problem = ImageInferenceProblem(
        model_factory=model_factory,
        parameters=parameters,
        likelihood=likelihood,
    )

    sampler = make_sampler(
        "dynesty",
        problem=problem,
    )

    result = sampler.run(
        nlive=20,
        sample="rwalk",
        dlogz=0.5,
        print_progress=False,
        seed=12345,
    )

    assert result.parameter_names == [
        "flux_bulge",
        "flux_disk",
    ]

    relative_tolerances = {
        "flux_bulge": 0.15,
        "flux_disk": 0.10,
    }

    for name, expected in truth.items():
        summary = result.summary(name)

        assert summary["median"] == pytest.approx(
            expected,
            rel=relative_tolerances[name],
        )

    assert np.isfinite(result.final_logz)
    assert result.logzerr is not None
    assert np.isfinite(result.logzerr[-1])
