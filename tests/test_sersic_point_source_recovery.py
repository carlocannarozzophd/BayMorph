"""Controlled recovery test for Sérsic plus point-source inference."""

from __future__ import annotations

import numpy as np
import pytest

from baymorph.fitting import (
    Parameter,
    ParameterSet,
    make_sampler,
    make_sersic_point_source_psf_sky_model,
)
from baymorph.fitting.image_problem import ImageInferenceProblem
from baymorph.likelihood import GaussianImageLikelihood
from baymorph.simulation.psf import GaussianPSF


def test_sersic_point_source_recovers_extended_and_unresolved_fluxes() -> None:
    """A controlled fit should recover the principal source parameters."""
    rng = np.random.default_rng(20260806)

    shape = (21, 21)
    noise_sigma = 1.5
    psf = GaussianPSF(fwhm=2.5)

    truth = {
        "flux_sersic": 400.0,
        "re": 3.5,
        "flux_point_source": 80.0,
    }

    fixed_parameters = {
        "n": 2.0,
        "q": 0.8,
        "theta": 0.0,
        "x0": 10.0,
        "y0": 10.0,
        "sky": 0.0,
    }

    truth_model = make_sersic_point_source_psf_sky_model(
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
                name="flux_sersic",
                lower=200.0,
                upper=650.0,
            ),
            Parameter(
                name="re",
                lower=2.0,
                upper=6.0,
            ),
            Parameter(
                name="flux_point_source",
                lower=10.0,
                upper=180.0,
            ),
        ]
    )

    def model_factory(
        *,
        flux_sersic: float,
        re: float,
        flux_point_source: float,
    ):
        return make_sersic_point_source_psf_sky_model(
            flux_sersic=flux_sersic,
            re=re,
            flux_point_source=flux_point_source,
            **fixed_parameters,
            psf=psf,
        )

    problem = ImageInferenceProblem(
        model_factory=model_factory,
        parameters=parameters,
        likelihood=likelihood,
    )

    backend = make_sampler(
        {
            "name": "dynesty",
            "mode": "static",
            "init": {
                "nlive": 20,
                "sample": "rwalk",
                "rstate": np.random.default_rng(
                    12345
                ),
            },
            "execution": {
                "processes": 1,
                "progress": False,
            },
            "run": {
                "dlogz": 0.5,
            },
        },
        problem=problem,
    )

    result = backend.fit()

    assert result.parameter_names == [
        "flux_sersic",
        "re",
        "flux_point_source",
    ]

    relative_tolerances = {
        "flux_sersic": 0.15,
        "re": 0.15,
        "flux_point_source": 0.25,
    }

    for name, expected in truth.items():
        summary = result.summary(name)

        assert summary["median"] == pytest.approx(
            expected,
            rel=relative_tolerances[name],
        )

        assert (
            summary["q16"]
            < expected
            < summary["q84"]
        )

    assert result.logz is not None
    assert np.isfinite(result.logz)
    assert result.logz_error is not None
    assert np.isfinite(result.logz_error)
