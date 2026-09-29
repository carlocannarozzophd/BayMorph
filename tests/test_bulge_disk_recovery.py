"""Controlled recovery test for bulge-disc inference."""

from __future__ import annotations

import numpy as np

from baymorph.fitting import (
    Parameter,
    ParameterSet,
    make_bulge_disk_psf_model,
    make_sampler,
)
from baymorph.fitting.image_problem import ImageInferenceProblem
from baymorph.likelihood import GaussianImageLikelihood
from baymorph.simulation.psf import GaussianPSF


def test_bulge_disk_sampler_recovers_linear_gaussian_posterior() -> None:
    """Recover the analytic posterior geometry of a controlled linear fit."""
    rng = np.random.default_rng(20260806)

    shape = (25, 25)
    noise_sigma = 1.5
    psf = GaussianPSF(fwhm=2.5)

    truth = np.asarray(
        [
            250.0,
            600.0,
        ],
        dtype=float,
    )

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
    }

    def render(
        flux_bulge: float,
        flux_disk: float,
    ) -> np.ndarray:
        model = make_bulge_disk_psf_model(
            flux_bulge=flux_bulge,
            flux_disk=flux_disk,
            **fixed_parameters,
            psf=psf,
        )
        return model.render(shape)

    noiseless = render(
        truth[0],
        truth[1],
    )

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
        return make_bulge_disk_psf_model(
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

    base = render(
        1.0,
        1.0,
    )

    bulge_template = (
        render(2.0, 1.0)
        - base
    )
    disk_template = (
        render(1.0, 2.0)
        - base
    )

    reconstructed_truth = (
        truth[0] * bulge_template
        + truth[1] * disk_template
    )

    np.testing.assert_allclose(
        reconstructed_truth,
        noiseless,
        rtol=1.0e-12,
        atol=1.0e-12,
    )

    design = np.column_stack(
        (
            bulge_template.ravel(),
            disk_template.ravel(),
        )
    )

    precision = (
        design.T @ design
        / noise_sigma**2
    )
    covariance = np.linalg.inv(
        precision
    )

    analytic_mean = covariance @ (
        design.T @ data.ravel()
        / noise_sigma**2
    )

    delta_truth = (
        analytic_mean - truth
    )

    truth_mahalanobis_sq = float(
        delta_truth.T
        @ precision
        @ delta_truth
    )

    # For two fitted parameters, Delta chi^2 = 5.991 corresponds to
    # the 95% joint Gaussian confidence region.
    assert truth_mahalanobis_sq < 5.991

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
        "flux_bulge",
        "flux_disk",
    ]

    sampled_location = np.asarray(
        [
            result.summary(
                "flux_bulge"
            )["median"],
            result.summary(
                "flux_disk"
            )["median"],
        ],
        dtype=float,
    )

    delta_sampler = (
        sampled_location
        - analytic_mean
    )

    sampler_mahalanobis_sq = float(
        delta_sampler.T
        @ precision
        @ delta_sampler
    )

    assert sampler_mahalanobis_sq < 1.0

    assert result.logz is not None
    assert np.isfinite(result.logz)

    assert result.logz_error is not None
    assert np.isfinite(result.logz_error)
