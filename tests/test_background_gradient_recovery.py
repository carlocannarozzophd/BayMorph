"""Controlled recovery test for residual background gradients."""

from __future__ import annotations

import numpy as np
import pytest

from baymorph.fitting import (
    Parameter,
    ParameterSet,
    make_sampler,
)
from baymorph.fitting.image_problem import ImageInferenceProblem
from baymorph.likelihood import GaussianImageLikelihood
from baymorph.pipeline.real_fit import _make_model_factory
from baymorph.simulation.psf import GaussianPSF


def _residual_plane_coefficients(
    residual: np.ndarray,
) -> tuple[float, float, float]:
    """Fit a diagnostic plane to one residual image."""
    ny, nx = residual.shape

    y, x = np.indices(
        residual.shape,
        dtype=float,
    )

    x -= 0.5 * (nx - 1)
    y -= 0.5 * (ny - 1)

    design = np.column_stack(
        (
            np.ones(
                residual.size,
                dtype=float,
            ),
            x.ravel(),
            y.ravel(),
        )
    )

    coefficients, *_ = np.linalg.lstsq(
        design,
        residual.ravel(),
        rcond=None,
    )

    return tuple(
        float(value)
        for value in coefficients
    )


def _fit_problem(
    *,
    model_factory,
    parameters: ParameterSet,
    data: np.ndarray,
    noise_sigma: float,
    seed: int,
):
    """Run one controlled Dynesty image fit."""
    likelihood = GaussianImageLikelihood(
        data=data,
        noise=np.full(
            data.shape,
            noise_sigma,
            dtype=float,
        ),
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
                "nlive": 30,
                "sample": "rwalk",
                "rstate": np.random.default_rng(
                    seed
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

    return backend.fit()


def test_plane_background_recovers_gradient_and_removes_residual_structure():
    """A fitted plane must recover a real gradient that a constant cannot."""
    rng = np.random.default_rng(
        20260817
    )

    shape = (
        25,
        25,
    )

    noise_sigma = 0.5

    psf = GaussianPSF(
        fwhm=2.5
    )

    fixed_morphology = {
        "n": 2.5,
        "q": 0.8,
        "theta": 0.2,
        "x0": 12.0,
        "y0": 12.0,
    }

    truth = {
        "flux": 500.0,
        "re": 4.0,
        "background_level": 1.5,
        "background_gradient_x": 0.08,
        "background_gradient_y": -0.05,
    }

    truth_factory = _make_model_factory(
        "single_sersic",
        psf,
        background_cfg={
            "model": "plane",
        },
    )

    truth_model = truth_factory(
        **fixed_morphology,
        **truth,
    )

    noiseless = truth_model.render(
        shape
    )

    data = (
        noiseless
        + rng.normal(
            loc=0.0,
            scale=noise_sigma,
            size=shape,
        )
    )

    plane_factory = _make_model_factory(
        "single_sersic",
        psf,
        background_cfg={
            "model": "plane",
        },
    )

    def plane_model_factory(
        **parameters,
    ):
        return plane_factory(
            **fixed_morphology,
            **parameters,
        )

    plane_parameters = ParameterSet(
        [
            Parameter(
                "flux",
                350.0,
                650.0,
            ),
            Parameter(
                "re",
                3.0,
                5.0,
            ),
            Parameter(
                "background_level",
                -2.0,
                5.0,
            ),
            Parameter(
                "background_gradient_x",
                -0.15,
                0.15,
            ),
            Parameter(
                "background_gradient_y",
                -0.15,
                0.15,
            ),
        ]
    )

    plane_result = _fit_problem(
        model_factory=plane_model_factory,
        parameters=plane_parameters,
        data=data,
        noise_sigma=noise_sigma,
        seed=12345,
    )

    assert plane_result.summary(
        "flux"
    )["median"] == pytest.approx(
        truth["flux"],
        rel=0.10,
    )

    assert plane_result.summary(
        "re"
    )["median"] == pytest.approx(
        truth["re"],
        rel=0.10,
    )

    assert plane_result.summary(
        "background_level"
    )["median"] == pytest.approx(
        truth["background_level"],
        abs=0.30,
    )

    assert plane_result.summary(
        "background_gradient_x"
    )["median"] == pytest.approx(
        truth["background_gradient_x"],
        abs=0.02,
    )

    assert plane_result.summary(
        "background_gradient_y"
    )["median"] == pytest.approx(
        truth["background_gradient_y"],
        abs=0.02,
    )

    best_plane_model = plane_model_factory(
        **plane_result.best_parameters
    )

    plane_residual = (
        data
        - best_plane_model.render(
            shape
        )
    )

    (
        _,
        plane_residual_gradient_x,
        plane_residual_gradient_y,
    ) = _residual_plane_coefficients(
        plane_residual
    )

    constant_factory = _make_model_factory(
        "single_sersic",
        psf,
        background_cfg={
            "model": "constant",
        },
    )

    def constant_model_factory(
        **parameters,
    ):
        return constant_factory(
            **fixed_morphology,
            **parameters,
        )

    constant_parameters = ParameterSet(
        [
            Parameter(
                "flux",
                350.0,
                650.0,
            ),
            Parameter(
                "re",
                3.0,
                5.0,
            ),
            Parameter(
                "background_level",
                -2.0,
                5.0,
            ),
        ]
    )

    constant_result = _fit_problem(
        model_factory=constant_model_factory,
        parameters=constant_parameters,
        data=data,
        noise_sigma=noise_sigma,
        seed=54321,
    )

    best_constant_model = constant_model_factory(
        **constant_result.best_parameters
    )

    constant_residual = (
        data
        - best_constant_model.render(
            shape
        )
    )

    (
        _,
        constant_residual_gradient_x,
        constant_residual_gradient_y,
    ) = _residual_plane_coefficients(
        constant_residual
    )

    plane_chi2 = float(
        np.sum(
            (
                plane_residual
                / noise_sigma
            )
            ** 2
        )
    )

    constant_chi2 = float(
        np.sum(
            (
                constant_residual
                / noise_sigma
            )
            ** 2
        )
    )

    assert abs(
        plane_residual_gradient_x
    ) < 0.02

    assert abs(
        plane_residual_gradient_y
    ) < 0.02

    assert abs(
        constant_residual_gradient_x
    ) > 0.04

    assert abs(
        constant_residual_gradient_y
    ) > 0.025

    assert plane_chi2 < (
        0.75
        * constant_chi2
    )
