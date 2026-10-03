"""Tests for correlated Gaussian integration in the real-fit pipeline."""

from types import SimpleNamespace

import json
import numpy as np
import pytest

from baymorph.fitting import Parameter, ParameterSet
from baymorph.fitting.priors import UniformPrior
from baymorph.likelihood import CorrelatedGaussianImageLikelihood
from baymorph.likelihood.vecchia_gaussian import (
    VecchiaCorrelatedGaussianImageLikelihood,
    build_vecchia_factorization,
)
from baymorph.pipeline.config import resolve_config
from baymorph.pipeline.real_fit import (
    _build_image_inference_problem,
    _compute_fit_statistics,
    load_real_fit_products,
    run_real_fit_from_config,
    save_real_fit_products,
)
from baymorph.results import RealFitProducts


def test_build_image_problem_passes_correlated_kernel():
    """Problem builder must pass fixed correlation state to the factory."""
    parameters = ParameterSet(
        [
            Parameter(
                "flux",
                prior=UniformPrior(
                    1.0,
                    2.0,
                ),
            )
        ]
    )

    kernel = np.array(
        [
            [0.01, 0.04, 0.01],
            [0.08, 1.00, 0.08],
            [0.01, 0.04, 0.01],
        ]
    )

    problem = _build_image_inference_problem(
        model_factory=lambda **kwargs: None,
        parameters=parameters,
        likelihood_cfg={
            "type": "correlated_gaussian",
            "include_normalization": False,
        },
        data_fit=np.zeros((4, 5)),
        noise_fit=np.ones((4, 5)),
        mask_fit=np.zeros(
            (4, 5),
            dtype=bool,
        ),
        correlation_kernel_fit=kernel,
    )

    assert isinstance(
        problem.likelihood,
        CorrelatedGaussianImageLikelihood,
    )

    np.testing.assert_array_equal(
        problem.likelihood.correlation_kernel,
        kernel,
    )


def test_correlated_fit_statistics_use_mahalanobis_quadratic():
    """Correlated statistics must use r.T C^-1 r, not diagonal chi2."""
    data = np.array(
        [
            [0.3, 0.9, -0.2],
            [1.1, 0.4, 0.6],
            [0.2, -0.1, 0.7],
        ]
    )

    model = np.zeros_like(
        data
    )

    noise = np.full_like(
        data,
        1.2,
    )

    mask = np.array(
        [
            [False, False, True],
            [False, False, False],
            [False, True, False],
        ]
    )

    kernel = np.array(
        [
            [0.01, 0.04, 0.01],
            [0.08, 1.00, 0.08],
            [0.01, 0.04, 0.01],
        ]
    )

    likelihood = CorrelatedGaussianImageLikelihood(
        data=data,
        noise=noise,
        correlation_kernel=kernel,
        mask=mask,
        include_normalization=False,
        cg_rtol=1.0e-12,
    )

    stats = _compute_fit_statistics(
        data=data,
        model_image=model,
        noise=noise,
        mask=mask,
        n_parameters=2,
        likelihood=likelihood,
    )

    assert stats["chi2"] == pytest.approx(
        likelihood.quadratic_form(
            model
        ),
        rel=1.0e-11,
        abs=1.0e-11,
    )

    assert stats["ndof"] == (
        likelihood.n_valid_pixels
        - 2
    )

    np.testing.assert_allclose(
        stats["norm_residual"][
            ~mask
        ],
        (
            (data - model)
            / noise
        )[
            ~mask
        ],
    )

    assert np.all(
        np.isnan(
            stats[
                "norm_residual"
            ][
                mask
            ]
        )
    )


def test_noise_model_fit_statistics_use_the_scale_of_ad047():
    """Residuals are standardized by sqrt(s^2 + max(m, 0) / g), not by s."""
    rng = np.random.default_rng(
        47
    )

    shape = (
        6,
        7,
    )

    yy, xx = np.indices(
        shape
    )

    source = 40.0 * np.exp(
        -0.5
        * (
            (xx - 3.0) ** 2
            + (yy - 2.5) ** 2
        )
        / 1.5**2
    )
    source[
        0,
        0,
    ] = -0.5

    background = 0.3
    model = source + background
    sky_scale = 1.2
    gain = 2.0

    data = model + rng.normal(
        0.0,
        sky_scale,
        shape,
    )

    noise = np.full(
        shape,
        sky_scale,
    )

    mask = np.zeros(
        shape,
        dtype=bool,
    )
    mask[
        5,
        6,
    ] = True

    kernel = np.array(
        [
            [0.01, 0.04, 0.01],
            [0.08, 1.00, 0.08],
            [0.01, 0.04, 0.01],
        ]
    )

    likelihood = VecchiaCorrelatedGaussianImageLikelihood(
        data=data,
        noise=noise,
        correlation_kernel=kernel,
        mask=mask,
        factorization=build_vecchia_factorization(
            mask,
            kernel,
            max_neighbors=24,
        ),
        max_neighbors=24,
        effective_gain=gain,
    )

    stats = _compute_fit_statistics(
        data=data,
        model_image=model,
        noise=noise,
        mask=mask,
        n_parameters=3,
        likelihood=likelihood,
        source_image=source,
    )

    scale = np.sqrt(
        sky_scale**2
        + np.clip(
            source,
            0.0,
            None,
        )
        / gain
    )

    np.testing.assert_allclose(
        stats["norm_residual"][
            ~mask
        ],
        (
            (data - model)
            / scale
        )[
            ~mask
        ],
        rtol=1.0e-12,
    )

    assert np.isnan(
        stats["norm_residual"][
            5,
            6,
        ]
    )

    assert scale.max() > 3.0 * sky_scale

    assert stats["chi2"] == pytest.approx(
        likelihood.quadratic_form(
            model,
            source,
        ),
        rel=1.0e-12,
    )

    assert stats["ndof"] == (
        likelihood.n_valid_pixels
        - 3
    )


def test_fit_persistence_preserves_exact_noise_and_likelihood_metadata(
    tmp_path,
    monkeypatch,
):
    """Saved fits must reload the exact statistical noise state."""
    config = resolve_config(
        {
            "object": {
                "name": "test",
            },
            "data": {
                "image_path": "image.fits",
            },
            "outputs": {
                "root": str(
                    tmp_path
                ),
            },
            "fit": {
                "models": [
                    "single_sersic",
                ],
            },
        }
    )

    fitted_noise = np.array(
        [
            [1.1, 1.2],
            [1.3, 1.4],
        ]
    )

    products = RealFitProducts(
        model_name="single_sersic",
        posterior=None,
        best_parameters={
            "flux": 1.0,
        },
        best_model_image=np.zeros(
            (2, 2)
        ),
        residual=np.zeros(
            (2, 2)
        ),
        norm_residual=np.zeros(
            (2, 2)
        ),
        chi2=2.5,
        ndof=3,
        reduced_chi2=2.5 / 3.0,
        noise=fitted_noise,
        likelihood_metadata={
            "type": (
                "correlated_gaussian"
            ),
            "fit_statistic": (
                "mahalanobis_chi2"
            ),
            "residual_diagnostic": (
                "marginal_standardized"
            ),
            "normalization_complete": False,
        },
    )

    monkeypatch.setattr(
        "baymorph.pipeline.real_fit."
        "load_image_preparation_products",
        lambda config: SimpleNamespace(
            noise=np.full(
                (2, 2),
                99.0,
            ),
        ),
    )

    save_real_fit_products(
        config,
        "single_sersic",
        products,
    )

    loaded = load_real_fit_products(
        config,
        "single_sersic",
    )

    np.testing.assert_array_equal(
        loaded.noise,
        fitted_noise,
    )

    assert (
        loaded.likelihood_metadata
        == products.likelihood_metadata
    )

    summary_path = (
        tmp_path
        / "test"
        / "fits"
        / "single_sersic"
        / "fit_summary.json"
    )

    summary = json.loads(
        summary_path.read_text()
    )

    assert (
        summary[
            "likelihood"
        ][
            "type"
        ]
        == "correlated_gaussian"
    )

    assert (
        summary[
            "likelihood"
        ][
            "normalization_complete"
        ]
        is False
    )


def test_run_real_fit_passes_prepared_likelihood_state(
    tmp_path,
    monkeypatch,
):
    """Real-fit execution must use one fixed prepared likelihood state."""
    from baymorph.pipeline.likelihood_state import (
        RealFitLikelihoodState,
    )

    resolved = {
        "object": {
            "name": "test",
        },
        "data": {
            "image_path": "missing.fits",
        },
        "outputs": {
            "root": str(
                tmp_path
                / "outputs"
            ),
            "overwrite": False,
        },
        "fit": {
            "models": [
                "single_sersic",
            ],
            "priors": {},
            "background": {
                "model": "constant",
                "priors": {},
            },
            "likelihood": {
                "type": (
                    "correlated_gaussian"
                ),
                "include_normalization": True,
            },
            "sampler": {
                "name": "nautilus",
            },
            "comparison": {
                "enabled": False,
                "metric": "bayes_factor",
            },
        },
    }

    image_products = SimpleNamespace(
        prepared_data=np.zeros(
            (2, 2),
            dtype=float,
        ),
        noise=np.ones(
            (2, 2),
            dtype=float,
        ),
        fitting_mask=np.zeros(
            (2, 2),
            dtype=bool,
        ),
        cutout_bounds={
            "xmin": 0,
            "xmax": 2,
            "ymin": 0,
            "ymax": 2,
        },
    )

    state = RealFitLikelihoodState(
        likelihood_type=(
            "correlated_gaussian"
        ),
        noise=np.full(
            (2, 2),
            1.5,
        ),
        correlation_kernel=np.ones(
            (1, 1)
        ),
        metadata={
            "type": (
                "correlated_gaussian"
            ),
            "normalization_complete": True,
        },
    )

    captured = {}

    monkeypatch.setattr(
        "baymorph.pipeline.real_fit.resolve_config",
        lambda config: resolved,
    )

    monkeypatch.setattr(
        "baymorph.pipeline.real_fit.validate_fit_priors",
        lambda config: None,
    )

    monkeypatch.setattr(
        "baymorph.pipeline.real_fit.make_output_tree",
        lambda config: None,
    )

    monkeypatch.setattr(
        "baymorph.pipeline.real_fit.save_fit_resolved_config",
        lambda *args, **kwargs: None,
    )

    monkeypatch.setattr(
        "baymorph.pipeline.real_fit."
        "load_image_preparation_products",
        lambda config: image_products,
    )

    monkeypatch.setattr(
        "baymorph.pipeline.real_fit._load_psf_from_config",
        lambda config: object(),
    )

    monkeypatch.setattr(
        "baymorph.pipeline.real_fit.Observation.from_fits",
        lambda path: None,
    )

    monkeypatch.setattr(
        "baymorph.pipeline.real_fit."
        "_target_reference_coordinates",
        lambda **kwargs: {
            "target_x": 1.0,
            "target_y": 1.0,
        },
    )

    def fake_prepare_state(
        config,
        image_products_arg,
    ):
        captured[
            "state_config"
        ] = config

        captured[
            "state_products"
        ] = image_products_arg

        return state

    monkeypatch.setattr(
        "baymorph.pipeline.real_fit."
        "prepare_real_fit_likelihood_state",
        fake_prepare_state,
    )

    products = RealFitProducts(
        model_name="single_sersic",
        posterior=None,
        best_parameters={},
        best_model_image=np.zeros(
            (2, 2)
        ),
        residual=np.zeros(
            (2, 2)
        ),
        norm_residual=np.zeros(
            (2, 2)
        ),
        chi2=1.0,
        ndof=1,
        reduced_chi2=1.0,
        noise=state.noise,
    )

    def fake_execute(
        **kwargs,
    ):
        captured[
            "execute"
        ] = kwargs

        return products

    monkeypatch.setattr(
        "baymorph.pipeline.real_fit."
        "_execute_single_model_fit",
        fake_execute,
    )

    monkeypatch.setattr(
        "baymorph.pipeline.real_fit."
        "save_real_fit_products",
        lambda *args, **kwargs: None,
    )

    monkeypatch.setattr(
        "baymorph.pipeline.real_fit."
        "_resolved_boundary_priors",
        lambda **kwargs: {},
    )

    monkeypatch.setattr(
        "baymorph.pipeline.real_fit."
        "_write_fit_completion_marker",
        lambda *args, **kwargs: None,
    )

    result = run_real_fit_from_config(
        {}
    )

    assert result is products

    assert (
        captured[
            "state_config"
        ]
        is resolved
    )

    assert (
        captured[
            "state_products"
        ]
        is image_products
    )

    assert (
        captured[
            "execute"
        ][
            "likelihood_state"
        ]
        is state
    )
