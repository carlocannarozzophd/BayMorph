"""Tests for the production multi-band fitting workflow."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from baymorph.fitting.multi_observation import (
    MultiObservationParameterLayout,
    MultiObservationParameterSpec,
)
from baymorph.fitting.parameters import Parameter
from baymorph.pipeline import multi_observation_fit


class DummyBackend:
    """Minimal backend recording the received fit request."""

    def __init__(
        self,
        result,
    ):
        self.result = result
        self.fit_kwargs = None

    def fit(
        self,
        **kwargs,
    ):
        self.fit_kwargs = kwargs
        return self.result


def _layout():
    return MultiObservationParameterLayout(
        observations=(
            "g",
            "r",
        ),
        specifications=[
            MultiObservationParameterSpec.shared_parameter(
                Parameter(
                    "re",
                    lower=1.0,
                    upper=10.0,
                )
            ),
            MultiObservationParameterSpec.independent(
                "flux",
                {
                    "g": Parameter(
                        "flux",
                        lower=10.0,
                        upper=100.0,
                    ),
                    "r": Parameter(
                        "flux",
                        lower=20.0,
                        upper=200.0,
                    ),
                },
            ),
            MultiObservationParameterSpec.shared_parameter(
                Parameter(
                    "n",
                    lower=0.5,
                    upper=6.0,
                )
            ),
            MultiObservationParameterSpec.shared_parameter(
                Parameter(
                    "q",
                    lower=0.2,
                    upper=1.0,
                )
            ),
            MultiObservationParameterSpec.shared_parameter(
                Parameter(
                    "theta",
                    lower=-1.57,
                    upper=1.57,
                )
            ),
            MultiObservationParameterSpec.shared_parameter(
                Parameter(
                    "x0",
                    lower=2.0,
                    upper=8.0,
                )
            ),
            MultiObservationParameterSpec.shared_parameter(
                Parameter(
                    "y0",
                    lower=2.0,
                    upper=8.0,
                )
            ),
            MultiObservationParameterSpec.independent(
                "background_level",
                {
                    "g": Parameter(
                        "background_level",
                        lower=-5.0,
                        upper=5.0,
                    ),
                    "r": Parameter(
                        "background_level",
                        lower=-5.0,
                        upper=5.0,
                    ),
                },
            ),
        ],
    )


def _products(
    shape=(11, 11),
):
    return SimpleNamespace(
        prepared_data=np.zeros(
            shape,
            dtype=float,
        ),
        noise=np.ones(
            shape,
            dtype=float,
        ),
        fitting_mask=np.zeros(
            shape,
            dtype=bool,
        ),
        target_x=(
            shape[1] - 1
        )
        / 2.0,
        target_y=(
            shape[0] - 1
        )
        / 2.0,
    )


def _config():
    return {
        "observations": {
            "g": {
                "config_path": "g.yaml",
            },
            "r": {
                "config_path": "r.yaml",
            },
        },
        "fit": {
            "models": [
                "single_sersic",
            ],
            # A complete likelihood without a calibration keeps these tests on
            # the joint mechanics; the noise model of AD-047 has its own tests.
            "likelihood": {
                "type": "poisson_gaussian",
            },
            "sampler": {
                "name": "dynesty",
                "nlive": 50,
                "sample": "auto",
                "seed": 4,
                "n_processes": 1,
            },
            "multi_observation": {
                "parameters": {},
            },
        },
    }


def test_run_multi_observation_inference_builds_problem_and_returns_result(
    monkeypatch,
):
    expected_result = SimpleNamespace(diagnostics={})
    backend = DummyBackend(
        expected_result
    )

    observation_configs = {
        "g": {
            "fit": {},
        },
        "r": {
            "fit": {},
        },
    }

    monkeypatch.setattr(
        multi_observation_fit,
        "resolve_multi_observation_config",
        lambda config, base_dir: (
            observation_configs,
            _layout(),
        ),
    )

    monkeypatch.setattr(
        multi_observation_fit,
        "load_image_preparation_products",
        lambda config: _products(),
    )

    monkeypatch.setattr(
        multi_observation_fit,
        "_load_psf_from_config",
        lambda config: np.ones(
            (3, 3),
            dtype=float,
        )
        / 9.0,
    )

    captured = {}

    def fake_make_sampler(
        config,
        *,
        problem,
    ):
        captured["config"] = config
        captured["problem"] = problem
        return backend

    monkeypatch.setattr(
        multi_observation_fit,
        "make_sampler",
        fake_make_sampler,
    )

    result = (
        multi_observation_fit
        .run_multi_observation_inference_from_config(
            _config()
        )
    )

    assert result is expected_result
    assert captured["problem"].parameter_layout.observations == (
        "g",
        "r",
    )
    assert captured["problem"].likelihood.observations == (
        "g",
        "r",
    )
    assert captured["problem"].representation_factory.observations == (
        "g",
        "r",
    )

    assert backend.fit_kwargs == {
        "progress_label": "multi_observation_single_sersic",
    }


def test_run_multi_observation_inference_supports_different_observation_shapes(
    monkeypatch,
):
    observation_configs = {
        "g": {},
        "r": {},
    }

    monkeypatch.setattr(
        multi_observation_fit,
        "resolve_multi_observation_config",
        lambda config, base_dir: (
            observation_configs,
            _layout(),
        ),
    )

    products = {
        id(
            observation_configs["g"]
        ): _products(
            (11, 11)
        ),
        id(
            observation_configs["r"]
        ): _products(
            (13, 17)
        ),
    }

    monkeypatch.setattr(
        multi_observation_fit,
        "load_image_preparation_products",
        lambda config: products[
            id(
                config
            )
        ],
    )

    monkeypatch.setattr(
        multi_observation_fit,
        "_load_psf_from_config",
        lambda config: np.ones(
            (3, 3),
            dtype=float,
        )
        / 9.0,
    )

    expected_result = SimpleNamespace(diagnostics={})
    backend = DummyBackend(
        expected_result
    )
    captured = {}

    def fake_make_sampler(
        config,
        *,
        problem,
    ):
        captured["problem"] = problem
        return backend

    monkeypatch.setattr(
        multi_observation_fit,
        "make_sampler",
        fake_make_sampler,
    )

    result = (
        multi_observation_fit
        .run_multi_observation_inference_from_config(
            _config()
        )
    )

    assert result is expected_result

    problem = captured["problem"]

    assert problem.likelihood.image_shapes == {
        "g": (11, 11),
        "r": (13, 17),
    }


def test_build_sky_parameter_transforms_uses_each_cutout_wcs(
    monkeypatch,
):
    from astropy.wcs import WCS

    observation_configs = {
        "g": {
            "data": {
                "image_path": "g.fits",
            },
        },
        "r": {
            "data": {
                "image_path": "r.fits",
            },
        },
    }

    products = {
        "g": SimpleNamespace(
            cutout_bounds={
                "xmin": 40,
                "xmax": 160,
                "ymin": 55,
                "ymax": 175,
            },
        ),
        "r": SimpleNamespace(
            cutout_bounds={
                "xmin": 20,
                "xmax": 120,
                "ymin": 30,
                "ymax": 130,
            },
        ),
    }

    def make_wcs(scale):
        wcs = WCS(naxis=2)
        wcs.wcs.crpix = [
            101.0,
            101.0,
        ]
        wcs.wcs.crval = [
            150.0,
            30.0,
        ]
        wcs.wcs.ctype = [
            "RA---TAN",
            "DEC--TAN",
        ]
        wcs.wcs.cdelt = np.array(
            [
                -scale / 3600.0,
                scale / 3600.0,
            ]
        )
        return wcs

    observations = {
        "g.fits": SimpleNamespace(
            metadata=SimpleNamespace(
                wcs=make_wcs(
                    0.2
                )
            )
        ),
        "r.fits": SimpleNamespace(
            metadata=SimpleNamespace(
                wcs=make_wcs(
                    0.4
                )
            )
        ),
    }

    monkeypatch.setattr(
        multi_observation_fit.Observation,
        "from_fits",
        lambda path: observations[path],
    )

    transforms = (
        multi_observation_fit
        ._build_sky_parameter_transforms(
            observation_configs=observation_configs,
            image_products=products,
            model_name="single_sersic",
        )
    )

    assert set(transforms) == {
        "g",
        "r",
    }

    assert transforms["g"] is not transforms["r"]

    g_x, g_y = (
        transforms["g"]
        .wcs.world_to_pixel_values(
            150.0,
            30.0,
        )
    )
    r_x, r_y = (
        transforms["r"]
        .wcs.world_to_pixel_values(
            150.0,
            30.0,
        )
    )

    assert g_x == pytest.approx(
        60.0
    )
    assert g_y == pytest.approx(
        45.0
    )

    assert r_x == pytest.approx(
        80.0
    )
    assert r_y == pytest.approx(
        70.0
    )


def test_build_sky_parameter_transforms_requires_wcs(
    monkeypatch,
):
    observation_configs = {
        "g": {
            "data": {
                "image_path": "g.fits",
            },
        },
    }

    products = {
        "g": SimpleNamespace(
            cutout_bounds={
                "xmin": 0,
                "xmax": 20,
                "ymin": 0,
                "ymax": 20,
            },
        ),
    }

    monkeypatch.setattr(
        multi_observation_fit.Observation,
        "from_fits",
        lambda path: SimpleNamespace(
            metadata=SimpleNamespace(
                wcs=None
            )
        ),
    )

    with pytest.raises(
        ValueError,
        match="valid celestial WCS",
    ):
        (
            multi_observation_fit
            ._build_sky_parameter_transforms(
                observation_configs=observation_configs,
                image_products=products,
                model_name="single_sersic",
            )
        )


def test_build_sky_parameter_transforms_requires_cutout_bounds(
    monkeypatch,
):
    from astropy.wcs import WCS

    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [
        11.0,
        11.0,
    ]
    wcs.wcs.crval = [
        150.0,
        30.0,
    ]
    wcs.wcs.ctype = [
        "RA---TAN",
        "DEC--TAN",
    ]
    wcs.wcs.cdelt = np.array(
        [
            -0.2 / 3600.0,
            0.2 / 3600.0,
        ]
    )

    observation_configs = {
        "g": {
            "data": {
                "image_path": "g.fits",
            },
        },
    }

    products = {
        "g": SimpleNamespace(
            cutout_bounds=None,
        ),
    }

    monkeypatch.setattr(
        multi_observation_fit.Observation,
        "from_fits",
        lambda path: SimpleNamespace(
            metadata=SimpleNamespace(
                wcs=wcs
            )
        ),
    )

    with pytest.raises(
        ValueError,
        match="saved cutout bounds",
    ):
        (
            multi_observation_fit
            ._build_sky_parameter_transforms(
                observation_configs=observation_configs,
                image_products=products,
                model_name="single_sersic",
            )
        )


def test_multi_observation_fit_builds_joint_and_observation_products(
    monkeypatch,
):
    """Production joint fitting should preserve one posterior and many grids."""
    from baymorph.results import (
        InferenceResult,
        MultiObservationFitProducts,
        PosteriorSamples,
    )

    observation_configs = {
        "g": {},
        "r": {},
    }

    layout = _layout()

    monkeypatch.setattr(
        multi_observation_fit,
        "resolve_multi_observation_config",
        lambda config, base_dir: (
            observation_configs,
            layout,
        ),
    )

    products = {
        id(
            observation_configs["g"]
        ): _products(
            (11, 11)
        ),
        id(
            observation_configs["r"]
        ): _products(
            (13, 17)
        ),
    }

    monkeypatch.setattr(
        multi_observation_fit,
        "load_image_preparation_products",
        lambda config: products[
            id(
                config
            )
        ],
    )

    monkeypatch.setattr(
        multi_observation_fit,
        "_load_psf_from_config",
        lambda config: np.ones(
            (3, 3),
            dtype=float,
        )
        / 9.0,
    )

    class ConstantModel:
        def __init__(
            self,
            **kwargs,
        ):
            del kwargs

        def render(
            self,
            shape,
        ):
            return np.ones(
                shape,
                dtype=float,
            )

    monkeypatch.setattr(
        multi_observation_fit,
        "ObservedModelFactory",
        lambda model_name, **options: ConstantModel,
    )

    names = layout.parameter_set.names

    values_by_name = {
        "re": 5.0,
        "flux__g": 50.0,
        "flux__r": 80.0,
        "n": 2.0,
        "q": 0.7,
        "theta": 0.0,
        "x0": 5.0,
        "y0": 5.0,
        "background_level__g": 0.0,
        "background_level__r": 0.0,
    }

    sample = np.asarray(
        [
            values_by_name[
                name
            ]
            for name in names
        ],
        dtype=float,
    )

    inference_result = InferenceResult(
        posterior=PosteriorSamples(
            samples=sample[
                None,
                :,
            ],
            weights=np.asarray(
                [
                    1.0,
                ]
            ),
            parameter_names=list(
                names
            ),
            log_likelihood=np.asarray(
                [
                    -1.0,
                ]
            ),
        ),
        diagnostics={},
        provenance={
            "backend": "test",
        },
    )

    backend = DummyBackend(
        inference_result
    )

    monkeypatch.setattr(
        multi_observation_fit,
        "make_sampler",
        lambda config, *, problem: backend,
    )

    result = (
        multi_observation_fit
        .run_multi_observation_fit_from_config(
            _config()
        )
    )

    assert isinstance(
        result,
        MultiObservationFitProducts,
    )

    assert result.posterior is inference_result

    assert tuple(
        result.observations
    ) == (
        "g",
        "r",
    )

    assert result.observations[
        "g"
    ].best_model_image.shape == (
        11,
        11,
    )

    assert result.observations[
        "r"
    ].best_model_image.shape == (
        13,
        17,
    )

    assert result.observations[
        "g"
    ].n_valid_pixels == 121

    assert result.observations[
        "r"
    ].n_valid_pixels == 221

    assert result.observations[
        "g"
    ].chi2 == pytest.approx(
        121.0
    )

    assert result.observations[
        "r"
    ].chi2 == pytest.approx(
        221.0
    )

    assert result.chi2 == pytest.approx(
        342.0
    )

    assert result.ndof == (
        342
        - layout.ndim
    )

    assert result.reduced_chi2 == pytest.approx(
        result.chi2
        / result.ndof
    )

    assert result.best_parameters == (
        inference_result.best_parameters
    )


def test_multi_observation_uses_observation_specific_likelihood_state(
    monkeypatch,
):
    """Joint inference must preserve each observation likelihood model."""
    observation_configs = {
        "g": {
            "fit": {
                "likelihood": {
                    "type": "gaussian",
                },
            },
        },
        "r": {
            "fit": {
                "likelihood": {
                    "type": "correlated_gaussian",
                    "include_normalization": False,
                },
            },
        },
    }

    monkeypatch.setattr(
        multi_observation_fit,
        "resolve_multi_observation_config",
        lambda config, base_dir: (
            observation_configs,
            _layout(),
        ),
    )

    products = {
        "g": _products(),
        "r": _products(),
    }

    for product in products.values():
        product.target_x = 5.0
        product.target_y = 5.0

    monkeypatch.setattr(
        multi_observation_fit,
        "load_image_preparation_products",
        lambda config: (
            products["g"]
            if config is observation_configs["g"]
            else products["r"]
        ),
    )

    monkeypatch.setattr(
        multi_observation_fit,
        "_load_psf_from_config",
        lambda config: np.ones(
            (3, 3),
            dtype=float,
        )
        / 9.0,
    )

    kernel = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=float,
    )

    def fake_prepare_likelihood_state(
        config,
        image_products,
    ):
        likelihood_type = config["fit"]["likelihood"]["type"]

        return SimpleNamespace(
            noise=np.ones(
                np.asarray(
                    image_products.prepared_data
                ).shape,
                dtype=float,
            ),
            correlation_kernel=(
                kernel
                if likelihood_type == "correlated_gaussian"
                else None
            ),
            vecchia_factorization=None,
            effective_gain=None,
        )

    monkeypatch.setattr(
        multi_observation_fit,
        "prepare_real_fit_likelihood_state",
        fake_prepare_likelihood_state,
    )

    problem = (
        multi_observation_fit
        .build_multi_observation_problem_from_config(
            _config()
        )
    )

    assert (
        problem.likelihood._likelihoods["g"].__class__.__name__
        == "GaussianImageLikelihood"
    )

    assert (
        problem.likelihood._likelihoods["r"].__class__.__name__
        == "CorrelatedGaussianImageLikelihood"
    )


def test_multi_observation_products_use_likelihood_consistent_statistics():
    """Joint products must use each observation's actual likelihood state."""
    from baymorph.likelihood import CorrelatedGaussianImageLikelihood
    from baymorph.results import (
        InferenceResult,
        PosteriorSamples,
    )

    data = np.asarray(
        [
            [1.0, 2.0, 0.5],
            [0.0, 1.5, 0.0],
            [0.5, 0.0, 1.0],
        ],
        dtype=float,
    )
    model_image = np.zeros_like(
        data
    )
    likelihood_noise = np.full_like(
        data,
        2.0,
    )
    preparation_noise = np.full_like(
        data,
        10.0,
    )
    mask = np.zeros_like(
        data,
        dtype=bool,
    )
    correlation_kernel = np.asarray(
        [
            [0.0, 0.05, 0.0],
            [0.05, 1.0, 0.05],
            [0.0, 0.05, 0.0],
        ],
        dtype=float,
    )

    likelihood = CorrelatedGaussianImageLikelihood(
        data=data,
        noise=likelihood_noise,
        correlation_kernel=correlation_kernel,
        mask=mask,
        include_normalization=False,
        cg_rtol=1.0e-12,
    )

    likelihood_metadata = {
        "type": "correlated_gaussian",
        "normalization_complete": False,
        "fit_statistic": "mahalanobis_chi2",
        "residual_diagnostic": "marginal_standardized",
    }

    problem = SimpleNamespace(
        parameter_layout=SimpleNamespace(
            observations=(
                "g",
            ),
        ),
        ndim=1,
        representation_factory=SimpleNamespace(
            parameter_transforms={},
        ),
        render_models=lambda sample: {
            "g": model_image,
        },
        render_predictions=lambda sample: (
            {"g": model_image},
            {"g": model_image},
        ),
        routed_parameters=lambda sample: {
            "g": {
                "flux": float(
                    sample[0]
                ),
            },
        },
    )

    context = SimpleNamespace(
        problem=problem,
        image_products={
            "g": SimpleNamespace(
                prepared_data=data,
                noise=preparation_noise,
                fitting_mask=mask,
            ),
        },
        likelihoods={
            "g": likelihood,
        },
        likelihood_states={
            "g": SimpleNamespace(
                noise=likelihood_noise,
                metadata=likelihood_metadata,
            ),
        },
    )

    inference_result = InferenceResult(
        posterior=PosteriorSamples(
            samples=np.asarray(
                [
                    [1.0],
                ],
                dtype=float,
            ),
            weights=np.asarray(
                [
                    1.0,
                ],
                dtype=float,
            ),
            parameter_names=[
                "flux__g",
            ],
            log_likelihood=np.asarray(
                [
                    -1.0,
                ],
                dtype=float,
            ),
        ),
        diagnostics={},
        provenance={
            "backend": "test",
        },
    )

    products = (
        multi_observation_fit
        ._build_multi_observation_fit_products(
            context=context,
            inference_result=inference_result,
            model_name="single_sersic",
        )
    )

    observation = products.observations["g"]

    assert observation.chi2 == pytest.approx(
        likelihood.quadratic_form(
            model_image
        ),
        rel=1.0e-11,
        abs=1.0e-11,
    )

    np.testing.assert_allclose(
        observation.noise,
        likelihood_noise,
    )
    np.testing.assert_allclose(
        observation.norm_residual,
        data / likelihood_noise,
    )

    assert observation.likelihood_metadata == likelihood_metadata
    assert products.chi2 == pytest.approx(
        observation.chi2
    )
    assert products.ndof == (
        likelihood.n_valid_pixels
        - problem.ndim
    )
