"""Tests for real-fit likelihood-state preparation."""

from types import SimpleNamespace

import numpy as np
import pytest

from baymorph.pipeline.likelihood_state import (
    prepare_real_fit_likelihood_state,
    require_complete_likelihood_normalization,
)


def _image_products(
    *,
    shape=(3, 4),
    noise=None,
):
    """Return minimal prepared observation products."""
    if noise is None:
        noise = np.full(
            shape,
            2.0,
            dtype=float,
        )

    return SimpleNamespace(
        prepared_data=np.zeros(
            shape,
            dtype=float,
        ),
        noise=np.asarray(
            noise,
            dtype=float,
        ),
        fitting_mask=np.zeros(
            shape,
            dtype=bool,
        ),
    )


def _calibration():
    """Return a deterministic sky-correlation calibration of AD-047."""
    return SimpleNamespace(
        correlation_kernel=np.array(
            [
                [0.0, 0.1, 0.0],
                [0.2, 1.0, 0.2],
                [0.0, 0.1, 0.0],
            ]
        ),
        metadata={
            "schema_version": 2,
            "factorization": "C = S R S",
            "correlation_model": "stationary_sky_autocovariance",
            "estimator": "masked_sky_autocovariance_v1",
            "max_lag": 1,
        },
    )


def test_independent_gaussian_reuses_prepared_noise():
    """Independent Gaussian state must preserve the prepared noise map."""
    config = {
        "fit": {
            "likelihood": {
                "type": "gaussian",
            },
        },
    }

    products = _image_products()

    state = prepare_real_fit_likelihood_state(
        config,
        products,
    )

    assert state.likelihood_type == "gaussian"

    np.testing.assert_array_equal(
        state.noise,
        products.noise,
    )

    assert state.correlation_kernel is None

    assert (
        state.metadata[
            "normalization_complete"
        ]
        is True
    )


def _correlated_config(**likelihood):
    """Return the configuration of an unnormalized correlated likelihood."""
    return {
        "fit": {
            "likelihood": {
                "type": "correlated_gaussian",
                "include_normalization": False,
                "effective_gain": {"source": "absent"},
                **likelihood,
            },
        },
    }


def _install_calibration(monkeypatch, calibration):
    """Serve ``calibration`` as the persisted, source-validated product."""
    monkeypatch.setattr(
        "baymorph.pipeline.likelihood_state."
        "load_correlated_noise_calibration",
        lambda config: calibration,
    )
    monkeypatch.setattr(
        "baymorph.pipeline.likelihood_state."
        "correlated_noise_calibration_path",
        lambda config: "correlated_gaussian_noise.npz",
    )
    monkeypatch.setattr(
        "baymorph.pipeline.likelihood_state."
        "validate_correlated_noise_calibration_sources",
        lambda *args, **kwargs: None,
    )


def test_correlated_gaussian_uses_the_prepared_noise_map(
    monkeypatch,
):
    """AD-047 takes the marginal scale from the prepared noise map."""
    noise = np.arange(12, dtype=float).reshape(3, 4) + 1.0
    calibration = _calibration()
    _install_calibration(monkeypatch, calibration)

    state = prepare_real_fit_likelihood_state(
        _correlated_config(),
        _image_products(noise=noise),
    )

    np.testing.assert_array_equal(state.noise, noise)
    np.testing.assert_array_equal(
        state.correlation_kernel,
        calibration.correlation_kernel,
    )
    assert state.metadata["marginal_model"] == "prepared_sky_scale"
    assert (
        state.metadata["correlation_estimator"]
        == "masked_sky_autocovariance_v1"
    )
    assert state.metadata["calibration_schema_version"] == 2
    assert state.metadata["fit_statistic"] == "mahalanobis_chi2"
    assert state.metadata["residual_diagnostic"] == "marginal_standardized"
    assert state.metadata["normalization_complete"] is False


def test_correlated_gaussian_rejects_a_noise_map_of_another_footprint(
    monkeypatch,
):
    """The prepared noise map must cover the prepared observation."""
    _install_calibration(monkeypatch, _calibration())

    with pytest.raises(
        ValueError,
        match="prepared observation footprint",
    ):
        prepare_real_fit_likelihood_state(
            _correlated_config(),
            _image_products(noise=np.ones((2, 2))),
        )


def test_correlated_gaussian_rejects_estimator_settings(
    monkeypatch,
):
    """AD-047 fixes the estimator, so the configuration cannot set it."""
    _install_calibration(monkeypatch, _calibration())

    with pytest.raises(ValueError, match="AD-047"):
        prepare_real_fit_likelihood_state(
            _correlated_config(calibration={"max_lag": 20}),
            _image_products(),
        )


def test_correlated_gaussian_requires_a_declared_gain(
    monkeypatch,
):
    """AD-047 gives the effective gain no default."""
    _install_calibration(monkeypatch, _calibration())
    config = _correlated_config()
    del config["fit"]["likelihood"]["effective_gain"]

    with pytest.raises(
        ValueError,
        match="requires fit.likelihood.effective_gain",
    ):
        prepare_real_fit_likelihood_state(config, _image_products())


def test_correlated_gaussian_takes_the_gain_from_the_measured_sky(
    monkeypatch,
):
    """The measured sky gives the gain as its level over its variance."""
    _install_calibration(monkeypatch, _calibration())
    products = _image_products()
    products.background = 320.0
    config = _correlated_config(effective_gain={"source": "measured_sky"})

    state = prepare_real_fit_likelihood_state(config, products)

    assert state.effective_gain.value == pytest.approx(320.0 / 2.0**2)
    assert (
        state.metadata["marginal_model"]
        == "prepared_sky_scale_with_source_variance"
    )
    assert state.metadata["effective_gain"]["sky_level"] == 320.0

    products.background = 0.5

    with pytest.raises(ValueError, match="below the 10"):
        prepare_real_fit_likelihood_state(config, products)


def test_independent_gaussian_rejects_an_effective_gain():
    """Only the correlated Gaussian of AD-047 models the source variance."""
    config = {
        "fit": {
            "likelihood": {
                "type": "gaussian",
                "effective_gain": {"source": "absent"},
            },
        },
    }

    with pytest.raises(ValueError, match="belongs to the correlated"):
        prepare_real_fit_likelihood_state(config, _image_products())


def test_correlated_gaussian_requires_persisted_calibration(
    monkeypatch,
):
    """Correlated fitting must never silently fall back to diagonal noise."""
    def raise_missing(
        config,
    ):
        del config

        raise FileNotFoundError(
            "missing"
        )

    monkeypatch.setattr(
        "baymorph.pipeline.likelihood_state."
        "load_correlated_noise_calibration",
        raise_missing,
    )

    config = {
        "fit": {
            "likelihood": {
                "type": (
                    "correlated_gaussian"
                ),
                "include_normalization": False,
                "effective_gain": {
                    "source": "absent",
                },
            },
        },
    }

    with pytest.raises(
        FileNotFoundError,
        match="requires a persisted",
    ):
        prepare_real_fit_likelihood_state(
            config,
            _image_products(),
        )


def test_correlated_gaussian_rejects_incomplete_absolute_normalization():
    """Absolute normalization must not be claimed without log|R_V|."""
    config = {
        "fit": {
            "likelihood": {
                "type": (
                    "correlated_gaussian"
                ),
                "include_normalization": True,
            },
        },
    }

    with pytest.raises(
        ValueError,
        match="factorization.type='nearest_neighbor'",
    ):
        prepare_real_fit_likelihood_state(
            config,
            _image_products(),
        )



@pytest.mark.parametrize(
    ("likelihood", "complete"),
    [
        ({"type": "gaussian"}, True),
        ({"type": "gaussian", "include_normalization": False}, False),
        ({"type": "cash"}, False),
        ({"type": "cash", "include_data_term": True}, True),
        ({"type": "poisson_gaussian"}, True),
    ],
)
def test_normalization_metadata_matches_likelihood_options(
    likelihood,
    complete,
):
    """Evidence status must reflect the effective normalization options."""
    state = prepare_real_fit_likelihood_state(
        {"fit": {"likelihood": likelihood}},
        _image_products(),
    )

    assert state.metadata["normalization_complete"] is complete


def test_evidence_gate_rejects_incomplete_observation():
    """An incomplete likelihood must block absolute evidence."""
    state = prepare_real_fit_likelihood_state(
        {
            "fit": {
                "likelihood": {
                    "type": "gaussian",
                    "include_normalization": False,
                },
            },
        },
        _image_products(),
    )

    with pytest.raises(
        ValueError,
        match="completely normalized likelihood",
    ):
        require_complete_likelihood_normalization({"g": state})


_UNNORMALIZED_LIKELIHOOD = {
    "type": "gaussian",
    "include_normalization": False,
}

_REGISTERED_BACKENDS = (
    "dynesty",
    "nautilus",
    "pocomc",
)


@pytest.mark.parametrize(
    "sampler_name",
    _REGISTERED_BACKENDS,
)
def test_real_fit_rejects_incomplete_normalization_before_outputs(
    monkeypatch,
    sampler_name,
):
    """No backend may fit a likelihood without complete normalization."""
    from baymorph.pipeline import real_fit

    resolved = {
        "data": {
            "image_path": "missing.fits",
        },
        "fit": {
            "likelihood": dict(
                _UNNORMALIZED_LIKELIHOOD
            ),
            "sampler": {
                "name": sampler_name,
            },
        },
    }
    image_products = _image_products()
    image_products.cutout_bounds = {
        "xmin": 0,
        "xmax": 4,
        "ymin": 0,
        "ymax": 3,
    }
    created = []

    monkeypatch.setattr(
        real_fit,
        "resolve_config",
        lambda config: resolved,
    )
    monkeypatch.setattr(
        real_fit,
        "validate_fit_priors",
        lambda config: None,
    )
    monkeypatch.setattr(
        real_fit,
        "load_image_preparation_products",
        lambda config: image_products,
    )
    monkeypatch.setattr(
        real_fit,
        "_load_psf_from_config",
        lambda config: object(),
    )
    monkeypatch.setattr(
        "baymorph.pipeline.real_fit.Observation.from_fits",
        lambda path: None,
    )
    monkeypatch.setattr(
        real_fit,
        "_target_reference_coordinates",
        lambda **kwargs: {
            "target_x": 1.0,
            "target_y": 1.0,
        },
    )
    monkeypatch.setattr(
        real_fit,
        "make_output_tree",
        created.append,
    )

    with pytest.raises(
        ValueError,
        match="completely normalized likelihood",
    ):
        real_fit.run_real_fit_from_config({})

    assert created == []


@pytest.mark.parametrize(
    "sampler_name",
    _REGISTERED_BACKENDS,
)
def test_joint_inference_rejects_incomplete_normalization_before_sampling(
    monkeypatch,
    sampler_name,
):
    """No backend may sample a joint likelihood without normalization."""
    from pathlib import Path

    from baymorph.pipeline import multi_observation_fit

    raw_config = {
        "fit": {
            "models": [
                "single_sersic",
            ],
            "sampler": {
                "name": sampler_name,
            },
        },
    }
    state = prepare_real_fit_likelihood_state(
        {
            "fit": {
                "likelihood": dict(
                    _UNNORMALIZED_LIKELIHOOD
                ),
            },
        },
        _image_products(),
    )
    context = SimpleNamespace(
        problem=object(),
        image_products={},
        observation_configs={},
        likelihood_states={
            "g": state,
        },
    )
    events = []

    monkeypatch.setattr(
        multi_observation_fit,
        "_load_multi_observation_config",
        lambda config: (
            raw_config,
            Path("."),
        ),
    )
    monkeypatch.setattr(
        multi_observation_fit,
        "_build_multi_observation_context",
        lambda config, base_dir: context,
    )
    monkeypatch.setattr(
        multi_observation_fit,
        "_estimate_pre_fit_noise_diagnostics",
        lambda **kwargs: events.append(
            "noise_diagnostic"
        ),
    )
    monkeypatch.setattr(
        multi_observation_fit,
        "_run_multi_observation_inference",
        lambda **kwargs: events.append(
            "sampler"
        ),
    )

    with pytest.raises(
        ValueError,
        match="completely normalized likelihood",
    ):
        (
            multi_observation_fit
            .run_multi_observation_inference_from_config({})
        )

    assert events == []


def test_likelihood_noise_override_is_rejected():
    """Prepared noise is authoritative for real-image inference."""
    with pytest.raises(
        ValueError,
        match="prepared-noise product is authoritative",
    ):
        prepare_real_fit_likelihood_state(
            {
                "fit": {
                    "likelihood": {
                        "type": "gaussian",
                        "noise": 9.0,
                    },
                },
            },
            _image_products(),
        )


def test_correlated_logdet_cannot_come_from_yaml():
    """A scalar YAML value cannot certify the masked covariance domain."""
    with pytest.raises(
        ValueError,
        match="configuration-supplied correlation_logdet",
    ):
        prepare_real_fit_likelihood_state(
            {
                "fit": {
                    "likelihood": {
                        "type": "correlated_gaussian",
                        "include_normalization": False,
                        "correlation_logdet": 1.0,
                    },
                },
            },
            _image_products(),
        )
