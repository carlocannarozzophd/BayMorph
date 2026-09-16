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


def _calibration(
    marginal_sigma,
):
    """Return a deterministic correlated-noise calibration."""
    return SimpleNamespace(
        marginal_sigma=np.asarray(
            marginal_sigma,
            dtype=float,
        ),
        correlation_kernel=np.array(
            [
                [0.0, 0.1, 0.0],
                [0.2, 1.0, 0.2],
                [0.0, 0.1, 0.0],
            ]
        ),
        metadata={
            "schema_version": 1,
            "factorization": "C = S R S",
            "correlation_model": (
                "empirical_stationary_2d"
            ),
            "marginal_model": (
                "quadratic_log_sigma"
            ),
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


def test_correlated_gaussian_uses_full_prepared_scale(
    monkeypatch,
):
    """Correlated state must use the calibration on the prepared footprint."""
    marginal_sigma = np.arange(
        12,
        dtype=float,
    ).reshape(
        3,
        4,
    ) + 1.0

    calibration = _calibration(
        marginal_sigma
    )

    monkeypatch.setattr(
        "baymorph.pipeline.likelihood_state."
        "load_correlated_noise_calibration",
        lambda config: calibration,
    )

    monkeypatch.setattr(
        "baymorph.pipeline.likelihood_state."
        "correlated_noise_calibration_path",
        lambda config: (
            "correlated_gaussian_noise.npz"
        ),
    )

    monkeypatch.setattr(
        "baymorph.pipeline.likelihood_state."
        "validate_correlated_noise_calibration_sources",
        lambda *args, **kwargs: None,
    )

    config = {
        "fit": {
            "likelihood": {
                "type": (
                    "correlated_gaussian"
                ),
                "include_normalization": False,
            },
        },
    }

    state = prepare_real_fit_likelihood_state(
        config,
        _image_products(),
    )

    np.testing.assert_array_equal(
        state.noise,
        marginal_sigma,
    )

    np.testing.assert_array_equal(
        state.correlation_kernel,
        calibration.correlation_kernel,
    )

    assert (
        state.metadata[
            "fit_statistic"
        ]
        == "mahalanobis_chi2"
    )

    assert (
        state.metadata[
            "residual_diagnostic"
        ]
        == "marginal_standardized"
    )

    assert (
        state.metadata[
            "normalization_complete"
        ]
        is False
    )


def test_correlated_gaussian_rejects_calibration_shape_mismatch(
    monkeypatch,
):
    """Calibration and prepared observation must have identical footprints."""
    monkeypatch.setattr(
        "baymorph.pipeline.likelihood_state."
        "load_correlated_noise_calibration",
        lambda config: _calibration(
            np.ones(
                (2, 2),
                dtype=float,
            )
        ),
    )

    monkeypatch.setattr(
        "baymorph.pipeline.likelihood_state."
        "validate_correlated_noise_calibration_sources",
        lambda *args, **kwargs: None,
    )

    config = {
        "fit": {
            "likelihood": {
                "type": (
                    "correlated_gaussian"
                ),
                "include_normalization": False,
            },
        },
    }

    with pytest.raises(
        ValueError,
        match="prepared observation footprint",
    ):
        prepare_real_fit_likelihood_state(
            config,
            _image_products(),
        )


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
